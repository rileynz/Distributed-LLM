from __future__ import annotations

import datetime
import hashlib
import hmac
import ipaddress
import json
import os
import socket
import ssl
import struct
import threading
import time
from pathlib import Path
from typing import Callable


FRAME_AUTH = 1
FRAME_AUTH_OK = 2
FRAME_OPEN = 3
FRAME_DATA = 4
FRAME_CLOSE = 5
FRAME_PING = 6
FRAME_PONG = 7
FRAME_ERROR = 8

HEADER = struct.Struct("!BII")
MAX_FRAME_BYTES = 1024 * 1024
READ_CHUNK_BYTES = 64 * 1024


class TunnelError(RuntimeError):
    pass


def _receive_exact(connection: socket.socket, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        chunk = connection.recv(length - len(chunks))
        if not chunk:
            raise EOFError("tunnel connection closed")
        chunks.extend(chunk)
    return bytes(chunks)


def receive_frame(connection: socket.socket) -> tuple[int, int, bytes]:
    kind, stream_id, length = HEADER.unpack(_receive_exact(connection, HEADER.size))
    if length > MAX_FRAME_BYTES:
        raise TunnelError("tunnel frame is too large")
    return kind, stream_id, _receive_exact(connection, length) if length else b""


def send_frame(
    connection: socket.socket,
    lock: threading.Lock,
    kind: int,
    stream_id: int = 0,
    payload: bytes = b"",
) -> None:
    if len(payload) > MAX_FRAME_BYTES:
        raise TunnelError("tunnel frame is too large")
    packet = HEADER.pack(kind, stream_id, len(payload)) + payload
    with lock:
        connection.sendall(packet)


def certificate_fingerprint(certificate_der: bytes) -> str:
    return hashlib.sha256(certificate_der).hexdigest()


def ensure_tls_identity(
    certificate_path: Path,
    private_key_path: Path,
    hostnames: tuple[str, ...] = (),
) -> str:
    certificate_path.parent.mkdir(parents=True, exist_ok=True)
    private_key_path.parent.mkdir(parents=True, exist_ok=True)
    if not certificate_path.is_file() or not private_key_path.is_file():
        try:
            from cryptography import x509
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import ec
            from cryptography.x509.oid import NameOID
        except ImportError as exc:
            raise TunnelError(
                "Encrypted portable-worker tunnels require the cryptography package"
            ) from exc

        key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "Distributed LLM Coordinator")
        ])
        now = datetime.datetime.now(datetime.timezone.utc)
        san_entries: list[x509.GeneralName] = [
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        ]
        for hostname in hostnames:
            try:
                san_entries.append(x509.IPAddress(ipaddress.ip_address(hostname)))
            except ValueError:
                if hostname:
                    san_entries.append(x509.DNSName(hostname))
        certificate = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=3650))
            .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
            .sign(key, hashes.SHA256())
        )
        private_key_path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
        try:
            os.chmod(private_key_path, 0o600)
            os.chmod(certificate_path, 0o644)
        except OSError:
            pass

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(certificate_path), str(private_key_path))
    certificate_pem = certificate_path.read_text(encoding="ascii")
    certificate_der = ssl.PEM_cert_to_DER_cert(certificate_pem)
    return certificate_fingerprint(certificate_der)


class _BrokerSession:
    def __init__(
        self,
        broker: "TunnelBroker",
        connection: ssl.SSLSocket,
        node_id: str,
    ):
        self.broker = broker
        self.connection = connection
        self.node_id = node_id
        self.send_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.streams: dict[int, socket.socket] = {}
        self.streams_lock = threading.RLock()
        self.next_stream_id = 1
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(16)
        self.listener.settimeout(1.0)
        self.endpoint = self.listener.getsockname()

    def start(self) -> None:
        threading.Thread(
            target=self._accept_local_connections,
            name=f"tunnel-proxy-{self.node_id}",
            daemon=True,
        ).start()
        try:
            while not self.stop_event.is_set():
                kind, stream_id, payload = receive_frame(self.connection)
                if kind == FRAME_DATA:
                    with self.streams_lock:
                        local = self.streams.get(stream_id)
                    if local:
                        local.sendall(payload)
                elif kind == FRAME_CLOSE:
                    self._close_stream(stream_id, notify=False)
                elif kind == FRAME_PING:
                    send_frame(self.connection, self.send_lock, FRAME_PONG)
                elif kind == FRAME_ERROR:
                    raise TunnelError(payload.decode("utf-8", errors="replace"))
                else:
                    raise TunnelError(f"unexpected tunnel frame {kind}")
        finally:
            self.close()

    def _accept_local_connections(self) -> None:
        while not self.stop_event.is_set():
            try:
                local, _address = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with self.streams_lock:
                stream_id = self.next_stream_id
                self.next_stream_id += 1
                self.streams[stream_id] = local
            try:
                send_frame(self.connection, self.send_lock, FRAME_OPEN, stream_id)
            except OSError:
                self._close_stream(stream_id, notify=False)
                break
            threading.Thread(
                target=self._pump_local_to_worker,
                args=(stream_id, local),
                name=f"tunnel-up-{self.node_id}-{stream_id}",
                daemon=True,
            ).start()

    def _pump_local_to_worker(self, stream_id: int, local: socket.socket) -> None:
        try:
            while not self.stop_event.is_set():
                payload = local.recv(READ_CHUNK_BYTES)
                if not payload:
                    break
                send_frame(
                    self.connection,
                    self.send_lock,
                    FRAME_DATA,
                    stream_id,
                    payload,
                )
        except (OSError, TunnelError):
            pass
        finally:
            self._close_stream(stream_id, notify=True)

    def _close_stream(self, stream_id: int, *, notify: bool) -> None:
        with self.streams_lock:
            local = self.streams.pop(stream_id, None)
        if local:
            try:
                local.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            local.close()
        if notify and not self.stop_event.is_set():
            try:
                send_frame(self.connection, self.send_lock, FRAME_CLOSE, stream_id)
            except OSError:
                pass

    def close(self) -> None:
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        try:
            self.listener.close()
        except OSError:
            pass
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.connection.close()
        with self.streams_lock:
            stream_ids = list(self.streams)
        for stream_id in stream_ids:
            self._close_stream(stream_id, notify=False)
        self.broker._remove(self)


class TunnelBroker:
    def __init__(
        self,
        host: str,
        port: int,
        certificate_path: Path,
        private_key_path: Path,
        authenticate: Callable[[str, str], bool],
        advertised_hosts: tuple[str, ...] = (),
    ):
        self.host = host
        self.port = port
        self.certificate_path = certificate_path
        self.private_key_path = private_key_path
        self.authenticate = authenticate
        self.advertised_hosts = advertised_hosts
        self.fingerprint = ""
        self.stop_event = threading.Event()
        self.sessions: dict[str, _BrokerSession] = {}
        self.sessions_lock = threading.RLock()
        self.listener: socket.socket | None = None
        self.thread: threading.Thread | None = None
        self.last_error = ""

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.fingerprint = ensure_tls_identity(
            self.certificate_path,
            self.private_key_path,
            self.advertised_hosts,
        )
        self.stop_event.clear()
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind((self.host, self.port))
        self.listener.listen(32)
        self.listener.settimeout(1.0)
        self.port = int(self.listener.getsockname()[1])
        self.thread = threading.Thread(
            target=self._accept_loop,
            name="portable-tunnel-broker",
            daemon=True,
        )
        self.thread.start()

    def _accept_loop(self) -> None:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(str(self.certificate_path), str(self.private_key_path))
        while not self.stop_event.is_set():
            try:
                raw, _address = self.listener.accept() if self.listener else (None, None)
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(
                target=self._authenticate_connection,
                args=(context, raw),
                daemon=True,
            ).start()

    def _authenticate_connection(
        self,
        context: ssl.SSLContext,
        raw: socket.socket,
    ) -> None:
        connection: ssl.SSLSocket | None = None
        lock = threading.Lock()
        try:
            raw.settimeout(15)
            connection = context.wrap_socket(raw, server_side=True)
            kind, stream_id, payload = receive_frame(connection)
            if kind != FRAME_AUTH or stream_id != 0 or len(payload) > 16_384:
                raise TunnelError("authentication frame required")
            request = json.loads(payload.decode("utf-8"))
            node_id = str(request.get("node_id", ""))
            token = str(request.get("node_token", ""))
            if not node_id or not token or not self.authenticate(node_id, token):
                send_frame(connection, lock, FRAME_ERROR, payload=b"authentication failed")
                return
            session = _BrokerSession(self, connection, node_id)
            with self.sessions_lock:
                old = self.sessions.get(node_id)
                self.sessions[node_id] = session
            if old:
                old.close()
            connection.settimeout(None)
            send_frame(
                connection,
                session.send_lock,
                FRAME_AUTH_OK,
                payload=json.dumps({
                    "endpoint_host": session.endpoint[0],
                    "endpoint_port": session.endpoint[1],
                }).encode("utf-8"),
            )
            session.start()
        except Exception as exc:
            self.last_error = str(exc)
        finally:
            if connection and not connection._closed:
                connection.close()
            elif not connection:
                raw.close()

    def _remove(self, session: _BrokerSession) -> None:
        with self.sessions_lock:
            if self.sessions.get(session.node_id) is session:
                self.sessions.pop(session.node_id, None)

    def endpoint(self, node_id: str) -> tuple[str, int] | None:
        with self.sessions_lock:
            session = self.sessions.get(node_id)
            if not session or session.stop_event.is_set():
                return None
            return str(session.endpoint[0]), int(session.endpoint[1])

    def connected(self, node_id: str) -> bool:
        return self.endpoint(node_id) is not None

    def close(self) -> None:
        self.stop_event.set()
        if self.listener:
            try:
                self.listener.close()
            except OSError:
                pass
        with self.sessions_lock:
            sessions = list(self.sessions.values())
        for session in sessions:
            session.close()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)


class TunnelClient:
    def __init__(
        self,
        coordinator_host: str,
        coordinator_port: int,
        node_id: str,
        node_token: str,
        expected_fingerprint: str,
        local_rpc_port: int,
    ):
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.node_id = node_id
        self.node_token = node_token
        self.expected_fingerprint = expected_fingerprint.lower().replace(":", "")
        self.local_rpc_port = local_rpc_port
        self.stop_event = threading.Event()
        self.connected_event = threading.Event()
        self.send_lock = threading.Lock()
        self.streams: dict[int, socket.socket] = {}
        self.streams_lock = threading.RLock()
        self.connection: ssl.SSLSocket | None = None
        self.thread: threading.Thread | None = None
        self.last_error = ""

    @property
    def running(self) -> bool:
        return self.connected_event.is_set()

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        if not self.expected_fingerprint:
            raise TunnelError("coordinator tunnel fingerprint is missing")
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run,
            name="portable-tunnel-client",
            daemon=True,
        )
        self.thread.start()

    def _run(self) -> None:
        failures = 0
        while not self.stop_event.is_set():
            try:
                self._connect_once()
                failures = 0
            except Exception as exc:
                self.last_error = str(exc)
                failures += 1
            self.connected_event.clear()
            self._close_streams()
            if not self.stop_event.is_set():
                self.stop_event.wait(min(30, 2 ** min(failures, 5)))

    def _connect_once(self) -> None:
        connection: ssl.SSLSocket | None = None
        try:
            raw = socket.create_connection(
                (self.coordinator_host, self.coordinator_port),
                timeout=15,
            )
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            connection = context.wrap_socket(raw, server_hostname=self.coordinator_host)
            actual = certificate_fingerprint(connection.getpeercert(binary_form=True))
            if not hmac.compare_digest(actual, self.expected_fingerprint):
                raise TunnelError("coordinator tunnel certificate fingerprint changed")
            connection.settimeout(None)
            self.connection = connection
            send_frame(
                connection,
                self.send_lock,
                FRAME_AUTH,
                payload=json.dumps({
                    "node_id": self.node_id,
                    "node_token": self.node_token,
                }).encode("utf-8"),
            )
            kind, stream_id, payload = receive_frame(connection)
            if kind != FRAME_AUTH_OK or stream_id != 0:
                message = payload.decode("utf-8", errors="replace")
                raise TunnelError(message or "coordinator rejected tunnel")
            self.connected_event.set()
            while not self.stop_event.is_set():
                kind, stream_id, payload = receive_frame(connection)
                if kind == FRAME_OPEN:
                    self._open_stream(stream_id)
                elif kind == FRAME_DATA:
                    with self.streams_lock:
                        local = self.streams.get(stream_id)
                    if local:
                        local.sendall(payload)
                elif kind == FRAME_CLOSE:
                    self._close_stream(stream_id, notify=False)
                elif kind == FRAME_PING:
                    send_frame(connection, self.send_lock, FRAME_PONG)
                elif kind == FRAME_ERROR:
                    raise TunnelError(payload.decode("utf-8", errors="replace"))
                else:
                    raise TunnelError(f"unexpected tunnel frame {kind}")
        finally:
            if self.connection is connection:
                self.connection = None
            if connection:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                connection.close()

    def _open_stream(self, stream_id: int) -> None:
        try:
            local = socket.create_connection(("127.0.0.1", self.local_rpc_port), timeout=8)
            local.settimeout(None)
        except OSError as exc:
            if self.connection:
                send_frame(
                    self.connection,
                    self.send_lock,
                    FRAME_ERROR,
                    stream_id,
                    str(exc).encode("utf-8"),
                )
                send_frame(self.connection, self.send_lock, FRAME_CLOSE, stream_id)
            return
        with self.streams_lock:
            self.streams[stream_id] = local
        threading.Thread(
            target=self._pump_rpc_to_coordinator,
            args=(stream_id, local),
            name=f"portable-rpc-{stream_id}",
            daemon=True,
        ).start()

    def _pump_rpc_to_coordinator(self, stream_id: int, local: socket.socket) -> None:
        try:
            while not self.stop_event.is_set():
                payload = local.recv(READ_CHUNK_BYTES)
                if not payload:
                    break
                connection = self.connection
                if not connection:
                    break
                send_frame(connection, self.send_lock, FRAME_DATA, stream_id, payload)
        except (OSError, TunnelError):
            pass
        finally:
            self._close_stream(stream_id, notify=True)

    def _close_stream(self, stream_id: int, *, notify: bool) -> None:
        with self.streams_lock:
            local = self.streams.pop(stream_id, None)
        if local:
            try:
                local.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            local.close()
        if notify and self.connection and self.connected_event.is_set():
            try:
                send_frame(self.connection, self.send_lock, FRAME_CLOSE, stream_id)
            except OSError:
                pass

    def _close_streams(self) -> None:
        with self.streams_lock:
            stream_ids = list(self.streams)
        for stream_id in stream_ids:
            self._close_stream(stream_id, notify=False)

    def close(self) -> None:
        self.stop_event.set()
        self.connected_event.clear()
        connection = self.connection
        self.connection = None
        if connection:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        self._close_streams()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)
