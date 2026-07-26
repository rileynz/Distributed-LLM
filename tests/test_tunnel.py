from __future__ import annotations

import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

from dllm.tunnel import TunnelBroker, TunnelClient


class EchoServer:
    def __init__(self):
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(8)
        self.listener.settimeout(0.5)
        self.port = self.listener.getsockname()[1]
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()

    def run(self):
        while not self.stop_event.is_set():
            try:
                connection, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self.echo, args=(connection,), daemon=True).start()

    @staticmethod
    def echo(connection):
        with connection:
            while True:
                data = connection.recv(65536)
                if not data:
                    return
                connection.sendall(data)

    def close(self):
        self.stop_event.set()
        self.listener.close()
        self.thread.join(timeout=2)


class TunnelTests(unittest.TestCase):
    def test_authenticated_outbound_tunnel_relays_binary_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            echo = EchoServer()
            echo.start()
            broker = TunnelBroker(
                "127.0.0.1",
                0,
                root / "cert.pem",
                root / "key.pem",
                lambda node_id, token: node_id == "worker-1" and token == "secret",
            )
            broker.start()
            client = TunnelClient(
                "127.0.0.1",
                broker.port,
                "worker-1",
                "secret",
                broker.fingerprint,
                echo.port,
            )
            client.start()
            try:
                deadline = time.monotonic() + 8
                endpoint = None
                while time.monotonic() < deadline:
                    endpoint = broker.endpoint("worker-1")
                    if endpoint:
                        break
                    time.sleep(0.05)
                self.assertIsNotNone(endpoint)
                with socket.create_connection(endpoint, timeout=3) as connection:
                    payload = (b"\x00binary-rpc-payload\xff" * 9000)
                    connection.sendall(payload)
                    received = bytearray()
                    while len(received) < len(payload):
                        received.extend(connection.recv(65536))
                    self.assertEqual(bytes(received), payload)
                self.assertTrue(client.running)
            finally:
                client.close()
                broker.close()
                echo.close()

    def test_wrong_token_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            broker = TunnelBroker(
                "127.0.0.1",
                0,
                root / "cert.pem",
                root / "key.pem",
                lambda _node_id, token: token == "right",
            )
            broker.start()
            client = TunnelClient(
                "127.0.0.1",
                broker.port,
                "worker",
                "wrong",
                broker.fingerprint,
                1,
            )
            client.start()
            try:
                time.sleep(0.6)
                self.assertFalse(client.running)
                self.assertIsNone(broker.endpoint("worker"))
            finally:
                client.close()
                broker.close()
