from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass


DISCOVERY_PORT = 47991
QUERY = b"DLLM_DISCOVER_V1"


@dataclass(frozen=True)
class Cluster:
    name: str
    url: str
    cluster_id: str
    address: str


class DiscoveryResponder:
    def __init__(self, name: str, url: str, cluster_id: str):
        self.payload = json.dumps({
            "protocol": 1,
            "name": name,
            "url": url,
            "cluster_id": cluster_id,
        }).encode("utf-8")
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def _serve(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", DISCOVERY_PORT))
            sock.settimeout(0.5)
            while not self.stop_event.is_set():
                try:
                    data, address = sock.recvfrom(2048)
                    if data == QUERY:
                        sock.sendto(self.payload, address)
                except socket.timeout:
                    continue
                except OSError:
                    break
        finally:
            sock.close()


def discover(timeout: float = 2.0) -> list[Cluster]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(0.2)
    results: dict[str, Cluster] = {}
    try:
        for address in ("255.255.255.255", "127.0.0.1"):
            try:
                sock.sendto(QUERY, (address, DISCOVERY_PORT))
            except OSError:
                pass
        deadline = time.monotonic() + max(0.1, timeout)
        while time.monotonic() < deadline:
            try:
                data, source = sock.recvfrom(4096)
                payload = json.loads(data.decode("utf-8"))
                if payload.get("protocol") != 1:
                    continue
                cluster = Cluster(
                    name=str(payload["name"]),
                    url=str(payload["url"]),
                    cluster_id=str(payload["cluster_id"]),
                    address=source[0],
                )
                results[cluster.cluster_id] = cluster
            except socket.timeout:
                continue
            except (OSError, ValueError, KeyError, UnicodeDecodeError):
                continue
    finally:
        sock.close()
    return sorted(results.values(), key=lambda item: (item.name, item.url))

