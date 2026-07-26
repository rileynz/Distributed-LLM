import os
import socket
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from dllm import backend, config, tunnel
from dllm.coordinator import CoordinatorState, make_handler
from dllm.http_utils import request_json


HARDWARE = {
    "hostname": "coordinator",
    "os": "Linux",
    "os_version": "test",
    "architecture": "x64",
    "cpu": "Test CPU",
    "logical_cores": 8,
    "cpu_flags": ["avx2"],
    "ram_total_gb": 16.0,
    "ram_free_gb": 12.0,
    "gpus": [],
    "lan_ip": "127.0.0.1",
    "memory_bandwidth_gbps": 20.0,
    "cpu_score": 20_000.0,
}


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.environment = patch.dict(os.environ, {"DLLM_HOME": self.temporary.name})
        self.environment.start()
        self.detect_patch = patch("dllm.coordinator.detect")
        detected = self.detect_patch.start()
        detected.return_value.to_dict.return_value = dict(HARDWARE)
        self.backend_patch = patch(
            "dllm.coordinator.backend.discover",
            return_value=backend.Tools(None, None, None),
        )
        self.backend_patch.start()
        self.cfg = dict(config.DEFAULTS)
        self.cfg.update({
            "configured": True,
            "role": "coordinator",
            "name": "coordinator",
            "cluster_name": "Test",
            "join_code": "123456",
        })
        self.state = CoordinatorState(self.cfg)

    def tearDown(self):
        self.state.close()
        self.backend_patch.stop()
        self.detect_patch.stop()
        self.environment.stop()
        self.temporary.cleanup()

    def _request(self):
        payload = {
            "join_code": "123456",
            "name": "worker",
            "hardware": {**HARDWARE, "hostname": "worker", "ram_free_gb": 6},
            "rpc_host": "127.0.0.1",
            "rpc_port": 50053,
        }
        code, result = self.state.enroll(payload, "127.0.0.1")
        self.assertEqual(code, 202)
        return result

    def test_bad_join_code_is_rejected(self):
        code, _ = self.state.enroll(
            {"join_code": "000000", "hardware": HARDWARE}, "127.0.0.1"
        )
        self.assertEqual(code, 403)

    def test_approval_and_authenticated_heartbeat(self):
        request = self._request()
        self.assertEqual(request["cluster_name"], "Test")
        code, _ = self.state.approve(request["request_id"])
        self.assertEqual(code, 200)
        code, approved = self.state.enrollment_status(request["request_id"])
        self.assertEqual(code, 200)
        code, response = self.state.heartbeat(
            {
                "node_id": approved["node_id"],
                "hardware": HARDWARE,
                "rpc_running": True,
                "rpc_host": "127.0.0.1",
                "rpc_port": 50053,
            },
            approved["node_token"],
        )
        self.assertEqual(code, 200)
        self.assertTrue(response["desired"]["rpc_running"])

    def test_repeated_request_replaces_old_pending_entry(self):
        first = self._request()
        payload = {
            "join_code": "123456",
            "node_id": first["node_id"],
            "name": "worker",
            "hardware": HARDWARE,
            "rpc_host": "127.0.0.1",
            "rpc_port": 50053,
        }
        code, second = self.state.enroll(payload, "127.0.0.2")
        self.assertEqual(code, 202)
        self.assertNotEqual(first["request_id"], second["request_id"])
        self.assertEqual(len(self.state.data["pending"]), 1)

    def test_http_management_requires_admin_code(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.state))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            code, payload = request_json(f"{base}/api/status")
            self.assertEqual(code, 200)
            self.assertEqual(payload["join_code"], "••••••")
            code, _ = request_json(
                f"{base}/api/model/stop", method="POST", payload={}
            )
            self.assertEqual(code, 403)
            code, _ = request_json(
                f"{base}/api/model/stop",
                method="POST",
                payload={},
                headers={"X-DLLM-Admin": "123456"},
            )
            self.assertEqual(code, 200)
            code, payload = request_json(f"{base}/api/local/join-code")
            self.assertEqual(code, 200)
            self.assertEqual(payload["join_code"], "123456")
        finally:
            server.shutdown()
            server.server_close()

    def test_custom_model_input_is_validated(self):
        code, payload = self.state.start_model(
            "custom", {"source": "not a repo", "size_gb": 4}
        )
        self.assertEqual(code, 400)
        self.assertIn("Hugging Face", payload["error"])

    def test_portable_worker_receives_tunnel_identity_and_loopback_endpoint(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            self.state.cfg["tunnel_port"] = probe.getsockname()[1]
        self.state.start_tunnel_broker()
        payload = {
            "join_code": "123456",
            "name": "portable",
            "hardware": {**HARDWARE, "hostname": "portable"},
            "rpc_port": 50052,
            "worker_connection_mode": "outbound",
        }
        code, pending = self.state.enroll(payload, "127.0.0.1")
        self.assertEqual(code, 202)
        self.state.approve(pending["request_id"])
        code, approved = self.state.enrollment_status(pending["request_id"])
        self.assertEqual(code, 200)
        self.assertEqual(approved["tunnel_port"], self.state.cfg["tunnel_port"])
        self.assertEqual(len(approved["tunnel_fingerprint"]), 64)

        client = tunnel.TunnelClient(
            "127.0.0.1",
            approved["tunnel_port"],
            approved["node_id"],
            approved["node_token"],
            approved["tunnel_fingerprint"],
            1,
        )
        client.start()
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if self.state.tunnel_broker.connected(approved["node_id"]):
                    break
                time.sleep(0.05)
            code, _ = self.state.heartbeat(
                {
                    "node_id": approved["node_id"],
                    "hardware": HARDWARE,
                    "rpc_running": True,
                    "rpc_port": 50052,
                    "worker_connection_mode": "outbound",
                    "tunnel_running": True,
                },
                approved["node_token"],
            )
            self.assertEqual(code, 200)
            worker = next(
                item for item in self.state.all_compute_nodes()
                if item["node_id"] == approved["node_id"]
            )
            self.assertTrue(worker["rpc_ready"])
            self.assertEqual(worker["rpc_host"], "127.0.0.1")
            self.assertNotEqual(worker["rpc_port"], 50052)
        finally:
            client.close()
