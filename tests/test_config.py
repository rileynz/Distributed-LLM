import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dllm import config


class ConfigTests(unittest.TestCase):
    def test_round_trip_and_private_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "config.json"
            value = dict(config.DEFAULTS)
            value.update({
                "configured": True,
                "role": "coordinator",
                "join_code": "123456",
                "cluster_name": "Test",
            })
            config.save(value, target)
            loaded = config.load(target)
            self.assertEqual(loaded["cluster_name"], "Test")
            self.assertEqual(loaded["join_code"], "123456")
            if os.name != "nt":
                self.assertEqual(target.stat().st_mode & 0o777, 0o600)

    def test_agent_requires_join_code_until_enrolled(self):
        cfg = dict(config.DEFAULTS)
        cfg.update({"configured": True, "role": "agent", "coordinator_url": "http://host:7000"})
        self.assertTrue(any("join code" in error for error in config.validate(cfg)))
        cfg["node_token"] = "saved-token"
        self.assertFalse(any("join code" in error for error in config.validate(cfg)))

    def test_invalid_port_is_rejected(self):
        cfg = dict(config.DEFAULTS)
        cfg["control_port"] = 70000
        self.assertIn("control_port must be between 1 and 65535", config.validate(cfg))

    def test_portable_outbound_worker_is_valid(self):
        cfg = dict(config.DEFAULTS)
        cfg.update({
            "configured": True,
            "role": "agent",
            "coordinator_url": "http://coordinator:7000",
            "join_code": "123456",
            "worker_connection_mode": "outbound",
            "tunnel_port": 7443,
        })
        self.assertEqual(config.validate(cfg), [])

    def test_data_dir_override(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"DLLM_HOME": directory}
        ):
            self.assertEqual(config.data_dir(), Path(directory).resolve())
