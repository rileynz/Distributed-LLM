import unittest

from dllm.allocator import fit_label, plan


def node(name, ram, speed, bandwidth, *, local=False, ready=True):
    return {
        "node_id": name,
        "name": name,
        "local": local,
        "approved": True,
        "rpc_ready": ready,
        "rtt_ms": 0.8,
        "hardware": {
            "ram_free_gb": ram,
            "logical_cores": 4,
            "cpu_score": speed,
            "memory_bandwidth_gbps": bandwidth,
            "gpus": [],
        },
    }


class AllocatorTests(unittest.TestCase):
    def test_mixed_ram_cluster_fits_14b(self):
        result = plan([
            node("32gb", 29, 30_000, 20, local=True),
            node("8gb", 6.5, 18_000, 12),
            node("4gb", 2.5, 8_000, 6),
        ], 9.0)
        self.assertTrue(result["fits"])
        self.assertIn(fit_label(result), {"Fast", "Should run", "May be slow"})
        included = [item for item in result["placements"] if item["included"]]
        self.assertAlmostEqual(sum(item["weight"] for item in included), 1.0, places=5)

    def test_tiny_slow_node_is_monitoring_only_when_not_needed(self):
        result = plan([
            node("fast", 16, 50_000, 30, local=True),
            node("tiny", 3.0, 500, 0.5),
        ], 2.5)
        tiny = next(item for item in result["placements"] if item["node_id"] == "tiny")
        self.assertFalse(tiny["included"])
        self.assertIn("not needed", tiny["reason"].lower())

    def test_unready_rpc_node_is_excluded(self):
        result = plan([
            node("local", 4, 20_000, 10, local=True),
            node("worker", 16, 20_000, 10, ready=False),
        ], 9)
        self.assertFalse(result["fits"])
        worker = next(item for item in result["placements"] if item["node_id"] == "worker")
        self.assertEqual(worker["reason"], "RPC backend is not ready")

    def test_low_memory_is_safe(self):
        result = plan([node("tiny", 0.8, 10_000, 5, local=True)], 0.6)
        self.assertFalse(result["fits"])
        self.assertFalse(result["placements"][0]["included"])

    def test_weights_never_exceed_safe_capacity(self):
        result = plan([
            node("fast-small", 3.5, 50_000, 30, local=True),
            node("slow-large", 16, 5_000, 8),
        ], 9.0)
        self.assertTrue(result["fits"])
        for placement in result["placements"]:
            if placement["included"]:
                maximum = placement["usable_ram_gb"] / result["estimated_required_gb"]
                self.assertLessEqual(placement["weight"], maximum + 1e-5)
