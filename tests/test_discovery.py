import time
import unittest

from dllm.discovery import DiscoveryResponder, discover


class DiscoveryTests(unittest.TestCase):
    def test_local_discovery(self):
        responder = DiscoveryResponder(
            "Test cluster", "http://127.0.0.1:7000", "test-cluster-id"
        )
        responder.start()
        try:
            time.sleep(0.1)
            found = discover(timeout=0.8)
            self.assertTrue(any(item.cluster_id == "test-cluster-id" for item in found))
        finally:
            responder.stop()

