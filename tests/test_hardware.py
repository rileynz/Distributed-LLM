import unittest

from dllm.hardware import normalized_architecture


class HardwareTests(unittest.TestCase):
    def test_architecture_aliases(self):
        for value in ("AMD64", "x86_64", "x64"):
            self.assertEqual(normalized_architecture(value), "x64")
        for value in ("aarch64", "ARM64"):
            self.assertEqual(normalized_architecture(value), "arm64")

