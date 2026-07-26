import io
import os
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from dllm import backend


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.release = {
            "tag_name": "b9999",
            "assets": [
                {
                    "name": "llama-b9999-bin-win-cpu-x64.zip",
                    "browser_download_url": "https://example/cpu.zip",
                    "size": 10,
                },
                {
                    "name": "llama-b9999-bin-win-vulkan-x64.zip",
                    "browser_download_url": "https://example/vulkan.zip",
                    "size": 12,
                },
                {
                    "name": "llama-b9999-bin-ubuntu-x64.tar.gz",
                    "browser_download_url": "https://example/linux.tgz",
                    "size": 15,
                },
            ],
        }

    def test_selects_matching_windows_variant(self):
        with patch("dllm.backend.platform.system", return_value="Windows"), patch(
            "dllm.backend.normalized_architecture", return_value="x64"
        ):
            selected = backend.select_asset(self.release, "vulkan")
        self.assertIn("vulkan", selected["name"])

    def test_specialized_variant_falls_back_to_cpu(self):
        release = {
            "assets": [self.release["assets"][0]],
            "tag_name": "b9999",
        }
        with patch("dllm.backend.platform.system", return_value="Windows"), patch(
            "dllm.backend.normalized_architecture", return_value="x64"
        ):
            selected = backend.select_asset(release, "cuda")
        self.assertIn("cpu", selected["name"])

    def test_generic_linux_package_beats_specialized_packages_for_cpu(self):
        release = {
            "tag_name": "b9999",
            "assets": [
                {
                    "name": "llama-b9999-bin-ubuntu-rocm-7.2-x64.tar.gz",
                    "browser_download_url": "https://example/rocm.tgz",
                    "size": 1000,
                },
                {
                    "name": "llama-b9999-bin-ubuntu-x64.tar.gz",
                    "browser_download_url": "https://example/cpu.tgz",
                    "size": 100,
                },
            ],
        }
        with patch("dllm.backend.platform.system", return_value="Linux"), patch(
            "dllm.backend.normalized_architecture", return_value="x64"
        ):
            selected = backend.select_asset(release, "cpu")
        self.assertEqual(selected["name"], "llama-b9999-bin-ubuntu-x64.tar.gz")

    def test_server_command_with_rpc_split(self):
        tools = backend.Tools("/bin/llama-server", "/bin/rpc-server", None)
        command = backend.server_command(
            tools,
            "Qwen/Qwen3-4B-GGUF:Q4_K_M",
            ["127.0.0.1:50052", "192.168.1.20:50052"],
            [0.7, 0.3],
        )
        self.assertIn("-hf", command)
        self.assertIn("--rpc", command)
        self.assertIn("0.700000,0.300000", command)

    def test_split_length_must_match(self):
        tools = backend.Tools("/bin/llama-server", "/bin/rpc-server", None)
        with self.assertRaises(ValueError):
            backend.server_command(tools, "repo:model", ["host:1"], [0.5, 0.5])

    def test_zip_path_traversal_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "bad.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../escape.txt", "bad")
            with self.assertRaises(RuntimeError):
                backend._safe_extract(archive, Path(directory) / "out")

    def test_valid_archive_extracts(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "good.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("bin/rpc-server", "binary")
            target = Path(directory) / "out"
            backend._safe_extract(archive, target)
            self.assertEqual((target / "bin" / "rpc-server").read_text(), "binary")

    def test_tar_link_escaping_destination_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "bad-link.tar"
            with tarfile.open(archive, "w") as output:
                member = tarfile.TarInfo("bin/rpc-server")
                member.type = tarfile.SYMTYPE
                member.linkname = "../../outside"
                output.addfile(member)
            with self.assertRaises(RuntimeError):
                backend._safe_extract(archive, Path(directory) / "out")
