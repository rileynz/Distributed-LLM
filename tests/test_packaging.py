from __future__ import annotations

import re
import unittest
from pathlib import Path

from dllm import __version__


PROJECT = Path(__file__).resolve().parent.parent


class PackagingTests(unittest.TestCase):
    def test_versions_match(self) -> None:
        pyproject = (PROJECT / "pyproject.toml").read_text(encoding="utf-8")
        installer = (
            PROJECT / "packaging" / "windows" / "installer.iss"
        ).read_text(encoding="utf-8")
        self.assertIn(f'version = "{__version__}"', pyproject)
        self.assertIn(f'#define MyAppVersion "{__version__}"', installer)

    def test_windows_installer_uses_pyinstaller_output(self) -> None:
        installer = (
            PROJECT / "packaging" / "windows" / "installer.iss"
        ).read_text(encoding="utf-8")
        spec = (
            PROJECT / "packaging" / "windows" / "DistributedLLM.spec"
        ).read_text(encoding="utf-8")
        self.assertIn(r"dist\windows\DistributedLLM\*", installer)
        self.assertNotIn(r"build\DistributedLLM\*", installer)
        self.assertIn('hiddenimports=["cryptography"]', spec)

    def test_windows_builds_use_setup_python_and_install_before_tests(self) -> None:
        for relative in (
            "packaging/windows/build.ps1",
            "portable/windows/build-windows.ps1",
        ):
            script = (PROJECT / relative).read_text(encoding="utf-8")
            self.assertIn("$env:pythonLocation", script, relative)
            self.assertLess(
                script.index("-m pip install"),
                script.index("-m unittest"),
                relative,
            )

    def test_linux_desktop_entry(self) -> None:
        desktop = (
            PROJECT / "packaging" / "linux" / "distributed-llm.desktop"
        ).read_text(encoding="utf-8")
        self.assertRegex(desktop, r"(?m)^Exec=distributed-llm$")
        self.assertRegex(desktop, r"(?m)^Terminal=true$")

    def test_release_workflow_has_all_installer_formats(self) -> None:
        workflow = (
            PROJECT / ".github" / "workflows" / "build-installers.yml"
        ).read_text(encoding="utf-8")
        for expected in (
            "Distributed-LLM-Setup-",
            "build-deb.sh",
            "build-run.sh",
            "build-windows.ps1",
            "build-linux.sh",
        ):
            self.assertIn(expected, workflow)
        for script in (
            "packaging/linux/build-deb.sh",
            "packaging/linux/build-run.sh",
            "portable/linux/build-linux.sh",
        ):
            self.assertIn(f"bash {script}", workflow)
            self.assertNotIn(f"./{script}", workflow)
        self.assertIn("python -m venv .windows-build-venv", workflow)
        self.assertIn('"DLLM_PYTHON=$python"', workflow)
        self.assertIn("import cryptography, sys", workflow)
        self.assertIn("Outdated Windows build files", workflow)

    def test_portable_worker_sources_are_in_the_repository(self) -> None:
        for relative in (
            "portable/README.md",
            "portable/worker_entry.py",
            "portable/windows/PortableWorker.spec",
            "portable/windows/build-windows.ps1",
            "portable/linux/build-linux.sh",
        ):
            self.assertTrue((PROJECT / relative).is_file(), relative)


if __name__ == "__main__":
    unittest.main()
