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
        self.assertIn(r"dist\windows\DistributedLLM\*", installer)
        self.assertNotIn(r"build\DistributedLLM\*", installer)

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
