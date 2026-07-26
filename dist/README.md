# Release files

This directory contains the locally built release artifacts for version 0.3.0.

## Included in the replacement-repository ZIP

- `distributed-llm_0.3.0_all.deb` — Debian/Ubuntu installer.
- `Distributed-LLM-Universal-0.3.0-Linux.run` — self-extracting Linux installer.
- `portable/Distributed-LLM-Portable-Worker-Linux-x64-0.3.0.tar.gz` — verified
  admin-free Linux x64 worker with its pinned CPU `llama.cpp` backend.
- SHA-256 checksum files for the included artifacts.

## Windows release files

Windows executables must be produced on a Windows build machine. This
repository includes the complete PyInstaller and Inno Setup sources plus a
GitHub Actions workflow that builds:

- `Distributed-LLM-Setup-0.3.0.exe`
- `Distributed-LLM-Portable-Worker-Windows-x64-0.3.0.zip`

Open the repository's **Actions** tab, select **Build installers**, and choose
**Run workflow**. Pushing a `v0.3.0` tag also builds every format and attaches
the results to a GitHub Release.

The portable worker does not request administrator privileges, install a
service, add a firewall rule, or need Python. It still respects operating
system and organization security policies and cannot install missing drivers.
