# Building the Windows installer

The Windows installer must be compiled on Windows because PyInstaller packages
the Python runtime for the operating system where it runs.

## Automatic GitHub build

1. Upload this project to GitHub.
2. Open the repository's **Actions** tab.
3. Select **Build installers**.
4. Select **Run workflow**.
5. When it finishes, download the `windows-installer` artifact.

Pushing a version tag such as `v0.3.0` builds both operating-system installers
and creates a GitHub Release containing:

- `Distributed-LLM-Setup-0.3.0.exe`
- `distributed-llm_0.3.0_all.deb`
- `Distributed-LLM-Universal-0.3.0-Linux.run`
- `Distributed-LLM-Portable-Worker-Windows-x64-0.3.0.zip`
- `Distributed-LLM-Portable-Worker-Linux-x64-0.3.0.tar.gz`
- `Distributed-LLM-Portable-Worker-Linux-arm64-0.3.0.tar.gz`
- SHA-256 checksum files

## Build on a Windows computer

Install Python 3.10 or newer and Inno Setup 6, open PowerShell in the repository,
and run:

```powershell
.\packaging\windows\build.ps1 -Version 0.3.0
```

The completed installer is written to:

```text
dist\windows\Distributed-LLM-Setup-0.3.0.exe
```

The installer is currently unsigned. Windows SmartScreen can therefore show an
unrecognized-app warning even when the checksum is correct.
