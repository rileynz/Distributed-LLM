Distributed LLM Universal 0.3.0 — Windows
================================================

Run Distributed-LLM-Setup-0.3.0.exe and follow the installer.

The installer:
  - installs a private Python runtime bundled by PyInstaller;
  - adds a Start menu shortcut;
  - can add a desktop shortcut;
  - can optionally start the app when you sign in; and
  - provides a normal Windows uninstaller.

On first launch, choose Create a cluster on one computer. On every other
computer, launch the app, choose Join, enter the six-digit code, and approve
the machine in the coordinator dashboard.

The installer is not code-signed. Windows SmartScreen may show an
"unrecognized app" warning until the project obtains a trusted signing
certificate. Only use installers downloaded from the project's official
release and compare the SHA-256 checksum.

The inference backend and GGUF models are downloaded after installation.
Use llama.cpp RPC only on a trusted private LAN or VPN. Never expose TCP port
50052 to the internet.
