# Changelog

## 0.3.0

- Added a first-class admin-free portable worker mode for Windows and Linux.
- Added a TLS-encrypted, token-authenticated outbound RPC tunnel so portable
  workers do not need an inbound firewall rule or LAN RPC listener.
- Added coordinator-side loopback proxy endpoints and binary stream
  multiplexing for `llama.cpp` RPC.
- Added coordinator certificate generation and worker-side SHA-256 certificate
  pinning.
- Added user-profile and beside-the-executable portable data modes.
- Added portable Windows x64, Linux x64, and Linux ARM64 packaging workflows
  with pinned CPU backends and SHA-256 release files.
- Added portable-worker documentation, source commands, dashboard connection
  status, and end-to-end authenticated tunnel tests.

## 0.2.0

- Added a complete per-user Windows installer build with bundled Python,
  Start-menu and optional desktop/startup shortcuts, uninstaller metadata,
  packaged startup testing, and SHA-256 output.
- Fixed the Windows installer source path so Inno Setup packages the actual
  PyInstaller application rather than the temporary build directory.
- Added a universal self-extracting Linux `.run` installer.
- Expanded the Debian package with a desktop launcher, application icon,
  documentation, uninstall helper, and hardened systemd settings.
- Added release CI for Windows and both Linux installer formats.

## 0.1.0

- Added one-command coordinator and worker setup.
- Added Windows x64/ARM64 and Linux x64/ARM64 capability detection.
- Added dependency-free dashboard and control API.
- Added LAN discovery with manual-address fallback.
- Added join codes, explicit node approval, token authentication, and rate
  limiting.
- Added official `llama.cpp` release selection and safe archive extraction.
- Added CPU, Vulkan, CUDA, and ROCm package preferences with CPU fallback.
- Added automatic worker RPC startup and local tensor caching.
- Added RAM/performance allocation and automatic exclusion of unhelpful nodes.
- Added Qwen3 0.6B, 1.7B, 4B, 8B, and 14B GGUF catalogue.
- Added OpenAI-compatible model serving through `llama-server`.
- Added headless Linux systemd service, `.deb` builder, portable setup, Windows
  installer recipe, and CI packaging.
