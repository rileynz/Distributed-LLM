# Distributed LLM Universal

A lightweight control app for combining compatible Windows and Linux machines
into one `llama.cpp` inference cluster. Model execution stays in official,
hardware-optimized `llama.cpp` binaries. Release 0.3 adds an admin-free worker
that keeps RPC on localhost and reaches the coordinator through an
authenticated outbound TLS tunnel.

## What this build does

- Runs from the same project on Windows x64, Windows ARM64, Linux x64, and
  Linux ARM64.
- Detects OS, architecture, CPU features, logical cores, free RAM, GPUs, LAN
  address, and a short memory/CPU benchmark.
- Discovers coordinators automatically on a LAN.
- Uses a six-digit join code, explicit coordinator approval, and persistent
  per-node tokens.
- Includes a separate admin-free portable worker for Windows and Linux.
- Multiplexes portable-worker RPC through one outbound TLS connection, with a
  pinned coordinator certificate and no inbound worker firewall rule.
- Downloads the matching CPU/Vulkan/CUDA/ROCm backend from official
  `llama.cpp` releases, with SHA-256 verification when GitHub supplies a digest.
  The default release is pinned so an upstream command change cannot silently
  break an existing installer.
- Runs low-power workers through `llama.cpp` RPC with its local tensor cache.
- Keeps unsuitable low-RAM machines connected for monitoring without forcing
  them into the inference path.
- Produces a RAM- and performance-weighted tensor split.
- Shows honest **Fast**, **Should run**, **May be slow**, and **Will not fit**
  model labels.
- Starts an OpenAI-compatible API and `llama.cpp` chat interface.
- Includes a browser dashboard, headless Linux service, Debian package builder,
  Windows installer workflow, and portable-worker release workflow.

It does not require PyTorch, Transformers, Node.js, Docker, or a full model on
each worker.

## Start on Windows or desktop Linux

Install Python 3.10 or newer, extract the project, install the small encrypted
tunnel dependency into your account, then run:

```bash
python -m pip install --user -r requirements.txt
python run.py
```

On Windows you can double-click `Start-Distributed-LLM.cmd`.

First launch asks:

1. **Create a cluster**, or
2. **Join an existing cluster**.

Creating a cluster opens the dashboard and offers to download the appropriate
`llama.cpp` backend. Joining discovers coordinators on the LAN and asks for the
six-digit code. The coordinator must approve the new machine in its dashboard.
Later launches remember the role and start it automatically.

## Admin-free portable worker

Use this when a worker must run without an installer, administrator rights,
`sudo`, a system service, Python, or an inbound firewall rule.

Download the matching archive from GitHub Releases:

```text
Distributed-LLM-Portable-Worker-Windows-x64-0.3.0.zip
Distributed-LLM-Portable-Worker-Linux-x64-0.3.0.tar.gz
Distributed-LLM-Portable-Worker-Linux-arm64-0.3.0.tar.gz
```

Extract the complete archive into a user-writable folder. On Windows,
double-click `DistributedLLM-Worker.exe`. On Linux, run:

```bash
./DistributedLLM-Worker portable-worker \
  --coordinator http://192.168.1.50:7000 \
  --code 123456 \
  --name portable-worker
```

Portable workers bind the native RPC backend only to `127.0.0.1`. They open an
outbound TLS connection to coordinator port `7443`, authenticate with their
approved node token, pin the coordinator certificate fingerprint learned at
enrollment, and carry multiple binary RPC streams through that connection.
The coordinator presents each connected worker to `llama-server` through a
loopback-only proxy endpoint.

Configuration and caches normally stay in the current user's application-data
directory. Pass `--portable-data` to the packaged worker to keep its data in a
folder beside the executable.

The complete portable source and build scripts are in [`portable/`](portable/).
The mode does not bypass device-owner policies or application controls, and it
cannot install missing GPU drivers. See
[`portable/README.md`](portable/README.md) for details.

## Headless Linux

The source version works directly:

```bash
python3 run.py join \
  --coordinator http://192.168.1.50:7000 \
  --code 123456 \
  --name worker-1
```

For a system service:

```bash
sudo bash packaging/linux/install.sh

sudo -u distributed-llm distributed-llm join \
  --coordinator http://192.168.1.50:7000 \
  --code 123456 \
  --name worker-1 \
  --configure-only

sudo systemctl enable --now distributed-llm
sudo systemctl status distributed-llm
journalctl -u distributed-llm -f
```

Approve the worker from the coordinator dashboard after starting it.

To run a headless Linux coordinator instead, use:

```bash
sudo -u distributed-llm distributed-llm setup --configure-only
sudo systemctl enable --now distributed-llm
```

Open `http://SERVER-IP:7000` from another device.

## Dashboard workflow

1. Install the backend if it is not ready.
2. Start all workers and approve pending machines.
3. Review model fit labels and the proposed per-machine allocation.
4. Choose context length, parallel request count, and KV-cache type.
5. Select **Download & run** on a model.
6. When the model reports ready, select **Open chat**.

The custom-model box also accepts any compatible GGUF file already on the
coordinator or a Hugging Face `owner/repository:quantization` reference.

The coordinator downloads the model. Workers receive only the tensors assigned
to their RPC devices and cache them locally. Subsequent model starts can reuse
that RPC cache.

## Included model catalogue

| Model | Format | Approximate weights | Intended use |
|---|---|---:|---|
| Qwen3 0.6B | Q8 | 0.64 GB | Backend test |
| Qwen3 1.7B | Q8 | 1.83 GB | Very weak devices |
| Qwen3 4B | Q4_K_M | 2.50 GB | Low-power default |
| Qwen3 8B | Q4_K_M | 5.03 GB | Medium cluster |
| Qwen3 14B | Q4_K_M | 9.00 GB | Larger mixed cluster |

Fit estimates add model-runtime and context headroom. They intentionally avoid
counting all installed RAM as usable.

## Commands

```bash
python run.py                   # first setup or start saved role
python run.py setup             # reconfigure
python run.py doctor            # hardware and backend report
python run.py status            # short live cluster status
python run.py install-backend   # install official llama.cpp tools
python run.py join --help       # headless worker options
python run.py portable-worker --help
```

Set `DLLM_HOME` to move configuration, logs, backend files, and caches to a
different data directory.

Set `DLLM_ADVERTISE_HOST` when a machine has several network interfaces and
automatic LAN-address selection picks the wrong one. Advanced testers can set
`DLLM_LLAMA_RELEASE=latest` or a specific release tag; the supported default is
intentionally pinned.

## Packaging

Build Debian/Ubuntu and universal Linux installers:

```bash
bash packaging/linux/build-deb.sh
bash packaging/linux/build-run.sh
sudo apt install ./dist/distributed-llm_0.3.0_all.deb
```

The Windows build uses a PyInstaller **one-directory** package to avoid the
slow temporary extraction caused by one-file executables, then wraps it in an
Inno Setup installer:

```powershell
.\packaging\windows\build.ps1 -Version 0.3.0
```

The Windows output is `dist/windows/Distributed-LLM-Setup-0.3.0.exe`. The
included GitHub Actions workflow builds and tests the Windows installer on a
real Windows runner, builds both Linux installer formats, and builds portable
workers for Windows x64, Linux x64, and Linux ARM64. Pushing a `v0.3.0` tag
creates a GitHub Release and attaches the finished installers and workers.
The Windows builders use the exact interpreter selected by
`actions/setup-python` and install dependencies before running tests, even when
the hosted runner has additional Python versions installed.

When replacing an existing GitHub repository, upload the **contents inside**
`Distributed-LLM-Final`, including `.github`, directly to the repository root.
Do not upload `Distributed-LLM-Final` as a nested folder. A correct Windows run
first shows **Create isolated Windows build environment**, Python 3.12, and a
successful `cryptography` import before starting the installer build.

Portable packages can also be built directly on their target operating system:

```powershell
.\portable\windows\build-windows.ps1 -Version 0.3.0
```

```bash
VERSION=0.3.0 bash portable/linux/build-linux.sh
```

## Supported and unsupported machines

The first release targets:

- Windows 10/11 x64
- Windows 11 ARM64
- Debian/Ubuntu-family Linux x64
- Linux ARM64
- Other modern 64-bit Linux distributions through the portable source package

A device with under roughly 1 GB of safe free memory remains monitoring-only.
Useful inference workers normally need at least 2 GB free. Ancient 32-bit
systems are not supported. GPU acceleration also depends on working vendor
drivers and an available matching `llama.cpp` release; CPU is the fallback.

## Security

The management layer validates join codes, requires explicit approval, stores
random node tokens, rate-limits enrollment attempts, bounds JSON request sizes,
and keeps configuration private to the account where the OS allows it.
Portable RPC tunnels use TLS 1.2 or newer, authenticate with the approved node
token, and pin the coordinator's generated certificate fingerprint.

Direct-mode `llama.cpp` RPC remains unencrypted and experimental. Use direct
workers only on a trusted LAN or private VPN and never expose or port-forward
TCP port `50052`. The initial enrollment/dashboard API currently uses HTTP and
is also intended for a trusted LAN. Portable workers do not expose their RPC
port to the network.

## Honest limitations

- More machines primarily add memory and throughput; they do not multiply
  single-response speed.
- A slow worker can still reduce speed when the model needs its RAM.
- Changing the active machine set requires restarting the model.
- If an RPC worker disconnects, the current request can fail and the model must
  be restarted with a new placement.
- Normal source/install packages download backend files at first use. Portable
  release builds bundle the pinned CPU RPC backend; specialized GPU use still
  depends on compatible drivers already installed on the worker.
- Automatic backend installation depends on current official release assets.
  If a specialized GPU package is unavailable, the installer falls back to CPU.
- Windows executables must be built on Windows or by the included Windows CI
  job. The repository ZIP includes their complete reproducible build inputs.

## Tests

```bash
python -m unittest discover -v
```

The tests cover configuration, hardware normalization, allocation, release
selection, secure archive extraction, command construction, discovery, node
approval/authentication, and HTTP control flow.

## GitHub Pages documentation

The complete static website is in [`docs/`](docs/). To publish it from GitHub:

1. Upload this project with `README.md` and `docs/` at the repository root.
2. Open **Settings → Pages** in the GitHub repository.
3. Under **Build and deployment**, select **Deploy from a branch**.
4. Select the `main` branch and the `/docs` folder, then save.

The included `docs/CNAME` keeps the existing `rileybylsma.tech` custom domain.
Remove or edit that file before publishing if the repository should use the
default `github.io` address instead.

The restored `docs/prototype.html`, diagrams, and whitepaper describe the
earlier custom-transformer prototype. The main `docs/index.html` describes the
current Universal release, which uses `llama.cpp` RPC for model execution.
