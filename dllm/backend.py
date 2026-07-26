from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from . import config
from .hardware import normalized_architecture


DEFAULT_RELEASE_TAG = "b10091"
GITHUB_RELEASE = "https://api.github.com/repos/ggml-org/llama.cpp/releases"


def _executable_names(base: str) -> tuple[str, ...]:
    suffix = ".exe" if os.name == "nt" else ""
    if base == "rpc":
        return tuple(f"{name}{suffix}" for name in (
            "rpc-server", "ggml-rpc-server", "llama-rpc-server"
        ))
    return (f"{base}{suffix}",)


def _find(names: tuple[str, ...], directory: Path | None = None) -> str | None:
    overrides = {
        "llama-server": "DLLM_LLAMA_SERVER",
        "llama-bench": "DLLM_LLAMA_BENCH",
        "rpc-server": "DLLM_RPC_SERVER",
        "ggml-rpc-server": "DLLM_RPC_SERVER",
        "llama-rpc-server": "DLLM_RPC_SERVER",
    }
    for name in names:
        override = os.environ.get(overrides.get(name.removesuffix(".exe"), ""))
        if override and Path(override).is_file():
            return str(Path(override).resolve())
        found = shutil.which(name)
        if found:
            return found
    root = directory or config.backend_dir()
    if root.exists():
        lowered = {name.lower() for name in names}
        for candidate in root.rglob("*"):
            if candidate.is_file() and candidate.name.lower() in lowered:
                return str(candidate.resolve())
    return None


@dataclass(frozen=True)
class Tools:
    server: str | None
    rpc_server: str | None
    bench: str | None

    @property
    def coordinator_ready(self) -> bool:
        return bool(self.server and self.rpc_server)

    @property
    def agent_ready(self) -> bool:
        return bool(self.rpc_server)


def discover(directory: Path | None = None) -> Tools:
    search_directory = directory
    if search_directory is None:
        candidates: list[Path] = []
        frozen_root = getattr(sys, "_MEIPASS", "")
        if frozen_root:
            candidates.append(Path(frozen_root) / "backend")
        candidates.append(Path(sys.executable).resolve().parent / "backend")
        candidates.append(Path(__file__).resolve().parent.parent / "portable" / "runtime" / "backend")
        search_directory = next(
            (candidate for candidate in candidates if candidate.exists()),
            None,
        )
    return Tools(
        server=_find(_executable_names("llama-server"), search_directory),
        rpc_server=_find(_executable_names("rpc"), search_directory),
        bench=_find(_executable_names("llama-bench"), search_directory),
    )


def release_metadata(timeout: float = 15.0) -> dict:
    requested = os.environ.get("DLLM_LLAMA_RELEASE", DEFAULT_RELEASE_TAG).strip()
    endpoint = (
        f"{GITHUB_RELEASE}/latest"
        if requested.lower() == "latest"
        else f"{GITHUB_RELEASE}/tags/{requested}"
    )
    request = urllib.request.Request(
        endpoint,
        headers={"User-Agent": "Distributed-LLM-Universal/0.3"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict) or not isinstance(payload.get("assets"), list):
        raise RuntimeError("GitHub did not return a valid llama.cpp release")
    return payload


def _asset_score(name: str, system: str, architecture: str, variant: str) -> int:
    value = name.lower()
    if not value.endswith((".zip", ".tar.gz", ".tgz")):
        return -10_000
    if any(token in value for token in ("cudart", "sha256", "source")):
        return -10_000
    score = 0
    if system == "windows":
        score += 30 if "win" in value else -100
    elif system == "linux":
        score += 30 if any(token in value for token in ("ubuntu", "linux")) else -100
    if architecture == "x64":
        score += 20 if any(token in value for token in ("x64", "x86_64")) else -50
    elif architecture == "arm64":
        score += 20 if any(token in value for token in ("arm64", "aarch64")) else -50
    wanted = "cpu" if variant == "auto" else variant
    specialized = ("cuda", "rocm", "vulkan", "sycl", "openvino", "hip")
    if wanted == "cpu" and any(token in value for token in specialized):
        score -= 80
    elif wanted == "cpu" and "cpu" not in value:
        score += 25
    elif wanted in value:
        score += 40
    elif wanted != "cpu":
        score -= 20
    if "cpu" in value and wanted == "cpu":
        score += 15
    if "rpc" in value:
        score += 3
    return score


def select_asset(release: dict, variant: str = "auto") -> dict:
    system = platform.system().lower()
    if system not in {"windows", "linux"}:
        raise RuntimeError("Automatic backend installation supports Windows and Linux")
    architecture = normalized_architecture()
    if architecture not in {"x64", "arm64"}:
        raise RuntimeError(f"No automatic llama.cpp build for {architecture}")
    choices = []
    for asset in release.get("assets", []):
        name = str(asset.get("name", ""))
        score = _asset_score(name, system, architecture, variant)
        if score > 0 and asset.get("browser_download_url"):
            choices.append((score, asset))
    if not choices and variant not in {"auto", "cpu"}:
        return select_asset(release, "cpu")
    if not choices:
        raise RuntimeError(
            f"No compatible llama.cpp release asset for {system} {architecture}"
        )
    choices.sort(key=lambda item: (item[0], item[1].get("size", 0)), reverse=True)
    return choices[0][1]


def _download(url: str, target: Path, expected_size: int = 0) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "Distributed-LLM-Universal/0.3"}
    )
    with urllib.request.urlopen(request, timeout=60) as response, open(target, "wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    if expected_size and target.stat().st_size != expected_size:
        raise RuntimeError("Backend download was incomplete")


def _verify_digest(path: Path, digest: str | None) -> bool:
    if not digest or not digest.startswith("sha256:"):
        return False
    expected = digest.split(":", 1)[1].lower()
    actual = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            actual.update(chunk)
    if actual.hexdigest().lower() != expected:
        raise RuntimeError("Backend SHA-256 verification failed")
    return True


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                target = (destination / member.filename).resolve()
                if root not in target.parents and target != root:
                    raise RuntimeError("Unsafe path in backend archive")
            source.extractall(destination)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as source:
            for member in source.getmembers():
                target = (destination / member.name).resolve()
                if root not in target.parents and target != root:
                    raise RuntimeError("Unsafe path in backend archive")
                if member.issym():
                    link_target = (target.parent / member.linkname).resolve()
                    if root not in link_target.parents and link_target != root:
                        raise RuntimeError("Unsafe symbolic link in backend archive")
                elif member.islnk():
                    link_target = (destination / member.linkname).resolve()
                    if root not in link_target.parents and link_target != root:
                        raise RuntimeError("Unsafe hard link in backend archive")
            source.extractall(destination, filter="data")
        return
    raise RuntimeError("Unsupported backend archive format")


def install(variant: str = "auto", destination: Path | None = None) -> dict:
    destination = destination or config.backend_dir()
    release = release_metadata()
    asset = select_asset(release, variant)
    tag = str(release.get("tag_name") or "unknown")
    install_root = destination / tag
    if discover(install_root).coordinator_ready:
        return {
            "tag": tag,
            "asset": asset["name"],
            "directory": str(install_root),
            "verified": bool(asset.get("digest")),
            "already_installed": True,
        }
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dllm-backend-") as temporary:
        archive = Path(temporary) / str(asset["name"])
        _download(str(asset["browser_download_url"]), archive, int(asset.get("size", 0)))
        verified = _verify_digest(archive, asset.get("digest"))
        staged = Path(temporary) / "extracted"
        _safe_extract(archive, staged)
        install_root.mkdir(parents=True, exist_ok=True)
        for item in staged.iterdir():
            target = install_root / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
    tools = discover(install_root)
    if not tools.coordinator_ready:
        raise RuntimeError(
            "The selected llama.cpp package did not include llama-server and RPC server"
        )
    for executable in (tools.server, tools.rpc_server, tools.bench):
        if executable and os.name != "nt":
            os.chmod(executable, os.stat(executable).st_mode | 0o111)
    manifest = {
        "tag": tag,
        "asset": asset["name"],
        "variant": variant,
        "verified_sha256": verified,
    }
    (install_root / "dllm-install.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {**manifest, "directory": str(install_root), "already_installed": False}


def rpc_command(tools: Tools, host: str, port: int, memory_mb: int = 0) -> list[str]:
    if not tools.rpc_server:
        raise RuntimeError("llama.cpp RPC server is not installed")
    command = [tools.rpc_server, "-c", "-H", host, "-p", str(port)]
    if memory_mb > 0:
        command += ["--mem", str(memory_mb)]
    return command


def server_command(
    tools: Tools,
    model: str,
    rpc_endpoints: list[str],
    tensor_split: list[float],
    *,
    port: int = 8080,
    context_size: int = 4096,
    parallel: int = 2,
    kv_cache: str = "q8_0",
) -> list[str]:
    if not tools.server:
        raise RuntimeError("llama-server is not installed")
    command = [
        tools.server,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--ctx-size", str(context_size),
        "--parallel", str(parallel),
        "--cont-batching",
        "--cache-type-k", kv_cache,
        "--cache-type-v", kv_cache,
        "--metrics",
    ]
    if model.lower().endswith(".gguf"):
        path = Path(model).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"GGUF model not found: {path}")
        command += ["--model", str(path)]
    else:
        command += ["-hf", model]
    if rpc_endpoints:
        if len(rpc_endpoints) != len(tensor_split):
            raise ValueError("each RPC endpoint must have one tensor weight")
        command += [
            "--n-gpu-layers", "all",
            "--rpc", ",".join(rpc_endpoints),
            "--split-mode", "layer",
            "--tensor-split", ",".join(f"{max(weight, 0.001):.6f}" for weight in tensor_split),
        ]
    return command


class ManagedProcess:
    def __init__(self, name: str, log_path: Path):
        self.name = name
        self.log_path = log_path
        self.process: subprocess.Popen | None = None
        self.command: list[str] = []
        self._log = None
        self._lock = threading.Lock()

    def start(self, command: list[str]) -> None:
        with self._lock:
            self.stop()
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(self.log_path, "a", encoding="utf-8")
            self.command = list(command)
            kwargs = {
                "stdout": self._log,
                "stderr": subprocess.STDOUT,
                "stdin": subprocess.DEVNULL,
            }
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            else:
                kwargs["start_new_session"] = True
            try:
                self.process = subprocess.Popen(command, **kwargs)
            except Exception:
                self._log.close()
                self._log = None
                self.command = []
                raise

    def stop(self, timeout: float = 8.0) -> None:
        process = self.process
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        self.process = None
        if self._log:
            self._log.close()
            self._log = None

    @property
    def running(self) -> bool:
        return bool(self.process and self.process.poll() is None)

    def status(self) -> dict:
        return {
            "name": self.name,
            "running": self.running,
            "pid": self.process.pid if self.running else None,
            "command": self.command,
            "log": str(self.log_path),
        }
