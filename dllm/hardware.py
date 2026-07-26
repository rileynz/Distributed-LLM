from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, asdict


def _run(command: list[str], timeout: float = 3.0) -> str:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
        return (completed.stdout or "").strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def normalized_architecture(value: str | None = None) -> str:
    current = (value or platform.machine()).lower().replace("-", "_")
    if current in {"amd64", "x86_64", "x64"}:
        return "x64"
    if current in {"aarch64", "arm64"}:
        return "arm64"
    return current or "unknown"


def memory_bytes() -> tuple[int, int]:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_physical), int(status.available_physical)
    try:
        values: dict[str, int] = {}
        with open("/proc/meminfo", encoding="utf-8") as stream:
            for line in stream:
                key, value = line.split(":", 1)
                values[key] = int(value.strip().split()[0]) * 1024
        return values.get("MemTotal", 0), values.get("MemAvailable", 0)
    except (OSError, ValueError):
        pass
    if hasattr(os, "sysconf"):
        try:
            page = os.sysconf("SC_PAGE_SIZE")
            return (
                int(page * os.sysconf("SC_PHYS_PAGES")),
                int(page * os.sysconf("SC_AVPHYS_PAGES")),
            )
        except (OSError, ValueError):
            pass
    return 0, 0


def cpu_name() -> str:
    if os.name == "nt":
        value = os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
        if value:
            return value
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                lower = line.lower()
                if lower.startswith(("model name", "hardware", "processor")):
                    value = line.split(":", 1)[-1].strip()
                    if value and not value.isdigit():
                        return value
    except OSError:
        pass
    return platform.processor() or platform.machine() or "Unknown CPU"


def cpu_flags() -> list[str]:
    flags: set[str] = set()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.lower().startswith(("flags", "features")):
                    flags.update(line.split(":", 1)[-1].lower().split())
    except OSError:
        identifier = os.environ.get("PROCESSOR_IDENTIFIER", "").lower()
        if "arm" in identifier:
            flags.add("neon")
    interesting = {
        "avx", "avx2", "avx512f", "avx_vnni", "avx512_vnni", "fma",
        "sse4_2", "neon", "asimd", "dotprod", "i8mm", "sve", "sve2",
    }
    return sorted(flags & interesting)


def gpu_names() -> list[str]:
    found: list[str] = []
    if shutil.which("nvidia-smi"):
        text = _run([
            "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"
        ])
        found.extend(line.strip() for line in text.splitlines() if line.strip())
    if os.name == "nt" and not found:
        text = _run([
            "powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
        ])
        found.extend(line.strip() for line in text.splitlines() if line.strip())
    if os.name != "nt" and shutil.which("lspci"):
        text = _run(["lspci"])
        for line in text.splitlines():
            if "VGA compatible controller" in line or "3D controller" in line:
                found.append(line.split(":", 2)[-1].strip())
    unique: list[str] = []
    for name in found:
        if name and name not in unique:
            unique.append(name)
    return unique


def lan_ip() -> str:
    override = os.environ.get("DLLM_ADVERTISE_HOST", "").strip()
    if override:
        return override
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("192.0.2.1", 9))
        return str(sock.getsockname()[0])
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"
    finally:
        sock.close()


def quick_benchmark(max_megabytes: int = 64) -> dict:
    total, available = memory_bytes()
    amount = max(8, min(max_megabytes, int(available / (1024 * 1024) / 16) or 8))
    size = amount * 1024 * 1024
    source = bytearray((index * 31) & 0xFF for index in range(size))
    started = time.perf_counter()
    target = bytearray(source)
    target[:] = source
    elapsed = max(time.perf_counter() - started, 0.000_001)
    bandwidth = (size * 2) / elapsed / 1_000_000_000

    payload = b"distributed-llm-benchmark" * 256
    started = time.perf_counter()
    digest = payload
    iterations = 20_000
    for _ in range(iterations):
        digest = hashlib.sha256(digest).digest()
    cpu_elapsed = max(time.perf_counter() - started, 0.000_001)
    del source, target
    return {
        "memory_bandwidth_gbps": round(bandwidth, 2),
        "cpu_score": round(iterations / cpu_elapsed, 2),
        "benchmark_megabytes": amount,
        "proof": digest.hex()[:8],
    }


@dataclass
class Hardware:
    hostname: str
    os: str
    os_version: str
    architecture: str
    cpu: str
    logical_cores: int
    cpu_flags: list[str]
    ram_total_gb: float
    ram_free_gb: float
    gpus: list[str]
    lan_ip: str
    memory_bandwidth_gbps: float = 0.0
    cpu_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def detect(run_benchmark: bool = True) -> Hardware:
    total, available = memory_bytes()
    benchmark = quick_benchmark() if run_benchmark else {}
    return Hardware(
        hostname=platform.node() or "unnamed-device",
        os=platform.system() or "Unknown",
        os_version=platform.release(),
        architecture=normalized_architecture(),
        cpu=cpu_name(),
        logical_cores=os.cpu_count() or 1,
        cpu_flags=cpu_flags(),
        ram_total_gb=round(total / (1024 ** 3), 2),
        ram_free_gb=round(available / (1024 ** 3), 2),
        gpus=gpu_names(),
        lan_ip=lan_ip(),
        memory_bandwidth_gbps=float(benchmark.get("memory_bandwidth_gbps", 0)),
        cpu_score=float(benchmark.get("cpu_score", 0)),
    )


def refresh_dynamic(report: dict) -> dict:
    """Refresh values that can change without rerunning the startup benchmark."""
    total, available = memory_bytes()
    updated = dict(report)
    updated["ram_total_gb"] = round(total / (1024 ** 3), 2)
    updated["ram_free_gb"] = round(available / (1024 ** 3), 2)
    updated["lan_ip"] = lan_ip()
    return updated
