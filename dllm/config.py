from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlparse


CONFIG_VERSION = 1

DEFAULTS = {
    "version": CONFIG_VERSION,
    "configured": False,
    "role": None,
    "name": "",
    "cluster_name": "Distributed LLM",
    "cluster_id": "",
    "coordinator_url": "",
    "join_code": "",
    "node_id": "",
    "node_token": "",
    "control_port": 7000,
    "rpc_port": 50052,
    "tunnel_port": 7443,
    "inference_port": 8080,
    "worker_connection_mode": "direct",
    "tunnel_fingerprint": "",
    "backend_variant": "auto",
    "backend_dir": "",
    "model": "",
    "context_size": 4096,
    "parallel_requests": 2,
    "kv_cache": "q8_0",
    "open_browser": True,
    "auto_exclude_slow_nodes": True,
    "trusted_lan_only": True,
}


def data_dir() -> Path:
    override = os.environ.get("DLLM_HOME")
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
        return base / "DistributedLLM"
    return Path.home() / ".local" / "share" / "distributed-llm"


def config_path() -> Path:
    override = os.environ.get("DLLM_CONFIG")
    return Path(override).expanduser().resolve() if override else data_dir() / "config.json"


def state_path() -> Path:
    return data_dir() / "coordinator-state.json"


def logs_dir() -> Path:
    return data_dir() / "logs"


def models_dir() -> Path:
    return data_dir() / "models"


def backend_dir() -> Path:
    return data_dir() / "backend"


def load(path: Path | None = None) -> dict:
    target = path or config_path()
    result = deepcopy(DEFAULTS)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            result.update(payload)
    except FileNotFoundError:
        pass
    except (OSError, json.JSONDecodeError) as exc:
        result["config_error"] = str(exc)
    result["version"] = CONFIG_VERSION
    return result


def validate(cfg: dict) -> list[str]:
    errors: list[str] = []
    if cfg.get("configured") and cfg.get("role") not in {"coordinator", "agent"}:
        errors.append("role must be coordinator or agent")
    if cfg.get("backend_variant") not in {
        "auto", "cpu", "vulkan", "cuda", "rocm", "metal"
    }:
        errors.append("unsupported backend variant")
    if cfg.get("kv_cache") not in {"q8_0", "f16"}:
        errors.append("kv_cache must be q8_0 or f16")
    if cfg.get("worker_connection_mode") not in {"direct", "outbound"}:
        errors.append("worker_connection_mode must be direct or outbound")
    for field in ("control_port", "rpc_port", "tunnel_port", "inference_port"):
        try:
            value = int(cfg.get(field))
            if value < 1 or value > 65535:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"{field} must be between 1 and 65535")
    if cfg.get("role") == "agent" and cfg.get("coordinator_url"):
        parsed = urlparse(str(cfg["coordinator_url"]))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            errors.append("coordinator_url must be a valid HTTP(S) URL")
    code = str(cfg.get("join_code", ""))
    if code and (len(code) != 6 or not code.isdigit()):
        errors.append("join_code must contain six digits")
    if (
        cfg.get("configured")
        and cfg.get("role") == "agent"
        and not cfg.get("node_token")
        and (len(code) != 6 or not code.isdigit())
    ):
        errors.append("an agent needs its six-digit join code")
    try:
        if int(cfg.get("context_size", 0)) < 512:
            errors.append("context_size must be at least 512")
        if int(cfg.get("parallel_requests", 0)) < 1:
            errors.append("parallel_requests must be at least 1")
    except (TypeError, ValueError):
        errors.append("context and parallel settings must be integers")
    return errors


def save(cfg: dict, path: Path | None = None) -> Path:
    target = path or config_path()
    merged = deepcopy(DEFAULTS)
    merged.update(cfg)
    merged["version"] = CONFIG_VERSION
    errors = validate(merged)
    if errors:
        raise ValueError("; ".join(errors))
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".config-", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(merged, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return target


def save_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
