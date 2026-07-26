from __future__ import annotations

import json
import urllib.error
import urllib.request


def request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
) -> tuple[int, dict]:
    body = None
    actual_headers = {"Accept": "application/json", **(headers or {})}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        actual_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url, data=body, headers=actual_headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return response.status, json.loads(raw.decode("utf-8")) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            payload_out = json.loads(raw.decode("utf-8")) if raw else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload_out = {"error": str(exc)}
        return exc.code, payload_out

