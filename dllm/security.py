from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from collections import defaultdict, deque


def new_join_code() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def new_token(bytes_count: int = 32) -> str:
    return secrets.token_urlsafe(bytes_count)


def hash_secret(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()


def verify_secret(value: str, salt: str, expected: str) -> bool:
    return hmac.compare_digest(hash_secret(value, salt), expected)


class RateLimiter:
    def __init__(self, attempts: int = 8, window_seconds: int = 60):
        self.attempts = attempts
        self.window_seconds = window_seconds
        self.events: dict[str, deque[float]] = defaultdict(deque)

    def allowed(self, key: str) -> bool:
        now = time.monotonic()
        queue = self.events[key]
        while queue and now - queue[0] > self.window_seconds:
            queue.popleft()
        if len(queue) >= self.attempts:
            return False
        queue.append(now)
        return True

