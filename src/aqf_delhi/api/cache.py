"""Cache abstraction for the API.

Provides a TTL-keyed ``MemoryCache`` (always available) and a lazy
``RedisCache`` wrapper used when ``AQF_REDIS_URL`` is set. If the redis
client is unavailable at construction time the factory *degrades to
memory* — the service never fails to boot because of a cache backend.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class Cache(Protocol):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any, ttl: int = 30) -> None: ...
    def delete(self, key: str) -> None: ...
    def entries(self) -> int: ...
    def name(self) -> str: ...


class MemoryCache:
    """Thread-safe in-process TTL cache (no external dependency)."""

    def __init__(self, ttl: int = 30) -> None:
        self._ttl = ttl
        self._data: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def name(self) -> str:
        return "memory"

    def entries(self) -> int:
        with self._lock:
            now = time.monotonic()
            return sum(1 for exp, _ in self._data.values() if exp > now)

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            exp, value = item
            if exp <= time.monotonic():
                self._data.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        with self._lock:
            self._data[key] = (time.monotonic() + (ttl if ttl is not None else self._ttl), value)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)


class RedisCache:
    """Redis-backed JSON cache. Construction raises if redis can't connect."""

    def __init__(self, url: str, *, key_prefix: str = "aqf:", ttl: int = 30) -> None:
        import redis  # lazy import

        self._prefix = key_prefix
        self._ttl = ttl
        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._client.ping()

    def name(self) -> str:
        return "redis"

    def entries(self) -> int:
        return int(self._client.dbsize())

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def get(self, key: str) -> Any | None:
        raw = self._client.get(self._key(key))
        return json.loads(raw) if raw is not None else None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        self._client.setex(
            self._key(key), ttl if ttl is not None else self._ttl, json.dumps(value)
        )

    def delete(self, key: str) -> None:
        self._client.delete(self._key(key))


def build_cache(redis_url: str | None = None, ttl: int = 30) -> Cache:
    """Return RedisCache when a working URL is given, else MemoryCache."""
    if redis_url:
        try:
            cache = RedisCache(redis_url, ttl=ttl)
            logger.info("cache backend: redis (%s)", redis_url)
            return cache
        except Exception as exc:  # pragma: no cover - depends on infra
            logger.warning("redis unavailable (%s); falling back to memory cache", exc)
    return MemoryCache(ttl=ttl)