"""
Persistent cache for deterministic LLM responses.

Redis is the production default because it supports cross-worker
deduplication and automatic TTL expiry. SQLite remains available as a
local fallback for environments that do not provide Redis.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def make_cache_key(namespace: str, payload: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(namespace.encode("utf-8"))
    digest.update(b"\0")
    digest.update(_stable_json(payload).encode("utf-8"))
    return digest.hexdigest()


class LLMCacheBackend(ABC):
    @abstractmethod
    def get(self, cache_key: str) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def set(self, cache_key: str, payload_json: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def healthcheck(self) -> dict[str, Any]:
        raise NotImplementedError


class SQLiteLLMCache(LLMCacheBackend):
    """
    Small persistent cache for structured LLM responses.
    """

    def __init__(self, path: str | None = None, ttl_seconds: int | None = None) -> None:
        self._path = Path(path or settings.LLM_CACHE_PATH)
        self._ttl_seconds = int(ttl_seconds or settings.LLM_CACHE_TTL_SECONDS)
        self._enabled = settings.LLM_CACHE_ENABLED
        self._lock = threading.Lock()
        if self._enabled:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        if not self._enabled:
            return
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_llm_cache_expires_at ON llm_cache(expires_at)"
            )
            conn.commit()

    def get(self, cache_key: str) -> str | None:
        if not self._enabled:
            return None
        now = int(time.time())
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json, expires_at FROM llm_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is None:
                return None
            if int(row["expires_at"]) <= now:
                conn.execute("DELETE FROM llm_cache WHERE cache_key = ?", (cache_key,))
                conn.commit()
                return None
            return str(row["payload_json"])

    def set(self, cache_key: str, payload_json: str) -> None:
        if not self._enabled:
            return
        now = int(time.time())
        expires_at = now + self._ttl_seconds
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO llm_cache(cache_key, payload_json, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at
                """,
                (cache_key, payload_json, now, expires_at),
            )
            conn.commit()

    def healthcheck(self) -> dict[str, Any]:
        return {
            "backend": "sqlite",
            "reachable": True,
            "path": str(self._path),
            "configured": self._enabled,
        }


class RedisLLMCache(LLMCacheBackend):
    """
    Redis-backed cache for production deployments.

    Each key uses Redis TTL so expired entries are evicted automatically.
    """

    def __init__(self, url: str | None = None, prefix: str | None = None, ttl_seconds: int | None = None) -> None:
        self._enabled = settings.LLM_CACHE_ENABLED and bool(url or settings.REDIS_URL)
        self._prefix = prefix or settings.REDIS_CACHE_PREFIX
        self._ttl_seconds = int(ttl_seconds or settings.LLM_CACHE_TTL_SECONDS)
        self._client = None
        if self._enabled:
            self._client = self._build_client(url or settings.REDIS_URL)

    @staticmethod
    def _build_client(url: str):
        try:
            import redis  # type: ignore[import-not-found]
        except Exception as exc:
            raise RuntimeError(f"Redis cache requested but redis package is unavailable: {exc}") from exc

        client = redis.Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=settings.REDIS_CONNECT_TIMEOUT_SECONDS,
            socket_timeout=settings.REDIS_CONNECT_TIMEOUT_SECONDS,
        )

        last_exc: Exception | None = None
        for attempt in range(1, max(1, settings.REDIS_CONNECT_RETRIES) + 1):
            try:
                client.ping()
                return client
            except Exception as exc:
                last_exc = exc
                if attempt >= max(1, settings.REDIS_CONNECT_RETRIES):
                    break
                sleep_for = settings.REDIS_CONNECT_BACKOFF_SECONDS * attempt
                logger.warning(
                    "Redis connection attempt %d/%d failed; retrying in %.2fs: %s",
                    attempt,
                    settings.REDIS_CONNECT_RETRIES,
                    sleep_for,
                    exc,
                )
                time.sleep(sleep_for)

        raise RuntimeError(f"Redis cache unavailable after retries: {last_exc}") from last_exc

    def _key(self, cache_key: str) -> str:
        return f"{self._prefix}{cache_key}"

    def get(self, cache_key: str) -> str | None:
        if not self._enabled or self._client is None:
            return None
        try:
            return self._client.get(self._key(cache_key))
        except Exception as exc:
            logger.warning("Redis cache get failed, falling back to miss: %s", exc)
            return None

    def set(self, cache_key: str, payload_json: str) -> None:
        if not self._enabled or self._client is None:
            return
        try:
            self._client.set(self._key(cache_key), payload_json, ex=self._ttl_seconds)
        except Exception as exc:
            logger.warning("Redis cache set failed, dropping cache write: %s", exc)

    def healthcheck(self) -> dict[str, Any]:
        if not self._enabled or self._client is None:
            return {
                "backend": "redis",
                "reachable": False,
                "configured": False,
                "error": "Redis cache is not configured.",
            }
        try:
            return {
                "backend": "redis",
                "reachable": bool(self._client.ping()),
                "configured": True,
                "url": settings.REDIS_URL,
            }
        except Exception as exc:
            return {
                "backend": "redis",
                "reachable": False,
                "configured": True,
                "url": settings.REDIS_URL,
                "error": str(exc),
            }


def build_llm_cache() -> LLMCacheBackend:
    backend = settings.LLM_CACHE_BACKEND.lower().strip()
    if backend == "redis" and settings.REDIS_URL:
        try:
            return RedisLLMCache()
        except Exception as exc:
            logger.warning("Redis cache unavailable, falling back to SQLite: %s", exc)
    return SQLiteLLMCache()


def build_llm_cache_healthcheck(*, strict_redis: bool = True) -> dict[str, Any]:
    backend = settings.LLM_CACHE_BACKEND.lower().strip()
    if backend == "redis":
        if not settings.REDIS_URL:
            return {
                "backend": "redis",
                "reachable": False,
                "configured": False,
                "error": "REDIS_URL is not configured.",
            }
        try:
            cache = RedisLLMCache()
            return cache.healthcheck()
        except Exception as exc:
            if strict_redis:
                return {
                    "backend": "redis",
                    "reachable": False,
                    "configured": True,
                    "url": settings.REDIS_URL,
                    "error": str(exc),
                }
            logger.warning("Redis healthcheck failed; falling back to SQLite: %s", exc)
    cache = SQLiteLLMCache()
    return cache.healthcheck()
