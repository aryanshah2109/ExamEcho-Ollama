"""
Small helper around the OpenAI Responses API.

It centralizes the structured-output call, cache lookup, and JSON
serialization so the individual generators stay focused on their prompt
logic and validation rules.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, TypeVar

from pydantic import BaseModel

from app.config import settings
from app.utils.llm_cache import LLMCacheBackend, build_llm_cache, make_cache_key
from ai_ml.exceptions import (
    LLMAuthenticationError,
    LLMPermissionError,
    LLMRateLimitError,
    LLMResponseError,
    LLMServiceError,
    LLMTimeoutError,
    LLMTransientError,
)
from ai_ml.model_creator import OpenAIClientLoader

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OpenAIStructuredCaller:
    """Execute cached structured-output calls against OpenAI."""

    _key_locks: dict[str, threading.Lock] = {}
    _key_locks_guard = threading.Lock()

    def __init__(self, client=None, cache: LLMCacheBackend | None = None) -> None:
        self._client = client
        self._cache = cache or build_llm_cache()

    def parse(
        self,
        *,
        cache_namespace: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        max_output_tokens: int,
        temperature: float | None = None,
        cache_inputs: dict[str, Any],
    ) -> T:
        """
        Return a validated structured response, reading from cache when possible.
        """
        client = self._client or OpenAIClientLoader.get_client()
        cache_key = make_cache_key(
            cache_namespace,
            {
                "model": model,
                "response_model": response_model.__name__,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
                "cache_inputs": cache_inputs,
            },
        )

        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for namespace=%s model=%s", cache_namespace, model)
            return response_model.model_validate_json(cached)

        lock = self._get_lock(cache_key)
        with lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit-after-wait for namespace=%s model=%s", cache_namespace, model)
                return response_model.model_validate_json(cached)

            logger.debug("Cache miss for namespace=%s model=%s", cache_namespace, model)

            try:
                response = client.responses.parse(
                    model=model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    text_format=response_model,
                    temperature=temperature if temperature is not None else settings.OPENAI_TEMPERATURE,
                    max_output_tokens=max_output_tokens,
                )
            except Exception as exc:
                raise self._classify_error(exc) from exc

            parsed = getattr(response, "output_parsed", None)
            if parsed is None:
                output_text = getattr(response, "output_text", None)
                if not output_text:
                    raise LLMResponseError("OpenAI response did not include parsed output or text content.")
                try:
                    parsed = response_model.model_validate_json(output_text)
                except Exception as exc:
                    raise LLMResponseError(f"Could not validate OpenAI output: {exc}") from exc

            self._cache.set(cache_key, parsed.model_dump_json())
            return parsed

    @classmethod
    def _get_lock(cls, cache_key: str) -> threading.Lock:
        with cls._key_locks_guard:
            lock = cls._key_locks.get(cache_key)
            if lock is None:
                lock = threading.Lock()
                cls._key_locks[cache_key] = lock
            return lock

    @staticmethod
    def _classify_error(exc: Exception) -> LLMServiceError:
        name = exc.__class__.__name__.lower()
        message = str(exc) or exc.__class__.__name__

        if "authentication" in name or "invalidapi" in name:
            return LLMAuthenticationError(message)
        if "permission" in name or "forbidden" in name:
            return LLMPermissionError(message)
        if "ratelimit" in name or "rate" in name:
            return LLMRateLimitError(message)
        if "timeout" in name:
            return LLMTimeoutError(message)
        if "connection" in name or "internalserver" in name or "serviceunavailable" in name:
            return LLMTransientError(message)
        return LLMResponseError(message)
