"""
AI usage accounting and response caching.

This module is independent of the backend and database. The caller supplies
token-usage metadata and is responsible for persisting any summaries.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple


# USD per 1 million tokens. Pass custom pricing for other models or updated rates.
DEFAULT_MODEL_PRICING_USD_PER_1M: Dict[str, Dict[str, float]] = {
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
}


def _normalise_usage(usage: Mapping[str, Any]) -> Dict[str, int]:
    """Accept GeminiClient keys and common input/output token aliases."""
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
    try:
        input_tokens = int(input_tokens or 0)
        output_tokens = int(output_tokens or 0)
        total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("Token counts must be non-negative integers.") from exc
    if min(input_tokens, output_tokens, total_tokens) < 0:
        raise ValueError("Token counts cannot be negative.")
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def calculate_cost(
    usage: Mapping[str, Any],
    model: str = "gemini-1.5-flash",
    pricing: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> float:
    """Estimate one model call's cost in USD using per-million-token rates."""
    rates = pricing if pricing is not None else DEFAULT_MODEL_PRICING_USD_PER_1M
    if model not in rates:
        raise ValueError(
            f"No token pricing configured for {model!r}; pass a pricing mapping."
        )
    tokens = _normalise_usage(usage)
    try:
        input_rate = float(rates[model]["input"])
        output_rate = float(rates[model]["output"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Pricing for {model!r} must contain numeric input and output rates."
        ) from exc
    if input_rate < 0 or output_rate < 0:
        raise ValueError("Token prices cannot be negative.")
    cost = (
        tokens["prompt_tokens"] * input_rate
        + tokens["completion_tokens"] * output_rate
    ) / 1_000_000
    return round(cost, 10)


@dataclass(frozen=True)
class UsageRecord:
    """Token and estimated cost information for one AI request."""
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float


class UsageTracker:
    """Thread-safe in-memory usage tracker; persistence belongs to the caller."""

    def __init__(
        self,
        pricing: Optional[Mapping[str, Mapping[str, float]]] = None,
    ) -> None:
        self.pricing = pricing or DEFAULT_MODEL_PRICING_USD_PER_1M
        self._records = []
        self._lock = threading.Lock()

    def track(self, usage: Mapping[str, Any], model: str) -> UsageRecord:
        """Record token usage for one completed model request."""
        tokens = _normalise_usage(usage)
        record = UsageRecord(
            model=model,
            prompt_tokens=tokens["prompt_tokens"],
            completion_tokens=tokens["completion_tokens"],
            total_tokens=tokens["total_tokens"],
            cost_usd=calculate_cost(usage, model, self.pricing),
        )
        with self._lock:
            self._records.append(record)
        return record

    def summary(self) -> Dict[str, Any]:
        """Return aggregate token counts and estimated spend by model."""
        with self._lock:
            records = list(self._records)
        by_model: Dict[str, Dict[str, Any]] = {}
        for record in records:
            totals = by_model.setdefault(record.model, {
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            })
            totals["requests"] += 1
            totals["prompt_tokens"] += record.prompt_tokens
            totals["completion_tokens"] += record.completion_tokens
            totals["total_tokens"] += record.total_tokens
            totals["cost_usd"] += record.cost_usd
        for totals in by_model.values():
            totals["cost_usd"] = round(totals["cost_usd"], 10)
        return {
            "requests": len(records),
            "prompt_tokens": sum(r.prompt_tokens for r in records),
            "completion_tokens": sum(r.completion_tokens for r in records),
            "total_tokens": sum(r.total_tokens for r in records),
            "cost_usd": round(sum(r.cost_usd for r in records), 10),
            "by_model": by_model,
        }

    def reset(self) -> None:
        """Clear in-memory usage records."""
        with self._lock:
            self._records.clear()


class HashCache:
    """Thread-safe in-memory LRU cache with TTL for generated AI responses."""

    def __init__(self, max_entries: int = 256, ttl_seconds: int = 3600) -> None:
        if max_entries < 1 or ttl_seconds < 1:
            raise ValueError("max_entries and ttl_seconds must be at least 1.")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(
        prompt: str,
        model: str,
        schema: Optional[Mapping[str, Any]] = None,
        temperature: float = 0.3,
    ) -> str:
        """Create a stable SHA-256 key for the complete generation request."""
        payload = {
            "prompt": prompt,
            "model": model,
            "schema": schema,
            "temperature": temperature,
        }
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Any:
        """Return a cached response, or None when absent or expired."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            expires_at, value = entry
            if expires_at <= time.monotonic():
                del self._entries[key]
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return deepcopy(value)

    def set(self, key: str, value: Any) -> None:
        """Store a response and evict least-recently-used entries if needed."""
        with self._lock:
            self._entries[key] = (
                time.monotonic() + self.ttl_seconds, deepcopy(value)
            )
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        """Remove cached responses and reset hit/miss statistics."""
        with self._lock:
            self._entries.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> Dict[str, int]:
        """Return cache hit, miss, and current-size statistics."""
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "entries": len(self._entries),
            }


# Module-level convenience instances. Create your own instances for isolated state.
_default_tracker = UsageTracker()
_default_cache = HashCache()


def track_usage(
    usage: Mapping[str, Any],
    model: str = "gemini-1.5-flash",
) -> UsageRecord:
    """Record usage in the module-level tracker."""
    return _default_tracker.track(usage, model)


def get_usage_summary() -> Dict[str, Any]:
    """Get aggregate statistics from the module-level tracker."""
    return _default_tracker.summary()


def get_cached_response(
    prompt: str,
    model: str = "gemini-1.5-flash",
    schema: Optional[Mapping[str, Any]] = None,
    temperature: float = 0.3,
) -> Any:
    """Look up a response for an identical generation request."""
    key = _default_cache.make_key(prompt, model, schema, temperature)
    return _default_cache.get(key)


def cache_response(
    prompt: str,
    response: Any,
    model: str = "gemini-1.5-flash",
    schema: Optional[Mapping[str, Any]] = None,
    temperature: float = 0.3,
) -> None:
    """Cache a response using the same parameters as the lookup."""
    key = _default_cache.make_key(prompt, model, schema, temperature)
    _default_cache.set(key, response)


def get_cache_stats() -> Dict[str, int]:
    """Return module-level cache statistics."""
    return _default_cache.stats()


__all__ = [
    "DEFAULT_MODEL_PRICING_USD_PER_1M",
    "HashCache",
    "UsageRecord",
    "UsageTracker",
    "calculate_cost",
    "cache_response",
    "get_cache_stats",
    "get_cached_response",
    "get_usage_summary",
    "track_usage",
]
