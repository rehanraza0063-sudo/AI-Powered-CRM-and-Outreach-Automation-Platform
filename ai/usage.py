"""
Token usage, cost tracking and in-memory hash cache (AI-20).

Two small building blocks plus a thin wrapper that combines them:

- ``UsageTracker``: records prompt/completion token counts and the estimated
  USD cost of every call, and exposes running totals.
- ``HashCache``: in-memory LRU cache keyed by the SHA-256 hash of the prompt
  input. An identical repeated prompt returns the cached output and costs
  0 new tokens.
- ``CachedGeminiClient``: wraps ``GeminiClient`` (or any object with the same
  ``generate(prompt, schema, temperature)`` method) so callers get caching and
  tracking without changing how they call it.

Follows the AI package rules: no backend imports, no database access and no
message sending. Everything lives in process memory and is lost on restart.

Cost estimation uses per-million-token prices. The defaults below are only a
starting point; override them with ``GEMINI_INPUT_COST_PER_1M_TOKENS`` and
``GEMINI_OUTPUT_COST_PER_1M_TOKENS`` and check them against Google's current
pricing page for the configured model.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, asdict
from typing import Any, Deque, Dict, List, Optional, Tuple

from ai.gemini_client import GeminiClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

# USD per 1,000,000 tokens as (input, output), matched by model-name prefix.
DEFAULT_PRICING: Dict[str, Tuple[float, float]] = {
    "gemini-1.5-flash": (0.075, 0.30),
}
FALLBACK_PRICING: Tuple[float, float] = (0.075, 0.30)

INPUT_COST_ENV = "GEMINI_INPUT_COST_PER_1M_TOKENS"
OUTPUT_COST_ENV = "GEMINI_OUTPUT_COST_PER_1M_TOKENS"


def _env_float(name: str) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r (expected a number).", name, raw)
        return None
    return value if value >= 0 else None


def get_pricing(model: Optional[str] = None) -> Tuple[float, float]:
    """Returns (input, output) USD price per 1M tokens for ``model``.

    Environment overrides win; otherwise the longest matching model prefix in
    ``DEFAULT_PRICING`` is used, falling back to ``FALLBACK_PRICING``.
    """
    input_price, output_price = FALLBACK_PRICING
    if model:
        matches = [p for p in DEFAULT_PRICING if model.startswith(p)]
        if matches:
            input_price, output_price = DEFAULT_PRICING[max(matches, key=len)]

    env_in = _env_float(INPUT_COST_ENV)
    env_out = _env_float(OUTPUT_COST_ENV)
    return (
        env_in if env_in is not None else input_price,
        env_out if env_out is not None else output_price,
    )


def compute_cost(prompt_tokens: int, completion_tokens: int, model: Optional[str] = None) -> float:
    """Estimated USD cost for one call."""
    input_price, output_price = get_pricing(model)
    cost = (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000
    return round(cost, 8)


# ---------------------------------------------------------------------------
# Usage records and tracker
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UsageRecord:
    """One recorded call. Cached calls have zero tokens and zero cost."""

    timestamp: float
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    cached: bool
    cache_key: Optional[str] = None
    saved_tokens: int = 0
    saved_cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def usage_dict(self) -> Dict[str, int]:
        """Same shape ``GeminiClient.generate`` returns as its usage value."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def normalize_usage(usage: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """Coerces a usage dict into non-negative ints and fills in the total."""

    def _count(key: str) -> int:
        try:
            return max(int((usage or {}).get(key) or 0), 0)
        except (TypeError, ValueError):
            return 0

    prompt = _count("prompt_tokens")
    completion = _count("completion_tokens")
    total = _count("total_tokens") or (prompt + completion)
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


class UsageTracker:
    """Thread-safe, in-memory log of token usage and estimated cost.

    Running totals are kept separately from the record log, so trimming old
    records (``max_records``) never changes the totals.
    """

    def __init__(self, max_records: int = 10_000):
        self._lock = threading.Lock()
        self._records: Deque[UsageRecord] = deque(maxlen=max_records)
        self._reset_totals()

    def _reset_totals(self) -> None:
        self._calls = 0
        self._cache_hits = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._cost_usd = 0.0
        self._tokens_saved = 0
        self._cost_saved_usd = 0.0

    def record_call(
        self,
        usage: Optional[Dict[str, Any]],
        model: str = "unknown",
        cache_key: Optional[str] = None,
    ) -> UsageRecord:
        """Records a real (non-cached) model call."""
        counts = normalize_usage(usage)
        record = UsageRecord(
            timestamp=time.time(),
            model=model,
            prompt_tokens=counts["prompt_tokens"],
            completion_tokens=counts["completion_tokens"],
            total_tokens=counts["total_tokens"],
            cost_usd=compute_cost(counts["prompt_tokens"], counts["completion_tokens"], model),
            cached=False,
            cache_key=cache_key,
        )
        self._add(record)
        return record

    def record_cache_hit(
        self,
        saved_usage: Optional[Dict[str, Any]] = None,
        model: str = "unknown",
        cache_key: Optional[str] = None,
    ) -> UsageRecord:
        """Records a cache hit: 0 new tokens, 0 new cost.

        ``saved_usage`` is the usage of the original call, used only to report
        how many tokens (and how much money) the cache avoided.
        """
        saved = normalize_usage(saved_usage)
        record = UsageRecord(
            timestamp=time.time(),
            model=model,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            cached=True,
            cache_key=cache_key,
            saved_tokens=saved["total_tokens"],
            saved_cost_usd=compute_cost(saved["prompt_tokens"], saved["completion_tokens"], model),
        )
        self._add(record)
        return record

    def _add(self, record: UsageRecord) -> None:
        with self._lock:
            self._records.append(record)
            self._calls += 1
            self._prompt_tokens += record.prompt_tokens
            self._completion_tokens += record.completion_tokens
            self._total_tokens += record.total_tokens
            self._cost_usd += record.cost_usd
            if record.cached:
                self._cache_hits += 1
                self._tokens_saved += record.saved_tokens
                self._cost_saved_usd += record.saved_cost_usd

    @property
    def records(self) -> List[UsageRecord]:
        """Snapshot of the most recent records (oldest first)."""
        with self._lock:
            return list(self._records)

    def summary(self) -> Dict[str, Any]:
        """Running totals across every recorded call."""
        with self._lock:
            calls = self._calls
            hits = self._cache_hits
            return {
                "calls": calls,
                "model_calls": calls - hits,
                "cache_hits": hits,
                "cache_hit_rate": round(hits / calls, 4) if calls else 0.0,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "total_tokens": self._total_tokens,
                "cost_usd": round(self._cost_usd, 8),
                "tokens_saved": self._tokens_saved,
                "cost_saved_usd": round(self._cost_saved_usd, 8),
            }

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._reset_totals()


# ---------------------------------------------------------------------------
# Hash cache
# ---------------------------------------------------------------------------

def make_cache_key(
    prompt: str,
    schema: Optional[Dict[str, Any]] = None,
    temperature: float = 0.3,
    model: str = "",
) -> str:
    """SHA-256 hex digest of the prompt input.

    The schema, temperature and model are part of the input, so the same prompt
    sent with different settings never shares a cache entry.
    """
    canonical = json.dumps(
        {"prompt": prompt, "schema": schema, "temperature": temperature, "model": model},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class HashCache:
    """Thread-safe in-memory LRU cache keyed by hash string.

    Values are deep-copied on the way in and out, so a caller that edits a
    returned dict cannot corrupt what is stored.
    """

    def __init__(self, max_entries: int = 256):
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._store: "OrderedDict[str, Any]" = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._hits += 1
                return copy.deepcopy(self._store[key])
            self._misses += 1
            return default

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = copy.deepcopy(value)
            self._store.move_to_end(key)
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._store

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._store),
                "max_entries": self._max_entries,
                "hits": self._hits,
                "misses": self._misses,
            }


# ---------------------------------------------------------------------------
# Cached + tracked client
# ---------------------------------------------------------------------------

_MISS = object()


class CachedGeminiClient:
    """Adds hash caching and usage tracking around a Gemini client.

    ``generate`` has the same signature and return shape as
    ``GeminiClient.generate``, so it can be dropped in wherever a client is
    accepted. On a cache hit it returns the stored output with usage of 0 tokens.

    Two threads sending the same new prompt at the same moment may both call the
    model; the later result simply overwrites the earlier one in the cache.
    Failed calls are neither cached nor recorded.
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        tracker: Optional[UsageTracker] = None,
        cache: Optional[HashCache] = None,
        cache_enabled: bool = True,
    ):
        self.client = client if client is not None else GeminiClient()
        self.tracker = tracker if tracker is not None else UsageTracker()
        self.cache = cache if cache is not None else HashCache()
        self.cache_enabled = cache_enabled

    @property
    def model(self) -> str:
        config = getattr(self.client, "config", None)
        return getattr(config, "gemini_model", None) or "unknown"

    def generate_with_record(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.3,
    ) -> Tuple[Dict[str, Any], UsageRecord]:
        """Like ``generate`` but returns the full ``UsageRecord`` (cost, cached flag)."""
        model = self.model
        key = make_cache_key(prompt, schema, temperature, model)

        if self.cache_enabled:
            entry = self.cache.get(key, _MISS)
            if entry is not _MISS:
                output, original_usage = entry
                logger.info("AI cache hit (%s...); 0 new tokens.", key[:12])
                record = self.tracker.record_cache_hit(original_usage, model=model, cache_key=key)
                return output, record

        output, usage = self.client.generate(prompt, schema=schema, temperature=temperature)
        record = self.tracker.record_call(usage, model=model, cache_key=key)
        if self.cache_enabled:
            self.cache.set(key, (output, normalize_usage(usage)))
        return output, record

    def generate(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.3,
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """Drop-in for ``GeminiClient.generate``: returns (output, usage_dict)."""
        output, record = self.generate_with_record(prompt, schema=schema, temperature=temperature)
        return output, record.usage_dict()

    def usage_summary(self) -> Dict[str, Any]:
        """Running token/cost totals plus cache statistics."""
        summary = self.tracker.summary()
        summary["cache"] = self.cache.stats()
        return summary
