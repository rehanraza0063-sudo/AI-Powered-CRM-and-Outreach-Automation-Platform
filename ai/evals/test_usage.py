"""
Unit tests for ai/usage.py (AI-20): token/cost tracking and the SHA-256 hash cache.

All tests run offline. They use a fake client or GeminiClient in mock mode, so CI
never makes an external API request.
"""

import threading

import pytest

from ai.gemini_client import GeminiClient
from ai.usage import (
    CachedGeminiClient,
    HashCache,
    UsageTracker,
    compute_cost,
    make_cache_key,
    normalize_usage,
)

ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class _FakeConfig:
    gemini_model = "gemini-1.5-flash"


class FakeClient:
    """Counts real model calls and returns usage in the same shape as GeminiClient."""

    def __init__(self, fail=False):
        self.config = _FakeConfig()
        self.calls = 0
        self.fail = fail

    def generate(self, prompt, schema=None, temperature=0.3):
        self.calls += 1
        if self.fail:
            raise RuntimeError("model unavailable")
        output = {"subject": "Hello", "body": prompt.upper()}
        usage = {"prompt_tokens": 120, "completion_tokens": 65, "total_tokens": 185}
        return output, usage


# --- AI-20 acceptance criterion -------------------------------------------------

def test_identical_input_returns_cached_output_and_reports_zero_new_tokens():
    fake = FakeClient()
    client = CachedGeminiClient(client=fake)

    first_output, first_usage = client.generate("Draft an email for Acme")
    second_output, second_usage = client.generate("Draft an email for Acme")

    # The second call is served from the cache: same output, no new model call.
    assert second_output == first_output
    assert fake.calls == 1

    # First call used tokens; the repeated call reports 0 new tokens.
    assert first_usage["total_tokens"] == 185
    assert second_usage == ZERO_USAGE


def test_cache_hit_has_zero_cost_and_reports_savings():
    client = CachedGeminiClient(client=FakeClient())

    _, first = client.generate_with_record("same prompt")
    _, second = client.generate_with_record("same prompt")

    assert first.cached is False
    assert first.cost_usd == pytest.approx(compute_cost(120, 65, "gemini-1.5-flash"))
    assert first.cost_usd > 0

    assert second.cached is True
    assert second.total_tokens == 0
    assert second.cost_usd == 0.0
    assert second.saved_tokens == 185
    assert second.saved_cost_usd == pytest.approx(first.cost_usd)


def test_repeated_identical_calls_only_hit_the_model_once():
    fake = FakeClient()
    client = CachedGeminiClient(client=fake)

    for _ in range(10):
        client.generate("repeat me")

    summary = client.usage_summary()
    assert fake.calls == 1
    assert summary["calls"] == 10
    assert summary["model_calls"] == 1
    assert summary["cache_hits"] == 9
    assert summary["total_tokens"] == 185  # only the first call cost tokens
    assert summary["tokens_saved"] == 9 * 185


def test_works_end_to_end_with_real_client_in_mock_mode(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK_MODE", "true")
    client = CachedGeminiClient(client=GeminiClient())

    first_output, first_usage = client.generate("Generate outreach draft")
    second_output, second_usage = client.generate("Generate outreach draft")

    assert second_output == first_output
    assert first_usage["total_tokens"] == 185
    assert second_usage == ZERO_USAGE


# --- Cache key behaviour -------------------------------------------------------

def test_cache_key_is_a_stable_sha256_hex_digest():
    key = make_cache_key("prompt", {"a": 1, "b": 2})
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)
    # Dict key order must not change the hash.
    assert key == make_cache_key("prompt", {"b": 2, "a": 1})
    assert key != make_cache_key("different prompt", {"a": 1, "b": 2})


def test_different_prompt_settings_do_not_share_a_cache_entry():
    fake = FakeClient()
    client = CachedGeminiClient(client=fake)

    client.generate("same prompt", temperature=0.1)
    client.generate("same prompt", temperature=0.9)
    client.generate("same prompt", schema={"type": "object"})

    assert fake.calls == 3


def test_different_prompts_each_cost_tokens():
    client = CachedGeminiClient(client=FakeClient())

    _, usage_a = client.generate("prompt a")
    _, usage_b = client.generate("prompt b")

    assert usage_a["total_tokens"] == 185
    assert usage_b["total_tokens"] == 185


# --- Cache safety --------------------------------------------------------------

def test_mutating_a_returned_output_does_not_corrupt_the_cache():
    client = CachedGeminiClient(client=FakeClient())

    first, _ = client.generate("x")
    first["body"] = "tampered"
    second, _ = client.generate("x")
    assert second["body"] == "X"

    second["body"] = "tampered again"
    third, _ = client.generate("x")
    assert third["body"] == "X"


def test_failed_call_is_not_cached_or_recorded():
    client = CachedGeminiClient(client=FakeClient(fail=True))

    with pytest.raises(RuntimeError):
        client.generate("x")

    assert client.tracker.summary()["calls"] == 0
    assert len(client.cache) == 0


def test_cache_can_be_disabled():
    fake = FakeClient()
    client = CachedGeminiClient(client=fake, cache_enabled=False)

    client.generate("x")
    _, usage = client.generate("x")

    assert fake.calls == 2
    assert usage["total_tokens"] == 185


def test_lru_eviction_drops_least_recently_used_entry():
    cache = HashCache(max_entries=2)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1  # touching "a" makes "b" the oldest
    cache.set("c", 3)

    assert "b" not in cache
    assert "a" in cache and "c" in cache
    assert cache.stats()["entries"] == 2


def test_hash_cache_rejects_invalid_size():
    with pytest.raises(ValueError):
        HashCache(max_entries=0)


# --- Tracker -------------------------------------------------------------------

def test_tracker_totals_and_cost():
    tracker = UsageTracker()
    tracker.record_call({"prompt_tokens": 100, "completion_tokens": 50}, model="gemini-1.5-flash")
    tracker.record_call({"prompt_tokens": 10, "completion_tokens": 5}, model="gemini-1.5-flash")

    summary = tracker.summary()
    assert summary["prompt_tokens"] == 110
    assert summary["completion_tokens"] == 55
    assert summary["total_tokens"] == 165
    assert summary["cost_usd"] == pytest.approx(compute_cost(110, 55, "gemini-1.5-flash"))


def test_tracker_totals_survive_record_trimming_and_reset():
    tracker = UsageTracker(max_records=2)
    for _ in range(5):
        tracker.record_call({"prompt_tokens": 10, "completion_tokens": 5})

    assert len(tracker.records) == 2
    assert tracker.summary()["calls"] == 5
    assert tracker.summary()["total_tokens"] == 75

    tracker.reset()
    assert tracker.summary()["calls"] == 0
    assert tracker.records == []


def test_normalize_usage_handles_missing_and_bad_values():
    assert normalize_usage(None) == ZERO_USAGE
    assert normalize_usage({"prompt_tokens": "7", "completion_tokens": -3}) == {
        "prompt_tokens": 7, "completion_tokens": 0, "total_tokens": 7,
    }
    assert normalize_usage({"prompt_tokens": "not a number"})["prompt_tokens"] == 0


def test_price_can_be_overridden_from_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_INPUT_COST_PER_1M_TOKENS", "1")
    monkeypatch.setenv("GEMINI_OUTPUT_COST_PER_1M_TOKENS", "2")
    assert compute_cost(1_000_000, 1_000_000, "gemini-1.5-flash") == pytest.approx(3.0)

    # An invalid override is ignored and the default price is used.
    monkeypatch.setenv("GEMINI_INPUT_COST_PER_1M_TOKENS", "not-a-number")
    assert compute_cost(1_000_000, 0, "gemini-1.5-flash") == pytest.approx(0.075)


def test_concurrent_calls_keep_totals_consistent():
    client = CachedGeminiClient(client=FakeClient())

    def work():
        for i in range(50):
            client.generate(f"prompt {i % 5}")

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert client.tracker.summary()["calls"] == 400
