"""Regression tests: cache keys must separate mock/live and different repeat counts."""

from src.clients.base import DecisionRequest, LLMRequest


def _req(repeats: int = 3) -> DecisionRequest:
    return DecisionRequest(state={"claim": "c", "evidence": "e"}, repeats=repeats)


def test_mock_and_live_keys_differ():
    ep = "https://example.invalid/decisions"
    assert _req().cache_key(ep, mode="mock") != _req().cache_key(ep, mode="live")


def test_repeat_count_changes_key():
    ep = "https://example.invalid/decisions"
    assert _req(1).cache_key(ep) != _req(3).cache_key(ep)


def test_key_is_deterministic():
    ep = "https://example.invalid/decisions"
    assert _req().cache_key(ep) == _req().cache_key(ep)


def test_llm_mock_and_live_keys_differ():
    req = LLMRequest(model="m", messages=[{"role": "user", "content": "hi"}])
    ep = "https://example.invalid/chat"
    assert req.cache_key(ep, mode="mock") != req.cache_key(ep, mode="live")
