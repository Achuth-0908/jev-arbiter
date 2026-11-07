"""Live-path behaviour of the Jev client, exercised offline with httpx.MockTransport."""

import tempfile
from pathlib import Path

import httpx
import pytest

from src.clients.cache import CacheStore
from src.clients.jev import DecisionSchemaError, JevDecisionsClient, parse_live_answer


def _ok(p: float) -> httpx.Response:
    return httpx.Response(200, json={
        "model": "typesafe/jev-1.13-test", "answers": {"is_supported": {"type": "noul", "noul": p}},
        "usage": {"cost": 1e-5},
    })


def _client(handler, tmp_path: Path, **kw) -> JevDecisionsClient:
    return JevDecisionsClient(
        api_key="test-key", mock_mode=False, cache_store=CacheStore(db_path=str(tmp_path / "c.db")),
        transport=httpx.MockTransport(handler), **kw,
    )


@pytest.fixture
def tmp(tmp_path):
    return tmp_path


def test_parse_live_answer_accepts_valid_and_rejects_everything_else():
    assert parse_live_answer({"answers": {"is_supported": {"noul": 0.4}}}) == 0.4
    for bad in ({}, {"answers": {}}, {"answers": {"is_supported": {"noul": "x"}}},
                {"answers": {"is_supported": {"noul": 1.5}}}, {"answers": {"is_supported": {"noul": True}}}, None):
        with pytest.raises(DecisionSchemaError):
            parse_live_answer(bad)


@pytest.mark.asyncio
async def test_success_records_resolved_version_and_cost(tmp):
    c = _client(lambda r: _ok(0.8), tmp)
    resp = await c.evaluate_claim("claim", "evidence", repeats=3)
    assert resp.clipped_probabilities["Yes"] == pytest.approx(0.8)
    assert resp.resolved_version == "typesafe/jev-1.13-test"
    assert resp.cost_usd == pytest.approx(3e-5)
    assert c.network_calls == 3
    await c.aclose()


@pytest.mark.asyncio
async def test_second_identical_call_makes_zero_network_requests(tmp):
    c = _client(lambda r: _ok(0.3), tmp)
    await c.evaluate_claim("c", "e", repeats=3)
    before = c.network_calls
    again = await c.evaluate_claim("c", "e", repeats=3)
    assert again.cached and c.network_calls == before
    await c.aclose()


@pytest.mark.asyncio
async def test_cache_salt_forces_a_fresh_call(tmp):
    c = _client(lambda r: _ok(0.3), tmp)
    await c.evaluate_claim("c", "e", repeats=1)
    n = c.network_calls
    await c.evaluate_claim("c", "e", repeats=1, cache_salt="nd1")
    assert c.network_calls == n + 1
    await c.aclose()


@pytest.mark.asyncio
async def test_permanent_4xx_is_not_retried(tmp):
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(401, json={"error": "bad key"})

    c = _client(handler, tmp, max_retries=4)
    with pytest.raises(RuntimeError, match="non-retryable HTTP 401"):
        await c.evaluate_claim("c", "e", repeats=1)
    assert len(calls) == 1
    await c.aclose()


@pytest.mark.asyncio
async def test_429_is_retried_honouring_retry_after_then_succeeds(tmp):
    state = {"n": 0}

    def handler(req):
        state["n"] += 1
        return httpx.Response(429, headers={"retry-after": "0"}) if state["n"] == 1 else _ok(0.6)

    c = _client(handler, tmp)
    resp = await c.evaluate_claim("c", "e", repeats=1)
    assert state["n"] == 2 and resp.clipped_probabilities["Yes"] == pytest.approx(0.6)
    await c.aclose()


@pytest.mark.asyncio
async def test_unrecognised_payload_raises_instead_of_returning_a_default(tmp):
    c = _client(lambda r: httpx.Response(200, json={"something": "else"}), tmp)
    with pytest.raises(DecisionSchemaError):
        await c.evaluate_claim("c", "e", repeats=1)
    # and nothing was cached for the bad payload
    assert c.cache.stats()["hits"] == 0
    await c.aclose()


@pytest.mark.asyncio
async def test_402_is_fatal_not_swallowed_and_not_retried(tmp):
    from src.clients.jev import FatalAPIError

    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(402, json={"error": "insufficient credits"})

    c = _client(handler, tmp, max_retries=4)
    with pytest.raises(FatalAPIError, match="insufficient credits"):
        await c.evaluate_claim("c", "e", repeats=1)
    assert len(calls) == 1
    await c.aclose()


def _ok_cost(p: float, cost: float = 1e-5) -> httpx.Response:
    return httpx.Response(200, json={"model": "m", "answers": {"is_supported": {"noul": p}}, "usage": {"cost": cost}})


@pytest.mark.asyncio
async def test_budget_cap_stops_before_overspending_and_keeps_cached_work(tmp):
    from src.clients.jev import BudgetExceededError, FatalAPIError

    c = _client(lambda r: _ok_cost(0.5), tmp, max_spend_usd=2.5e-5)
    first = await c.evaluate_claim("a", "e", repeats=1)       # spends 1e-5
    await c.evaluate_claim("b", "e", repeats=1)               # spends 1e-5 (total 2e-5)
    with pytest.raises(BudgetExceededError) as exc:
        await c.evaluate_claim("c", "e", repeats=1)           # est 1e-5 would reach 3e-5 > cap
    assert isinstance(exc.value, FatalAPIError)               # run-level code treats it as fatal
    assert c.network_calls == 2                               # the over-budget call never hit the network
    assert (await c.evaluate_claim("a", "e", repeats=1)).cached   # finished work is still free to read
    assert first.cost_usd == pytest.approx(1e-5)
    await c.aclose()


@pytest.mark.asyncio
async def test_derived_view_uses_first_repeat_of_cached_larger_entry_with_no_network(tmp):
    seq = iter([0.2, 0.6, 0.9])
    c = _client(lambda r: _ok_cost(next(seq)), tmp, derive_from_repeats=3)
    full = await c.evaluate_claim("c", "e", repeats=3)
    assert full.clipped_probabilities["Yes"] == pytest.approx((0.2 + 0.6 + 0.9) / 3, abs=1e-3)
    n = c.network_calls
    one = await c.evaluate_claim("c", "e", repeats=1)
    assert c.network_calls == n and one.cached
    assert one.clipped_probabilities["Yes"] == pytest.approx(0.2)        # the first stored repeat
    assert c.derived_from_cache == 1
    again = await c.evaluate_claim("c", "e", repeats=1)                  # now stored under its own key
    assert again.cached and c.derived_from_cache == 1
    await c.aclose()


@pytest.mark.asyncio
async def test_derive_does_not_cross_different_criteria_or_salt(tmp):
    c = _client(lambda r: _ok_cost(0.5), tmp, derive_from_repeats=3)
    await c.evaluate_claim("c", "e", repeats=3)
    n = c.network_calls
    await c.evaluate_claim("c", "e", repeats=1, criteria="a different question")
    await c.evaluate_claim("c", "e", repeats=1, cache_salt="nd1")
    assert c.network_calls == n + 2                                       # both needed fresh calls
    await c.aclose()
