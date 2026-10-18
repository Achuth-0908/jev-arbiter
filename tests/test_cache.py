"""Tests for SQLite-backed content-addressed cache store."""

import shutil
import tempfile
from pathlib import Path
import pytest

from src.clients.cache import CacheStore


@pytest.fixture
def temp_cache():
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_cache.db"
    store = CacheStore(db_path=str(db_path))
    yield store
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_cache_miss_and_set(temp_cache: CacheStore):
    key = "test_key_123"
    assert temp_cache.get(key) is None
    assert temp_cache.misses == 1
    assert temp_cache.hits == 0

    payload = {"status": "ok", "prediction": 0.88}
    temp_cache.set(key=key, endpoint="test_ep", model="test_model", response_payload=payload, latency_ms=120.0, cost_usd=0.002)

    cached = temp_cache.get(key)
    assert cached is not None
    res_data, lat, cost = cached
    assert res_data == payload
    assert lat == 120.0
    assert cost == 0.002
    assert temp_cache.hits == 1


def test_100_calls_zero_network_requests(temp_cache: CacheStore):
    """Phase 0 gate requirement: 100 repeated queries hit cache with zero network calls."""
    key = "deterministic_sha256_hash_abc"
    payload = {"result": "success", "probabilities": {"Yes": 0.95, "No": 0.05}}

    # Initial priming call (1 miss)
    assert temp_cache.get(key) is None
    temp_cache.set(key=key, endpoint="ep", model="jev", response_payload=payload, latency_ms=50.0, cost_usd=0.00025)

    # 100 repeated requests
    for _ in range(100):
        entry = temp_cache.get(key)
        assert entry is not None
        assert entry[0]["probabilities"]["Yes"] == 0.95

    stats = temp_cache.stats()
    assert stats["hits"] == 100
    assert stats["misses"] == 1
    assert stats["hit_rate"] >= 0.99
    assert stats["saved_latency_ms"] == 50.0 * 100
