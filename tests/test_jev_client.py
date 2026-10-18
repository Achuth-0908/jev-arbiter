"""Tests for JevDecisionsClient, repeats, clipping, and mock simulation."""

import shutil
import tempfile
from pathlib import Path
import pytest

from src.clients.base import canonical_json_hash
from src.clients.cache import CacheStore
from src.clients.jev import JevDecisionsClient


@pytest.fixture
def temp_jev_client():
    temp_dir = tempfile.mkdtemp()
    cache_path = Path(temp_dir) / "jev_cache.db"
    store = CacheStore(db_path=str(cache_path))
    client = JevDecisionsClient(cache_store=store, mock_mode=True)
    yield client
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_jev_evaluate_claim_and_repeats(temp_jev_client: JevDecisionsClient):
    claim = "The Hubble Space Telescope was launched in 1990."
    evidence = "The Hubble Space Telescope was launched into low Earth orbit in 1990 and remains in operation."

    resp = await temp_jev_client.evaluate_claim(claim, evidence, repeats=3)

    assert resp.model == "typesafe/jev-1.13"
    assert len(resp.results) == 3
    assert not resp.cached
    assert "Yes" in resp.clipped_probabilities
    assert "No" in resp.clipped_probabilities

    # Probability clipping check: [0.01, 0.99]
    p_yes = resp.clipped_probabilities["Yes"]
    assert 0.01 <= p_yes <= 0.99
    assert p_yes > 0.70  # Strong factual alignment

    # Second call should be instant cache hit
    resp2 = await temp_jev_client.evaluate_claim(claim, evidence, repeats=3)
    assert resp2.cached
    assert resp2.clipped_probabilities["Yes"] == p_yes


@pytest.mark.asyncio
async def test_jev_hallucination_detection(temp_jev_client: JevDecisionsClient):
    claim = "Neil Armstrong landed on the Sun in 1840."
    evidence = "Neil Armstrong was an American astronaut who was the first person to walk on the Moon in 1969."

    resp = await temp_jev_client.evaluate_claim(claim, evidence, repeats=3)
    assert resp.clipped_probabilities["Yes"] < 0.40
    assert resp.clipped_probabilities["No"] > 0.60


def test_canonical_hash_invariance():
    """Hash must be invariant to dict key insertion order."""
    d1 = {"model": "jev", "options": ["Yes", "No"], "state": {"claim": "A", "evidence": "B"}}
    d2 = {"state": {"evidence": "B", "claim": "A"}, "options": ["Yes", "No"], "model": "jev"}
    assert canonical_json_hash(d1) == canonical_json_hash(d2)
