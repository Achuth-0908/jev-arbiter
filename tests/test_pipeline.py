"""End-to-end integration test for the multi-agent arbitration pipeline."""

import shutil
import tempfile
from pathlib import Path
import pytest

from src.clients.cache import CacheStore
from src.clients.jev import JevDecisionsClient
from src.fuse.engine import ArbitrationEngine
from src.verify.verifier import JevClaimVerifier


@pytest.fixture
def isolated_cache():
    temp_dir = tempfile.mkdtemp()
    cache_path = Path(temp_dir) / "pipeline_test.db"
    store = CacheStore(db_path=str(cache_path))
    yield store
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_full_arbitration_pipeline(isolated_cache: CacheStore):
    # Setup mock verifier and arbitration engine with isolated test cache
    jev_client = JevDecisionsClient(mock_mode=True, cache_store=isolated_cache)
    verifier = JevClaimVerifier(client=jev_client)
    engine = ArbitrationEngine(
        verifier=verifier,
        acceptance_threshold=0.60,
        abstention_threshold=0.35,
        default_mean_phi=0.30,
    )

    query = "What did Alexander Fleming discover in 1928?"
    candidates = {
        "agent_claude": "Alexander Fleming discovered penicillin in 1928. It revolutionized antibacterial medicine. Confidence: 0.95",
        "agent_llama": "In 1928, Alexander Fleming discovered penicillin at St Mary's Hospital. Confidence: 0.90",
        "agent_gemini": "Alexander Fleming invented the polio vaccine in 1955. Confidence: 0.85",
    }
    references = [
        "Alexander Fleming was a Scottish physician and microbiologist, best known for discovering the world's first broadly effective antibiotic substance, which he named penicillin, in 1928 at St Mary's Hospital.",
        "Jonas Salk developed the first successful polio vaccine in the 1950s.",
    ]

    res = await engine.arbitrate(
        query=query,
        candidate_responses=candidates,
        reference_documents=references,
        strategy="calibrated_fusion",
    )

    assert res.strategy == "calibrated_fusion"
    assert len(res.arbitrated_claims) >= 1
    assert len(res.accepted_claims) >= 1
    # Penicillin claim should be accepted
    assert any("penicillin" in c.lower() for c in res.accepted_claims)
    # Polio vaccine hallucination should be rejected
    assert any("polio" in c.lower() for c in res.rejected_claims)
    assert res.final_answer != ""
    assert "penicillin" in res.final_answer.lower()
    assert "polio" not in res.final_answer.lower()
    assert res.effective_independent_votes > 1.0
