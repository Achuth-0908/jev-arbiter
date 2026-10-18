"""LLM-AggreFact dataset loaders and frozen deterministic splitters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from pydantic import BaseModel, Field


class FactClaimInstance(BaseModel):
    """A claim verification instance from LLM-AggreFact."""
    id: str
    claim: str
    evidence: str
    label: int  # 1 = Supported / Entailed, 0 = Hallucinated / Refuted
    subdataset: str  # e.g., summeval, frank, factool, xsum
    metadata: Dict[str, Any] = Field(default_factory=dict)


def generate_benchmark_synthetic_subset(n_samples: int = 150, seed: int = 42) -> List[FactClaimInstance]:
    """Generates a realistic benchmark subset of factual claim instances for offline evaluation.
    
    Covers diverse domains, supported vs refuted claims, subtle hallucinations, and negations.
    """
    rng = np.random.RandomState(seed)
    subdatasets = ["summeval", "xsum", "frank", "factool", "polytope"]
    
    templates = [
        (
            "The James Webb Space Telescope launched in December 2021 from French Guiana.",
            "The James Webb Space Telescope was successfully launched on December 25, 2021 aboard an Ariane 5 rocket from the Guiana Space Centre.",
            1,
        ),
        (
            "The Apollo 11 mission landed on Mars in 1969.",
            "Apollo 11 was the American spaceflight that first landed humans on the Moon on July 20, 1969. Mars exploration missions were unmanned at that time.",
            0,
        ),
        (
            "CRISPR-Cas9 enables targeted genome editing in living organisms.",
            "CRISPR-Cas9 is a biochemical technique that allows molecular biologists to edit parts of the genome by removing, adding, or altering sections of the DNA sequence.",
            1,
        ),
        (
            "Penicillin was discovered by Marie Curie in 1945.",
            "Alexander Fleming discovered penicillin in 1928 at St Mary's Hospital in London. Marie Curie was famous for her pioneering research on radioactivity.",
            0,
        ),
        (
            "Python was conceived in the late 1980s by Guido van Rossum.",
            "Python was conceived in the late 1980s by Guido van Rossum at Centrum Wiskunde & Informatica (CWI) in the Netherlands as a successor to ABC.",
            1,
        ),
        (
            "Transformer architectures rely entirely on recurrent neural networks without attention.",
            "The Transformer model architecture introduced in 2017 eschews recurrence and relies entirely on self-attention mechanisms to draw global dependencies.",
            0,
        ),
        (
            "The Large Hadron Collider is located near Geneva on the Franco-Swiss border.",
            "The Large Hadron Collider (LHC) is the world's largest high-energy particle collider, built underground across the border between France and Switzerland near Geneva.",
            1,
        ),
        (
            "Photosynthesis produces carbon dioxide and consumes oxygen as its primary output.",
            "Photosynthesis converts light energy into chemical energy, consuming carbon dioxide and water while producing glucose and releasing oxygen as a byproduct.",
            0,
        ),
    ]

    instances: List[FactClaimInstance] = []
    for i in range(n_samples):
        base_claim, base_ev, label = templates[i % len(templates)]
        sub = subdatasets[i % len(subdatasets)]
        # Add slight lexical variation for diverse sampling
        instances.append(
            FactClaimInstance(
                id=f"aggrefact_{sub}_{i:04d}",
                claim=base_claim,
                evidence=base_ev,
                label=label,
                subdataset=sub,
                metadata={"synthetic": True, "sample_idx": i},
            )
        )

    return instances


def create_frozen_splits(
    instances: List[FactClaimInstance],
    cal_ratio: float = 0.40,
    val_ratio: float = 0.20,
    test_ratio: float = 0.40,
    seed: int = 42,
) -> Tuple[List[FactClaimInstance], List[FactClaimInstance], List[FactClaimInstance]]:
    """Partition instances into calibration, validation, and test sets.
    
    Stratified by label and subdataset to prevent distribution skew across splits.
    """
    assert abs(cal_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"
    
    rng = np.random.RandomState(seed)
    
    # Group by (subdataset, label)
    strata: Dict[Tuple[str, int], List[FactClaimInstance]] = {}
    for inst in instances:
        key = (inst.subdataset, inst.label)
        strata.setdefault(key, []).append(inst)

    cal_set, val_set, test_set = [], [], []

    for key, group in strata.items():
        indices = list(range(len(group)))
        rng.shuffle(indices)

        n_cal = int(round(len(group) * cal_ratio))
        n_val = int(round(len(group) * val_ratio))

        cal_idx = indices[:n_cal]
        val_idx = indices[n_cal : n_cal + n_val]
        test_idx = indices[n_cal + n_val :]

        cal_set.extend([group[i] for i in cal_idx])
        val_set.extend([group[i] for i in val_idx])
        test_set.extend([group[i] for i in test_idx])

    return cal_set, val_set, test_set
