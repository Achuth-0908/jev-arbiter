"""Export (id, claim, evidence snippet) for calibration + validation claims, using the SAME retrieval as Jev.

Writes data/nli/inputs.parquet (git-ignored: it contains text from the licensed dataset).
The test split is never exported here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.aggrefact import build_protocol_splits
from src.retrieval.bm25 import retrieve_evidence


def main() -> None:
    cal, val, _test = build_protocol_splits(200, 200)  # test list is built by the helper but not exported
    rows = []
    for inst in cal + val:
        snippet, bm25 = retrieve_evidence(inst.evidence, inst.claim, 120, 20, 2)
        rows.append({"id": inst.id, "claim": inst.claim, "snippet": snippet})
    out = ROOT / "data" / "nli" / "inputs.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {len(rows)} rows -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
