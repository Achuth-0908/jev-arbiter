"""Standalone NLI scorer: P(entailment) of a claim given an evidence snippet. No project imports.

usage: python -I scripts/nli_score.py IN.parquet OUT.parquet MODEL_ID CACHE_DIR

Safety: weights are loaded with use_safetensors=True and trust_remote_code=False (never executes
downloaded code). Label order is read from the model config, not assumed.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BATCH = 16
MAX_LEN = 512


def main(src: str, dst: str, model_id: str, cache_dir: str) -> None:
    df = pd.read_parquet(src)
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, cache_dir=cache_dir, use_safetensors=True, trust_remote_code=False
    ).eval()
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    ent = next(i for i, lab in id2label.items() if lab.startswith("entail"))
    print("labels:", id2label, "| entailment index:", ent, "| revision:", getattr(model.config, "_commit_hash", None), flush=True)

    # sort by length so batches pad little; restore the original order afterwards
    order = np.argsort([len(s) + len(c) for s, c in zip(df["snippet"], df["claim"])])
    probs = np.zeros((len(df), model.config.num_labels), dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for b in range(0, len(order), BATCH):
            idx = order[b : b + BATCH]
            enc = tok(list(df["snippet"].values[idx]), list(df["claim"].values[idx]),
                      truncation="only_first", max_length=MAX_LEN, padding=True, return_tensors="pt")
            probs[idx] = torch.softmax(model(**enc).logits, dim=-1).numpy()
            if (b // BATCH) % 10 == 0:
                print(f"  {b + len(idx)}/{len(df)}  ({time.time() - t0:.0f}s)", flush=True)

    out = pd.DataFrame({"id": df["id"].values, "p_entail": probs[:, ent]})
    for i, lab in id2label.items():
        out[f"p_{lab}"] = probs[:, i]
    out.to_parquet(dst, index=False)
    print(f"done in {time.time() - t0:.0f}s -> {dst}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])
