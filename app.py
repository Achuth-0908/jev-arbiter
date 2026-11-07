"""Streamlit Interactive Platform for Calibration-Weighted Multi-Agent Arbitration with Jev."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.calibrate.metrics import (
    brier_score,
    expected_calibration_error,
    auroc_score,
    reliability_curve,
)
from src.calibrate.scalers import TemperatureScaler, IsotonicCalibrator
from src.clients.cache import CacheStore
from src.clients.jev import JevDecisionsClient
from src.data.loaders import generate_benchmark_synthetic_subset, create_frozen_splits
from src.data.qa_loader import get_sample_multihop_instances
from src.eval.runner import BenchmarkRunner
from src.fuse.effective_votes import (
    compute_pairwise_error_correlations,
    calculate_discounted_agreement,
)
from src.fuse.engine import ArbitrationEngine
from src.verify.verifier import JevClaimVerifier

load_dotenv()

st.set_page_config(
    page_title="Jev-Arbiter | Calibration-Weighted Multi-Agent Arbitration",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.warning(
    "DEMO ONLY: this app runs on a tiny synthetic fixture and hand-labelled toy questions, so its numbers are "
    "NOT evidence. Real, leakage-safe results are produced by `python -m src.runs phase1` (see results/runs/).",
    icon="⚠️",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title { font-size: 2.1rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
    .subtitle { font-size: 1.05rem; color: #64748B; margin-bottom: 1.5rem; }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .status-accepted { color: #15803D; font-weight: 600; background-color: #DCFCE7; padding: 3px 8px; border-radius: 4px; }
    .status-rejected { color: #B91C1C; font-weight: 600; background-color: #FEE2E2; padding: 3px 8px; border-radius: 4px; }
    .status-abstained { color: #B45309; font-weight: 600; background-color: #FEF3C7; padding: 3px 8px; border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_shared_cache() -> CacheStore:
    return CacheStore(db_path=".cache/streamlit_calls.db")


cache_store = get_shared_cache()

RUNS_DIR = Path(__file__).resolve().parent / "results" / "runs"


def load_latest_run():
    """Newest Phase 1 run that scored the test split (else the newest run), or None. Reads artifacts only."""
    dirs = sorted(RUNS_DIR.glob("phase1-*"), key=lambda p: p.stat().st_mtime, reverse=True) if RUNS_DIR.exists() else []
    loaded = []
    for d in dirs:
        try:
            man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
            met = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        loaded.append((d, man, met))
    if not loaded:
        return None
    d, man, met = next((x for x in loaded if x[1].get("test_scored")), loaded[0])
    frames = {}
    for split in ("val", "test"):
        f = d / f"records_{split}.parquet"
        if f.exists():
            frames[split] = pd.read_parquet(f)
    return {"dir": d, "manifest": man, "metrics": met, "frames": frames}


def fmt_ci(m, key):
    """'point [lo, hi]' from a metrics.json cell, tolerating missing values."""
    v = (m or {}).get(key)
    if not isinstance(v, dict) or v.get("point") is None:
        return "n/a"
    lo, hi = v.get("lo"), v.get("hi")
    return f"{v['point']:.3f}" if lo is None or hi is None else f"{v['point']:.3f} [{lo:.3f}, {hi:.3f}]"

# Sidebar Configuration
st.sidebar.title("⚙️ System Controls")

api_mode = st.sidebar.radio(
    "API Runtime Mode",
    ["Mock Mode (Offline, Deterministic, $0)", "Live OpenRouter Decisions API"],
    index=0,
)
is_mock = "Mock Mode" in api_mode

openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
if not is_mock:
    user_key = st.sidebar.text_input(
        "OpenRouter API Key",
        value=openrouter_key,
        type="password",
        help="Targeting openrouter.ai/api/alpha/decisions, model typesafe/jev-1.13",
    )
    if user_key:
        openrouter_key = user_key

st.sidebar.markdown("---")
st.sidebar.subheader("Arbitration Hyperparameters")

strategy_choice = st.sidebar.selectbox(
    "Arbitration Strategy",
    [
        "calibrated_fusion",
        "calibrated_jev",
        "raw_jev",
        "self_confidence",
        "majority_vote",
    ],
    format_func=lambda s: {
        "calibrated_fusion": "Calibrated Fusion (Jev + N_eff + Conf + BM25)",
        "calibrated_jev": "Calibrated Jev Alone",
        "raw_jev": "Raw Jev Argmax",
        "self_confidence": "Self-Confidence Weighted Vote",
        "majority_vote": "Naive Majority Vote",
    }[s],
)

acceptance_tau = st.sidebar.slider("Acceptance Threshold (tau)", 0.50, 0.95, 0.70, 0.05)
abstention_tau = st.sidebar.slider("Abstention Threshold", 0.10, 0.50, 0.40, 0.05)
mean_phi = st.sidebar.slider("Agent Pairwise Error Correlation (phi_bar)", 0.0, 0.90, 0.35, 0.05)

k_agents = 4
n_eff = k_agents / (1.0 + (k_agents - 1) * mean_phi)
st.sidebar.info(f"**Effective Independent Votes (N_eff):** {n_eff:.2f} / {k_agents}")

cache_stats = cache_store.stats()
st.sidebar.caption(f"Cache Hits: {cache_stats['hits']} | Misses: {cache_stats['misses']} | Hit Rate: {cache_stats['hit_rate']*100:.1f}%")

# Main Navigation Tabs
tabs = st.tabs([
    "🧪 Live Arbitration Playground",
    "📈 Calibration & Reliability",
    "🔗 N_eff Formula (illustration)",
    "📊 Phase 1 Results (real runs)",
])


# TAB 1: Live Arbitration Playground
with tabs[0]:
    st.markdown("<div class='main-title'>Multi-Agent Claim-Level Arbitration</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Arbitrate conflicting candidate outputs per atomic claim using TypeSafe Jev calibrated verification.</div>",
        unsafe_allow_html=True,
    )

    sample_instances = get_sample_multihop_instances()
    example_names = [f"Example {i+1}: {inst.question[:65]}..." for i, inst in enumerate(sample_instances)] + ["Custom Query"]
    selected_example = st.selectbox("Select Benchmark Example or Custom Query", example_names, index=0)

    if selected_example == "Custom Query":
        col_q1, col_q2 = st.columns([1, 1])
        with col_q1:
            query = st.text_input("User Query", "What year was the Python programming language first released?")
            ref_text = st.text_area(
                "Reference Grounding Passages (One per line)",
                "Python was created by Guido van Rossum and first released on February 20, 1991.\nPython 2.0 was released in 2000.\nPython 3.0 was released in 2008.",
                height=120,
            )
            passages = [p.strip() for p in ref_text.split("\n") if p.strip()]
        with col_q2:
            st.markdown("**Candidate Agent Outputs**")
            agent_1 = st.text_input("Claude 3.5 Haiku", "Python was officially released by Guido van Rossum in 1991. Confidence: 0.95")
            agent_2 = st.text_input("Llama 3.1 70B", "Guido van Rossum released Python in 1991. Confidence: 0.92")
            agent_3 = st.text_input("Gemini Flash", "Python was created by Guido van Rossum and launched in 2005. Confidence: 0.85")
            agent_4 = st.text_input("Mistral Nemo", "Python was developed at CWI and released in 1991. Confidence: 0.90")
            candidate_responses = {
                "agent_claude": agent_1,
                "agent_llama": agent_2,
                "agent_gemini": agent_3,
                "agent_mistral": agent_4,
            }
    else:
        ex_idx = example_names.index(selected_example)
        inst = sample_instances[ex_idx]
        query = inst.question
        passages = inst.passages
        candidate_responses = inst.candidate_responses
        st.markdown(f"**Query:** *{query}*")

    with st.expander("👁️ View Candidate Agent Reasoning & Grounding Context", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Candidate Agent Responses:**")
            for ag, txt in candidate_responses.items():
                st.markdown(f"- **{ag}**: {txt}")
        with c2:
            st.markdown("**Reference Grounding Passages:**")
            for idx, p in enumerate(passages):
                st.markdown(f"**[{idx+1}]** {p}")

    if st.button("🚀 Run Arbitration Engine", type="primary", use_container_width=True):
        jev_client = JevDecisionsClient(
            api_key=openrouter_key if not is_mock else "",
            cache_store=cache_store,
            mock_mode=is_mock,
        )
        verifier = JevClaimVerifier(client=jev_client)
        engine = ArbitrationEngine(
            verifier=verifier,
            acceptance_threshold=acceptance_tau,
            abstention_threshold=abstention_tau,
            default_mean_phi=mean_phi,
        )

        with st.spinner("Extracting atomic claims, querying Jev Decisions Noul, and computing calibrated fusion..."):
            res = asyncio.run(
                engine.arbitrate(
                    query=query,
                    candidate_responses=candidate_responses,
                    reference_documents=passages,
                    strategy=strategy_choice,
                    mean_phi=mean_phi,
                )
            )

        st.success(f"Arbitration Complete in {res.total_latency_ms:.1f}ms | API Cost: ${res.total_cost_usd:.5f}")

        # Summary KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accepted Claims", len(res.accepted_claims))
        k2.metric("Rejected Claims", len(res.rejected_claims))
        k3.metric("Abstained Claims", len(res.abstained_claims))
        k4.metric("Effective Votes (N_eff)", f"{res.effective_independent_votes:.2f}")

        # Claim Table
        st.subheader("Sub-Claim Arbitration Breakdown")
        table_rows = []
        for ac in res.arbitrated_claims:
            table_rows.append({
                "Claim Proposition": ac.canonical_claim,
                "Supporting Agents": ", ".join(ac.supporting_agents),
                "Vote Count": f"{ac.agreement_count}/4",
                "Eff. Agree": f"{ac.effective_agreement_ratio:.2f}",
                "BM25": f"{ac.bm25_score:.2f}",
                "Raw Jev": f"{ac.raw_jev_prob:.2f}",
                "Calibrated Jev": f"{ac.calibrated_jev_prob:.2f}",
                "Fused P": f"{ac.fused_prob:.2f}",
                "Decision": ac.status,
            })
        df_claims = pd.DataFrame(table_rows)
        st.dataframe(df_claims, use_container_width=True)

        # Synthesized Output
        st.subheader("🎯 Synthesized Answer (Only Verified Claims)")
        st.markdown(
            f"""
            <div style="background-color: #F1F5F9; border-left: 5px solid #2563EB; padding: 1rem; border-radius: 4px; font-size: 1.1rem;">
            {res.final_answer}
            </div>
            """,
            unsafe_allow_html=True,
        )


# TAB 2: Calibration & Reliability Probe
with tabs[1]:
    st.markdown("<div class='main-title'>Jev Calibration & Reliability</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Real reliability diagram from the latest Phase 1 run's per-claim records (LLM-AggreFact).</div>",
        unsafe_allow_html=True,
    )

    run_cal = load_latest_run()
    frames = (run_cal or {}).get("frames", {})
    split_name = "test" if "test" in frames else ("val" if "val" in frames else None)
    if run_cal is None or split_name is None or "p_selected" not in frames[split_name].columns:
        st.info("No Phase 1 run records found. Run `python -m src.runs phase1` to produce them (git-ignored: they derive from a licensed dataset).")
    else:
        rec = frames[split_name]
        rec = rec[(rec["error"] == "") & rec["raw_p"].notna()]
        sel = run_cal["metrics"].get("selected_calibrator", "selected")
        fig_r, ax_r = plt.subplots(figsize=(6.5, 5))
        ax_r.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        for col, label, style, color in (("raw_p", "Raw Jev", "s-", "#DC2626"), ("p_selected", f"Recalibrated ({sel})", "o-", "#2563EB")):
            _, accs, confs, counts = reliability_curve(rec["label"].values, rec[col].values, n_bins=10)
            xs = [c for c, n in zip(confs, counts) if n >= 10]
            ys = [a for a, n in zip(accs, counts) if n >= 10]
            ax_r.plot(xs, ys, style, color=color, label=label)
        ax_r.set_xlabel("Mean predicted P(supported)")
        ax_r.set_ylabel("Observed fraction supported")
        ax_r.set_title(f"Reliability, {split_name.upper()} split, n={len(rec)} (bins with >= 10 claims)")
        ax_r.legend()
        ax_r.grid(True, alpha=0.3)
        st.pyplot(fig_r)
        st.caption(
            f"Run {run_cal['manifest']['run_id']}. Pooled over 11 datasets with different positive rates, so the pooled curve hides "
            "per-dataset miscalibration (see the Phase 1 Results tab and docs/JEV_FINDINGS.md)."
        )

    st.markdown("---")
    st.markdown("##### Synthetic fixture demo (mock Jev, 8 templated claims: **not evidence**)")
    if st.button("Run synthetic-fixture calibration demo", type="secondary"):
        jev_client = JevDecisionsClient(mock_mode=True, cache_store=cache_store)
        verifier = JevClaimVerifier(client=jev_client)
        runner = BenchmarkRunner(verifier)

        instances = generate_benchmark_synthetic_subset(n_samples=120, seed=42)
        cal_set, val_set, test_set = create_frozen_splits(instances, cal_ratio=0.40, val_ratio=0.20, test_ratio=0.40, seed=42)

        with st.spinner("Evaluating held-out calibration split and test split..."):
            probe_results = asyncio.run(runner.run_phase1_calibration_probe(cal_set, test_set))

        c1, c2, c3 = st.columns(3)
        raw_m = probe_results["raw_jev"]
        temp_m = probe_results["temperature_scaled"]
        iso_m = probe_results["isotonic_scaled"]

        with c1:
            st.markdown("### Raw Jev-1.13")
            st.metric("ECE", f"{raw_m['ece']:.4f}")
            st.metric("Brier Score", f"{raw_m['brier']:.4f}")
            st.metric("AUROC", f"{raw_m['auroc']:.4f}")
            st.metric("Accuracy", f"{raw_m['accuracy']*100:.1f}%")

        with c2:
            st.markdown(f"### Temperature Scaled (T={probe_results['fitted_temperature']:.2f})")
            st.metric("ECE", f"{temp_m['ece']:.4f}", delta=f"{temp_m['ece'] - raw_m['ece']:.4f}", delta_color="inverse")
            st.metric("Brier Score", f"{temp_m['brier']:.4f}", delta=f"{temp_m['brier'] - raw_m['brier']:.4f}", delta_color="inverse")
            st.metric("AUROC", f"{temp_m['auroc']:.4f}")
            st.metric("Accuracy", f"{temp_m['accuracy']*100:.1f}%")

        with c3:
            st.markdown("### Isotonic Recalibrated")
            st.metric("ECE", f"{iso_m['ece']:.4f}", delta=f"{iso_m['ece'] - raw_m['ece']:.4f}", delta_color="inverse")
            st.metric("Brier Score", f"{iso_m['brier']:.4f}", delta=f"{iso_m['brier'] - raw_m['brier']:.4f}", delta_color="inverse")
            st.metric("AUROC", f"{iso_m['auroc']:.4f}")
            st.metric("Accuracy", f"{iso_m['accuracy']*100:.1f}%")

        st.caption("Mock verifier on a templated fixture: these numbers only check that the code runs. See the real diagram above.")


# TAB 3: Error Correlation & N_eff
with tabs[2]:
    st.markdown("<div class='main-title'>Effective Independent Votes: the Formula</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>An illustration of N_eff = k / (1 + (k-1) phi_bar). This is a formula plot, not a measurement.</div>",
        unsafe_allow_html=True,
    )

    phi_vals = np.linspace(0.0, 0.95, 100)
    fig_corr, ax_corr = plt.subplots(figsize=(8, 4.5))

    for k in [3, 5, 9]:
        n_eff_curve = k / (1.0 + (k - 1) * phi_vals)
        ax_corr.plot(phi_vals, n_eff_curve, label=f"k = {k} Agents")

    ax_corr.set_xlabel("Average Pairwise Error Correlation (phi_bar)")
    ax_corr.set_ylabel("Effective Independent Votes (N_eff)")
    ax_corr.set_title("Information Saturation under Error Correlation: k / (1 + (k - 1) * phi_bar)")
    ax_corr.legend()
    ax_corr.grid(True, alpha=0.3)
    st.pyplot(fig_corr)

    st.markdown(
        """
        > **What the formula says:** with k = 9 voters and an average error correlation of $\\bar{\\phi} = 0.40$,
        > N_eff = 9 / (1 + 8 x 0.40) = **2.14** independent votes.
        >
        > **What it does not say:** nothing here is measured for LLM agents. In this repo's one real test (Jev vs a generic
        > NLI model on 2,200 claims) the measured error correlation was 0.41, yet fusing the two added nothing, because
        > N_eff ignores that one voter was much less accurate. See `docs/JEV_FINDINGS.md`, section 5b.
        """
    )


# TAB 4: Real Phase 1 results, read from run artifacts (nothing on this tab is typed in)
with tabs[3]:
    st.markdown("<div class='main-title'>Phase 1 Results (real runs)</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Read from results/runs/ artifacts. Jev as a single claim verifier on LLM-AggreFact; "
        "the multi-agent arbitration stages have NOT been evaluated.</div>",
        unsafe_allow_html=True,
    )

    run = load_latest_run()
    if run is None:
        st.info(
            "No Phase 1 run found. Produce one with `python -m src.runs phase1` (live, needs OPENROUTER_API_KEY) or "
            "`python -m src.runs phase1 --mode mock` (offline pipeline check; mock scores are not evidence). "
            "Run artifacts are git-ignored because they derive from a licensed dataset."
        )
    else:
        man, met = run["manifest"], run["metrics"]
        scored_test = bool(man.get("test_scored") and met.get("test"))
        split_label = "TEST" if scored_test else "VALIDATION"
        section = met["test"] if scored_test else met["validation"]
        if man.get("mode") == "mock":
            st.warning("This run used the MOCK verifier: its scores only check that the pipeline works. They are not evidence about Jev.")
        reg = man.get("registration") or {}
        st.caption(
            f"Run {man['run_id']} | {man['mode']} mode | requested {man['jev_model']}, "
            f"resolved {', '.join(man.get('resolved_versions', [])) or 'n/a'} | showing the {split_label} split | "
            f"pre-registration committed: {reg.get('committed', 'n/a')}"
        )

        rows = []
        for name in ("raw", "temperature", "isotonic"):
            m = section.get(name)
            if m:
                rows.append({
                    "Scores": name, "n": m.get("n"), "AUROC": fmt_ci(m, "auroc"), "Brier": fmt_ci(m, "brier"),
                    "ECE": fmt_ci(m, "ece"), "Accuracy": fmt_ci(m, "accuracy"),
                })
        st.markdown(f"#### {split_label} metrics, 95% cluster-bootstrap CIs (clustered by source document)")
        st.table(pd.DataFrame(rows))

        gate = met.get("gate1_test") if scored_test else met.get("gate1_validation_preview")
        if gate:
            st.markdown("#### Gate 1 " + ("verdict (binding: scored once on the test split)" if scored_test else "preview on validation (not binding)"))
            st.table(pd.DataFrame([
                {"Criterion": k, "Value": f"{v['value']:.4f}", "Target": f"{v['op']} {v['target']}", "Result": "PASS" if v["pass"] else "FAIL"}
                for k, v in gate["criteria"].items()
            ]))
            st.markdown(f"**Overall: {'PASS' if gate['all_pass'] else 'FAIL'}**")

        per = met.get("per_dataset_test") if scored_test else met.get("per_dataset_validation")
        pts = []
        for ds, d in (per or {}).items():
            a = (d.get("raw") or {}).get("auroc")
            if isinstance(a, dict) and a.get("point") is not None:
                pts.append((ds, a["point"], a.get("lo"), a.get("hi"), (d.get("raw") or {}).get("n")))
        pts.sort(key=lambda t: t[1])
        if pts:
            fig_d, ax_d = plt.subplots(figsize=(8, 4.8))
            for y, (ds, p, lo, hi, _n) in enumerate(pts):
                ax_d.plot([p], [y], "o", color="#2563EB", zorder=3)
                if lo is not None and hi is not None:
                    ax_d.plot([lo, hi], [y, y], "-", color="#93C5FD", zorder=2)
            ax_d.set_yticks(range(len(pts)))
            ax_d.set_yticklabels([f"{t[0]} (n={t[4]})" for t in pts])
            ax_d.axvline(0.5, color="#94A3B8", ls=":", label="chance")
            ax_d.set_xlabel("Raw AUROC within dataset (95% CI)")
            ax_d.set_title(f"Per-dataset discrimination, {split_label} split")
            ax_d.legend(loc="lower right")
            ax_d.grid(True, axis="x", alpha=0.3)
            st.pyplot(fig_d)
            st.caption("Within-dataset AUROC is the honest measure of claim-level skill: pooled AUROC is inflated by differing positive rates.")

        c = man.get("cost", {})
        if c.get("claims_scored"):
            st.caption(
                f"Cost accounting for this run: {c['claims_scored']} claims, cold-run equivalent ${c['cold_run_equivalent_usd']:.4f} "
                f"(about ${c['cold_run_equivalent_usd'] / c['claims_scored']:.6f} per claim), spent this run ${c['spent_this_run_usd']:.4f} "
                "(cached calls cost nothing). Full write-up: docs/JEV_FINDINGS.md"
            )
