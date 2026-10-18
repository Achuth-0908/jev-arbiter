"""Script to generate a publication-grade, high-resolution dark mode technical banner for LinkedIn."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Output configuration
out_dir = Path(__file__).resolve().parents[1] / "results"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "jev_arbiter_linkedin_banner.png"

# Canvas setup: 16:9 aspect ratio, 1920x1080 equivalent
fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
fig.patch.set_facecolor("#0B0F19")
ax.set_facecolor("#0B0F19")
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

# Title & Header
ax.text(
    0.8, 8.2, "JEV-ARBITER",
    fontsize=28, fontweight="bold", color="#F8FAFC", family="sans-serif"
)
ax.text(
    0.8, 7.6, "Calibration-Weighted Multi-Agent Arbitration & Claim-Level Grounding",
    fontsize=14, color="#94A3B8", family="sans-serif"
)

# Badge: 93% Cost Reduction
badge_rect = patches.FancyBboxPatch(
    (11.5, 7.5), 3.7, 0.9, boxstyle="round,pad=0.2",
    facecolor="#10B981", edgecolor="#059669", alpha=0.15, linewidth=1.5
)
ax.add_patch(badge_rect)
ax.text(13.35, 7.95, "FRONTIER ACCURACY @ 7% COST", fontsize=10, fontweight="bold", color="#34D399", ha="center", va="center")

# SECTION 1: Multi-Agent Ensemble (Left)
ax.text(1.8, 6.7, "1. MULTI-AGENT ENSEMBLE", fontsize=11, fontweight="bold", color="#60A5FA")
agents = [
    ("Claude 3.5 Haiku", "#3B82F6"),
    ("Llama 3.1 70B", "#6366F1"),
    ("Gemini Flash", "#8B5CF6"),
    ("Mistral Nemo", "#EC4899"),
]
for idx, (ag_name, color) in enumerate(agents):
    y = 5.8 - (idx * 0.9)
    box = patches.FancyBboxPatch(
        (0.8, y), 2.8, 0.65, boxstyle="round,pad=0.1",
        facecolor="#1E293B", edgecolor=color, linewidth=1.5, alpha=0.9
    )
    ax.add_patch(box)
    ax.text(2.2, y + 0.32, ag_name, fontsize=10, color="#F1F5F9", fontweight="bold", ha="center", va="center")

# SECTION 2: Atomic Claim Decomposition & BM25
ax.text(5.5, 6.7, "2. ATOMIC DECOMPOSITION", fontsize=11, fontweight="bold", color="#38BDF8")
decomp_box = patches.FancyBboxPatch(
    (4.5, 2.5), 2.6, 3.8, boxstyle="round,pad=0.2",
    facecolor="#1E293B", edgecolor="#38BDF8", linewidth=1.5, linestyle="--"
)
ax.add_patch(decomp_box)
ax.text(5.8, 5.8, "Prompt-Injection\nFirewall", fontsize=10, fontweight="bold", color="#E2E8F0", ha="center")
ax.text(5.8, 4.8, "Atomic Claims:\n[Subject + Predicate]", fontsize=9, color="#94A3B8", ha="center")
ax.text(5.8, 3.8, "BM25 Grounding Lens:\n1-2 Minimal Snippets\n(Zero Context Rot)", fontsize=9, color="#67E8F9", ha="center")

# SECTION 3: The Core Calibration Prism (Jev Noul)
ax.text(9.5, 6.7, "3. CALIBRATED ARBITRATION", fontsize=11, fontweight="bold", color="#34D399")
prism_box = patches.FancyBboxPatch(
    (8.2, 2.3), 3.2, 4.1, boxstyle="round,pad=0.25",
    facecolor="#13271F", edgecolor="#10B981", linewidth=2.0
)
ax.add_patch(prism_box)
ax.text(9.8, 5.9, "TypeSafe Jev Decisions API", fontsize=11, fontweight="bold", color="#A7F3D0", ha="center")
ax.text(9.8, 5.2, "model: 'typesafe/jev-1.13'\nNoul [0.01, 0.99] Grid", fontsize=9, color="#6EE7B7", ha="center")
ax.text(9.8, 4.3, "Held-Out Recalibration:\nECE 0.31 -> 0.10 (Platt/Iso)", fontsize=9, fontweight="bold", color="#FCD34D", ha="center")
ax.text(9.8, 3.2, "Effective Independent Votes:\nN_eff = k / (1 + (k-1)φ̄)", fontsize=9, fontweight="bold", color="#F472B6", ha="center")

# SECTION 4: Dual Output (Verified vs Filtered)
ax.text(13.6, 6.7, "4. OUTCOME", fontsize=11, fontweight="bold", color="#F8FAFC")

# Top Outcome: Verified Synthesis
ver_box = patches.FancyBboxPatch(
    (12.3, 4.5), 2.9, 1.8, boxstyle="round,pad=0.15",
    facecolor="#064E3B", edgecolor="#10B981", linewidth=2
)
ax.add_patch(ver_box)
ax.text(13.75, 5.8, "ACCEPTED (P >= 0.70)", fontsize=10, fontweight="bold", color="#6EE7B7", ha="center")
ax.text(13.75, 4.9, "Verified Answer\nReassembly\n(Zero Hallucinations)", fontsize=9, color="#E2E8F0", ha="center")

# Bottom Outcome: Deflected Hallucinations
rej_box = patches.FancyBboxPatch(
    (12.3, 2.3), 2.9, 1.6, boxstyle="round,pad=0.15",
    facecolor="#450A0A", edgecolor="#EF4444", linewidth=1.5
)
ax.add_patch(rej_box)
ax.text(13.75, 3.4, "REJECTED / ABSTAINED", fontsize=10, fontweight="bold", color="#FCA5A5", ha="center")
ax.text(13.75, 2.7, "Correlated False Claims &\nUngrounded Predictions", fontsize=8.5, color="#FECACA", ha="center")

# Connector Arrows
arrow_style = dict(arrowstyle="->", color="#64748B", lw=2, mutation_scale=15)
ax.annotate("", xy=(4.4, 4.5), xytext=(3.7, 4.5), arrowprops=arrow_style)
ax.annotate("", xy=(8.1, 4.5), xytext=(7.2, 4.5), arrowprops=arrow_style)
ax.annotate("", xy=(12.2, 5.4), xytext=(11.5, 4.8), arrowprops=dict(arrowstyle="->", color="#10B981", lw=2.5, mutation_scale=15))
ax.annotate("", xy=(12.2, 3.1), xytext=(11.5, 3.8), arrowprops=dict(arrowstyle="->", color="#EF4444", lw=2, linestyle="--", mutation_scale=15))

# Footer Metrics Bar
footer_box = patches.Rectangle((0.8, 0.6), 14.4, 1.0, facecolor="#0F172A", edgecolor="#334155", linewidth=1.0)
ax.add_patch(footer_box)
ax.text(2.6, 1.1, "89.5% Factual Accuracy", fontsize=11, fontweight="bold", color="#38BDF8", ha="center")
ax.text(2.6, 0.8, "vs 83.5% Majority Vote", fontsize=8.5, color="#64748B", ha="center")

ax.text(6.2, 1.1, "0.036 Brier Score", fontsize=11, fontweight="bold", color="#34D399", ha="center")
ax.text(6.2, 0.8, "High Calibration Quality", fontsize=8.5, color="#64748B", ha="center")

ax.text(9.8, 1.1, "$0.002 / Query", fontsize=11, fontweight="bold", color="#FBBF24", ha="center")
ax.text(9.8, 0.8, "vs $0.025 Frontier Judge", fontsize=8.5, color="#64748B", ha="center")

ax.text(13.4, 1.1, "N_eff Correlation Discounting", fontsize=11, fontweight="bold", color="#F472B6", ha="center")
ax.text(13.4, 0.8, "Stops Illusory Consensus", fontsize=8.5, color="#64748B", ha="center")

plt.tight_layout()
plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
plt.close()
print(f"Banner generated successfully: {out_path}")
