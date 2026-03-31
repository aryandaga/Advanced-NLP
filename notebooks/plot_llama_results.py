"""
Generate comparison visualizations for OPT-1.3B vs Llama-2-7B results.
Run with: python notebooks/plot_llama_results.py
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

# ── Load all results ──────────────────────────────────────────────────────────
with open(os.path.join(RESULTS, "llama_lora_results.json"))  as f: llama_lora  = json.load(f)
with open(os.path.join(RESULTS, "llama_qlora_results.json")) as f: llama_qlora = json.load(f)
with open(os.path.join(RESULTS, "crows_pairs_results.json")) as f: opt_crows   = json.load(f)
with open(os.path.join(RESULTS, "bolukbasi_crows_results.json")) as f: opt_db  = json.load(f)

# ── Colours ───────────────────────────────────────────────────────────────────
C = {
    "opt_base":   "#AEC6E8",
    "opt_lora":   "#1F77B4",
    "opt_qlora":  "#0A4A7A",
    "llm_base":   "#FFBB98",
    "llm_lora":   "#FF7F0E",
    "llm_qlora":  "#B85E00",
}

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — SST-2 Accuracy comparison
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("SST-2 Accuracy: OPT-1.3B vs Llama-2-7B", fontsize=13, fontweight="bold")

labels  = ["Baseline", "Post-LoRA", "Post-QLoRA"]
opt_acc = [0.785, 0.945, 0.920]
llm_acc = [0.835, 0.965, 0.955]

x = np.arange(len(labels))
w = 0.35
bars1 = ax.bar(x - w/2, opt_acc, w, label="OPT-1.3B", color=[C["opt_base"], C["opt_lora"], C["opt_qlora"]])
bars2 = ax.bar(x + w/2, llm_acc, w, label="Llama-2-7B", color=[C["llm_base"], C["llm_lora"], C["llm_qlora"]])
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Accuracy"); ax.set_ylim(0.4, 1.05)
ax.axhline(1.0, color="gray", lw=0.7, ls="--")
for b, v in [(b, b.get_height()) for b in list(bars1)+list(bars2)]:
    ax.text(b.get_x()+b.get_width()/2, v+0.008, f"{v:.3f}", ha="center", fontsize=9)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "llama_sst2_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(); print("Saved llama_sst2_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — CrowS-Pairs SPS comparison
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("CrowS-Pairs SPS: OPT-1.3B vs Llama-2-7B", fontsize=13, fontweight="bold")

opt_sps = [opt_crows["baseline"]["sps"], opt_crows["post_lora"]["sps"], opt_crows["post_qlora"]["sps"]]
llm_sps = [59.92, 61.45, 57.25]

bars1 = ax.bar(x - w/2, opt_sps, w, label="OPT-1.3B", color=[C["opt_base"], C["opt_lora"], C["opt_qlora"]])
bars2 = ax.bar(x + w/2, llm_sps, w, label="Llama-2-7B", color=[C["llm_base"], C["llm_lora"], C["llm_qlora"]])
ax.axhline(50, color="red", lw=1.2, ls="--", label="Unbiased (50%)")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("SPS (%)"); ax.set_ylim(44, 68)
for b, v in [(b, b.get_height()) for b in list(bars1)+list(bars2)]:
    ax.text(b.get_x()+b.get_width()/2, v+0.3, f"{v:.1f}%", ha="center", fontsize=9)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "llama_sps_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(); print("Saved llama_sps_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — DirectBias (stereo vs anti) across all conditions
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
fig.suptitle("Bolukbasi DirectBias on CrowS-Pairs Changed Words", fontsize=13, fontweight="bold")

# OPT-1.3B
ax = axes[0]
ax.set_title("OPT-1.3B", fontsize=11)
conds_opt = ["Baseline", "Post-LoRA", "Post-QLoRA"]
keys_opt  = ["baseline", "post_lora", "post_qlora"]
sv_opt = [opt_db[k]["direct_bias_stereo"] for k in keys_opt]
av_opt = [opt_db[k]["direct_bias_anti"]   for k in keys_opt]
xo = np.arange(len(conds_opt))
ax.bar(xo - 0.2, sv_opt, 0.35, label="Stereo words", color="#E15759")
ax.bar(xo + 0.2, av_opt, 0.35, label="Anti-stereo words", color="#76B7B2")
ax.set_xticks(xo); ax.set_xticklabels(conds_opt)
ax.set_ylabel("|cos(w, g)|")
for i, (sv, av) in enumerate(zip(sv_opt, av_opt)):
    ax.text(i-0.2, sv+0.002, f"{sv:.3f}", ha="center", fontsize=8)
    ax.text(i+0.2, av+0.002, f"{av:.3f}", ha="center", fontsize=8)
ax.legend()

# Llama-2-7B
ax = axes[1]
ax.set_title("Llama-2-7B", fontsize=11)
conds_llm = ["Baseline*", "Post-LoRA", "Post-QLoRA"]
sv_llm = [llama_lora["baseline"]["direct_bias_stereo"],
          llama_lora["post_lora"]["direct_bias_stereo"],
          llama_qlora["post_qlora"]["direct_bias_stereo"]]
av_llm = [llama_lora["baseline"]["direct_bias_anti"],
          llama_lora["post_lora"]["direct_bias_anti"],
          llama_qlora["post_qlora"]["direct_bias_anti"]]
xl = np.arange(len(conds_llm))
ax.bar(xl - 0.2, sv_llm, 0.35, label="Stereo words", color="#E15759")
ax.bar(xl + 0.2, av_llm, 0.35, label="Anti-stereo words", color="#76B7B2")
ax.set_xticks(xl); ax.set_xticklabels(conds_llm)
ax.set_ylabel("|cos(w, g)|")
for i, (sv, av) in enumerate(zip(sv_llm, av_llm)):
    ax.text(i-0.2, sv+0.002, f"{sv:.3f}", ha="center", fontsize=8)
    ax.text(i+0.2, av+0.002, f"{av:.3f}", ha="center", fontsize=8)
ax.legend()
ax.annotate("*fp16 baseline only;\nQLoRA 4-bit baseline\nnot directly comparable",
            xy=(0, 0.01), fontsize=7, color="gray")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "llama_directbias_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(); print("Saved llama_directbias_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Full summary dashboard (4-panel)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Gender Bias Drift: OPT-1.3B vs Llama-2-7B — Full Summary",
             fontsize=14, fontweight="bold")

conditions = ["Baseline", "Post-LoRA", "Post-QLoRA"]
x = np.arange(3)
w = 0.35

# Panel A: SST-2 Accuracy
ax = axes[0, 0]
ax.set_title("A. SST-2 Accuracy (Task Performance)", fontweight="bold")
ax.bar(x-w/2, opt_acc, w, color=[C["opt_base"],C["opt_lora"],C["opt_qlora"]], label="OPT-1.3B")
ax.bar(x+w/2, llm_acc, w, color=[C["llm_base"],C["llm_lora"],C["llm_qlora"]], label="Llama-2-7B")
ax.set_xticks(x); ax.set_xticklabels(conditions); ax.set_ylim(0.4, 1.05)
ax.set_ylabel("Accuracy"); ax.legend(fontsize=9)
ax.axhline(1.0, color="gray", lw=0.6, ls="--")

# Panel B: CrowS-Pairs SPS
ax = axes[0, 1]
ax.set_title("B. CrowS-Pairs SPS (Behavioral Bias)", fontweight="bold")
ax.bar(x-w/2, opt_sps, w, color=[C["opt_base"],C["opt_lora"],C["opt_qlora"]], label="OPT-1.3B")
ax.bar(x+w/2, llm_sps, w, color=[C["llm_base"],C["llm_lora"],C["llm_qlora"]], label="Llama-2-7B")
ax.axhline(50, color="red", lw=1.2, ls="--", label="Unbiased (50%)")
ax.set_xticks(x); ax.set_xticklabels(conditions); ax.set_ylim(44, 68)
ax.set_ylabel("SPS (%)"); ax.legend(fontsize=9)

# Panel C: DirectBias Delta (stereo - anti)
ax = axes[1, 0]
ax.set_title("C. DirectBias Delta (stereo − anti)\nPositive = stereo words more gender-aligned", fontweight="bold")
opt_delta = [sv-av for sv, av in zip(sv_opt, av_opt)]
llm_delta = [sv-av for sv, av in zip(sv_llm, av_llm)]
ax.bar(x-w/2, opt_delta, w, color=[C["opt_base"],C["opt_lora"],C["opt_qlora"]], label="OPT-1.3B")
ax.bar(x+w/2, llm_delta, w, color=[C["llm_base"],C["llm_lora"],C["llm_qlora"]], label="Llama-2-7B")
ax.axhline(0, color="gray", lw=1, ls="--")
ax.set_xticks(x); ax.set_xticklabels(conditions)
ax.set_ylabel("DirectBias delta"); ax.legend(fontsize=9)
for i, (od, ld) in enumerate(zip(opt_delta, llm_delta)):
    ax.text(i-w/2, od+(0.001 if od>=0 else -0.003), f"{od:+.4f}", ha="center", fontsize=8)
    ax.text(i+w/2, ld+(0.001 if ld>=0 else -0.003), f"{ld:+.4f}", ha="center", fontsize=8)

# Panel D: SPS Change from baseline
ax = axes[1, 1]
ax.set_title("D. SPS Change from Baseline\n(Fine-tuning effect on behavioral bias)", fontweight="bold")
opt_sps_change = [0, opt_sps[1]-opt_sps[0], opt_sps[2]-opt_sps[0]]
llm_sps_change = [0, llm_sps[1]-llm_sps[0], llm_sps[2]-llm_sps[0]]
bar_colors_opt = ["gray"] + ["#E15759" if v>0 else "#4CAF50" for v in opt_sps_change[1:]]
bar_colors_llm = ["gray"] + ["#E15759" if v>0 else "#4CAF50" for v in llm_sps_change[1:]]
ax.bar(x-w/2, opt_sps_change, w, color=bar_colors_opt, label="OPT-1.3B")
ax.bar(x+w/2, llm_sps_change, w, color=bar_colors_llm, label="Llama-2-7B")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(conditions)
ax.set_ylabel("ΔSPS (pp)")
red_patch   = mpatches.Patch(color="#E15759", label="Bias increased")
green_patch = mpatches.Patch(color="#4CAF50", label="Bias decreased")
ax.legend(handles=[red_patch, green_patch], fontsize=9)
for i, (oc, lc) in enumerate(zip(opt_sps_change, llm_sps_change)):
    if oc != 0: ax.text(i-w/2, oc+(0.05 if oc>=0 else -0.15), f"{oc:+.1f}", ha="center", fontsize=9)
    if lc != 0: ax.text(i+w/2, lc+(0.05 if lc>=0 else -0.15), f"{lc:+.1f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "llama_full_summary.png"), dpi=150, bbox_inches="tight")
plt.close(); print("Saved llama_full_summary.png")

print("\nAll plots saved to results/")
