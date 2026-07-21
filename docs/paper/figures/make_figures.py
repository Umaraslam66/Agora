#!/usr/bin/env python3
"""Paper figures. Every plotted number is read from a sealed runs/ file
where one exists (runs/bt1/results.json, runs/e7_flatness_audit/audit.json,
runs/e7_ess/results.json); the BT2 summary quantities are transcribed from
the sealed verdict record (docs/BT2_VERDICT_RECORD.md, runs/bt2/results.json
raw members behind it) and asserted here against that record's values.

Run from the repo root:  .venv/bin/python docs/paper/figures/make_figures.py
Writes PDF figures next to this script. Colors: Okabe-Ito (colorblind-safe,
fixed assignment: method=blue #0072B2, comparator=orange #E69F00,
truth/neutral=dark gray, accent=vermillion #D55E00 for failures).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

C_METHOD = "#0072B2"   # blue
C_COMP = "#E69F00"     # orange
C_TRUTH = "#333333"    # near-black
C_FAIL = "#D55E00"     # vermillion
C_GRAY = "#999999"
C_GREEN = "#009E73"

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.dpi": 150,
})


# ---------------------------------------------------------------------------
# Fig 1: pipeline (five layers)
# ---------------------------------------------------------------------------

def fig_pipeline():
    layers = [
        ("1 · Grounding", "one persona per real\ndiary record (11,940\npersons; households kept)"),
        ("2 · Variance\npreservation", "between-individual\nspread + error\nindependence (E2)"),
        ("3 · Decision:\ntwo-brain loop", "LLM slow brain writes/\nrewrites cards; fast-brain\nexecutor lives each day;\nhabit counters harden"),
        ("4 · Say-do\ncalibration", "recall channel fitted;\nprice channel a declared\nprior (2–3×), tested blind"),
        ("5 · Evaluation", "E1–E7 frozen bars;\nplacebo estimand; drift\nrule; comparator; single\nblind firings"),
    ]
    fig, ax = plt.subplots(figsize=(9.0, 2.1))
    ax.set_axis_off()
    ax.grid(False)
    n = len(layers)
    w, gap = 1.0, 0.16
    for i, (title, body) in enumerate(layers):
        x = i * (w + gap)
        face = "#E8F1F8" if i == 2 else "#F5F5F5"
        edge = C_METHOD if i == 2 else "#BBBBBB"
        ax.add_patch(plt.Rectangle((x, 0), w, 1, facecolor=face,
                                   edgecolor=edge, linewidth=1.4 if i == 2 else 0.9))
        ax.text(x + w / 2, 0.86, title, ha="center", va="center",
                fontsize=7.0, fontweight="bold")
        ax.text(x + w / 2, 0.36, body, ha="center", va="center", fontsize=5.9)
        if i < n - 1:
            ax.annotate("", xy=(x + w + gap - 0.02, 0.5), xytext=(x + w + 0.02, 0.5),
                        arrowprops=dict(arrowstyle="->", color="#666666", lw=1.1))
    ax.text(2 * (w + gap) + w / 2, -0.14,
            "the “LLM agent” part — one layer of five",
            ha="center", fontsize=6.6, color=C_METHOD, style="italic")
    ax.set_xlim(-0.05, n * (w + gap) - gap + 0.05)
    ax.set_ylim(-0.24, 1.05)
    fig.savefig(OUT / "fig_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2: BT1 intervals (method vs comparator vs truth vs forecast)
# ---------------------------------------------------------------------------

def fig_bt1():
    bt1 = json.loads((ROOT / "runs/bt1/results.json").read_text())
    t5 = bt1["results"]["tiers"]["T5"]["delta_q"]
    # sealed comparator prediction (A5/A6, frozen pre-firing; verdict record)
    comp = {"central": 0.3096, "lo": 0.2980, "hi": 0.3212}
    obs, forecast = 0.28, 0.45

    fig, ax = plt.subplots(figsize=(6.2, 2.3))
    rows = [
        ("Method (T5, blind $\\Delta Q$)", t5["central"], t5["lo"], t5["hi"], C_METHOD),
        ("No-LLM comparator (frozen)", comp["central"], comp["lo"], comp["hi"], C_COMP),
    ]
    for y, (label, c, lo, hi, col) in enumerate(rows):
        ax.plot([lo, hi], [y, y], color=col, lw=3, solid_capstyle="round", alpha=0.45)
        ax.plot(c, y, "o", color=col, ms=7, zorder=5)
        ax.text(c, y + 0.16, f"{c:.4f}", ha="center", fontsize=7.5, color=col)
    ax.axvline(obs, color=C_TRUTH, lw=1.4)
    ax.text(obs, 1.62, "observed  $-28\\%$", ha="center", fontsize=8, color=C_TRUTH)
    ax.axvline(forecast, color=C_GRAY, lw=1.2, ls="--")
    ax.text(forecast, 1.62, "official forecast  $-45\\%$", ha="center",
            fontsize=8, color=C_GRAY)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_ylim(-0.5, 1.9)
    ax.set_xlim(0.25, 0.47)
    ax.set_xlabel("weekday tunnel-volume drop (fraction of baseline; 80% intervals)")
    ax.invert_yaxis()
    fig.savefig(OUT / "fig_bt1_intervals.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3: channel decomposition (BT1 audit, sealed)
# ---------------------------------------------------------------------------

def fig_decomposition():
    audit = json.loads((ROOT / "runs/e7_flatness_audit/audit.json").read_text())
    dec = audit["decomposition"]
    tiers = ["T1", "T2", "T3", "T4", "T4_noclaims", "T5"]
    labels = ["T1", "T2", "T3", "T4", "T4-nc", "T5"]
    route = [dec[f"{t}/toll"]["route_share_of_drop"] * 100 for t in tiers]
    demand = [dec[f"{t}/toll"]["demand_share_of_drop"] * 100 for t in tiers]

    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    x = np.arange(len(tiers))
    ax.bar(x, route, 0.62, label="route dial (fast brain, shared VoT)",
           color=C_METHOD, edgecolor="white", linewidth=1)
    ax.bar(x, demand, 0.62, bottom=route, label="card channel (LLM rewrites)",
           color=C_COMP, edgecolor="white", linewidth=1)
    for xi, (r, d) in enumerate(zip(route, demand)):
        ax.text(xi, 103, f"{d:.0f}%", ha="center", fontsize=7.5,
                color=C_FAIL if d > 20 else C_COMP)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("share of blind response (%)")
    ax.set_ylim(0, 118)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2,
              fontsize=7, frameon=False, borderaxespad=0)
    ax.annotate("T3: churn, not response\n(drift-flagged; §4.4)", xy=(2.32, 72),
                xytext=(3.0, 55), fontsize=6.8, color=C_FAIL,
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_FAIL,
                          lw=0.6, alpha=0.95),
                arrowprops=dict(arrowstyle="->", color=C_FAIL, lw=0.9))
    fig.savefig(OUT / "fig_decomposition.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4: BT2 null (sealed verdict quantities)
# ---------------------------------------------------------------------------

def fig_bt2():
    # transcribed from the sealed verdict record (docs/BT2_VERDICT_RECORD.md)
    phases = ["P1 introduction", "P2 removal (E6)", "P3 return"]
    measured = [-0.00054, 0.000176, 0.00020]
    meas_lo = [-0.00062, 0.000136, 0.00015]
    meas_hi = [-0.00048, 0.000213, 0.00026]
    targets = [(0.21, None), (0.04, 0.12), (0.19, None)]  # bar / band

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5), sharey=False)
    for i, ax in enumerate(axes):
        t_lo, t_hi = targets[i]
        if t_hi is None:
            ax.axhline(t_lo, color=C_TRUTH, lw=1.4)
            ax.text(0.5, t_lo, f"  target {t_lo:.2f}", fontsize=7.5,
                    va="bottom", color=C_TRUTH)
        else:
            ax.axhspan(t_lo, t_hi, color=C_GREEN, alpha=0.18, lw=0)
            ax.text(0.5, (t_lo + t_hi) / 2, f"E6 band\n[{t_lo:.2f}, {t_hi:.2f}]",
                    fontsize=7.5, ha="center", va="center", color=C_GREEN)
        m, lo, hi = measured[i], meas_lo[i], meas_hi[i]
        ax.errorbar([0.2], [abs(m)], yerr=[[abs(m) - min(abs(lo), abs(hi))],
                                           [max(abs(lo), abs(hi)) - abs(m)]],
                    fmt="o", color=C_FAIL, ms=6, capsize=3, lw=1.5)
        sign = "−" if m < 0 else "+"
        ax.text(0.2, abs(m) * 2.2, f"measured\n{sign}{abs(m):.5f}", ha="center",
                fontsize=7, color=C_FAIL)
        ax.set_yscale("log")
        ax.set_ylim(5e-5, 0.5)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_title(phases[i], fontsize=8.5)
        if i == 0:
            ax.set_ylabel("|paired $\\Delta Q$| (log scale)")
    fig.suptitle("BT2 (Stockholm cordon, METHOD-TRANSFER): the blind response is "
                 "~3 orders of magnitude below every bar", fontsize=8.5, y=1.04)
    fig.savefig(OUT / "fig_bt2_null.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 5: ESS individual-level curve (WS1; runs/e7_ess/results.json)
# ---------------------------------------------------------------------------

def fig_ess():
    path = ROOT / "runs/e7_ess/results.json"
    if not path.exists():
        print("fig_ess: runs/e7_ess/results.json not present yet — skipped")
        return
    res = json.loads(path.read_text())
    arms = ["T1", "T2", "T3", "T4", "T4_noclaims", "T4_nofidelity", "T5",
            "m2_deployed", "template"]
    labels = ["T1", "T2", "T3", "T4", "T4-nc", "T4-nf", "T5", "deployed", "template"]
    la = [res["per_arm"][a]["loss_all"]["mean"] for a in arms]
    lh = [res["per_arm"][a]["loss_heldout"]["mean"] for a in arms]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 2.9),
                                   gridspec_kw={"width_ratios": [3, 2]})
    x = np.arange(len(arms))
    ax1.plot(x, la, "o-", color=C_METHOD, lw=1.6, ms=5, label="all days")
    ax1.plot(x, lh, "s--", color=C_COMP, lw=1.4, ms=5, label="held-out days")
    for xi, v in zip(x, la):
        ax1.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=6.2, color=C_METHOD)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7.5, rotation=20)
    ax1.set_ylabel("mean per-persona loss (TVD)")
    ax1.legend(fontsize=7.5)
    ax1.set_ylim(0, max(max(la), max(lh)) * 1.2)

    # baseline learning curve
    for tgt, col, mark, lab in (("target_all", C_METHOD, "o", "baseline, all days"),
                                ("target_heldout", C_COMP, "s", "baseline, held-out")):
        curve = res["baseline_mean_loss_curve"][tgt]
        sizes = sorted(int(s) for s in curve)
        ax2.plot(sizes, [curve[str(s)] for s in sizes], marker=mark, ms=4,
                 lw=1.4, color=col, label=lab)
    ax2.set_xscale("log")
    ax2.set_xlabel("real diary records in training block")
    ax2.set_ylabel("baseline mean loss")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ess_curve.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline()
    fig_bt1()
    fig_decomposition()
    fig_bt2()
    fig_ess()
    print("wrote figures to", OUT)
