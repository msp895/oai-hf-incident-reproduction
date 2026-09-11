"""Part 2 -- automated auditing: one figure, two panels sharing the per-step colours.

Left:  elicitation rate (valid hit = signature AND scenario_valid) per step, on a 0-0.5 axis with
       Wilson 95% whiskers and the rate printed above each bar.
Right: P(>=1 valid elicitation) vs compute cost for all four steps + the whole incident. Closed
       form (no Monte Carlo): each step's cost-to-first-hit is Exponential(lambda) with
       lambda = -ln(1-p)/c_bar; the incident is their hypoexponential sum, so it is <= every step
       at all budgets. Each curve is shaded with the band from the Wilson interval on its p.

Reads the committed JSON caches (part2_counts.json, part2_cost_data.json), so it renders without
the (gitignored) raw logs; if a cache is missing it is rebuilt from the scored logs.

    uv run python analysis/plot_part2.py   ->  results/part2_petri/figures/part2.png
"""
import glob
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from inspect_ai.log import read_eval_log_sample_summaries

from alignment_auditor.petri.cost_model import reviewer_usage_cost, usage_cost

from _style import GRID, INK, MUTED, SURFACE, INCIDENT, STEP_COLORS, apply_axes

REPO = Path(__file__).resolve().parent.parent
OUTDIR = REPO / "results/part2_petri/figures"
COUNTS_CACHE = OUTDIR / "part2_counts.json"
COST_CACHE = OUTDIR / "part2_cost_data.json"

# step -> (descriptive bar label, scored-log dirs to pool). Step 2's rare rate is pooled over
# several runs of the one step2 config (~1506 audits, same seed).
STEPS = [
    ("Step 1", "Writes to shared\nregistry", ["part2_step1_registry_write"]),
    ("Step 2", "Requests help",
     ["part2_step2_reachout", "part2_step2_reachout_peeraction_n256", "part2_step2_reachout_n1000"]),
    ("Step 3", "Shares exploit", ["part2_step3_exploit_share"]),
    ("Step 4", "Hacks\nHugging Face", ["part2_step4_hf_cheat"]),
]

# type sizes
TITLE, AXLABEL, TICK, BARVAL, DESC, STEPTICK, LEG = 19, 15, 13, 16, 12, 14.5, 13


# --- data helpers ---------------------------------------------------------------------------
def _val(sample, want):
    for _n, sc in (sample.scores or {}).items():
        v = sc.value
        if isinstance(v, dict) and want in v:
            return v[want]
    return None


def _judge_cost(sample):
    for sc in (sample.scores or {}).values():
        u = (sc.metadata or {}).get("judge_usage")
        if u and u.get("model"):
            return reviewer_usage_cost(u)
    return 0.0


def _scored(dirs):
    """Per-audit (cost, signature, scenario_valid) for the pooled dirs, from the scored-log
    summaries (model_usage + scores + score metadata; ~0.2s/file)."""
    rows = []
    for d in dirs:
        cf = glob.glob(str(REPO / f"logs/{d}/scored/*custom*.eval"))
        vf = glob.glob(str(REPO / f"logs/{d}/scored/*validity*.eval"))
        if not cf or not vf:
            continue
        roll, sig, jc = {}, {}, {}
        for s in read_eval_log_sample_summaries(cf[0]):
            k = (s.id, s.epoch)
            roll[k] = sum(usage_cost(m, u) for m, u in (s.model_usage or {}).items())
            sig[k] = _val(s, "signature")
            jc[k] = _judge_cost(s)
        vld, jv = {}, {}
        for s in read_eval_log_sample_summaries(vf[0]):
            k = (s.id, s.epoch)
            vld[k] = _val(s, "scenario_valid")
            jv[k] = _judge_cost(s)
        for k in roll:
            if sig[k] is None or vld.get(k) is None:
                continue
            rows.append((roll[k] + jc.get(k, 0.0) + jv.get(k, 0.0), bool(sig[k]) and bool(vld[k])))
    return rows


def counts():
    """{label: (hits, n)} for the rate panel, cached to JSON."""
    if COUNTS_CACHE.exists():
        return {k: tuple(v) for k, v in json.loads(COUNTS_CACHE.read_text()).items()}
    c = {}
    for label, _sub, dirs in STEPS:
        rows = _scored(dirs)
        c[label] = (sum(1 for _cost, win in rows if win), len(rows))
    COUNTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    COUNTS_CACHE.write_text(json.dumps({k: list(v) for k, v in c.items()}))
    return c


def cost_data():
    """[(costs, wins)] per step for the cost panel, cached to JSON."""
    if COST_CACHE.exists():
        d = json.loads(COST_CACHE.read_text())
        return [(np.array(d[l]["costs"], float), np.array(d[l]["wins"], bool)) for l, _s, _d in STEPS]
    loaded = []
    for _l, _s, dirs in STEPS:
        rows = _scored(dirs)
        loaded.append((np.array([c for c, _w in rows], float), np.array([w for _c, w in rows], bool)))
    COST_CACHE.parent.mkdir(parents=True, exist_ok=True)
    COST_CACHE.write_text(json.dumps(
        {l: {"costs": c.tolist(), "wins": w.tolist()} for (l, _s, _d), (c, w) in zip(STEPS, loaded)}))
    return loaded


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def rate_from(p, cbar):
    return 0.0 if (p <= 0 or cbar <= 0) else -math.log1p(-p) / cbar


def step_cdf(lam, xs):
    return 1.0 - np.exp(-lam * xs)


def incident_cdf(lams, xs):
    """Hypoexponential CDF 1 - sum_i A_i e^(-lam_i X), A_i = prod_{j!=i} lam_j/(lam_j-lam_i)."""
    lams = np.asarray(lams, float)
    A = np.array([np.prod([lams[j] / (lams[j] - lams[i])
                           for j in range(len(lams)) if j != i]) for i in range(len(lams))])
    return 1.0 - (A[:, None] * np.exp(-lams[:, None] * xs[None, :])).sum(axis=0)


def dollar_ticks(ax, lo, hi):
    ax.set_xscale("log", base=2)
    ax.set_xlim(lo, hi)
    ticks = [2.0 ** k for k in range(int(np.ceil(np.log2(lo))), int(np.floor(np.log2(hi))) + 1)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"${t:g}" if t >= 1 else f"${t:.2f}" for t in ticks])


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    c = counts()
    loaded = cost_data()
    lams = [rate_from(float(w.mean()), float(cc.mean())) for cc, w in loaded]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16.0, 6.0), dpi=200,
                                   gridspec_kw={"width_ratios": [1.0, 1.75]})
    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.28, wspace=0.17)

    # ---- left: elicitation rate by step (0-0.5) ----
    apply_axes(fig, axL, grid_axis="y")
    ymax = 0.52
    for x, (label, _sub, _dirs) in enumerate(STEPS):
        k, n = c[label]
        p = k / n if n else 0.0
        lo, hi = wilson(k, n)
        axL.bar(x, p, width=0.64, color=STEP_COLORS[label], edgecolor="black", linewidth=0.8, zorder=3)
        axL.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor="black",
                     elinewidth=1.6, capsize=5, capthick=1.6, zorder=4)
        axL.text(x, hi + ymax * 0.03, f"{p:.2f}", ha="center", va="bottom", fontsize=BARVAL,
                 color=INK, zorder=5)
        print(f"{label}: {k}/{n} = {p:.1%}")
    axL.set_xticks(range(len(STEPS)))
    axL.set_xticklabels([s[0] for s in STEPS], fontsize=STEPTICK, fontweight="bold", color=INK)
    for x, (_label, sub, _dirs) in enumerate(STEPS):
        axL.text(x, -ymax * 0.085, sub, ha="center", va="top", fontsize=DESC, color=MUTED)
    axL.set_ylim(0, ymax)
    axL.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    axL.set_yticklabels([f"{t:.1f}" for t in (0, 0.1, 0.2, 0.3, 0.4, 0.5)], fontsize=TICK, color=INK)
    axL.set_ylabel("Elicitation rate", fontsize=AXLABEL, color=INK)
    axL.set_title("Elicitation rate by step", fontsize=TITLE, fontweight="bold", color=INK, pad=10)

    # ---- right: P(>=1 elicitation) vs compute cost ----
    apply_axes(fig, axR, grid_axis="both")
    axR.set_ylim(-0.03, 1.03)
    axR.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axR.tick_params(labelsize=TICK)
    xs = np.exp(np.linspace(np.log(0.25), np.log(4096), 300))
    for (label, _sub, _dirs), (costs, wins), lam in zip(STEPS, loaded, lams):
        col = STEP_COLORS[label]
        axR.plot(xs, step_cdf(lam, xs), "-", color=col, linewidth=2.6, zorder=3)
        plo, phi = wilson(int(wins.sum()), len(wins))
        cbar = float(costs.mean())
        axR.fill_between(xs, step_cdf(rate_from(plo, cbar), xs), step_cdf(rate_from(phi, cbar), xs),
                         color=col, alpha=0.15, linewidth=0, zorder=2)
    axR.plot(xs, incident_cdf(lams, xs), "--", color=INCIDENT, linewidth=3.2, zorder=6)
    lo = [rate_from(wilson(int(w.sum()), len(w))[0], float(cc.mean())) for cc, w in loaded]
    hi = [rate_from(wilson(int(w.sum()), len(w))[1], float(cc.mean())) for cc, w in loaded]
    axR.fill_between(xs, incident_cdf(lo, xs), incident_cdf(hi, xs), color=INCIDENT,
                     alpha=0.13, linewidth=0, zorder=5)
    dollar_ticks(axR, xs[0], xs[-1])
    axR.tick_params(labelsize=TICK)
    axR.set_xlabel("Compute cost (USD)", fontsize=AXLABEL, color=INK)
    axR.set_ylabel("P(≥1 elicitation)", fontsize=AXLABEL, color=INK)
    axR.set_title("Probability of elicitation vs. compute cost", fontsize=TITLE, fontweight="bold",
                  color=INK, pad=10)

    # ---- shared legend below ----
    handles = [Line2D([0], [0], color=STEP_COLORS[l], lw=3.4) for l, _s, _d in STEPS]
    handles.append(Line2D([0], [0], color=INCIDENT, lw=3.4, linestyle="--"))
    labels = [l for l, _s, _d in STEPS] + ["Whole incident (all 4)"]
    leg = fig.legend(handles, labels, loc="lower center", ncol=5, frameon=True, fontsize=LEG,
                     labelcolor=INK, handlelength=1.7, columnspacing=1.9, borderpad=0.7,
                     bbox_to_anchor=(0.5, 0.02))
    leg.get_frame().set_facecolor(SURFACE)
    leg.get_frame().set_edgecolor(GRID)
    leg.get_frame().set_linewidth(1)

    out = OUTDIR / "part2.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
