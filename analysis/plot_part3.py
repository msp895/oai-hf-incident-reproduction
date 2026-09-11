"""Part 3 -- cost to first valid Step-2 elicitation: best-of-N vs in-context RL.

x = compute cost (USD, log2), y = P(>=1 valid Step-2 elicitation). Two arms:

  * best-of-N (Part 2, naive iid)  -- the bare auditor; cost-to-first-hit is the geometric wait at
    the measured mean $/audit, closed-form CDF 1-e^(-lambda X), shaded with the Wilson band on p.
  * in-context RL (Part 3)          -- the reviewer-driven memory loop; one real dollar cost per
    replicate to its first valid hit (cost_model.cost_to_first_hit). With few replicates we fit a
    log-normal (probability-plot regression) and carry the uncertainty with a bootstrap band; the
    replicates are the dots the fit passes through.

A double-headed arrow at the 80th-percentile row marks the compute-cost gap between the two arms.

    uv run python analysis/plot_part3.py   ->  results/part3_in_context_rl/figures/part3.png
"""
import json
import math
from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from alignment_auditor.petri.cost_model import cost_to_first_hit, load_memory_run

from _style import GRID, INK, MUTED, SURFACE, apply_axes

REPO = Path(__file__).resolve().parent.parent
OUTDIR = REPO / "results/part3_in_context_rl/figures"
MEMORY_RUN = "part3_step2_memory"
PART2_CACHE = REPO / "results/part2_petri/figures/part2_cost_data.json"

# Two distinct METHOD colours (this figure compares methods on Step 2, not the four steps).
AMBER = "#E8833A"   # in-context RL (the hero)
AMBER_D = "#C96A24"
IID = "#7E8A97"     # best-of-N baseline (naive iid sampling; recessive slate)

PCTL = 0.80
RNG = np.random.default_rng(0)
BOOT = 2000
_erf = np.vectorize(math.erf)
_N = NormalDist()
TITLE, AXLABEL, TICK, LEG, GAP = 19, 15, 13, 14, 16


def normal_cdf(x, mu, sigma):
    sigma = max(sigma, 1e-9)
    return 0.5 * (1.0 + _erf((np.log(x) - mu) / (sigma * math.sqrt(2))))


def plotting_pos(n):
    return (np.arange(1, n + 1) - 0.5) / n


def lognormal_fit(vals):
    """Log-normal by probability-plot regression: least-squares of ln(cost) on Phi^-1((i-0.5)/n)."""
    cs = np.sort(vals)
    z = np.array([_N.inv_cdf(p) for p in plotting_pos(len(cs))])
    A = np.vstack([np.ones(len(cs)), z]).T
    mu, sigma = np.linalg.lstsq(A, np.log(cs), rcond=None)[0]
    return float(mu), float(max(sigma, 1e-9))


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def iid_step2():
    """(k, n, c_bar) for the Part-2 Step-2 best-of-N arm, from the cost cache."""
    d = json.loads(PART2_CACHE.read_text())["Step 2"]
    costs = np.array(d["costs"], float)
    wins = np.array(d["wins"], bool)
    return int(wins.sum()), len(wins), float(costs.mean())


def rate_from(p, cbar):
    return 0.0 if (p <= 0 or cbar <= 0) else -math.log1p(-p) / cbar


def dollar_ticks(ax, lo, hi):
    ax.set_xscale("log", base=2)
    ax.set_xlim(lo, hi)
    ticks = [2.0 ** k for k in range(int(np.ceil(np.log2(lo))), int(np.floor(np.log2(hi))) + 1)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"${t:g}" if t >= 1 else f"${t:.2f}" for t in ticks], fontsize=TICK)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # in-context RL: dollar cost to first valid hit per replicate + log-normal fit
    records = load_memory_run(MEMORY_RUN)
    costs = np.sort(np.array([c for c in cost_to_first_hit(records) if np.isfinite(c)], float))
    mu, sigma = lognormal_fit(costs)

    # best-of-N: rate + Wilson band on p = k/n
    k, n, cbar = iid_step2()
    lam = rate_from(k / n, cbar)
    plo, phi = wilson(k, n)
    lam_lo, lam_hi = rate_from(plo, cbar), rate_from(phi, cbar)

    # 80th-percentile costs + the x-saving
    c_ic = math.exp(mu + sigma * _N.inv_cdf(PCTL))
    c_bon = -math.log(1 - PCTL) / lam
    ratio = c_bon / c_ic
    print(f"P={PCTL:.2f}: best-of-N ${c_bon:,.0f}  in-context ${c_ic:,.0f}  -> {ratio:.1f}x")

    xs = np.exp(np.linspace(np.log(32), np.log(2048), 400))
    fig, ax = plt.subplots(figsize=(9.6, 6.3), dpi=200)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.90, bottom=0.13)
    apply_axes(fig, ax, grid_axis="both")
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    ax.fill_between(xs, 1 - np.exp(-lam_lo * xs), 1 - np.exp(-lam_hi * xs), color=IID,
                    alpha=0.15, linewidth=0, zorder=2)
    ax.plot(xs, 1 - np.exp(-lam * xs), "-", color=IID, linewidth=2.6, zorder=3, label="best-of-N")

    boot = np.array([normal_cdf(xs, *lognormal_fit(RNG.choice(costs, size=len(costs), replace=True)))
                     for _ in range(BOOT)])
    blo, bhi = np.percentile(boot, [5, 95], axis=0)
    ax.fill_between(xs, blo, bhi, color=AMBER, alpha=0.16, linewidth=0, zorder=4)
    ax.plot(xs, normal_cdf(xs, mu, sigma), "-", color=AMBER, linewidth=3.2, zorder=6,
            label="in-context RL")
    ax.scatter(costs, plotting_pos(len(costs)), s=52, color=AMBER_D, edgecolors=SURFACE,
               linewidths=1.2, zorder=7)

    # 80th-percentile gap: dotted guide + a double-headed arrow between the curves
    ax.plot([xs[0], xs[-1]], [PCTL, PCTL], ":", color=MUTED, lw=1.3, zorder=8)
    ax.annotate("", xy=(c_bon, PCTL), xytext=(c_ic, PCTL), zorder=9,
                arrowprops=dict(arrowstyle="<->", color=INK, lw=2.2, shrinkA=0, shrinkB=0))
    ax.text(math.sqrt(c_ic * c_bon), PCTL + 0.035, f"{ratio:.1f}×", ha="center", va="bottom",
            fontsize=GAP, fontweight="bold", color=INK, zorder=10)
    ax.text(xs[0] * 1.05, PCTL + 0.02, f"{int(PCTL * 100)}%", ha="left", va="bottom",
            fontsize=TICK, color=MUTED, zorder=10)

    dollar_ticks(ax, xs[0], xs[-1])
    ax.tick_params(labelsize=TICK)
    ax.set_xlabel("Compute cost (USD)", fontsize=AXLABEL, color=INK)
    ax.set_ylabel("P(≥1 elicitation)", fontsize=AXLABEL, color=INK)
    ax.set_title("Cost to first Step-2 elicitation", fontsize=TITLE, fontweight="bold",
                 color=INK, pad=10)
    leg = ax.legend(loc="center left", bbox_to_anchor=(0.04, 0.62), frameon=True, fontsize=LEG,
                    labelcolor=INK, handlelength=1.7, borderpad=0.7)
    leg.get_frame().set_facecolor(SURFACE)
    leg.get_frame().set_edgecolor(GRID)
    leg.get_frame().set_linewidth(1)

    out = OUTDIR / "part3.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
