"""Overview block diagram: how the three pipelines elicit a signature behaviour.

Three panels sharing five rows (input, environment, target, judge, runs), so the eye can compare
what fills each slot:

  * Manual reproduction (§2)     -- a hand-built Docker sandbox; the target runs real tools and the
                                    judge scores against server-side ground truth.
  * Automated audit, best-of-N (§3.1)  -- the Auditor (GLM 5.2) invents the scenario and role-plays the
                                    environment; the judge reads only the transcript.
  * Automated audit, in-context   -- the same audit run in sequential waves, with a Reviewer that
    RL (§3.2)                        reads each wave's results and proposes the next wave's ideas.

Pure schematic -- no data inputs. Same palette / typography as analysis/plot_part*.py.

    results/figures/pipeline_diagram.png

Optional argv[1] overrides the output path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO = Path(__file__).resolve().parent.parent
OUTDIR = REPO / "results/figures"

# --- Claude palette (matches results/*/figures) ------------------------------------------
BG      = "#FBFAF8"   # warm off-white ground
INK     = "#2C3E50"   # navy primary text
MUTED   = "#8A8A84"   # grey secondary text
GRID    = "#ECEAE4"   # recessive gridline
AMBER   = "#E8833A"   # accent (the auditor/reviewer -- the automated method)
AMBER_D = "#C96A24"
AMBER_L = "#FBE6D6"   # light amber tint for highlighted boxes
SLATE   = "#7E8A97"   # hand-built environment
SLATE_L = "#EFEEE9"
CARD    = "#F4F2EC"   # audit container
CARD_E  = "#D8D5CD"
WHITE   = "#FFFFFF"

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})

# --- canvas -------------------------------------------------------------------------------
XMAX, YMAX = 17.1, 9.94
FIG_W = 13.0
fig, ax = plt.subplots(figsize=(FIG_W, FIG_W * YMAX / XMAX))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, XMAX)
ax.set_ylim(0, YMAX)
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(0, 0, 1, 1)

PT = FIG_W / XMAX * 72.0          # points per data unit


def box(cx, cy, w, h, lines, fill=WHITE, edge=INK, lw=1.2, ls="-", r=0.14, z=5):
    """Rounded box centred at (cx, cy). lines = [(text, size_pt, color, weight)] top->bottom."""
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fill, ec=edge, lw=lw, ls=ls, zorder=z))
    heights = [s * 1.32 / PT * (1 + t.count("\n")) for t, s, _, _ in lines]
    total = sum(heights)
    y = cy + total / 2
    for (t, s, c, wgt), hh in zip(lines, heights):
        ax.text(cx, y - hh / 2, t, ha="center", va="center", fontsize=s, color=c,
                fontweight=wgt, zorder=z + 1, linespacing=1.25)
        y -= hh


def arrow(p0, p1, color=INK, lw=1.3, z=4):
    ax.annotate("", xy=p1, xytext=p0, zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0,
                                mutation_scale=11))


def route(points, color=INK, lw=1.3, z=4):
    """Polyline with an arrowhead on the last segment."""
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, zorder=z, solid_capstyle="round")
    arrow(points[-2], points[-1], color=color, lw=lw, z=z)


def label(x, y, t, size=7.6, color=MUTED, ha="center", va="center", rot=0, style="normal", z=6):
    ax.text(x, y, t, fontsize=size, color=color, ha=ha, va=va, rotation=rot, fontstyle=style,
            zorder=z, linespacing=1.2)


# --- shared row geometry -----------------------------------------------------------------
Y_TITLE, Y_SUB = 9.35, 8.98
Y_IN, H_IN, W_IN = 8.3, 0.72, 3.5
Y_ENV, H_ENV = 6.55, 1.3
Y_TGT, H_TGT = 4.55, 1.0
Y_JDG, H_JDG = 2.7, 1.3
Y_RUN = 1.05
W = 2.6                                  # model box width

CONT_Y0, CONT_Y1 = 1.85, 7.45            # audit container (panels 2 & 3)


def env_to_target(cx, down_txt, up_txt):
    """Two vertical arrows between the environment row and the target row."""
    y0, y1 = Y_ENV - H_ENV / 2, Y_TGT + H_TGT / 2
    arrow((cx - 0.38, y0), (cx - 0.38, y1))
    arrow((cx + 0.38, y1), (cx + 0.38, y0))
    label(cx - 0.52, (y0 + y1) / 2, down_txt, ha="right")
    label(cx + 0.52, (y0 + y1) / 2, up_txt, ha="left")


def target_to_judge(cx, txt):
    y0, y1 = Y_TGT - H_TGT / 2, Y_JDG + H_JDG / 2
    arrow((cx, y0), (cx, y1))
    label(cx + 0.14, (y0 + y1) / 2, txt, ha="left")


def runs(cx, main, note=""):
    # `main` sits level with the "RUNS" row label (centred at Y_RUN) across all panels; an
    # optional `note` sits just below it (close, but clearing the centred main line).
    label(cx, Y_RUN, main, size=9.2, color=INK, va="center")
    if note:
        label(cx, Y_RUN - 0.20, note, va="top")


def container(cx, half_w, title, stack=True):
    for k in ((2, 1) if stack else ()):
        off = 0.13 * k
        ax.add_patch(FancyBboxPatch((cx - half_w - off, CONT_Y0 - off), 2 * half_w,
                                    CONT_Y1 - CONT_Y0, boxstyle="round,pad=0,rounding_size=0.2",
                                    fc=CARD, ec=CARD_E, lw=1.0, zorder=1))
    ax.add_patch(FancyBboxPatch((cx - half_w, CONT_Y0), 2 * half_w, CONT_Y1 - CONT_Y0,
                                boxstyle="round,pad=0,rounding_size=0.2",
                                fc=CARD, ec=CARD_E, lw=1.0, zorder=2))
    label(cx - half_w + 0.12, CONT_Y1 - 0.16, title, ha="left", va="center", style="italic")


def title(cx, main, sub):
    ax.text(cx, Y_TITLE, main, fontsize=12.5, color=INK, ha="center", va="center",
            fontweight="bold", zorder=6)
    label(cx, Y_SUB, sub, size=8.5)


# --- row labels ---------------------------------------------------------------------------
for y, t in ((Y_IN, "INPUT"), (Y_ENV, "ENVIRONMENT"), (Y_TGT, "TARGET"), (Y_JDG, "JUDGE"),
             (Y_RUN, "RUNS")):
    label(0.12, y, t, size=6.8, ha="left")

# --- panel separators ---------------------------------------------------------------------
for x in (5.0, 9.6):
    ax.plot([x, x], [0.5, 9.6], color=GRID, lw=1.2, zorder=0)

# =========================================================================================
# Panel 1 - Manual reproduction
# =========================================================================================
cx = 2.75
title(2.95, "Manual Reproduction", "§2 · hand-built Docker environments")

box(cx, Y_IN, W_IN, H_IN,
    [("Humans hand-write the task and\nhand-build the Docker environment", 8.3, INK, "normal")],
    fill="none", edge=MUTED, lw=1.0, ls=(0, (3, 2.5)), r=0.1)
arrow((cx, Y_IN - H_IN / 2), (cx, Y_ENV + H_ENV / 2))

box(cx, Y_ENV, W, H_ENV,
    [("Docker sandbox", 10.5, INK, "bold"),
     ("mock Artifactory, seeded notes\nreal tools, real server logs", 7.4, MUTED, "normal")],
    fill=SLATE_L, edge=SLATE)
env_to_target(cx, "task,\nreal tool\nresults", "bash\ncommands")

box(cx, Y_TGT, W, H_TGT, [("Target", 10.5, INK, "bold"), ("GLM 5.2", 8.5, MUTED, "normal")])
target_to_judge(cx, "transcript")

box(cx, Y_JDG, W, H_JDG,
    [("Judge", 10.5, INK, "bold"), ("Opus 4.8", 8.5, MUTED, "normal"),
     ("reads the transcript +\nthe server-side logs", 7.2, MUTED, "normal")])
runs(cx, "K trajectories")

# server-side ground truth: env -> judge, routed down the right side
xr = cx + W / 2 + 0.45
route([(cx + W / 2, Y_ENV - 0.25), (xr, Y_ENV - 0.25), (xr, Y_JDG), (cx + W / 2, Y_JDG)])
label(xr + 0.16, (Y_ENV + Y_JDG) / 2 - 0.1, "server-side logs", rot=90)

# =========================================================================================
# Panel 2 - Automated audit, best-of-N
# =========================================================================================
cx = 7.3
title(cx, "Automated Audit – best-of-N", "§3.1 · built on Petri")

box(cx, Y_IN, W_IN, H_IN,
    [("High-level description of\nthe signature behaviour", 8.3, INK, "normal")],
    fill="none", edge=MUTED, lw=1.0, ls=(0, (3, 2.5)), r=0.1)
container(cx, 1.65, "one audit", stack=False)
arrow((cx, Y_IN - H_IN / 2), (cx, Y_ENV + H_ENV / 2))

box(cx, Y_ENV, W, H_ENV,
    [("Auditor", 10.5, INK, "bold"), ("GLM 5.2", 8.5, MUTED, "normal"),
     ("invents scenario + task,\nrole-plays the environment", 7.4, MUTED, "normal")],
    fill=AMBER_L, edge=AMBER, lw=1.4)
env_to_target(cx, "task, then\nsimulated\ntool results", "tool\ncalls")

box(cx, Y_TGT, W, H_TGT, [("Target", 10.5, INK, "bold"), ("GLM 5.2", 8.5, MUTED, "normal")])
target_to_judge(cx, "transcript")

box(cx, Y_JDG, W, H_JDG,
    [("Judge", 10.5, INK, "bold"), ("Opus 4.8", 8.5, MUTED, "normal"),
     ("reads the transcript only\nscores behaviour + scenario validity", 7.2, MUTED, "normal")])
runs(cx, "N independent audits")

# =========================================================================================
# Panel 3 - Automated audit, in-context RL
# =========================================================================================
cx = 11.75
cr = 15.6                                            # reviewer column
title(13.3, "Automated Audit – in-context RL", "§3.2 · built on Petri")

box(cx, Y_IN, W_IN, H_IN,
    [("High-level description of\nthe signature behaviour", 8.3, INK, "normal")],
    fill="none", edge=MUTED, lw=1.0, ls=(0, (3, 2.5)), r=0.1)
container(cx, 1.65, "wave of W audits")
arrow((cx, Y_IN - H_IN / 2), (cx, Y_ENV + H_ENV / 2))

box(cx, Y_ENV, W, H_ENV,
    [("Auditor", 10.5, INK, "bold"), ("GLM 5.2", 8.5, MUTED, "normal"),
     ("each auditor in the wave is\ngiven a different idea to try", 7.4, MUTED, "normal")],
    fill=AMBER_L, edge=AMBER, lw=1.4)
env_to_target(cx, "task, then\nsimulated\ntool results", "tool\ncalls")

box(cx, Y_TGT, W, H_TGT, [("Target", 10.5, INK, "bold"), ("GLM 5.2", 8.5, MUTED, "normal")])
target_to_judge(cx, "transcript")

box(cx, Y_JDG, W, H_JDG,
    [("Judge", 10.5, INK, "bold"), ("Opus 4.8", 8.5, MUTED, "normal"),
     ("reads the transcript only\nscores behaviour + scenario validity", 7.2, MUTED, "normal")])
runs(cx, "M sequential waves", "W audits per wave · stop at the first valid hit")

# Reviewer
H_REV = 1.7
box(cr, Y_ENV, W, H_REV,
    [("Reviewer", 10.5, INK, "bold"), ("GLM 5.2", 8.5, MUTED, "normal"),
     ("reads the wave's verdicts,\nkeeps a playbook of wins,\nnear misses & dead ends", 7.4, MUTED,
      "normal")],
    fill=AMBER_L, edge=AMBER, lw=1.4)

# description -> reviewer (branch off the input box)
route([(cx + W_IN / 2, Y_IN), (cr, Y_IN), (cr, Y_ENV + H_REV / 2)])
# results -> reviewer
route([(cx + W / 2, Y_JDG), (cr, Y_JDG), (cr, Y_ENV - H_REV / 2)])
label((cx + W / 2 + cr) / 2 + 0.2, Y_JDG + 0.34, "summary of the\nwave's results")
# reviewer -> next wave's auditors
arrow((cr - W / 2, Y_ENV), (cx + W / 2, Y_ENV), color=AMBER_D, lw=1.5)
label((cr - W / 2 + cx + W / 2) / 2 + 0.05, Y_ENV + 0.34, "one idea\nper audit", color=AMBER_D)
label((cr - W / 2 + cx + W / 2) / 2 + 0.05, Y_ENV - 0.34, "next wave", color=AMBER_D)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTDIR / "pipeline_diagram.png"
    fig.savefig(out, dpi=200, facecolor=BG)
    print(out)


if __name__ == "__main__":
    main()
