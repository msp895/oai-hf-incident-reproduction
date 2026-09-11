"""Social-preview card (1.91:1, Twitter/LinkedIn/Slack) for the LessWrong post.

Headline on the left; on the right, three lab logos inside a dashed "Docker sandbox" trading
messages (one asks for help, one posts an exploit) and the third breaking out to hack Hugging
Face. Pure schematic, no data inputs.

    uv run python analysis/plot_social_preview.py   ->  results/figures/social_preview_G_tri_logos.png

Optional argv[1] overrides the output directory. Logos are read from analysis/logos/<slug>.svg
(or .png); a labeled text tile is drawn if a logo file is missing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

REPO = Path(__file__).resolve().parent.parent
OUTDIR = REPO / "results/figures"

# --- palette (same family as the Figure 1 step diagrams) ------------------------------------
BG       = "#FFFFFF"
INK      = "#2C3E50"
MUTED    = "#8A8A84"
CARD_E   = "#D8D5CD"
AMBER    = "#D97A12"
AMBER_LL = "#FDF3E8"
HF_GREY  = "#B1B5BB"
KEY_Y    = "#F2C94C"
DASH     = "#B8BCC2"

# Palatino everywhere (same serif stack as analysis/_style.py)
for _fp in (os.path.expanduser("~/.fonts/texgyrepagella-regular.otf"),
            os.path.expanduser("~/.fonts/texgyrepagella-bold.otf")):
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["TeX Gyre Pagella", "Palatino", "P052", "DejaVu Serif"],
                     "text.color": INK})

XMAX, YMAX = 19.1, 10.0
FIG_W = 12.0

MODEL_COLORS = {"GPT 5.6 Sol": "#12A594", "Qwen 3.8 Max": "#F08A24", "Muse Spark 1.3": "#3B8FE0"}
# Each fictional model maps to the real lab behind it -> analysis/logos/<slug>.{svg,png}
COMPANY = {"GPT 5.6 Sol": ("OpenAI", "openai"), "Qwen 3.8 Max": ("Qwen", "qwen"),
           "Muse Spark 1.3": ("Meta", "meta")}
LOGO_DIR = REPO / "analysis/logos"
# per-logo size nudge (the OpenAI blossom carries lots of internal whitespace, so it reads small)
LOGO_SCALE = {"openai": 1.4}


# =========================================================================================
# primitives
# =========================================================================================
class Canvas:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(FIG_W, FIG_W * YMAX / XMAX))
        self.fig.patch.set_facecolor(BG)
        ax = self.ax
        ax.set_facecolor(BG)
        ax.set_xlim(0, XMAX)
        ax.set_ylim(0, YMAX)
        ax.set_aspect("equal")
        ax.axis("off")
        self.fig.subplots_adjust(0, 0, 1, 1)

    def rbox(self, x, y, w, h, fill="#FFFFFF", edge=CARD_E, lw=1.2, ls="-", r=0.18, z=3):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                         fc=fill, ec=edge, lw=lw, ls=ls, zorder=z))

    def text(self, x, y, t, size=9, color=INK, weight="normal", ha="center", va="center", z=8,
             **kw):
        self.ax.text(x, y, t, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
                     zorder=z, **kw)

    def arrow(self, p0, p1, color=AMBER, lw=2.2, ls="-", z=6):
        self.ax.annotate("", xy=p1, xytext=p0, zorder=z,
                         arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                         shrinkA=0, shrinkB=0, mutation_scale=16))

    def key(self, cx, cy, s=0.16, z=9):
        """Little key glyph (no emoji font on this box)."""
        ax = self.ax
        ax.add_patch(Circle((cx - 1.1 * s, cy), 0.9 * s, fc=KEY_Y, ec="#B8922E", lw=1.0, zorder=z))
        ax.add_patch(Circle((cx - 1.1 * s, cy), 0.35 * s, fc="#FFFFFF", ec="#B8922E", lw=0.8,
                            zorder=z + 1))
        ax.add_patch(Rectangle((cx - 0.3 * s, cy - 0.22 * s), 2.4 * s, 0.44 * s, fc=KEY_Y,
                               ec="#B8922E", lw=1.0, zorder=z))
        for dx in (1.3, 1.8):
            ax.add_patch(Rectangle((cx + dx * s, cy - 0.6 * s), 0.3 * s, 0.4 * s, fc=KEY_Y,
                                   ec="#B8922E", lw=1.0, zorder=z))

    def hf_box(self, cx, cy, w=2.7, h=1.35, hot=True):
        self.rbox(cx - w / 2, cy - h / 2, w, h, fill=AMBER_LL if hot else "#F1F2F4",
                  edge=AMBER if hot else CARD_E, lw=2.4 if hot else 1.2, r=0.22, z=3)
        self.text(cx, cy + 0.33, "Hugging Face", size=14, weight="bold",
                  color=INK if hot else HF_GREY)
        self.text(cx - 0.34, cy - 0.34, "answer key", size=11, color=MUTED)
        self.key(cx + 0.86, cy - 0.34, s=0.18)

    def save(self, name, outdir):
        outdir.mkdir(parents=True, exist_ok=True)
        out = outdir / f"social_preview_{name}.png"
        self.fig.savefig(out, dpi=200, facecolor=BG)
        plt.close(self.fig)
        return out


def _load_logo(slug):
    """Return an RGBA array for analysis/logos/<slug>.png|.svg, or None if absent."""
    import matplotlib.image as mpimg
    png = LOGO_DIR / f"{slug}.png"
    if png.exists():
        return mpimg.imread(str(png))
    svg = LOGO_DIR / f"{slug}.svg"
    if svg.exists():
        try:
            import io
            import cairosvg
            buf = io.BytesIO(cairosvg.svg2png(url=str(svg), output_width=512))
            return mpimg.imread(buf)
        except Exception:
            return None
    return None


def logo_tile(c, x, y, name, color, s=1.0, z=8):
    """Brand tile for a model: the lab's official logo if the file is present, otherwise a
    labeled placeholder."""
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    label, slug = COMPANY.get(name, (name, name.lower()))
    img = _load_logo(slug)
    if img is not None:
        px_per_unit = FIG_W / XMAX * c.fig.dpi
        zoom = (0.66 * s * LOGO_SCALE.get(slug, 1.0) * px_per_unit) / max(img.shape[0], img.shape[1])
        c.ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=zoom), (x, y), frameon=False,
                                       zorder=z + 1))
    else:
        c.text(x, y, label, size=13, color=color, weight="bold", z=z + 1)


def bubble(c: Canvas, x, y, w, h, text, tail="bl", color=INK, fill="#FFFFFF", size=9.5, z=9):
    """Speech bubble with a bottom-left ('bl'), bottom-centre ('bc') or bottom-right ('br') tail."""
    c.rbox(x, y, w, h, fill=fill, edge=color, lw=1.4, r=0.16, z=z)
    tx = {"bl": x + 0.45, "br": x + w - 0.45, "bc": x + w / 2}[tail]
    tip = {"bl": -0.32, "br": 0.32, "bc": 0.0}[tail]
    tri = [(tx - 0.18, y + 0.01), (tx + 0.18, y + 0.01), (tx + tip, y - 0.3)]
    yline = y + 0.012
    c.ax.add_patch(plt.Polygon(tri, closed=True, fc=fill, ec=color, lw=1.4, zorder=z))
    c.ax.plot([tri[0][0] + 0.03, tri[1][0] - 0.03], [yline, yline], color=fill, lw=2.2,
              zorder=z + 1)
    c.text(x + w / 2, y + h / 2, text, size=size, color=color, weight="bold", z=z + 2)


def _headline(c: Canvas, lines, x=0.7, mid=5.0, size=21):
    step = size * 0.043
    top = mid + step * len(lines) / 2 + 0.3
    for i, line in enumerate(lines):
        c.text(x, top - i * step, line, size=size, weight="bold", ha="left", va="top")


# =========================================================================================
# the card
# =========================================================================================
def social_preview(outdir, name="G_tri_logos"):
    """Three-lab triangle: GPT top-centre, Qwen bottom-left, Muse bottom-right, which then
    hacks Hugging Face."""
    c = Canvas()
    _headline(c, ("Reproducing the", "OpenAI–HuggingFace", "incident with",
                  "public models"), size=26, mid=5.0)

    sx, sy, sw, sh = 6.95, 1.2, 8.3, 7.7
    c.rbox(sx, sy, sw, sh, fill="none", edge=DASH, lw=1.4, ls=(0, (5, 4)), r=0.3, z=1)
    c.text(sx + 0.45, sy + sh - 0.42, "Docker sandbox  ·  no internet", size=12,
           color=MUTED, weight="bold", ha="left")

    cx = sx + sw / 2
    top, bot = 6.6, 3.15           # Qwen/Muse on the bottom row; GPT at the apex
    gpt, qwen, muse = "GPT 5.6 Sol", "Qwen 3.8 Max", "Muse Spark 1.3"
    pos = {gpt: (cx, top), qwen: (cx - 2.6, bot), muse: (cx + 2.6, bot)}

    def edge(a, b, style="<|-|>", shrinkA=42, shrinkB=42, z=4, offA=(0.0, 0.0)):
        p0 = (pos[a][0] + offA[0], pos[a][1] + offA[1])
        c.ax.annotate("", xy=pos[b], xytext=p0, zorder=z,
                      arrowprops=dict(arrowstyle=style, color=DASH, lw=1.6, shrinkA=shrinkA,
                                      shrinkB=shrinkB, mutation_scale=17))
    # leave GPT from its lower corners so the arrows clear the "GPT 5.6 Sol" label
    edge(gpt, qwen, offA=(-0.42, -0.1))
    edge(gpt, muse, offA=(0.42, -0.1))
    edge(qwen, muse, shrinkA=36, shrinkB=36)

    for n, (x, y) in pos.items():
        logo_tile(c, x, y, n, MODEL_COLORS[n], s=1.12)
        c.text(x, y - 1.08, n, size=12, color=MUTED, weight="bold", z=10,
               bbox=dict(facecolor=BG, edgecolor="none", boxstyle="round,pad=0.25"))

    bubble(c, pos[gpt][0] + 0.35, pos[gpt][1] + 0.82, 2.5, 0.66, "stuck, please help?",
           tail="bl", color=INK, size=10.5)
    bubble(c, pos[qwen][0] - 1.125, pos[qwen][1] + 1.1, 2.25, 0.66, "exploit here",
           tail="bc", color=INK, fill="#FFFFFF", size=10.5)

    hx, hy = 17.2, pos[muse][1]
    c.ax.add_patch(Rectangle((sx + sw - 0.16, hy - 0.32), 0.32, 1.08, fc=BG, ec="none", zorder=2))
    c.hf_box(hx, hy, w=3.1, h=1.65)
    c.arrow((pos[muse][0] + 0.85, hy), (hx - 1.55, hy), lw=3.2)
    c.text((pos[muse][0] + 0.85 + hx - 1.55) / 2, hy + 0.34, "hacks", size=13, color=AMBER,
           weight="bold", bbox=dict(facecolor=BG, edgecolor="none", boxstyle="round,pad=0.15"))
    return c.save(name, outdir)


if __name__ == "__main__":
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTDIR
    print(social_preview(outdir))
