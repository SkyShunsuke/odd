#!/usr/bin/env python3
"""Compact 2x1 figure (1.83 x 2.30 in canvas): velocity discrepancy vs training
budget (top) and Delta-accuracy vs denoising steps per budget (bottom).

Styling follows artifacts/design_ablations/scripts/plot_noise.py
(Palatino text, light y-grid, dotted y=0 reference, all-s x ticks, 0.22 spines,
framed legend, Delta metric scaled x100), while the epoch color coding of the
earlier draft (plasma, range 0-0.8) is retained and links the two panels.
"""
import argparse, os, warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

EPS = [10, 20, 40, 80, 300]
SCALE = 100.0          # Delta mAD in percentage points, as in plot_noise.py
TICK_FS = 5.2
LABEL_FS = 6.0
LEGEND_FS = 4.8


def _palatino_font(size: float) -> FontProperties:
    """Resolve Palatino or a metrically similar family (as in plot_noise.py)."""
    for family in ("Palatino", "Palatino Linotype", "Book Antiqua",
                   "TeX Gyre Pagella", "URW Palladio L", "P052"):
        prop = FontProperties(family=family, weight="normal", size=size)
        try:
            font_manager.findfont(prop, fallback_to_default=False)
            return prop
        except ValueError:
            continue
    warnings.warn("Palatino-compatible font not found; falling back to DejaVu Serif.")
    return FontProperties(family="DejaVu Serif", weight="normal", size=size)


def _style_axis(ax, tick_font):
    ax.grid(axis="y", color="0.90", linewidth=0.45, alpha=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("0.22")
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(axis="both", which="major", direction="out",
                   length=2.2, width=0.55, colors="0.18", pad=1.8)
    ax.tick_params(axis="both", which="minor", direction="out",
                   length=1.4, width=0.55, colors="0.18")
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontproperties(tick_font)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--agg', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    mad = pd.read_csv(os.path.join(args.agg, 'mad_steps.csv'))
    if 'seed' in mad:
        mad = mad[mad.seed == 0]
    mg = pd.read_csv(os.path.join(args.agg, 'merged.csv'))

    cmap_ep = plt.get_cmap('plasma')
    ep_color = {ep: cmap_ep(i / (len(EPS) - 1) * 0.8) for i, ep in enumerate(EPS)}

    tick_font = _palatino_font(TICK_FS)
    palatino = tick_font.get_name()

    with mpl.rc_context({
        "font.family": palatino,
        "font.serif": [palatino],
        "font.size": LABEL_FS,
        "axes.labelsize": LABEL_FS,
        "axes.labelpad": 1.5,
        "mathtext.fontset": "custom",
        "mathtext.rm": palatino,
        "mathtext.it": f"{palatino}:italic",
        "mathtext.bf": f"{palatino}:bold",
        "mathtext.bfit": f"{palatino}:bold:italic",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(1.83, 2.30), constrained_layout=True)

        # ---------- top: discrepancy vs training budget (t0 = 0.1) ----------
        bsel = mg[np.isclose(mg.t0, 0.1)].sort_values('epoch')
        ax1.plot(bsel.epoch, bsel.relerr_first, color="0.45", linewidth=1.0,
                 solid_capstyle="round", solid_joinstyle="round", zorder=2)
        ax1.plot(bsel.epoch, bsel.relerr_first, linestyle="None", marker="o",
                 markersize=2.2, markerfacecolor="0.45", markeredgecolor="0.45",
                 markeredgewidth=0.0, zorder=3)
        for ep in EPS:      # colored markers link to the bottom panel
            r = bsel[bsel.epoch == ep]
            ax1.plot(r.epoch, r.relerr_first, linestyle="None", marker="o",
                     markersize=2.6, markerfacecolor=ep_color[ep],
                     markeredgecolor=ep_color[ep], markeredgewidth=0.0,
                     zorder=3.5)
        ax1.set_xscale("log")
        epoch_ticks = [5, 20, 80, 300]
        ax1.set_xticks(epoch_ticks)
        ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax1.xaxis.set_minor_formatter(NullFormatter())
        # Labels are set without explicit fontproperties so the custom Palatino
        # mathtext (mathtext.it etc.) applies to "$...$" runs, exactly as in
        # plot_noise.py / dataset_ablations plot.py.
        ax1.set_xlabel("Training epochs")
        # \mathbfit is matplotlib's \boldsymbol equivalent (bold italic).
        ax1.set_ylabel(r"$\delta\!:\!\!=\|\mathbfit{v}_\theta-"
                       r"\mathbfit{v}^{*}\|\,/\,\|\mathbfit{v}^{*}\|$")
        ax1.margins(x=0.05)
        _style_axis(ax1, tick_font)

        handles = [Line2D([], [], color=ep_color[ep], linewidth=1.15,
                          marker="o", markersize=2.9,
                          markerfacecolor=ep_color[ep],
                          markeredgecolor=ep_color[ep], markeredgewidth=0.0,
                          label=f"{ep} ep")
                   for ep in EPS]
        legend_font = tick_font.copy()
        legend_font.set_size(LEGEND_FS)
        legend = ax1.legend(
            handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0),
            bbox_transform=ax1.transAxes, ncol=1, prop=legend_font,
            frameon=True, fancybox=False, shadow=False, framealpha=0.94,
            facecolor="white", edgecolor="0.78", borderpad=0.28,
            labelspacing=0.26, handlelength=1.35, handletextpad=0.42,
            borderaxespad=0.0)
        legend.get_frame().set_linewidth(0.55)
        legend.set_zorder(10)
        for text_artist in legend.get_texts():
            text_artist.set_color("0.16")

        # ---------- bottom: Delta accuracy vs denoising steps ----------
        ks = None
        for ep in EPS:
            sel = mad[(mad.tag == f'ep{ep}') & (np.isclose(mad.t0, 0.1))].sort_values('K')
            ks = sel.K.to_numpy(dtype=float)
            base = float(sel[sel.K == 1]['mad'].iloc[0])
            dy = (sel['mad'].to_numpy() - base) * SCALE
            c = ep_color[ep]
            ax2.plot(ks, dy, color=c, alpha=0.9, linewidth=1.0,
                     solid_capstyle="round", solid_joinstyle="round", zorder=2)
            ax2.plot(ks, dy, linestyle="None", marker="o", markersize=2.2,
                     markerfacecolor=c, markeredgecolor=c, markeredgewidth=0.0,
                     alpha=0.9, zorder=3.5)
        ax2.set_xscale("log")
        ax2.margins(x=0.025)
        reference_line = ax2.axhline(
            0.0, xmin=0.0, xmax=1.0, color="#000000", alpha=1.0,
            linewidth=0.8, linestyle=(0, (1.5, 3.0)), dash_capstyle="round",
            clip_on=False, zorder=3)
        reference_line.set_gid("diff-reference-y0")
        ax2.set_xticks([1, 4, 20, 80])
        ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax2.xaxis.set_minor_formatter(NullFormatter())
        ax2.set_xlabel(r"Number of denoising steps ($K$)")
        ax2.set_ylabel(r"ΔAccuracy (vs. $K$=1)")
        _style_axis(ax2, tick_font)

        for ax in (ax1, ax2):
            for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
                label.set_fontproperties(tick_font)

        fig.canvas.draw()
        for gridline in ax2.get_ygridlines():
            ydata = gridline.get_ydata()
            if len(ydata) > 0 and np.isclose(ydata[0], 0.0, atol=1e-12):
                gridline.set_visible(False)

        for ext in ('pdf', 'png'):
            fig.savefig(os.path.join(args.out, f'velocity_gap_budget.{ext}'),
                        dpi=600, bbox_inches='tight', pad_inches=0.02)
    print('saved', os.path.join(args.out, 'velocity_gap_budget.pdf'))


if __name__ == '__main__':
    main()
