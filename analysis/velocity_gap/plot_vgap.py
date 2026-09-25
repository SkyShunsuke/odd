#!/usr/bin/env python3
"""Figure: discrepancy between learned and optimal velocity vs denoising-step optimality.

Panels:
 (a) relerr(t) along the K=20 reconstruction trajectory, per perturbation strength t0 (final model)
 (b) relerr at the start point / trajectory mean vs training epoch (t0=0.1)
 (c) mAD(K) - mAD(1) vs K for training budgets (t0=0.1)
 (d) mAD(K) - mAD(1) vs K for noise strengths (final model)
Reads the tables produced by aggregate.py.
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.fontsize': 6.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 1.4, 'lines.markersize': 3.5,
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--agg', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tc = pd.read_csv(os.path.join(args.agg, 'vgap_timecurves.csv'))
    mad = pd.read_csv(os.path.join(args.agg, 'mad_steps.csv'))
    mg = pd.read_csv(os.path.join(args.agg, 'merged.csv'))

    fig, axes = plt.subplots(1, 4, figsize=(9.6, 1.95))
    cmap_t0 = plt.get_cmap('viridis')
    cmap_ep = plt.get_cmap('plasma')

    # ---- (a) relerr along trajectory, per t0, ep300
    ax = axes[0]
    t0s = [0.1, 0.3, 0.5, 0.7, 0.9]
    for i, t0 in enumerate(t0s):
        sel = tc[(tc.tag == 'ep300') & (np.isclose(tc.t0, t0))].sort_values('t')
        ax.plot(sel.t, sel.relerr_all, color=cmap_t0(i / (len(t0s) - 1) * 0.85),
                label=f'$t_0$={t0:g}')
    ax.set_xlabel('trajectory time $t$')
    ax.set_ylabel(r'$\|v_\theta - v^*\| \, / \, \|v^*\|$')
    ax.set_title('(a) discrepancy along trajectory', loc='left')
    ax.legend(frameon=False, ncol=2, handlelength=1.2, columnspacing=0.8)

    # ---- (b) relerr vs training epoch at t0=0.1
    ax = axes[1]
    bsel = mg[np.isclose(mg.t0, 0.1)].sort_values('epoch')
    ax.plot(bsel.epoch, bsel.relerr_first, 'o-', color='#0072B2', label='at start point $t_0$')
    ax.plot(bsel.epoch, bsel.relerr_traj, 's--', color='#D55E00', label='trajectory mean')
    ax.set_xscale('log')
    ax.set_xlabel('training epochs')
    ax.set_ylabel(r'$\|v_\theta - v^*\| \, / \, \|v^*\|$')
    ax.set_title('(b) discrepancy vs training budget', loc='left')
    ax.legend(frameon=False)

    # ---- helpers for (c)/(d): mean over seeds, Delta mAD(K)
    def delta_curve(tag, t0):
        sel = mad[(mad.tag == tag) & (np.isclose(mad.t0, t0))]
        g = sel.groupby('K')['mad'].agg(['mean', 'std', 'count'])
        base = g.loc[1, 'mean']
        return g.index.values, g['mean'].values - base, g['std'].values, g['count'].values

    # ---- (c) budget axis
    ax = axes[2]
    eps = [10, 20, 40, 80, 300]
    for i, ep in enumerate(eps):
        K, d, s, n = delta_curve(f'ep{ep}', 0.1)
        rel = mg[(mg.epoch == ep) & np.isclose(mg.t0, 0.1)].relerr_first.iloc[0]
        c = cmap_ep(i / (len(eps) - 1) * 0.8)
        ax.plot(K, d, 'o-', color=c, label=f'ep{ep} ($\\delta$={rel:.2f})')
        if np.isfinite(s).all() and (n > 1).all():
            ax.fill_between(K, d - s, d + s, color=c, alpha=0.15, lw=0)
    ax.axhline(0.0, color='0.4', lw=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('denoising steps $K$')
    ax.set_ylabel('mAD$(K)$ $-$ mAD$(1)$')
    ax.set_title('(c) step optimality vs budget ($t_0$=0.1)', loc='left')
    ax.set_ylim(-0.021, 0.0155)
    ax.legend(frameon=False, handlelength=1.0, fontsize=6, labelspacing=0.25,
              borderaxespad=0.2, loc='lower left')

    # ---- (d) noise axis
    ax = axes[3]
    t0s = [0.1, 0.3, 0.5, 0.7, 0.9]
    for i, t0 in enumerate(t0s):
        K, d, s, n = delta_curve('ep300', t0)
        rel = mg[(mg.epoch == 300) & np.isclose(mg.t0, t0)].relerr_first.iloc[0]
        c = cmap_t0(i / (len(t0s) - 1) * 0.85)
        ax.plot(K, d, 'o-', color=c, label=f'$t_0$={t0:g} ($\\delta$={rel:.2f})')
        if np.isfinite(s).all() and (n > 1).all():
            ax.fill_between(K, d - s, d + s, color=c, alpha=0.15, lw=0)
    ax.axhline(0.0, color='0.4', lw=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('denoising steps $K$')
    ax.set_ylabel('mAD$(K)$ $-$ mAD$(1)$')
    ax.set_title('(d) step optimality vs noise strength', loc='left')
    ax.legend(frameon=False, handlelength=1.0, fontsize=6, labelspacing=0.25,
              borderaxespad=0.2, loc='upper right')

    fig.tight_layout(pad=0.4)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(args.out, f'velocity_gap.{ext}'), dpi=200,
                    bbox_inches='tight')
    print('saved', os.path.join(args.out, 'velocity_gap.pdf'))


if __name__ == '__main__':
    main()
