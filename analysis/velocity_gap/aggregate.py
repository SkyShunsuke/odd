"""Aggregate velocity-gap npz files and mAD-sweep CSVs into tidy tables.

Outputs (into --out):
  vgap_conditions.csv : one row per (tag, t0): trajectory/time-resolved discrepancy stats
  vgap_timecurves.csv : per (tag, t0, step): mean relerr / cos over images (all & normal-only)
  mad_steps.csv       : per (tag, t0, K): average mAD + img_auroc etc.
  merged.csv          : condition-level join for the scatter panel
"""
import argparse, glob, json, os, re
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vgap', required=True)
    ap.add_argument('--mad', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--use-seeds', action='store_true',
                    help='include seed>0 sweeps in the condition-level stats '
                         '(default: seed 0 only, extra seeds only listed in mad_steps.csv)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---- velocity gap ----
    cond_rows, time_rows = [], []
    for f in sorted(glob.glob(os.path.join(args.vgap, 'vgap_*.npz'))):
        m = re.match(r'vgap_(?P<tag>[^_]+)_t0(?P<t0>[\d.]+)_K(?P<K>\d+)\.npz', os.path.basename(f))
        if not m:
            continue
        d = np.load(f, allow_pickle=True)
        tag, t0, K = m['tag'], float(m['t0']), int(m['K'])
        anom = d['anom'] > 0
        relerr, cos, tgrid = d['relerr'], d['cossim'], d['tgrid']
        for k in range(K):
            time_rows.append(dict(
                tag=tag, t0=t0, k=k, t=float(tgrid[k]),
                relerr_all=float(relerr[:, k].mean()),
                relerr_normal=float(relerr[~anom, k].mean()),
                relerr_anom=float(relerr[anom, k].mean()),
                cos_all=float(cos[:, k].mean()),
                cos_normal=float(cos[~anom, k].mean()),
                relerr_def=float(np.nanmean(d['relerr_def'][:, k])),
            ))
        # time-weighted stats over the trajectory
        cond_rows.append(dict(
            tag=tag, t0=t0, K=K,
            relerr_traj=float(relerr.mean()),
            relerr_traj_normal=float(relerr[~anom].mean()),
            relerr_traj_median=float(np.median(relerr)),
            relerr_first=float(relerr[:, 0].mean()),
            relerr_first_normal=float(relerr[~anom, 0].mean()),
            one_minus_cos_traj=float(1 - cos.mean()),
            one_minus_cos_first=float(1 - cos[:, 0].mean()),
            cos_traj=float(cos.mean()),
        ))
    pd.DataFrame(time_rows).to_csv(os.path.join(args.out, 'vgap_timecurves.csv'), index=False)
    cond = pd.DataFrame(cond_rows)
    cond.to_csv(os.path.join(args.out, 'vgap_conditions.csv'), index=False)

    # ---- mAD sweeps (seed 0 at tag/t0/, extra seeds at tag/t0/seedN/) ----
    rows = []
    files = glob.glob(os.path.join(args.mad, '*', '*', 'eval_results_*.csv')) \
        + glob.glob(os.path.join(args.mad, '*', '*', 'seed*', 'eval_results_*.csv'))
    for f in sorted(files):
        parts = f.split(os.sep)
        if parts[-2].startswith('seed'):
            seed = int(parts[-2].replace('seed', ''))
            tag, t0s = parts[-4], parts[-3]
        else:
            seed = 0
            tag, t0s = parts[-3], parts[-2]
        t0 = float(t0s.replace('t0', '').replace('p', '.'))
        K = int(re.search(r'_(\d+)\.csv', f).group(1))
        df = pd.read_csv(f)
        avg = df[df.category == 'average'].iloc[0]
        rows.append(dict(tag=tag, t0=t0, K=K, seed=seed, mad=float(avg['mad']),
                         img_auroc=float(avg['img_auroc']), px_auroc=float(avg['px_auroc']),
                         px_aupro=float(avg['px_aupro'])))
    mad_all = pd.DataFrame(rows).sort_values(['tag', 't0', 'K', 'seed'])
    mad_all.to_csv(os.path.join(args.out, 'mad_steps.csv'), index=False)
    if not args.use_seeds:
        mad_all = mad_all[mad_all.seed == 0]
    mad = mad_all[mad_all.seed == 0]

    # ---- condition-level merge (seed mean when seeds exist; seed spread reported) ----
    piv = mad_all.pivot_table(index=['tag', 't0'], columns='K', values='mad', aggfunc='mean')
    piv_std = mad_all.pivot_table(index=['tag', 't0'], columns='K', values='mad',
                                  aggfunc='std', dropna=False)
    nseeds = mad_all.groupby(['tag', 't0']).seed.nunique()
    merged = []
    for (tag, t0), row in piv.iterrows():
        mad1 = row.get(1, np.nan)
        multi = row.drop(labels=[1], errors='ignore')
        best_multi = multi.max()
        kstar = row.idxmax()
        sel = cond[(cond.tag == tag) & (cond.t0 == t0)]
        if len(sel) == 0:
            continue
        s = sel.iloc[0]
        merged.append(dict(
            tag=tag, t0=t0, epoch=int(tag.replace('ep', '')),
            mad1=float(mad1), best_multi=float(best_multi), kstar=int(kstar),
            onestep_adv=float(mad1 - best_multi),
            mad20=float(row.get(20, np.nan)),
            drop20=float(mad1 - row.get(20, np.nan)),
            n_seeds=int(nseeds.get((tag, t0), 1)),
            mad1_std=float(piv_std.loc[(tag, t0)].get(1, np.nan))
            if (tag, t0) in piv_std.index else np.nan,
            mad20_std=float(piv_std.loc[(tag, t0)].get(20, np.nan))
            if (tag, t0) in piv_std.index else np.nan,
            relerr_traj=s.relerr_traj, relerr_first=s.relerr_first,
            relerr_first_normal=s.relerr_first_normal,
            one_minus_cos_traj=s.one_minus_cos_traj,
            one_minus_cos_first=s.one_minus_cos_first,
        ))
    pd.DataFrame(merged).sort_values(['tag', 't0']).to_csv(
        os.path.join(args.out, 'merged.csv'), index=False)
    print(pd.DataFrame(merged).sort_values(['epoch', 't0']).to_string())


if __name__ == '__main__':
    main()
