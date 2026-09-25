"""mAD-vs-denoising-steps sweep from cached latents, with the perturbation noise
PAIRED across step counts K (same z1, same eps for every K -> differences are due
to the number of steps, not noise realization).

Protocol identical to src/vfad/eval_recon.py::evaluate_recon:
  x_t0 = t0 z1 + (1-t0) eps ; Euler K steps of the learned field t0 -> 1;
  score = channel-summed latent MSE, bilinear-upsampled to 256; img score diff+sum.

Writes eval_results_<K>.csv (same schema as artifacts/design_ablations rawdata)
into --out/<tag>/t0<t0>/.

Run from repo root:
  python sweep_mad.py --cache <dir> --ckpt <path> --t0 0.1 --ks 1 2 4 8 20 40 80 200 \
      --out <dir> --tag ep300 --device cuda:0
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.getcwd())
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

from src.utils.adeval.eval_utils import (
    calculate_img_metrics, calculate_px_metrics, divide_by_class, aggregate_px_values)
from measure_vgap import load_vf
from src.flow_matching.velocity_model import WrappedModel

METRICS = ['img_auroc', 'img_ap', 'img_f1max', 'px_auroc', 'px_ap', 'px_f1max', 'px_aupro']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--t0', type=float, required=True)
    ap.add_argument('--ks', type=int, nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = os.path.join(args.out, args.tag, f't0{args.t0:g}'.replace('.', 'p'))
    if args.seed != 0:
        out_dir = os.path.join(out_dir, f'seed{args.seed}')
    os.makedirs(out_dir, exist_ok=True)

    test = torch.load(os.path.join(args.cache, 'test_latents.pt'), map_location='cpu')
    mask_data = torch.load(os.path.join(args.cache, 'test_masks256.pt'), map_location='cpu')
    names = torch.load(os.path.join(args.cache, 'class_names.pt'))
    num_classes = len(names)
    assert torch.equal(mask_data['index'], test['index'])

    vf, feat_sz = load_vf(args.ckpt, num_classes, device)
    wrapped = WrappedModel(vf.model, vf.path, vf.pred_type,
                           cfg_interval=vf.cfg_interval, cfg_scale=vf.cfg_scale,
                           eps=vf.div_eps)

    g = torch.Generator(device='cpu').manual_seed(args.seed)
    noise_base = torch.randn((1, *feat_sz), generator=g).to(device)

    N = test['z'].shape[0]
    clslabels = test['clslabel'].numpy()
    img_gts = (test['label'].numpy() > 0).astype(np.uint8)
    px_gts = mask_data['mask256'].numpy()

    for K in args.ks:
        csv_path = os.path.join(out_dir, f'eval_results_{K}.csv')
        if os.path.exists(csv_path):
            print(f'skip existing {csv_path}')
            continue
        tgrid = torch.linspace(args.t0, 1.0, K + 1)
        mse_all = np.zeros((N, 256, 256), np.float32)
        with torch.no_grad():
            for s in tqdm(range(0, N, args.batch), desc=f'{args.tag} t0={args.t0} K={K}'):
                z1 = test['z'][s:s + args.batch].to(device, torch.float32)
                cls = test['clslabel'][s:s + args.batch].to(device)
                b = z1.shape[0]
                eps = noise_base.expand(b, -1, -1, -1)
                x = args.t0 * z1 + (1.0 - args.t0) * eps
                for k in range(K):
                    t_vec = torch.full((b,), float(tgrid[k]), device=device)
                    v = wrapped(x, t_vec, y=cls)
                    x = x + float(tgrid[k + 1] - tgrid[k]) * v
                mse = F.mse_loss(x, z1, reduction='none').sum(dim=1, keepdim=True)
                mse = F.interpolate(mse, size=(256, 256), mode='bilinear',
                                    align_corners=False).squeeze(1)
                mse_all[s:s + b] = mse.cpu().numpy()

        img_scores = aggregate_px_values(agg_method='diff+sum', px_values=mse_all)
        res = {}
        for c, sel_scores in divide_by_class(img_scores, clslabels).items():
            row = calculate_img_metrics(
                gt_labels=img_gts[clslabels == c], pred_scores=sel_scores,
                metrics=[m for m in METRICS if m.startswith('img_')])
            row.update({k: float(v) for k, v in calculate_px_metrics(
                gt_masks=px_gts[clslabels == c], pred_scores=mse_all[clslabels == c],
                metrics=[m for m in METRICS if m.startswith('px_')],
                device=device).items()})
            res[names[c]] = row
        avg = {m: float(np.mean([res[cn][m] for cn in res])) for m in METRICS}
        res['average'] = avg
        for cn in res:
            res[cn]['mad'] = float(np.mean([res[cn][m] for m in METRICS]))
        df = pd.DataFrame.from_dict(res, orient='index')
        df.index.name = 'category'
        df.reset_index(inplace=True)
        df.to_csv(csv_path, index=False)
        print(f'K={K}: mAD={res["average"]["mad"]:.4f} img_auroc={avg["img_auroc"]:.4f} -> {csv_path}')


if __name__ == '__main__':
    main()
