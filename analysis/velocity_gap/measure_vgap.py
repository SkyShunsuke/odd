"""Measure the discrepancy between the learned velocity v_theta and the closed-form
optimal velocity v* along the deployed reconstruction trajectory.

Protocol (matches src/vfad/eval_recon.py::evaluate_recon):
  x_{t0} = t0 * z1 + (1 - t0) * eps            (CondOT perturbation, shared eps)
  Euler steps k = 0..K-1 on the LEARNED field from t0 to 1.
At every visited state (x_k, t_k) we evaluate both fields:
  v_theta(x_k, t_k, y)   -- DiT through WrappedModel (deployed field)
  v*(x_k, t_k)           -- closed-form optimal velocity of the empirical FM path
                            for the class-conditional training distribution:
                            v* = (sum_i w_i x_i - x_k) / (1 - t_k),
                            w_i = softmax(-||x_k - t_k x_i||^2 / (2 (1-t_k)^2))
                            over the training latents x_i of the query's class.

Outputs per (ckpt, t0): npz with per-image per-step relerr / cosine / norms
plus per-image metadata, and one summary row appended to a CSV.

Run from repo root, e.g.:
  python measure_vgap.py --cache <dir> --ckpt <path> --t0 0.1 --steps 20 \
      --out <outdir> --tag ep300 --device cuda:0
"""
import argparse, os, sys, math, json
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.getcwd())
from src.models import init_model
from src.backbones import get_backbone_feature_shape
from src.flow_matching.velocity_model import VelocityField, WrappedModel

MODEL_PARAMS = dict(
    class_dropout_prob=0.0, cond_channels=768, depth=12, hidden_size=768,
    learn_sigma=False, mlp_ratio=4.0, model_name='dit', num_heads=12,
    patch_size=1, t_scale=1000, use_class_labels=True,
)
FM_PARAMS = dict(
    scheduler_name='affine_prob', solver_name='euler',
    loss_type='velocity', pred_type='velocity', train_steps=-1,
    t_scheduler_train='linear', t_scheduler_infer='linear',
    t_mu=-0.8, t_sigma=0.8,
)


def load_vf(ckpt_path, num_classes, device):
    feat_sz = get_backbone_feature_shape(model_name='efficientnet-b4')
    model = init_model(input_sz=feat_sz, num_classes=num_classes, **MODEL_PARAMS).to(device)
    vf = VelocityField(model=model, input_sz=feat_sz, **FM_PARAMS)
    sd = torch.load(ckpt_path, map_location='cpu')['model']
    sd = {k.removeprefix('module.'): v for k, v in sd.items()
          if not k.removeprefix('module.').startswith('ema_model.')}
    missing, unexpected = vf.load_state_dict(sd, strict=False)
    assert not [k for k in missing if not k.startswith('solver.')], f'missing keys: {missing}'
    assert not unexpected, f'unexpected keys: {unexpected}'
    vf.to(device).eval()
    return vf, feat_sz


class ClassClosedFM:
    """Image-level closed-form optimal CondOT velocity, per-class reference sets."""

    def __init__(self, train_z, train_cls, device):
        self.refs = {}
        for c in train_cls.unique().tolist():
            r = train_z[train_cls == c].to(device, torch.float32)
            self.refs[c] = r.view(r.shape[0], -1)          # (Nc, D)

    @torch.no_grad()
    def v_opt(self, xt, cls, t):
        """xt (B,C,H,W); cls (B,); scalar t in [0,1). Returns v* with same shape."""
        B = xt.shape[0]
        q_all = xt.view(B, -1).to(torch.float32)           # (B, D)
        out = torch.empty_like(q_all)
        one_minus_t = 1.0 - t
        temp = 2.0 * one_minus_t ** 2
        for c in cls.unique().tolist():
            sel = (cls == c)
            q = q_all[sel]                                  # (b, D)
            r = self.refs[c]                                # (Nc, D)
            # -||q - t r||^2 / temp, computed without materializing diffs
            scores = (2.0 * t * (q @ r.T)
                      - (q * q).sum(1, keepdim=True)
                      - (t ** 2) * (r * r).sum(1).unsqueeze(0)) / temp
            w = torch.softmax(scores, dim=1)                # (b, Nc)
            mean_ref = w @ r                                # (b, D)
            out[sel] = (mean_ref - q) / one_minus_t
        return out.view_as(xt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--t0', type=float, required=True)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', required=True, help='checkpoint tag, e.g. ep300')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    base = f'vgap_{args.tag}_t0{args.t0:g}_K{args.steps}'
    if args.seed != 0:
        base += f'_seed{args.seed}'
    if os.path.exists(os.path.join(args.out, base + '.npz')):
        print(f'skip existing {base}.npz')
        return

    test = torch.load(os.path.join(args.cache, 'test_latents.pt'), map_location='cpu')
    train = torch.load(os.path.join(args.cache, 'train_latents.pt'), map_location='cpu')
    names = torch.load(os.path.join(args.cache, 'class_names.pt'))
    num_classes = len(set(train['clslabel'].tolist()))

    vf, feat_sz = load_vf(args.ckpt, num_classes, device)
    wrapped = WrappedModel(vf.model, vf.path, vf.pred_type,
                           cfg_interval=vf.cfg_interval, cfg_scale=vf.cfg_scale,
                           eps=vf.div_eps)
    cfm = ClassClosedFM(train['z'], train['clslabel'], device)

    # shared perturbation noise, fixed across ckpts / t0 / K for comparability
    g = torch.Generator(device='cpu').manual_seed(args.seed)
    noise_base = torch.randn((1, *feat_sz), generator=g).to(device)

    K = args.steps
    tgrid = torch.linspace(args.t0, 1.0, K + 1)[:K]         # times where fields are evaluated
    N = test['z'].shape[0]
    relerr = np.zeros((N, K), np.float32)
    cossim = np.zeros((N, K), np.float32)
    vnorm_th = np.zeros((N, K), np.float32)
    vnorm_opt = np.zeros((N, K), np.float32)
    # same quantities restricted to defect cells (16x16 mask), anomalous images only
    relerr_def = np.full((N, K), np.nan, np.float32)

    order = torch.arange(N)
    with torch.no_grad():
        for s in tqdm(range(0, N, args.batch), desc=f'{args.tag} t0={args.t0}'):
            idx = order[s:s + args.batch]
            idx_np = idx.numpy()
            z1 = test['z'][idx].to(device, torch.float32)
            cls = test['clslabel'][idx].to(device)
            m16 = test['mask16'][idx].to(device).bool()      # (b,16,16)
            b = z1.shape[0]

            eps = noise_base.expand(b, -1, -1, -1)
            x = args.t0 * z1 + (1.0 - args.t0) * eps         # CondOT perturbation
            for k in range(K):
                t_k = float(tgrid[k])
                t_vec = torch.full((b,), t_k, device=device)
                v_th = wrapped(x, t_vec, y=cls).float()
                v_op = cfm.v_opt(x, cls, t_k)

                diff = (v_th - v_op).flatten(1)
                nrm_op = v_op.flatten(1).norm(dim=1)
                nrm_th = v_th.flatten(1).norm(dim=1)
                relerr[idx_np, k] = (diff.norm(dim=1) / nrm_op.clamp_min(1e-12)).cpu().numpy()
                cossim[idx_np, k] = ((v_th.flatten(1) * v_op.flatten(1)).sum(1)
                                     / (nrm_th * nrm_op).clamp_min(1e-12)).cpu().numpy()
                vnorm_th[idx_np, k] = nrm_th.cpu().numpy()
                vnorm_opt[idx_np, k] = nrm_op.cpu().numpy()

                has_def = m16.flatten(1).any(1)
                if has_def.any():
                    d2 = ((v_th - v_op) ** 2).sum(1)         # (b,16,16)
                    o2 = (v_op ** 2).sum(1)
                    mm = m16.float()
                    re_def = torch.sqrt((d2 * mm).flatten(1).sum(1)
                                        / (o2 * mm).flatten(1).sum(1).clamp_min(1e-12))
                    sub = idx_np[has_def.cpu().numpy()]
                    relerr_def[sub, k] = re_def[has_def].cpu().numpy()

                # Euler step of the DEPLOYED field (learned trajectory)
                h = (tgrid[k + 1] if k + 1 < K else torch.tensor(1.0)) - tgrid[k]
                x = x + float(h) * v_th

    np.savez_compressed(
        os.path.join(args.out, base + '.npz'),
        relerr=relerr, cossim=cossim, vnorm_th=vnorm_th, vnorm_opt=vnorm_opt,
        relerr_def=relerr_def, tgrid=tgrid.numpy(),
        clslabel=test['clslabel'].numpy(), anom=test['label'].numpy(),
        class_names=json.dumps({int(k): v for k, v in names.items()}),
    )

    # summary row: trajectory means over all / normal-only images
    anom = test['label'].numpy() > 0
    row = dict(tag=args.tag, ckpt=args.ckpt, t0=args.t0, K=K,
               relerr_traj_all=float(relerr.mean()),
               relerr_traj_normal=float(relerr[~anom].mean()),
               relerr_t0_all=float(relerr[:, 0].mean()),
               relerr_t0_normal=float(relerr[~anom, 0].mean()),
               cos_traj_all=float(cossim.mean()),
               cos_traj_normal=float(cossim[~anom].mean()),
               cos_t0_all=float(cossim[:, 0].mean()),
               relerr_def_traj=float(np.nanmean(relerr_def)),
               )
    import csv
    csv_path = os.path.join(args.out, 'vgap_summary.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(json.dumps(row, indent=2))


if __name__ == '__main__':
    main()
