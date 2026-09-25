# One-Step Denoising is Optimal for Flow-Matching Anomaly Detection

Anonymized code release for reproducing the **one-step optimality** observation:
for a well-trained flow-matching (FM) generative model over frozen-backbone
latents, reconstruction-based anomaly detection is best with a **single** Euler
denoising step — additional steps only hurt, and the multi-step penalty grows
with the discrepancy between the learned velocity and the closed-form optimal
velocity.

The generative model (a DiT) is trained in the latent feature space of a frozen
pretrained encoder (EfficientNet-B4 by default), not in pixel space. Anomaly
scores come from latent reconstruction error, upsampled to a pixel anomaly map.

## Setup

```bash
pip install -r requirements.txt
mkdir -p logs          # required before any run (logging opens logs/app_rank*.log)
bash scripts/download_mvtec.sh   # MVTec AD -> data/mvtec_ad
```

Everything runs through `torchrun` (even single-GPU) and **must be launched from
the repo root** (imports are absolute, e.g. `from src.flow_matching import ...`).

Supported datasets: `mvtec_ad`, `visa`, `mpdd` (and `*_all` multi-class
variants). Place VisA / MPDD under `data/visa`, `data/mpdd` manually.

## 1. Train the FM model

Multi-class MVTec AD (15 categories jointly, class-conditional DiT-B on
EfficientNet-B4 latents, CondOT path, 300 epochs):

```bash
bash scripts/train.sh    # configs/train/mvtec_dit_fm.yaml, 8 GPUs by default
```

Outputs go to `logs/mvtec_dit_fm/` (`checkpoints/`, `tb_logs/`, per-epoch eval
CSVs). `logging.ckpt.ckpt_epoch_list` saves the intermediate checkpoints used by
the training-budget analysis below.

## 2. Denoising-step sweep (main observation)

```bash
bash scripts/eval.sh     # configs/eval/mvtec_dit_fm_steps.yaml
```

One invocation sweeps `logging.eval.ablation_steps = [1, 2, 4, 8, 20, 40, 80, 200]`
and writes `eval_results_<K>.csv` (per-category image/pixel metrics) into
`logs/mvtec_dit_fm/results/`. The summary metric **mAD** is the mean of the seven
configured metrics; K = 1 attains the best mAD, and mAD decreases monotonically
in K for the well-trained model. `logging.eval.inv_stop_time` is the CondOT
perturbation time t0 (x_t0 = t0*z + (1-t0)*eps; smaller t0 = more noise).

`--img_metric_only` (append inside the script) skips the expensive pixel
metrics (PRO/AU-PRO).

## 3. Closed-form optimal denoiser baseline

The optimal velocity field of the empirical FM path has a closed form (a
softmax-weighted average over training latents), so the optimal denoiser can be
evaluated without training:

```bash
bash scripts/eval_optim.sh   # configs/eval/optimal_denoiser.yaml, single GPU
```

For the exact optimal denoiser the number of steps is (near-)irrelevant — mAD is
flat in K — which isolates the one-step advantage of the *learned* model as an
effect of imperfect velocity approximation.

## 4. Velocity-gap analysis (learned vs optimal velocity)

`analysis/velocity_gap/` quantifies delta = ||v_theta - v*|| / ||v*|| along the
deployed K-step Euler reconstruction trajectory and relates it to the multi-step
penalty mAD(1) - mAD(K), over both the noise axis (t0 in 0.1..0.9 at the final
checkpoint) and the training-budget axis (checkpoints from epoch 5 to 300 at
t0 = 0.1). All scripts run from the repo root:

```bash
# one-time caches (backbone is frozen, so latents are checkpoint-independent)
python analysis/velocity_gap/cache_latents.py --out analysis/velocity_gap/cache
python analysis/velocity_gap/cache_masks.py   --out analysis/velocity_gap/cache

# all (checkpoint, t0) conditions, one job per GPU
# (RUN_DIR defaults to logs/mvtec_dit_fm; override via env if needed)
python analysis/velocity_gap/run_all.py

# aggregate + figures
python analysis/velocity_gap/aggregate.py \
    --vgap logs/mvtec_dit_fm/velocity_gap --mad logs/mvtec_dit_fm/mad_sweeps \
    --out  logs/mvtec_dit_fm/velocity_gap/agg
python analysis/velocity_gap/plot_vgap.py        --agg logs/mvtec_dit_fm/velocity_gap/agg --out figs
python analysis/velocity_gap/plot_vgap_budget.py --agg logs/mvtec_dit_fm/velocity_gap/agg --out figs
```

Expected findings: (i) the relative velocity error collapses onto one master
curve in trajectory time t; (ii) K = 1 is optimal for every well-trained
condition, with the multi-step penalty growing with delta; (iii) one-step
optimality breaks only under severe under-training (early checkpoints), and
non-monotonically.

## Config reference

Each run is a single YAML file (`meta / data / model / flow_matching / opt /
logging / resume`). Sub-dicts are splatted directly into the factories
(`build_dataset`, `init_model`, ...), so YAML keys are function kwargs.
`model:` holds the trained DiT; the frozen encoder is nested at
`model.backbone`. The framework identity is the `scheduler` x `pred_type` x
`loss_type` triple (this release ships the plain CondOT FM configuration;
`flow_matching.vf_type: plain`).

Note (train vs eval configs): `global_seed` / `use_bfloat16` live under `opt:`
for training and under `meta:` for evaluation.

## Acknowledgements / third-party code

- `src/flow_matching/` (except `velocity_model.py`, `closed_fm.py`,
  `shortcut_models.py`) is adapted from Meta's
  [flow_matching](https://github.com/facebookresearch/flow_matching) library
  (CC BY-NC 4.0; headers preserved).
- `src/utils/adeval/` is a vendored copy of
  [ADEval](https://pypi.org/project/adeval/) (fast AUROC/AUPR/AUPRO).
- `src/backbones/efficientnet.py` follows the standard PyTorch EfficientNet
  reimplementation.
