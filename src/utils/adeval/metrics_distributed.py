import torch
import torch.distributed as dist
from src.utils.distributed import concat_all_gather

@torch.no_grad()
def f1_max_gpu_hist(scores: torch.Tensor,
                    labels: torch.Tensor,
                    n_bins: int = 1001,
                    eps: float = 1e-8,
                    distributed: bool = False,
                    device: torch.device = None,
                    chunk_size: int = 64_000_000):
    """
    Memory-efficient F1-max on GPU.
    scores : (N,)  float32/float16, already in [0,1] (will be min-max normalised again)
    labels : (N,)  bool / {0,1} tensor   (1=anomaly)
    n_bins : number of threshold bins (≥2)
    eps    : numerical stabiliser
    distributed : if True, aggregate histograms across processes (DDP)
    device : device to build the histograms on; scores/labels may live on CPU
             and are streamed to it in chunks of `chunk_size` elements, so
             peak GPU memory stays O(chunk_size) regardless of N (the previous
             whole-tensor scatter_add allocated several N-sized temporaries
             and OOM'd on very large pixel sets).
    """
    if device is None:
        device = scores.device if scores.is_cuda else torch.device('cuda')

    g_min = scores.min().to(device)
    g_max = scores.max().to(device)
    if distributed and dist.is_available() and dist.is_initialized():
        # compute global min/max over scores
        local_minmax = torch.stack([g_min, g_max])  # (2,)
        gathered = concat_all_gather(local_minmax).view(-1, 2)  # (world, 2)
        g_min = gathered[:, 0].min()
        g_max = gathered[:, 1].max()

    pos_per_bin = torch.zeros(n_bins, device=device, dtype=torch.int64)
    neg_per_bin = torch.zeros_like(pos_per_bin)

    n = scores.numel()
    for s in range(0, n, chunk_size):
        sc = scores[s:s + chunk_size].to(device, non_blocking=True)
        lb = labels[s:s + chunk_size].to(device, non_blocking=True).bool()
        sc = (sc - g_min) / (g_max - g_min + eps)
        sc = torch.clamp(sc, 0.0, 1.0 - eps)
        bi = (sc * (n_bins - 1)).long()
        pos_per_bin += torch.bincount(bi[lb], minlength=n_bins)
        neg_per_bin += torch.bincount(bi[~lb], minlength=n_bins)

    if distributed and dist.is_available() and dist.is_initialized():
        # sum histograms over all processes
        dist.all_reduce(pos_per_bin, op=dist.ReduceOp.SUM)
        dist.all_reduce(neg_per_bin, op=dist.ReduceOp.SUM)

    tp_cum = pos_per_bin.flip(0).cumsum(0).flip(0).to(torch.float32)
    fp_cum = neg_per_bin.flip(0).cumsum(0).flip(0).to(torch.float32)

    total_pos = tp_cum[0]
    fn_cum = total_pos - tp_cum

    denom = 2 * tp_cum + fp_cum + fn_cum + eps
    f1 = (2 * tp_cum) / denom

    best = torch.argmax(f1)
    thr = best.float() / (n_bins - 1)

    return f1[best], thr