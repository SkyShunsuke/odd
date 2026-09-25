import os
import torch
import numpy as np  
import torch.nn as nn

from src.flow_matching.path.scheduler import CondOTScheduler, VPScheduler, \
    CosineScheduler, EDMScheduler, LinearVPScheduler
from src.flow_matching.path import AffineProbPath
from src.flow_matching.solver import Solver, ODESolver

def build_scheduler(scheduler_name: str, scheduler_params: dict) -> CondOTScheduler:
    """Build scheduler for flow matching.

    Args:
        scheduler_name (str): Name of the scheduler.
        scheduler_params (dict): Parameters for the scheduler.

    Returns:
        CondOTScheduler: Configured scheduler.
    """
    if scheduler_name == 'affine_prob':
        scheduler = CondOTScheduler(**scheduler_params)
        return AffineProbPath(scheduler)
    elif scheduler_name == 'vp':
        scheduler = VPScheduler(**scheduler_params)
        return AffineProbPath(scheduler)
    elif scheduler_name == 'cosine':
        scheduler = CosineScheduler(**scheduler_params)
        return AffineProbPath(scheduler)
    elif scheduler_name == 'linear_vp':
        scheduler = LinearVPScheduler(**scheduler_params)
        return AffineProbPath(scheduler)
    elif scheduler_name == 'edm':
        scheduler = EDMScheduler(**scheduler_params)
        return AffineProbPath(scheduler)
    else:
        raise ValueError(f'Unknown scheduler name: {scheduler_name}')
    
def build_solver(model: nn.Module, solver_name: str, solver_params: dict) -> Solver:
    """Build solver for flow matching.

    Args:
        model (nn.Module): The velocity model.
        solver_name (str): Name of the solver.
        solver_params (dict): Parameters for the solver.

    Returns:
        Solver: Configured solver.
    """
    return ODESolver(model, **solver_params)

def logit_t(bs: int, device: torch.device, mu=0., sigma=1.) -> torch.Tensor:
    # -- sample t from gaussian
    t = torch.randn(bs, device=device) * sigma + mu
    # -- apply sigmoid
    t = torch.sigmoid(t)
    return t

class LocalizedClosedFM(nn.Module):
    def __init__(
        self,
        train_samples,
        query_batch_size=4096,
        ref_batch_size=4096,
        store_dtype=None,
        bandwidth_min=0.,
    ):
        super().__init__()

        assert train_samples.ndim == 4
        N, C, H, W = train_samples.shape

        # Important:
        # (N, C, H, W) -> (N, H, W, C) -> (N*H*W, C)
        train_flat = (
            train_samples
            .detach()
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(-1, C)
        )

        if store_dtype is not None:
            train_flat = train_flat.to(store_dtype)

        # moves together with model.to(device)
        self.register_buffer("train_samples", train_flat, persistent=False)

        self.query_batch_size = query_batch_size
        self.ref_batch_size = ref_batch_size
        
        self.bandwidth_min = bandwidth_min
        
        self.path = build_scheduler("affine_prob", {})
        
    def perturb(self, x1, x0, t):
        """Perturb the data points for training.
        Args:
            x1 (torch.Tensor): The ending points.
            x0 (torch.Tensor): The starting points.
            t (torch.Tensor): The time steps.
        Returns:
            torch.Tensor: The perturbed points.
        """
        path_sample = self.path.sample(t=t, x_0=x0, x_1=x1)
        return path_sample.x_t


    @torch.no_grad()
    def get_optimal_v(self, xt, t):
        """
        Computes

            v(x_t, t) = ( E_w[x_ref] - x_t ) / (1 - t)

        where

            w_i ∝ exp( -||x_t - t x_ref_i||^2 / (2(1-t)^2) )

        without materializing w of shape (B*H*W, N*H*W).
        """

        assert xt.ndim == 4
        B, C, H, W = xt.shape
        device = xt.device

        assert torch.all(t == t[0]), \
            "All t values must be the same for computing weights."

        t0 = t.flatten()[0].to(device=device, dtype=torch.float32)
        one_minus_t = 1.0 - t0

        if one_minus_t <= 0:
            raise ValueError("t must be strictly smaller than 1.")

        temp = 2.0 * (one_minus_t.square() + self.bandwidth_min ** 2)

        # (B, C, H, W) -> (B*H*W, C)
        x_query_all = (
            xt
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(-1, C)
        )

        x_ref_all = self.train_samples
        if x_ref_all.device != device:
            raise RuntimeError(
                "train_samples is on a different device. "
                "Call model.to(xt.device) before sampling."
            )

        Q = x_query_all.shape[0]
        R = x_ref_all.shape[0]

        out = torch.empty((Q, C), device=device, dtype=xt.dtype)

        for q_start in range(0, Q, self.query_batch_size):
            q_end = min(q_start + self.query_batch_size, Q)

            q = x_query_all[q_start:q_end].to(torch.float32)
            q_size = q.shape[0]

            # Online log-sum-exp state
            # m: running max logit
            # z: running sum exp(logit - m)
            # s: running weighted sum of x_ref
            m = torch.full(
                (q_size, 1),
                -torch.inf,
                device=device,
                dtype=torch.float32,
            )
            z = torch.zeros(
                (q_size, 1),
                device=device,
                dtype=torch.float32,
            )
            s = torch.zeros(
                (q_size, C),
                device=device,
                dtype=torch.float32,
            )

            q_norm = (q * q).sum(dim=1, keepdim=True)

            for r_start in range(0, R, self.ref_batch_size):
                r_end = min(r_start + self.ref_batch_size, R)

                r = x_ref_all[r_start:r_end].to(torch.float32)

                r_norm = (r * r).sum(dim=1).unsqueeze(0)

                # score = -||q - t r||^2 / temp
                #
                # ||q - t r||^2
                # = ||q||^2 - 2t q r^T + t^2 ||r||^2
                #
                # score
                # = (-||q||^2 + 2t q r^T - t^2 ||r||^2) / temp
                scores = (
                    2.0 * t0 * (q @ r.T)
                    - q_norm
                    - t0.square() * r_norm
                ) / temp

                chunk_max = scores.max(dim=1, keepdim=True).values
                m_new = torch.maximum(m, chunk_max)

                old_scale = torch.exp(m - m_new)

                # scores becomes exp(scores - m_new)
                scores.sub_(m_new)
                scores.exp_()

                z = z * old_scale + scores.sum(dim=1, keepdim=True)
                s = s * old_scale + scores @ r

                m = m_new

                # optional: release references earlier
                del r, r_norm, scores, chunk_max, m_new, old_scale

            mean_ref = s / z.clamp_min(1e-30)

            v = (mean_ref - q) / one_minus_t
            out[q_start:q_end] = v.to(dtype=xt.dtype)

            del q, q_norm, m, z, s, mean_ref, v

        v_opt = (
            out
            .view(B, H, W, C)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return v_opt

    @torch.no_grad()
    def sample(
        self,
        x0,
        steps: int,
        start_t: float = 0.0,
        return_intermediate: bool = False,
        **model_kwargs,
    ):
        timesteps = torch.linspace(
            start_t,
            1.0,
            steps + 1,
            device=x0.device,
            dtype=x0.dtype,
        )

        step_size = (1.0 - start_t) / steps

        x_in = x0
        x_traj = [x0]

        for i in range(steps):
            t_in = timesteps[i].expand(x_in.shape[0])
            opt_v = self.get_optimal_v(x_in, t_in)

            x_in = x_in + opt_v * step_size

            if return_intermediate:
                x_traj.append(x_in)

        if return_intermediate:
            return x_in, x_traj

        return x_in

class ClosedFM(nn.Module):
    def __init__(self, train_samples):
        super(ClosedFM, self).__init__()
        self.train_samples = train_samples  # (N, C, H, W)
    
    def compute_w(self, xt, t):
        temp = 2 * (1 - t) ** 2
        x_query = xt.unsqueeze(0)  # (1, B, C, H, W)
        x_ref = self.train_samples.unsqueeze(1)  # (N, 1, C, H, W)
        # all t must be same
        assert torch.all(t == t[0]), "All t values must be the same for computing weights." 
        l2_dist = torch.sum((x_query - t[0] * x_ref) ** 2, dim=(2, 3, 4))  # (N, B)
        # apply softmax to get weights
        w = torch.softmax(-l2_dist / temp, dim=0).transpose(0, 1)  # (B, N)
        return w  # (B, N)
    
    @torch.no_grad()
    def get_optimal_v(self, xt, t):
        w = self.compute_w(xt, t) # (B, N)
        x_ref = self.train_samples.unsqueeze(0)  # (1, N, C, H, W)
        x_query = xt.unsqueeze(1)  # (B, 1, C, H, W)
        
        t = t[0]
        w = w.view(w.shape[0], w.shape[1], 1, 1, 1)  # (B, N, 1, 1, 1)
        v_opt = torch.sum(w * (x_ref - x_query)/(1-t), dim=1)  # (B, C, H, W)
        return v_opt  # (B, C, H, W)

    @torch.no_grad()
    def sample(self, x0, steps: int, start_t: float=0.0, return_intermediate: bool=False, **model_kwargs):
        # - define timesteps
        timesteps = torch.linspace(start_t, 1., steps + 1).to(x0.device)
        step_size = (1.0 - start_t) / steps
        
        x_in = x0
        v_traj = []
        for i in range(steps):
            # print(f"Sampling step {i+1}/{steps}...")
            t_in = timesteps[i]
            t_in = t_in.expand(x_in.shape[0])
            opt_v = self.get_optimal_v(x_in, t_in)
            x_in = x_in + opt_v * step_size
            v_traj.append(opt_v)
        samples = x_in
        if return_intermediate:
            return samples, v_traj
        return samples
    
class SmoothedClosedFM(LocalizedClosedFM):
    """Optimal velocity field for the Gaussian-smoothed empirical distribution
    p_sigma = (1/N) sum_i N(x_i, sigma^2 I).

    sigma=0 reduces exactly to LocalizedClosedFM.
    """

    def __init__(self, train_samples, sigma=0.1, **kwargs):
        kwargs.pop("bandwidth_min", None)   # its role is replaced by sigma
        super().__init__(train_samples, **kwargs)
        self.sigma = float(sigma)

    @torch.no_grad()
    def get_optimal_v(self, xt, t):
        B, C, H, W = xt.shape
        device = xt.device

        assert torch.all(t == t[0])
        t0 = t.flatten()[0].to(device=device, dtype=torch.float32)
        one_minus_t = 1.0 - t0

        sigma2 = self.sigma ** 2
        gamma2 = one_minus_t.square() + t0.square() * sigma2   # γ_t²
        if gamma2 <= 0:
            raise ValueError("gamma_t^2 must be positive (increase sigma or use t < 1).")

        temp = 2.0 * gamma2                                    # modification (1)
        c_t = (t0 * sigma2 - one_minus_t) / gamma2             # coefficient of modification (2)

        x_query_all = xt.permute(0, 2, 3, 1).contiguous().view(-1, C)
        x_ref_all = self.train_samples
        Q, R = x_query_all.shape[0], x_ref_all.shape[0]
        out = torch.empty((Q, C), device=device, dtype=xt.dtype)

        for q_start in range(0, Q, self.query_batch_size):
            q_end = min(q_start + self.query_batch_size, Q)
            q = x_query_all[q_start:q_end].to(torch.float32)
            q_size = q.shape[0]

            m = torch.full((q_size, 1), -torch.inf, device=device, dtype=torch.float32)
            z = torch.zeros((q_size, 1), device=device, dtype=torch.float32)
            s = torch.zeros((q_size, C), device=device, dtype=torch.float32)
            q_norm = (q * q).sum(dim=1, keepdim=True)

            for r_start in range(0, R, self.ref_batch_size):
                r = x_ref_all[r_start:min(r_start + self.ref_batch_size, R)].to(torch.float32)
                r_norm = (r * r).sum(dim=1).unsqueeze(0)

                scores = (2.0 * t0 * (q @ r.T) - q_norm - t0.square() * r_norm) / temp

                chunk_max = scores.max(dim=1, keepdim=True).values
                m_new = torch.maximum(m, chunk_max)
                old_scale = torch.exp(m - m_new)
                scores.sub_(m_new); scores.exp_()

                z = z * old_scale + scores.sum(dim=1, keepdim=True)
                s = s * old_scale + scores @ r
                m = m_new
                del r, r_norm, scores, chunk_max, m_new, old_scale

            mean_ref = s / z.clamp_min(1e-30)                  # x̄

            # v = x_bar + c_t (x_t - t x_bar)   <- modification (2)
            v = mean_ref + c_t * (q - t0 * mean_ref)
            out[q_start:q_end] = v.to(dtype=xt.dtype)
            del q, q_norm, m, z, s, mean_ref, v

        return out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()