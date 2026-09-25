import tqdm
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal

from src.flow_matching.path.scheduler import CondOTScheduler, VPScheduler, \
    CosineScheduler, EDMScheduler, LinearVPScheduler
from src.flow_matching.path import AffineProbPath
from src.flow_matching.solver import Solver, ODESolver

import logging
logger = logging.getLogger(__name__)

PRED_TYPES = ['velocity']
LOSS_TYPES = ['velocity']


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
    
class WrappedModel(nn.Module):
    """A wrapper for the velocity model to handle additional conditioning information."""
    def __init__(self, model: nn.Module, path, pred_type: str, cfg_interval=[0.1, 1.0], cfg_scale=1.0, eps=0.05):
        super(WrappedModel, self).__init__()
        self.model = model
        self.path = path
        self.pred_type = pred_type
        self.cfg_interval = cfg_interval
        self.cfg_scale = cfg_scale
        self.eps = eps

    @staticmethod
    def _batch_time(t: torch.Tensor, batch_size: int, device, dtype) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, device=device, dtype=dtype)
        else:
            t = t.to(device=device, dtype=dtype)
        if t.ndim == 0:
            t = t.expand(batch_size)
        if t.ndim != 1 or t.shape[0] != batch_size:
            raise ValueError(
                f"Expected scalar time or shape ({batch_size},), got {tuple(t.shape)}."
            )
        return t
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, y=None, **extras) -> torch.Tensor:
        t = self._batch_time(t, x.shape[0], x.device, x.dtype)
        assert self.cfg_scale == 1.0, "Classifier-free guidance is not supported in this wrapper."
        
        out = self.model(x, t, y=y, **extras)
        return out


class ShortcutVelocityField(nn.Module):
    def __init__(self, model: nn.Module, input_sz: tuple, scheduler_name: str,  solver_name: str, \
        pred_type:str='velocity', loss_type:str='velocity', loss_fn:str='mse', shortcut_factor=7, shortcut_ratio: float=0.25, div_eps: float=0.05, scheduler_params: dict=None, solver_params: dict = None, ema_decay: float=0.999, tau: float=0.1, **kwargs
    ):
        """Velocity field module for flow matching.
        Args:
            model (nn.Module): The neural network model representing the velocity field.
            input_sz (tuple): The size of the input data.
            scheduler_name (str): The name of the scheduler to use.
            solver_name (str): The name of the solver to use.
            pred_type (str, optional): Type of prediction ('data', 'noise', 'velocity'). Defaults to 'velocity'.
            loss_type (str, optional): Type of loss ('data', 'noise', 'velocity'). Defaults to 'velocity'.
            shortcut_factor (int, optional): The factor by which to shortcut the computation. Defaults to 7.
            shortcut_ratio (float, optional): The ratio for the shortcut computation. Defaults to 0.4.
            scheduler_params (dict, optional): Additional parameters for the scheduler. Defaults to None.
            solver_params (dict, optional): Additional parameters for the solver. Defaults to None.
            ema_decay (float, optional): The decay rate for the exponential moving average of model parameters. Defaults to 0.999.
            tau (float, optional): The tau parameter for the shortcut computation. Defaults to 0.1.
        Usage: 
            - initialization
            vf = ShortcutVelocityField(model, input_sz, scheduler_name, solver_name, scheduler_params, solver_params)
            - training
            loss = vf(x1, y)
            - sampling
            x0 = torch.randn(batch_size, *input_sz).to(device)
            x1 = vf.sample(x0, y, steps=10)
            - inversion
            x0 = vf.invert(x1, y, steps=10)
            - density estimation
            log_prob = vf.log_prob(x1, y, steps=10, solver='euler', exact=False, hte_acc=10)
        Assumptions:
            - The model takes input of shape (batch_size, *input_sz) and returns output of the same shape.
            - The scheduler and solver are implemented elsewhere and are compatible with this module.
        """
        assert pred_type in PRED_TYPES, f"pred_type must be one of {PRED_TYPES}"
        assert loss_type in LOSS_TYPES, f"loss_type must be one of {LOSS_TYPES}"
        super(ShortcutVelocityField, self).__init__()

        self.model = model
        self.input_sz = input_sz
        self.pred_type = pred_type
        self.loss_type = loss_type
        self.loss_fn = loss_fn
        self.shortcut_factor = shortcut_factor
        self.div_eps = div_eps
        self.scheduler_name = scheduler_name
        self.solver_name = solver_name
        self.scheduler_params = scheduler_params
        self.solver_params = solver_params
        self.div_eps = div_eps
        self.shortcut_ratio = shortcut_ratio
        self.tau = tau
        self.scheduler_name = scheduler_name
        self.path = build_scheduler(scheduler_name, scheduler_params or {})
        self.is_edm = isinstance(self.path.scheduler, EDMScheduler)
        self.solver = build_solver(model, solver_name, solver_params or {})
        
        self.ema_model = copy.deepcopy(model)
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
        self.ema_decay = ema_decay

    @staticmethod
    def _expand_batch_scalar(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if value.ndim == 0:
            value = value.expand(reference.shape[0])
        if value.ndim != 1 or value.shape[0] != reference.shape[0]:
            raise ValueError(
                f"Expected scalar or batch vector of length {reference.shape[0]}, "
                f"got {tuple(value.shape)}."
            )
        return value.reshape(value.shape[0], *([1] * (reference.ndim - 1)))

    def _backbone_dtype(self, fallback: torch.dtype) -> torch.dtype:
        try:
            dtype = next(self.model.parameters()).dtype
        except StopIteration:
            dtype = fallback
        # EDM's solver carries states in float64, while the network normally
        # remains in float32/bfloat16.
        return torch.float32 if dtype == torch.float64 else dtype
    
    def update_ema(self):
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
            
    def compute_shortcut_loss(self, x0, x1, path_sample, t, y=None, d=None, **model_kwargs):
        """Compute the loss for the shortcut portion of the batch."""
        
        # first compute base prediction at larger step
        xt = path_sample.x_t
        t = path_sample.t
        base_v = self.model(xt, t, y=y, d=d, **model_kwargs)
        
        # next derive more finer path as target
        self.ema_model.eval()
        with torch.no_grad():
            d_half = d / 2
            v_1 = self.ema_model(xt, t, y=y, d=d_half, **model_kwargs)
            t_nxt = t + d_half
            xt_nxt = xt + v_1 * self._expand_batch_scalar(d_half, xt)
            v_2 = self.ema_model(xt_nxt, t_nxt, y=y, d=d_half, **model_kwargs)
            target_v = (v_1 + v_2) / 2

        return (base_v - target_v.detach()).square().mean()

    def compute_fmloss(self, x0, x1, path_sample, y=None, d=None, **model_kwargs):
        """Compute a representation-consistent affine-path or EDM loss."""
        xt = path_sample.x_t
        t = path_sample.t
        v = path_sample.dx_t
        assert d is not None, "Shortcut timestep 'd' must be provided to compute_loss."

        pred = self.model(xt, t, y=y, d=d, **model_kwargs)

        if self.loss_fn == 'mse':
            return (pred - v).square().mean()
        raise NotImplementedError(f"Loss function {self.loss_fn} not implemented.")

    def forward(self, x1, y=None, **model_kwargs):
        """Sample a path point and evaluate the configured training objective."""
        bs = x1.shape[0]
        num_shortcut = int(self.shortcut_ratio * bs)
        num_fm = bs - num_shortcut
        
        device = x1.device
        x0 = torch.randn_like(x1)
        
        # prepare flow matching (FM) 
        idxs = torch.randint(low=0, high=2 ** self.shortcut_factor, size=(num_fm,), device=device, dtype=x1.dtype)
        t_fm = self.tau + (1 - self.tau) * idxs / (2 ** self.shortcut_factor)
        d_fm = (1 - self.tau) / (2 ** self.shortcut_factor)
        d_fm = torch.full_like(t_fm, fill_value=d_fm)
        x0_fm, x1_fm, y_fm = x0[:num_fm], x1[:num_fm], y[:num_fm]
        path_sample_fm = self.path.sample(t=t_fm, x_0=x0_fm, x_1=x1_fm)
        fmloss = self.compute_fmloss(x0_fm, x1_fm, path_sample_fm, y=y_fm, d=d_fm, **model_kwargs)
        
        # prepare shortcut models (SM)
        k = torch.randint(low=0, high=self.shortcut_factor, size=(num_shortcut,), device=device)
        sections = 2 ** k
        d_base = (1 - self.tau) / sections.to(x1.dtype)
        
        j = torch.floor(
            torch.rand(num_shortcut, device=device) * sections
        ).to(x1.dtype)
        
        t_sm = self.tau + j * d_base
        x0_sm, x1_sm, y_sm = x0[num_fm:], x1[num_fm:], y[num_fm:]

        # compute loss for the shortcut models
        path_sample_sm = self.path.sample(t=t_sm, x_0=x0_sm, x_1=x1_sm)
        smloss = self.compute_shortcut_loss(x0_sm, x1_sm, path_sample_sm, t_sm, y=y_sm, d=d_base, **model_kwargs)
        
        w = self.shortcut_ratio
        loss = (1 - w) * fmloss + w * smloss
        
        return loss

    def sample(self, x0, y, steps: int, return_intermediate: bool=False, solver_name: str='euler', solver_params: dict=None, start_t: float=0.1, **model_kwargs):
        """Sample from the velocity field using the specified solver.
        Args:
            x0 (torch.Tensor): The starting points.
            y (torch.Tensor): Additional conditioning information.
            steps (int): Number of steps for the solver.
            return_intermediate (bool, optional): Whether to return intermediate results. Defaults to False.
            solver_name (str, optional): The name of the solver to use. Defaults to 'euler'.
            solver_params (dict, optional): Additional parameters for the solver. Defaults to None.
            start_t (float, optional): The starting time for sampling. Defaults to 0.0.
        Returns:
            torch.Tensor: The sampled points.
        Usage: 
            x0 = torch.randn(batch_size, *input_sz).to(device)
            x1 = vf.sample(x0, y, steps=10)  # (batch_size, *input_sz)
            x1_inter = vf.sample(x0, y, steps=10, return_intermediate=True)  # List of intermediate results
        """
        assert steps in [2 ** i for i in range(self.shortcut_factor + 1)]
        
        # - build solver
        model = WrappedModel(
            self.model,
            self.path,
            self.pred_type,
            eps=self.div_eps,
        )
        # solver = build_solver(model, solver_name, solver_params or {})

        # Fixed-step explicit solvers.  The original implementation accepted
        # ``solver_name`` but always executed Euler; supporting midpoint/Heun
        # is useful for the SI and Score-SDE probability-flow ODE cases while
        # preserving the existing return value (effective velocity per step).
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}.")
        method = solver_name.lower()
        if method not in {"euler"}:
            raise ValueError(
                f"Unsupported fixed-step solver '{solver_name}'. "
                "Choose one of: euler."
            )

        timesteps = torch.linspace(
            start_t, 1.0, steps + 1, device=x0.device, dtype=torch.float32
        )
        assert start_t == self.tau, f"start_t ({start_t}) must be equal to self.tau ({self.tau})"
        v_traj = []
        x_in = x0
        for i in range(steps):
            t_cur_scalar = timesteps[i]
            t_next_scalar = timesteps[i + 1]
            h = (t_next_scalar - t_cur_scalar).to(x_in.dtype)
            t_cur = t_cur_scalar.expand(x_in.shape[0])
            d_cur = torch.full_like(t_cur, fill_value=h)

            if method == "euler":
                v_eff = model(x_in, t_cur, y=y, d=d_cur, **model_kwargs)
            else: 
                raise NotImplementedError(f"Solver method '{method}' is not implemented.")

            x_in = x_in + h * v_eff
            v_traj.append(v_eff)

        if return_intermediate:
            return x_in, v_traj
        return x_in
            

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