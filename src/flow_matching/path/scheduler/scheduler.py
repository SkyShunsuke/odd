# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing import Union, Optional

import torch

from torch import Tensor


@dataclass
class SchedulerOutput:
    r"""Represents a sample of a conditional-flow generated probability path.

    Attributes:
        alpha_t (Tensor): :math:`\alpha_t`, shape (...).
        sigma_t (Tensor): :math:`\sigma_t`, shape (...).
        d_alpha_t (Tensor): :math:`\frac{\partial}{\partial t}\alpha_t`, shape (...).
        d_sigma_t (Tensor): :math:`\frac{\partial}{\partial t}\sigma_t`, shape (...).

    """

    alpha_t: Tensor = field(metadata={"help": "alpha_t"})
    sigma_t: Tensor = field(metadata={"help": "sigma_t"})
    d_alpha_t: Tensor = field(metadata={"help": "Derivative of alpha_t."})
    d_sigma_t: Tensor = field(metadata={"help": "Derivative of sigma_t."})


class Scheduler(ABC):
    """Base Scheduler class."""

    @abstractmethod
    def __call__(self, t: Tensor) -> SchedulerOutput:
        r"""
        Args:
            t (Tensor): times in [0,1], shape (...).

        Returns:
            SchedulerOutput: :math:`\alpha_t,\sigma_t,\frac{\partial}{\partial t}\alpha_t,\frac{\partial}{\partial t}\sigma_t`
        """
        ...

    @abstractmethod
    def snr_inverse(self, snr: Tensor) -> Tensor:
        r"""
        Computes :math:`t` from the signal-to-noise ratio :math:`\frac{\alpha_t}{\sigma_t}`.

        Args:
            snr (Tensor): The signal-to-noise, shape (...)

        Returns:
            Tensor: t, shape (...)
        """
        ...


class ConvexScheduler(Scheduler):
    @abstractmethod
    def __call__(self, t: Tensor) -> SchedulerOutput:
        r"""Scheduler for convex paths.

        Args:
            t (Tensor): times in [0,1], shape (...).

        Returns:
            SchedulerOutput: :math:`\alpha_t,\sigma_t,\frac{\partial}{\partial t}\alpha_t,\frac{\partial}{\partial t}\sigma_t`
        """
        ...

    @abstractmethod
    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        r"""
        Computes :math:`t` from :math:`\kappa_t`.

        Args:
            kappa (Tensor): :math:`\kappa`, shape (...)

        Returns:
            Tensor: t, shape (...)
        """
        ...

    def snr_inverse(self, snr: Tensor) -> Tensor:
        r"""
        Computes :math:`t` from the signal-to-noise ratio :math:`\frac{\alpha_t}{\sigma_t}`.

        Args:
            snr (Tensor): The signal-to-noise, shape (...)

        Returns:
            Tensor: t, shape (...)
        """
        kappa_t = snr / (1.0 + snr)

        return self.kappa_inverse(kappa=kappa_t)


class CondOTScheduler(ConvexScheduler):
    """CondOT Scheduler."""

    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=1 - t,
            d_alpha_t=torch.ones_like(t),
            d_sigma_t=-torch.ones_like(t),
        )

    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        return kappa


class PolynomialConvexScheduler(ConvexScheduler):
    """Polynomial Scheduler."""

    def __init__(self, n: Union[float, int]) -> None:
        assert isinstance(
            n, (float, int)
        ), f"`n` must be a float or int. Got {type(n)=}."
        assert n > 0, f"`n` must be positive. Got {n=}."

        self.n = n

    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t**self.n,
            sigma_t=1 - t**self.n,
            d_alpha_t=self.n * (t ** (self.n - 1)),
            d_sigma_t=-self.n * (t ** (self.n - 1)),
        )

    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        return torch.pow(kappa, 1.0 / self.n)


class VPScheduler(Scheduler):
    """Variance Preserving Scheduler."""

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max
        super().__init__()

    def __call__(self, t: Tensor) -> SchedulerOutput:
        b = self.beta_min
        B = self.beta_max
        T = 0.5 * (1 - t) ** 2 * (B - b) + (1 - t) * b
        dT = -(1 - t) * (B - b) - b

        return SchedulerOutput(
            alpha_t=torch.exp(-0.5 * T),
            sigma_t=torch.sqrt(1 - torch.exp(-T)),
            d_alpha_t=-0.5 * dT * torch.exp(-0.5 * T),
            d_sigma_t=0.5 * dT * torch.exp(-T) / torch.sqrt(1 - torch.exp(-T)),
        )

    def snr_inverse(self, snr: Tensor) -> Tensor:
        T = -torch.log(snr**2 / (snr**2 + 1))
        b = self.beta_min
        B = self.beta_max
        t = 1 - ((-b + torch.sqrt(b**2 + 2 * (B - b) * T)) / (B - b))
        return t


class LinearVPScheduler(Scheduler):
    """Linear Variance Preserving Scheduler."""

    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=(1 - t**2) ** 0.5,
            d_alpha_t=torch.ones_like(t),
            d_sigma_t=-t / (1 - t**2) ** 0.5,
        )

    def snr_inverse(self, snr: Tensor) -> Tensor:
        return torch.sqrt(snr**2 / (1 + snr**2))


class CosineScheduler(Scheduler):
    """Cosine Scheduler."""

    def __call__(self, t: Tensor) -> SchedulerOutput:
        pi = torch.pi
        return SchedulerOutput(
            alpha_t=torch.sin(pi / 2 * t),
            sigma_t=torch.cos(pi / 2 * t),
            d_alpha_t=pi / 2 * torch.cos(pi / 2 * t),
            d_sigma_t=-pi / 2 * torch.sin(pi / 2 * t),
        )

    def snr_inverse(self, snr: Tensor) -> Tensor:
        return 2.0 * torch.atan(snr) / torch.pi


class EDMScheduler(Scheduler):
    r"""EDM's unscaled perturbation path, parameterized directly by noise level.

    The independent variable supplied as ``t`` is the EDM noise standard
    deviation :math:`\sigma` rather than a normalized time in ``[0, 1]``.  In
    the notation of :class:`AffineProbPath`,

    .. math::

        X_\sigma = X_1 + \sigma X_0,

    so ``alpha_t = 1``, ``sigma_t = t``, and differentiation is with respect
    to :math:`\sigma`.  Consequently, converting an :math:`x_1` prediction to
    velocity gives exactly EDM's probability-flow direction

    .. math::

        \frac{dX}{d\sigma} = \frac{X-D_\theta(X,\sigma)}{\sigma}.

    The remaining constructor arguments are stored here so that the existing
    ``VelocityField`` configuration plumbing can carry EDM training and
    sampling hyperparameters without introducing another configuration block.
    """

    def __init__(
        self,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        S_churn: float = 0.0,
        S_min: float = 0.0,
        S_max: float = float("inf"),
        S_noise: float = 1.0,
    ) -> None:
        if P_std <= 0:
            raise ValueError(f"P_std must be positive, got {P_std}.")
        if sigma_data <= 0:
            raise ValueError(f"sigma_data must be positive, got {sigma_data}.")
        if not 0 < sigma_min < sigma_max:
            raise ValueError(
                f"Expected 0 < sigma_min < sigma_max, got {sigma_min}, {sigma_max}."
            )
        if rho <= 0:
            raise ValueError(f"rho must be positive, got {rho}.")

        self.P_mean = float(P_mean)
        self.P_std = float(P_std)
        self.sigma_data = float(sigma_data)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self.S_churn = float(S_churn)
        self.S_min = float(S_min)
        self.S_max = float(S_max)
        self.S_noise = float(S_noise)

    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=torch.ones_like(t),
            sigma_t=t,
            d_alpha_t=torch.zeros_like(t),
            d_sigma_t=torch.ones_like(t),
        )

    def snr_inverse(self, snr: Tensor) -> Tensor:
        # alpha / sigma = 1 / sigma for the EDM path.
        return torch.reciprocal(snr)

    def sample_train_sigma(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Draw the log-normal noise levels used by the EDM training loss."""
        rnd_normal = torch.randn(batch_size, device=device, dtype=dtype)
        return torch.exp(rnd_normal * self.P_std + self.P_mean)

    def time_grid(
        self,
        n_sampling_steps: int,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        rho: Optional[float] = None,
        append_zero: bool = True,
    ) -> Tensor:
        """Return the power-law noise grid from EDM Algorithm 2."""
        if n_sampling_steps < 2:
            raise ValueError(
                f"EDM sampling requires at least two steps, got {n_sampling_steps}."
            )
        sigma_min = self.sigma_min if sigma_min is None else float(sigma_min)
        sigma_max = self.sigma_max if sigma_max is None else float(sigma_max)
        rho = self.rho if rho is None else float(rho)
        if not 0 < sigma_min < sigma_max:
            raise ValueError(
                f"Expected 0 < sigma_min < sigma_max, got {sigma_min}, {sigma_max}."
            )
        if rho <= 0:
            raise ValueError(f"rho must be positive, got {rho}.")

        indices = torch.arange(n_sampling_steps, device=device, dtype=dtype)
        ramp = indices / (n_sampling_steps - 1)
        sigma_steps = (
            sigma_max ** (1.0 / rho)
            + ramp * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
        ) ** rho
        if append_zero:
            sigma_steps = torch.cat([sigma_steps, torch.zeros_like(sigma_steps[:1])])
        return sigma_steps

class DiscreteVPScheduler(Scheduler):
    def __init__(
        self,
        n_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        betas: Optional[Tensor] = None,
    ) -> None:
        if betas is None:
            betas = torch.linspace(beta_start, beta_end, n_steps, dtype=torch.float64)
        betas = betas.detach().to(torch.float64).flatten()
        assert torch.all((betas > 0) & (betas < 1)), "betas must lie in (0, 1)."

        self.n_steps = int(betas.numel())
        self.betas = betas
        # Lambda_k = -log alpha_bar_k (k = 0..N, Lambda_0 = 0). Slope on [k, k+1): dLambda_k = -log(1-beta_{k+1}) > 0
        self._dlambda = -torch.log1p(-betas)  # (N,)
        self._lambdas = torch.cat(
            [torch.zeros(1, dtype=torch.float64), torch.cumsum(self._dlambda, dim=0)]
        )  # (N+1,)

    # ------------------------------------------------------------------ utilities
    @property
    def alphas_cumprod(self) -> Tensor:
        """Table of alpha_bar_k (k = 0..N, alpha_bar_0 = 1), float64, shape (N+1,)."""
        return torch.exp(-self._lambdas)

    def time_grid(
        self, n_sampling_steps: int, device=None, dtype: torch.dtype = torch.float32
    ) -> Tensor:
        s = torch.linspace(self.n_steps, 0, n_sampling_steps + 1).round()
        return (1.0 - s / self.n_steps).to(device=device, dtype=dtype)

    def __call__(self, t: Tensor) -> SchedulerOutput:
        lambdas = self._lambdas.to(t.device)
        dlambda = self._dlambda.to(t.device)
        n = self.n_steps

        s = (1.0 - t.to(torch.float64)) * n          
        k = s.floor().clamp(0, n - 1).long()           
        lam = lambdas[k] + (s - k) * dlambda[k]        
        dlam_dt = -n * dlambda[k]                      # dΛ/dt = (dΛ/ds)(ds/dt), ds/dt = -N

        exp_neg = torch.exp(-lam)
        alpha_t = torch.exp(-0.5 * lam)
        sigma_t = torch.sqrt(1.0 - exp_neg)
        return SchedulerOutput(
            alpha_t=alpha_t.to(t.dtype),
            sigma_t=sigma_t.to(t.dtype),
            d_alpha_t=(-0.5 * dlam_dt * alpha_t).to(t.dtype),
            d_sigma_t=(0.5 * dlam_dt * exp_neg / sigma_t).to(t.dtype),
        )

    def snr_inverse(self, snr: Tensor) -> Tensor:
        lam = torch.log1p(snr.to(torch.float64) ** (-2))
        lam = lam.clamp(0.0, float(self._lambdas[-1]))
        lambdas = self._lambdas.to(snr.device)
        dlambda = self._dlambda.to(snr.device)
        k = (torch.searchsorted(lambdas, lam, right=True) - 1).clamp(0, self.n_steps - 1)
        frac = (lam - lambdas[k]) / dlambda[k]
        return (1.0 - (k + frac) / self.n_steps).to(snr.dtype)
