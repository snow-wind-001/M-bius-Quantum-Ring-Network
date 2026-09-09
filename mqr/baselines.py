from __future__ import annotations

import torch
import torch.nn as nn


class _ImplicitSinkhornFunction(torch.autograd.Function):
    """Autograd bridge for a converged matrix-scaling pullback."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor, iterations: int, temperature: float):
        log_H = logits / float(temperature)
        for _ in range(int(iterations)):
            log_H = log_H - torch.logsumexp(log_H, dim=1, keepdim=True)
            log_H = log_H - torch.logsumexp(log_H, dim=0, keepdim=True)
        H = torch.exp(log_H)
        ctx.save_for_backward(H)
        ctx.temperature = float(temperature)
        return H

    @staticmethod
    def backward(ctx, grad_H: torch.Tensor):
        (H,) = ctx.saved_tensors
        weighted = H * grad_H
        rhs = -torch.cat((weighted.sum(dim=1), weighted.sum(dim=0)))
        dim = H.size(0)
        identity = torch.eye(dim, device=H.device, dtype=H.dtype)
        block = torch.cat(
            (
                torch.cat((identity, H), dim=1),
                torch.cat((H.transpose(0, 1), identity), dim=1),
            ),
            dim=0,
        )
        potentials = torch.linalg.lstsq(
            block, rhs.unsqueeze(1)
        ).solution.squeeze(1)
        centered = (
            grad_H
            + potentials[:dim].unsqueeze(1)
            + potentials[dim:].unsqueeze(0)
        )
        grad_logits = (H * centered) / ctx.temperature
        return grad_logits, None, None


class SinkhornDoublyStochasticParam(nn.Module):
    """Matched ``N^2``-parameter doubly stochastic transition baseline.

    Positive logits are normalized in log space by alternating row and column
    scaling.  :meth:`implicit_logit_pullback` differentiates the converged
    matrix-scaling solution without unrolling the normalization iterations.
    This makes Sinkhorn a fair mathematical baseline for MQR's analytical
    Cayley pullback; finite forward normalization remains the only approximation.
    """

    def __init__(
        self,
        dim: int,
        *,
        iterations: int = 20,
        temperature: float = 1.0,
        init_scale: float = 0.01,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if init_scale < 0:
            raise ValueError("init_scale must be non-negative")
        self.dim = int(dim)
        self.iterations = int(iterations)
        self.temperature = float(temperature)
        self.logits = nn.Parameter(torch.randn(dim, dim) * float(init_scale))

    @property
    def effective_dof(self) -> int:
        """Dimension of the interior of the Birkhoff polytope."""

        return (self.dim - 1) ** 2

    def doubly_stochastic(self, *, iterations: int | None = None) -> torch.Tensor:
        """Return the finite-iteration log-space Sinkhorn normalization."""

        count = self.iterations if iterations is None else int(iterations)
        if count <= 0:
            raise ValueError("iterations must be positive")
        log_H = self.logits / self.temperature
        for _ in range(count):
            log_H = log_H - torch.logsumexp(log_H, dim=1, keepdim=True)
            log_H = log_H - torch.logsumexp(log_H, dim=0, keepdim=True)
        return torch.exp(log_H)

    def forward(self) -> torch.Tensor:
        return self.doubly_stochastic()

    def doubly_stochastic_implicit(self) -> torch.Tensor:
        """Return Sinkhorn scaling with a converged implicit backward pass."""

        return _ImplicitSinkhornFunction.apply(
            self.logits,
            self.iterations,
            self.temperature,
        )

    @torch.no_grad()
    def implicit_logit_pullback(
        self,
        grad_H: torch.Tensor,
        *,
        H: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pull ``dL/dH`` back through converged Sinkhorn scaling.

        For ``H = diag(u) exp(Z) diag(v)``, perturbations have the form

        ``dH = H * (dZ + da[:,None] + db[None,:])``.

        Row/column conservation gives a singular ``2N`` potential system.  Its
        Moore--Penrose solution fixes the irrelevant gauge and yields the same
        gradient for ``Z``.  The final division accounts for
        ``Z = logits / temperature``.
        """

        if grad_H.shape != (self.dim, self.dim):
            raise ValueError(
                f"grad_H must be [{self.dim}, {self.dim}], got {tuple(grad_H.shape)}"
            )
        if H is None:
            H = self.doubly_stochastic()
        if H.shape != (self.dim, self.dim):
            raise ValueError(f"H must be [{self.dim}, {self.dim}]")
        H = H.to(device=self.logits.device, dtype=self.logits.dtype)
        grad_H = grad_H.to(device=H.device, dtype=H.dtype)

        weighted = H * grad_H
        rhs = -torch.cat((weighted.sum(dim=1), weighted.sum(dim=0)))
        identity = torch.eye(self.dim, device=H.device, dtype=H.dtype)
        block = torch.cat(
            (
                torch.cat((identity, H), dim=1),
                torch.cat((H.transpose(0, 1), identity), dim=1),
            ),
            dim=0,
        )
        potentials = torch.linalg.lstsq(block, rhs.unsqueeze(1)).solution.squeeze(1)
        row_potential = potentials[: self.dim]
        column_potential = potentials[self.dim :]
        centered = (
            grad_H
            + row_potential.unsqueeze(1)
            + column_potential.unsqueeze(0)
        )
        return (H * centered) / self.temperature

    @torch.no_grad()
    def doubly_stochastic_errors(
        self, *, H: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return maximum row- and column-sum errors."""

        if H is None:
            H = self.doubly_stochastic()
        row_error = (H.sum(dim=1) - 1.0).abs().amax()
        column_error = (H.sum(dim=0) - 1.0).abs().amax()
        return row_error, column_error
