from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unitary import CayleyUnistochasticParam


@dataclass(frozen=True)
class RingState:
    """Container for the ring hidden state."""

    h: torch.Tensor  # [B, N]


class HamiltonianInjectionLoRA(nn.Module):
    """
    LoRA-style low-rank injection: J(x) = W_up(W_down(x)).

    HTML spec:
      J(x_in) = W_up W_down x_in, where W_down in R^{r×d}, W_up in R^{N×r}.
    """

    def __init__(self, input_dim: int, hidden_dim: int, rank: int):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if rank <= 0:
            raise ValueError("rank must be positive")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rank = rank

        self.down = nn.Linear(input_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, input_dim]
        return self.up(self.down(x))


class LocalProjectiveReadout(nn.Module):
    """
    Local projective sampling readout.

    HTML spec:
      y_out = W_readout * (h*_S)
    where S is a subset of nodes (e.g. first k nodes).
    """

    def __init__(self, hidden_dim: int, output_dim: int, sample_indices: Sequence[int]):
        super().__init__()
        if hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("hidden_dim and output_dim must be positive")
        if len(sample_indices) <= 0:
            raise ValueError("sample_indices must be non-empty")

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.register_buffer("sample_indices", torch.tensor(list(sample_indices), dtype=torch.long))
        # HTML spec: y_out = W_readout · (h*_S)  (no bias term)
        self.readout = nn.Linear(len(sample_indices), output_dim, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, hidden_dim]
        hs = h.index_select(dim=1, index=self.sample_indices)  # [B, |S|]
        return self.readout(hs)


class MoebiusQuantumRing(nn.Module):
    """
    Möbius Quantum Ring (MQR) model (vector input).

    Implements the "headless ring" dynamics described in the HTML:
      - H = |U|^2, U from Cayley transform of skew-Hermitian A
      - Fixed-point relaxation:
            h <- (1 - alpha) * h * H^T + alpha * J(x)
      - Local projective readout from a subset of nodes
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        alpha: float = 0.1,
        relaxation_steps: int = 20,
        lora_rank: int = 16,
        readout_dim: int = 16,
        sample_indices: Optional[Sequence[int]] = None,
        readout_mode: str = "linear",
        proto_tau: float = 1.0,
        # Online alternating inference/training extensions (optional)
        base_unitary_init: str = "identity",
        base_unitary_scale: float = 0.01,
        base_unitary_seed: Optional[int] = None,
        learnable_state_targets: bool = False,
    ):
        super().__init__()
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if relaxation_steps <= 0:
            raise ValueError("relaxation_steps must be positive")
        if readout_dim <= 0:
            raise ValueError("readout_dim must be positive")
        if sample_indices is None:
            if readout_dim > hidden_dim:
                raise ValueError("readout_dim cannot be larger than hidden_dim")
            sample_indices = list(range(readout_dim))
        else:
            if len(sample_indices) != readout_dim:
                raise ValueError("len(sample_indices) must equal readout_dim")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.alpha = float(alpha)
        self.relaxation_steps = int(relaxation_steps)

        readout_mode = str(readout_mode)
        if readout_mode not in ("linear", "proto"):
            raise ValueError('readout_mode must be "linear" or "proto"')
        if proto_tau <= 0:
            raise ValueError("proto_tau must be positive")
        self.readout_mode = readout_mode
        self.proto_tau = float(proto_tau)

        # ----------------------------
        # Optional: freeze a "world-model" unitary U_base and learn only a "policy" unitary U_policy.
        #
        # We always learn U_policy via `self.unitary_param`. If `base_unitary_init == "identity"`,
        # then U_total == U_policy (backward compatible). If "random", we sample a fixed unitary
        # U_base and use U_total = U_policy @ U_base.
        # ----------------------------
        base_unitary_init = str(base_unitary_init)
        if base_unitary_init not in ("identity", "random"):
            raise ValueError('base_unitary_init must be "identity" or "random"')
        if base_unitary_scale <= 0:
            raise ValueError("base_unitary_scale must be positive")

        self.base_unitary_init = base_unitary_init
        self._use_base_unitary = base_unitary_init == "random"

        # Register a buffer so it moves with .to(device) and is saved in state_dict.
        self.register_buffer("U_base", torch.eye(hidden_dim, dtype=torch.cfloat))
        if self._use_base_unitary:
            g = torch.Generator()
            if base_unitary_seed is not None:
                g.manual_seed(int(base_unitary_seed))
            A_real = torch.randn(hidden_dim, hidden_dim, generator=g) * float(base_unitary_scale)
            A_imag = torch.randn(hidden_dim, hidden_dim, generator=g) * float(base_unitary_scale)
            A = 0.5 * torch.complex(A_real - A_real.T, A_imag + A_imag.T)  # skew-Hermitian
            I = torch.eye(hidden_dim, dtype=torch.cfloat)
            U_base = torch.linalg.solve(I + A, I - A)  # unitary by Cayley
            self.U_base.copy_(U_base)

        self.unitary_param = CayleyUnistochasticParam(hidden_dim)
        self.injection = HamiltonianInjectionLoRA(input_dim, hidden_dim, lora_rank)
        self.readout = LocalProjectiveReadout(hidden_dim, output_dim, sample_indices)

        # Optional: learnable GT equilibrium targets (e.g. class prototypes) in hidden-state space.
        if learnable_state_targets:
            self.state_targets = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.01)
        else:
            self.register_parameter("state_targets", None)

        if self.readout_mode == "proto" and self.state_targets is None:
            raise ValueError('readout_mode="proto" requires learnable_state_targets=True')

    def _unitary_total(self) -> torch.Tensor:
        """Return U_total (complex) used by the forward/inference ring."""
        U = self.unitary_param.unitary()  # [N, N] complex
        if self._use_base_unitary:
            U = U @ self.U_base.to(device=U.device, dtype=U.dtype)
        return U

    def _current_H(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return H = |U_total|^2 (real) on the requested device/dtype."""
        H = self._unitary_total().abs().pow(2)
        return H.to(device=device, dtype=dtype)

    def _sampled_state(self, h: torch.Tensor) -> torch.Tensor:
        """Return the locally sampled state h_S used by readout/prototypes."""
        return h.index_select(dim=1, index=self.readout.sample_indices)

    def _proto_logits_from_state(self, h: torch.Tensor) -> torch.Tensor:
        """
        Prototype-distance logits on the sampled subspace.

        logits_{b,c} = - || h_{b,S} - P_{c,S} ||^2 / (2 * tau)
        """
        if self.state_targets is None:
            raise RuntimeError("Prototype readout requires learnable state_targets")
        hs = self._sampled_state(h)  # [B, K]
        P = self.state_targets.index_select(dim=1, index=self.readout.sample_indices)  # [C, K]

        # Squared Euclidean distances: ||x-p||^2 = ||x||^2 - 2 x·p + ||p||^2
        x2 = hs.pow(2).sum(dim=1, keepdim=True)  # [B, 1]
        p2 = P.pow(2).sum(dim=1).unsqueeze(0)  # [1, C]
        xp = hs @ P.transpose(0, 1)  # [B, C]
        dist2 = x2 - 2.0 * xp + p2  # [B, C]
        return (-0.5 * dist2 / self.proto_tau).to(dtype=h.dtype)

    def _logits_from_state(self, h: torch.Tensor) -> torch.Tensor:
        """Compute logits from the equilibrium state according to the chosen readout mode."""
        if self.readout_mode == "linear":
            return self.readout(h)
        if self.readout_mode == "proto":
            return self._proto_logits_from_state(h)
        raise RuntimeError(f"Unknown readout_mode: {self.readout_mode}")

    def forward(
        self,
        x: torch.Tensor,
        *,
        state: Optional[RingState] = None,
        return_state: bool = False,
        return_H: bool = False,
    ):
        """
        Args:
            x: [B, input_dim]
            state: optional previous state (to mimic "use previous frame state" in HTML)
            return_state: return RingState(h*) if True
            return_H: return H if True (diagnostic)
        """
        if x.dim() != 2 or x.size(-1) != self.input_dim:
            raise ValueError(f"x must be [B, {self.input_dim}], got {tuple(x.shape)}")

        # 1) Injection J(x)
        J = self.injection(x)  # [B, N]

        # 2) Unistochastic connection H = |U|^2
        H = self._current_H(device=x.device, dtype=J.dtype)  # [N, N] (real)
        Ht = H.transpose(0, 1)  # [N, N]

        # 3) Relaxation to the fixed point h*
        if state is None:
            h = torch.zeros(x.size(0), self.hidden_dim, device=x.device, dtype=J.dtype)
        else:
            if state.h.shape != (x.size(0), self.hidden_dim):
                raise ValueError(f"state.h must be [B, {self.hidden_dim}], got {tuple(state.h.shape)}")
            h = state.h

        a = self.alpha
        for _ in range(self.relaxation_steps):
            h = (1.0 - a) * (h @ Ht) + a * J

        # 4) Readout (linear local sampling OR prototype-distance logits)
        y = self._logits_from_state(h)

        if return_state and return_H:
            return y, RingState(h=h), H
        if return_state:
            return y, RingState(h=h)
        if return_H:
            return y, H
        return y

    @torch.no_grad()
    def compute_adjoint_state(
        self,
        h_star: torch.Tensor,
        grad_y: torch.Tensor,
        *,
        steps: int = 20,
    ) -> torch.Tensor:
        """
        Compute the adjoint state h^† by the fixed-point iteration described in the HTML.

        This is a diagnostic/educational utility mirroring the "Holomorphic Equilibrium Propagation"
        section. The training scripts in this repo still rely on PyTorch autograd by default.

        Args:
            h_star: fixed point state h* of shape [B, N]
            grad_y: gradient at the output y, shape [B, output_dim]
            steps: number of iterations for the adjoint fixed-point solver
        """
        if h_star.dim() != 2 or h_star.size(1) != self.hidden_dim:
            raise ValueError(f"h_star must be [B, {self.hidden_dim}]")
        if grad_y.dim() != 2 or grad_y.size(1) != self.output_dim:
            raise ValueError(f"grad_y must be [B, {self.output_dim}]")

        # Map output gradient to a hidden-state gradient source on the sampled subspace S.
        if self.readout_mode == "linear":
            # y = W_readout · h_S  => dL/dh_S = dL/dy · W
            grad_hs = grad_y @ self.readout.readout.weight  # [B, |S|]
        elif self.readout_mode == "proto":
            # logits_c = -||h_S - P_c||^2 / (2*tau)
            # dL/dh_S = sum_c dL/dlogits_c * dlogits_c/dh_S
            #        = -(1/tau) * (h_S * sum_c g_c - g @ P_S)
            # (for CE gradients, sum_c g_c = 0, but we keep the explicit form for robustness.)
            if self.state_targets is None:
                raise RuntimeError("readout_mode='proto' requires learnable state_targets")
            hs = self._sampled_state(h_star)  # [B, |S|]
            P = self.state_targets.index_select(dim=1, index=self.readout.sample_indices)  # [C, |S|]
            sum_g = grad_y.sum(dim=1, keepdim=True)  # [B, 1]
            grad_hs = -((hs * sum_g) - (grad_y @ P)) / self.proto_tau  # [B, |S|]
        else:
            raise RuntimeError(f"Unknown readout_mode: {self.readout_mode}")

        grad_h = torch.zeros_like(h_star)
        grad_h.index_copy_(1, self.readout.sample_indices, grad_hs)

        return self.compute_adjoint_state_from_grad_h(h_star, grad_h, steps=steps)

    @torch.no_grad()
    def compute_adjoint_state_from_grad_h(
        self,
        h_star: torch.Tensor,
        grad_h: torch.Tensor,
        *,
        steps: int = 20,
    ) -> torch.Tensor:
        """
        Compute adjoint state h^† given an explicit gradient source ∇_{h*} L.

        This is the core "reverse ring" iteration:
            h^† = (1-α) h^† H + α ∇_{h*} L
        """
        if h_star.shape != grad_h.shape:
            raise ValueError("h_star and grad_h must have the same shape")
        if h_star.dim() != 2 or h_star.size(1) != self.hidden_dim:
            raise ValueError(f"h_star must be [B, {self.hidden_dim}]")
        if steps <= 0:
            raise ValueError("steps must be positive")

        H = self._current_H(device=h_star.device, dtype=h_star.dtype)  # [N, N]
        h_dag = torch.zeros_like(h_star)
        a = self.alpha
        for _ in range(steps):
            # Row-vector form of: h^† = (1-α) H^T h^† + α ∇_{h*} L
            h_dag = (1.0 - a) * (h_dag @ H) + a * grad_h

        return h_dag

    @torch.no_grad()
    def approx_grad_H(
        self,
        h_star: torch.Tensor,
        h_dag: torch.Tensor,
        *,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Approximate ∂L/∂H ≈ h^† ⊗ h* (outer product), aggregated over batch.

        Returns:
            grad_H: [N, N]
        """
        if h_star.shape != h_dag.shape:
            raise ValueError("h_star and h_dag must have the same shape")
        if h_star.dim() != 2 or h_star.size(1) != self.hidden_dim:
            raise ValueError(f"h_star must be [B, {self.hidden_dim}]")

        grad_H = h_dag.transpose(0, 1) @ h_star  # [N, N]
        if normalize:
            grad_H = grad_H / max(1, h_star.size(0))
        return grad_H

    @torch.no_grad()
    def eqprop_update_step(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor],
        *,
        lr: float,
        unitary_lr_ratio: float = 0.5,
        injection_lr_ratio: float = 1.0,
        readout_lr_ratio: float = 1.0,
        adjoint_steps: int = 20,
        normalize_grad_H: bool = True,
        state: Optional[RingState] = None,
        # Optional state-target learning (GT equilibrium targets, e.g. class prototypes)
        state_target_weight: float = 0.0,
        state_target_lr_ratio: float = 1.0,
    ) -> dict:
        """
        Strict training step following the HTML "Holomorphic Equilibrium Propagation" section.

        Key properties:
          - No BPTT / no autograd through the relaxation loop.
          - Inference: relaxation to a fixed point h* (real ring).
          - Update: adjoint fixed point h^†, then ∂L/∂H ≈ h^† ⊗ h*.
          - Manifold update: ΔA ∝ skew( U^† · ( (∂L/∂H) ⊙ U ⊙ \bar U ) ).

        Args:
            x: [B, input_dim]
            target:
              - hard labels [B] (int64), or
              - soft labels [B, output_dim] (float), e.g. Mixup.
            lr: base learning rate
            unitary_lr_ratio: multiplier for unitary manifold parameters
            injection_lr_ratio: multiplier for LoRA injection parameters
            readout_lr_ratio: multiplier for readout parameters
            adjoint_steps: iterations for adjoint fixed-point solver
            normalize_grad_H: average outer product over batch if True
            state: optional previous state h (HTML mentions using previous frame state)

        Returns:
            dict with keys: loss, logits, h_star, h_dag, unitary_error
        """
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if unitary_lr_ratio < 0 or injection_lr_ratio < 0 or readout_lr_ratio < 0:
            raise ValueError("lr ratios must be non-negative")
        if adjoint_steps <= 0:
            raise ValueError("adjoint_steps must be positive")

        # ----------------------------
        # 1) Inference (Real Ring): relaxation to fixed point h*
        # ----------------------------
        logits, ring_state, _H = self.forward(x, state=state, return_state=True, return_H=True)
        h_star = ring_state.h  # [B, N]

        # If no GT is provided, we only do the forward ring (online inference).
        if target is None:
            unitary_error = self.unitary_param.unitary_error_fro().real.to(dtype=torch.float32).item()
            return {
                "loss": 0.0,
                "loss_cls": 0.0,
                "loss_state": 0.0,
                "logits": logits,
                "h_star": h_star,
                "h_dag": torch.zeros_like(h_star),
                "unitary_error": unitary_error,
                "did_update": False,
            }

        # ----------------------------
        # 2) Loss + dL/dy (no autograd through relaxation)
        # ----------------------------
        if target.dim() == 1:
            # Hard labels: standard cross-entropy
            loss_cls = F.cross_entropy(logits, target, reduction="mean")
            probs = torch.softmax(logits, dim=1)
            grad_y = probs
            grad_y[torch.arange(target.size(0), device=target.device), target] -= 1.0
            grad_y = grad_y / max(1, target.size(0))
        elif target.dim() == 2:
            # Soft labels: -sum(target * log_softmax)
            if target.size(1) != self.output_dim:
                raise ValueError(f"Soft target must be [B, {self.output_dim}]")
            log_probs = F.log_softmax(logits, dim=1)
            loss_cls = -(target * log_probs).sum(dim=1).mean()
            probs = torch.softmax(logits, dim=1)
            grad_y = (probs - target) / max(1, target.size(0))
        else:
            raise ValueError("target must be [B] or [B, C]")

        # Convert output gradient to a full hidden-state gradient source ∇_{h*} L (from readout).
        grad_state_targets_cls = None
        if self.readout_mode == "linear":
            grad_hs_out = grad_y @ self.readout.readout.weight  # [B, |S|]
        elif self.readout_mode == "proto":
            # logits_c = -||h_S - P_c||^2 / (2*tau)
            hs = self._sampled_state(h_star)  # [B, |S|]
            P_s = self.state_targets.index_select(dim=1, index=self.readout.sample_indices)  # [C, |S|]
            sum_g = grad_y.sum(dim=1, keepdim=True)  # [B, 1]
            grad_hs_out = -((hs * sum_g) - (grad_y @ P_s)) / self.proto_tau  # [B, |S|]

            # Gradient to prototypes (only on sampled dims S):
            # dL/dP_c = sum_b g_{b,c} * (hs_b - P_c)/tau
            s = grad_y.sum(dim=0)  # [C]
            grad_P_s = (grad_y.transpose(0, 1) @ hs - s.unsqueeze(1) * P_s) / self.proto_tau  # [C, |S|]
            grad_state_targets_cls = torch.zeros_like(self.state_targets)
            grad_state_targets_cls.index_copy_(1, self.readout.sample_indices, grad_P_s)
        else:
            raise RuntimeError(f"Unknown readout_mode: {self.readout_mode}")

        grad_h = torch.zeros_like(h_star)
        grad_h.index_copy_(1, self.readout.sample_indices, grad_hs_out)

        # Optional: add a GT equilibrium-state matching loss in hidden space.
        loss_state = torch.tensor(0.0, device=h_star.device, dtype=h_star.dtype)
        grad_state_targets = None
        if state_target_weight > 0:
            if self.state_targets is None:
                raise ValueError("state_target_weight > 0 requires learnable_state_targets=True at init")

            B = h_star.size(0)
            N = h_star.size(1)
            denom = float(B * N)

            if target.dim() == 1:
                proto = self.state_targets.index_select(0, target)  # [B, N]
                err = h_star - proto
                loss_state = 0.5 * err.pow(2).mean()
                grad_h = grad_h + (state_target_weight * (err / denom))

                # d/d proto: (proto - h) / (B*N)
                grad_proto_batch = (proto - h_star) / denom  # [B, N]
                grad_state_targets = torch.zeros_like(self.state_targets)
                grad_state_targets.index_add_(0, target, grad_proto_batch)
            else:
                # Soft label target: proto = target @ P
                proto = target @ self.state_targets  # [B, N]
                err = h_star - proto
                loss_state = 0.5 * err.pow(2).mean()
                grad_h = grad_h + (state_target_weight * (err / denom))

                grad_state_targets = target.transpose(0, 1) @ ((proto - h_star) / denom)  # [C, N]

        # ----------------------------
        # 3) Adjoint fixed point + dL/dH
        # ----------------------------
        h_dag = self.compute_adjoint_state_from_grad_h(h_star, grad_h, steps=adjoint_steps)
        grad_H = self.approx_grad_H(h_star, h_dag, normalize=normalize_grad_H)  # [N, N]

        # ----------------------------
        # 4) Parameter updates (Readout / Injection / Unitary manifold)
        # ----------------------------
        # 4.1 Readout update: y = W_readout · h*_S
        lr_readout = lr * readout_lr_ratio
        if lr_readout > 0 and self.readout_mode == "linear":
            hs = h_star.index_select(1, self.readout.sample_indices)  # [B, |S|]
            grad_W_readout = grad_y.transpose(0, 1) @ hs  # [C, |S|]
            self.readout.readout.weight.data.add_(grad_W_readout, alpha=-lr_readout)

        # 4.2 Injection update: J(x) = W_up W_down x
        lr_inj = lr * injection_lr_ratio
        if lr_inj > 0:
            z = self.injection.down(x)  # [B, r]
            grad_W_up = h_dag.transpose(0, 1) @ z  # [N, r]
            dz = h_dag @ self.injection.up.weight  # [B, r]
            grad_W_down = dz.transpose(0, 1) @ x  # [r, d]

            self.injection.up.weight.data.add_(grad_W_up, alpha=-lr_inj)
            self.injection.down.weight.data.add_(grad_W_down, alpha=-lr_inj)

        # 4.3 Unitary manifold update (core "phase / Lie algebra" step)
        lr_u = lr * unitary_lr_ratio
        if lr_u > 0:
            # We update the learnable "policy" unitary parameters. If a frozen U_base is used,
            # we must pull gradients back through U_total = U_policy U_base, treating U_base constant.
            U_policy = self.unitary_param.unitary()  # [N, N] complex
            if self._use_base_unitary:
                U_base = self.U_base.to(device=U_policy.device, dtype=U_policy.dtype)
                U_total = U_policy @ U_base
            else:
                U_base = None
                U_total = U_policy

            H_u = U_total.abs().pow(2)  # [N, N] real

            # HTML: ΔA ∝ skew( U^† · ( (∂L/∂H) ⊙ U ⊙ \bar U ) )
            inner_total = (grad_H * H_u).to(dtype=U_policy.dtype)  # cast real -> complex
            # Pullback through right-multiplication by constant U_base: dU_total = dU_policy U_base
            inner_policy = inner_total if U_base is None else (inner_total @ U_base.conj().transpose(-2, -1))
            M = U_policy.conj().transpose(-2, -1) @ inner_policy
            delta_A = 0.5 * (M - M.conj().transpose(-2, -1))  # skew-Hermitian

            # Map ΔA back to the stored parameters:
            # A = 0.5[(R - R^T) + i(I + I^T)], so updating R by ΔRe(A) (skew),
            # and I by ΔIm(A) (sym) yields an exact ΔA at the A-level.
            self.unitary_param.A_real.data.add_(delta_A.real, alpha=-lr_u)
            self.unitary_param.A_imag.data.add_(delta_A.imag, alpha=-lr_u)

        # Optional: update learnable GT equilibrium targets (e.g. class prototypes).
        lr_state = lr * state_target_lr_ratio
        if lr_state > 0 and self.state_targets is not None:
            grad_total = None
            if grad_state_targets_cls is not None:
                grad_total = grad_state_targets_cls
            if grad_state_targets is not None:
                if grad_total is None:
                    grad_total = torch.zeros_like(self.state_targets)
                grad_total = grad_total + (state_target_weight * grad_state_targets)
            if grad_total is not None:
                self.state_targets.data.add_(grad_total, alpha=-lr_state)

        unitary_error = self.unitary_param.unitary_error_fro().real.to(dtype=torch.float32).item()

        return {
            "loss": float((loss_cls + state_target_weight * loss_state).item()),
            "loss_cls": float(loss_cls.item()),
            "loss_state": float(loss_state.item()),
            "logits": logits,
            "h_star": h_star,
            "h_dag": h_dag,
            "unitary_error": unitary_error,
            "did_update": True,
        }

    @torch.no_grad()
    def get_orthogonal_loss(self) -> torch.Tensor:
        """
        Backward-compatible name from the previous codebase.

        Returns a diagnostic unitarity error ||U^H U - I||_F.
        """
        return self.unitary_param.unitary_error_fro().real.to(dtype=torch.float32)


class MoebiusQuantumRingImageClassifier(nn.Module):
    """Thin image wrapper for CIFAR-like inputs. Flattens images and feeds MoebiusQuantumRing."""

    def __init__(
        self,
        *,
        img_size: int = 32,
        in_channels: int = 3,
        num_classes: int = 100,
        hidden_dim: int = 384,
        alpha: float = 0.1,
        relaxation_steps: int = 20,
        lora_rank: int = 16,
        readout_dim: int = 16,
        readout_mode: str = "linear",
        proto_tau: float = 1.0,
        base_unitary_init: str = "identity",
        base_unitary_scale: float = 0.01,
        base_unitary_seed: Optional[int] = None,
        learnable_state_targets: bool = False,
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_classes = num_classes

        input_dim = in_channels * img_size * img_size
        self.ring = MoebiusQuantumRing(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            alpha=alpha,
            relaxation_steps=relaxation_steps,
            lora_rank=lora_rank,
            readout_dim=readout_dim,
            readout_mode=readout_mode,
            proto_tau=proto_tau,
            base_unitary_init=base_unitary_init,
            base_unitary_scale=base_unitary_scale,
            base_unitary_seed=base_unitary_seed,
            learnable_state_targets=learnable_state_targets,
        )

    def forward(self, x: torch.Tensor):
        if x.dim() != 4:
            raise ValueError(f"Expected image tensor [B,C,H,W], got {tuple(x.shape)}")
        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected C={self.in_channels}, got {x.size(1)}")
        if x.size(2) != self.img_size or x.size(3) != self.img_size:
            raise ValueError(f"Expected H=W={self.img_size}, got {(x.size(2), x.size(3))}")

        x_vec = x.flatten(1)  # [B, C*H*W]
        return self.ring(x_vec)

    @torch.no_grad()
    def get_orthogonal_loss(self) -> torch.Tensor:
        """Expose the same helper used by legacy training scripts."""
        return self.ring.get_orthogonal_loss()

    @torch.no_grad()
    def eqprop_update_step(
        self,
        images: torch.Tensor,
        target: torch.Tensor,
        *,
        lr: float,
        unitary_lr_ratio: float = 0.5,
        injection_lr_ratio: float = 1.0,
        readout_lr_ratio: float = 1.0,
        adjoint_steps: int = 20,
        normalize_grad_H: bool = True,
        state_target_weight: float = 0.0,
        state_target_lr_ratio: float = 1.0,
    ) -> dict:
        """Image wrapper for `MoebiusQuantumRing.eqprop_update_step`."""
        x_vec = images.flatten(1)
        return self.ring.eqprop_update_step(
            x_vec,
            target,
            lr=lr,
            unitary_lr_ratio=unitary_lr_ratio,
            injection_lr_ratio=injection_lr_ratio,
            readout_lr_ratio=readout_lr_ratio,
            adjoint_steps=adjoint_steps,
            normalize_grad_H=normalize_grad_H,
            state_target_weight=state_target_weight,
            state_target_lr_ratio=state_target_lr_ratio,
        )


def create_mobius_model(
    num_classes: int = 100,
    img_size: int = 32,
    in_channels: int = 3,
    *,
    embed_dim: int = 384,
    depth: int = 20,
    alpha: float = 0.1,
    lora_rank: int = 16,
    readout_dim: int = 16,
    readout_mode: str = "linear",
    proto_tau: float = 1.0,
    base_unitary_init: str = "identity",
    base_unitary_scale: float = 0.01,
    base_unitary_seed: Optional[int] = None,
    learnable_state_targets: bool = False,
    **_ignored,
) -> MoebiusQuantumRingImageClassifier:
    """
    Backward-compatible factory used by existing scripts.

    Mapping from old args:
      - embed_dim -> hidden_dim (ring nodes N)
      - depth     -> relaxation_steps (K)
    """
    return MoebiusQuantumRingImageClassifier(
        img_size=img_size,
        in_channels=in_channels,
        num_classes=num_classes,
        hidden_dim=embed_dim,
        alpha=alpha,
        relaxation_steps=depth,
        lora_rank=lora_rank,
        readout_dim=readout_dim,
        readout_mode=readout_mode,
        proto_tau=proto_tau,
        base_unitary_init=base_unitary_init,
        base_unitary_scale=base_unitary_scale,
        base_unitary_seed=base_unitary_seed,
        learnable_state_targets=learnable_state_targets,
    )

