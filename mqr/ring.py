from __future__ import annotations

import math
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

    def __init__(self, input_dim: int, hidden_dim: int, rank: int, *, activation: str = "none"):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if rank <= 0:
            raise ValueError("rank must be positive")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rank = rank

        activation = str(activation)
        if activation not in ("none", "relu", "tanh", "gelu"):
            raise ValueError('activation must be one of: "none", "relu", "tanh", "gelu"')
        self.activation = activation

        self.down = nn.Linear(input_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)

    def _act(self, z: torch.Tensor) -> torch.Tensor:
        if self.activation == "none":
            return z
        if self.activation == "relu":
            return F.relu(z)
        if self.activation == "tanh":
            return torch.tanh(z)
        if self.activation == "gelu":
            return F.gelu(z)
        raise RuntimeError(f"Unknown activation: {self.activation}")

    def _act_prime(self, *, pre: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Element-wise derivative of activation w.r.t pre-activation."""
        if self.activation == "none":
            return torch.ones_like(pre)
        if self.activation == "relu":
            return (pre > 0).to(dtype=pre.dtype)
        if self.activation == "tanh":
            return 1.0 - act.pow(2)
        if self.activation == "gelu":
            # gelu(x) = x * Phi(x); d/dx = Phi(x) + x * phi(x)
            inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
            phi = 0.5 * (1.0 + torch.erf(pre * inv_sqrt2))
            pdf = 0.3989422804014327 * torch.exp(-0.5 * pre.pow(2))  # 1/sqrt(2pi) * exp(-x^2/2)
            return phi + pre * pdf
        raise RuntimeError(f"Unknown activation: {self.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, input_dim]
        pre = self.down(x)
        z = self._act(pre)
        return self.up(z)


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
        inj_activation: str = "none",
        state_activation: str = "none",
        h_mix_beta: float = 1.0,
        learnable_h_mix_beta: bool = False,
        dynamics_mode: str = "unistochastic",
        measurement: str = "identity",
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

        inj_activation = str(inj_activation)
        if inj_activation not in ("none", "relu", "tanh", "gelu"):
            raise ValueError('inj_activation must be one of: "none", "relu", "tanh", "gelu"')
        self.inj_activation = inj_activation

        state_activation = str(state_activation)
        if state_activation not in ("none", "relu", "tanh"):
            raise ValueError('state_activation must be one of: "none", "relu", "tanh"')
        self.state_activation = state_activation

        dynamics_mode = str(dynamics_mode)
        if dynamics_mode not in ("unistochastic", "unitary"):
            raise ValueError('dynamics_mode must be "unistochastic" or "unitary"')
        self.dynamics_mode = dynamics_mode

        measurement = str(measurement)
        if self.dynamics_mode == "unitary" and measurement == "identity":
            # Default measurement for complex states: magnitude.
            measurement = "abs"
        if measurement not in ("identity", "abs", "real"):
            raise ValueError('measurement must be "identity", "abs", or "real"')
        self.measurement = measurement
        self.meas_eps = 1e-8

        if self.dynamics_mode == "unitary" and self.state_activation != "none":
            # Keep the complex dynamics well-defined and stable; nonlinearity can be introduced via measurement.
            raise ValueError('state_activation must be "none" when dynamics_mode="unitary"')

        # ----------------------------
        # Optional self-retention mixing:
        #   H_eff = (1 - beta) I + beta H,   beta in [0, 1]
        # This reduces the "over-mixing" tendency of doubly stochastic propagation while preserving
        # doubly-stochasticity (convex combination of DS matrices) and contraction (beta does not
        # change ||H_eff^T||_∞ = 1).
        # ----------------------------
        if not (0.0 <= float(h_mix_beta) <= 1.0):
            raise ValueError("h_mix_beta must be in [0, 1]")
        self.learnable_h_mix_beta = bool(learnable_h_mix_beta)
        if self.learnable_h_mix_beta:
            b = float(h_mix_beta)
            b = min(max(b, 1e-4), 1.0 - 1e-4)
            self.h_mix_beta_param = nn.Parameter(torch.tensor(math.log(b / (1.0 - b)), dtype=torch.float32))
            self.h_mix_beta = None
        else:
            self.register_parameter("h_mix_beta_param", None)
            self.h_mix_beta = float(h_mix_beta)

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
        self.injection = HamiltonianInjectionLoRA(input_dim, hidden_dim, lora_rank, activation=inj_activation)
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

    def _h_mix_beta_value(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return beta in [0,1] for H_eff = (1-beta)I + beta H."""
        if self.h_mix_beta_param is None:
            return torch.tensor(float(self.h_mix_beta), device=device, dtype=dtype)
        return torch.sigmoid(self.h_mix_beta_param).to(device=device, dtype=dtype)

    def _current_H_base(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return H = |U_total|^2 (real, doubly stochastic)."""
        H = self._unitary_total().abs().pow(2)
        return H.to(device=device, dtype=dtype)

    def _current_H(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return H_eff used by the relaxation dynamics."""
        H = self._current_H_base(device=device, dtype=dtype)
        beta = self._h_mix_beta_value(device=device, dtype=dtype)
        if float(beta.detach().cpu().item()) == 1.0:
            return H
        I = torch.eye(self.hidden_dim, device=device, dtype=dtype)
        return (1.0 - beta) * I + beta * H

    def _state_act(self, z: torch.Tensor) -> torch.Tensor:
        """Element-wise activation inside the ring relaxation (optional)."""
        if self.state_activation == "none":
            return z
        if self.state_activation == "relu":
            return F.relu(z)
        if self.state_activation == "tanh":
            return torch.tanh(z)
        raise RuntimeError(f"Unknown state_activation: {self.state_activation}")

    def _state_act_prime_from_h(self, h: torch.Tensor) -> torch.Tensor:
        """
        Derivative of the ring activation evaluated at the fixed point.

        We intentionally compute σ'(·) from the post-activation h* for stability and simplicity:
        - relu: σ'(u)=1[u>0] ≈ 1[h*>0]
        - tanh: σ'(u)=1-tanh(u)^2 = 1-(h*)^2
        """
        if self.state_activation == "none":
            return torch.ones_like(h)
        if self.state_activation == "relu":
            return (h > 0).to(dtype=h.dtype)
        if self.state_activation == "tanh":
            return 1.0 - h.pow(2)
        raise RuntimeError(f"Unknown state_activation: {self.state_activation}")

    def _measured_state(self, h: torch.Tensor) -> torch.Tensor:
        """
        Measurement / observation of the internal state used for readout and state-level losses.

        - identity: use h directly (real-valued setting)
        - abs:      use |h| (enables nonlinearity when h is complex)
        - real:     use Re(h)
        """
        if self.measurement == "identity":
            return h
        if self.measurement == "abs":
            return h.abs()
        if self.measurement == "real":
            return h.real
        raise RuntimeError(f"Unknown measurement: {self.measurement}")

    def _pullback_measured_grad(self, h: torch.Tensor, grad_meas: torch.Tensor) -> torch.Tensor:
        """
        Pull back gradient from measurement space to the internal state space.

        Args:
            h: internal state (real or complex)
            grad_meas: gradient w.r.t measured state (real)
        """
        if self.measurement == "identity":
            return grad_meas.to(dtype=h.dtype)
        if self.measurement == "real":
            if torch.is_complex(h):
                return torch.complex(grad_meas, torch.zeros_like(grad_meas))
            return grad_meas
        if self.measurement == "abs":
            if not torch.is_complex(h):
                # |h| for real h is piecewise; we use sign as subgradient.
                return grad_meas * torch.sign(h)
            denom = h.abs().clamp_min(self.meas_eps)
            return torch.complex(grad_meas, torch.zeros_like(grad_meas)) * (h / denom)
        raise RuntimeError(f"Unknown measurement: {self.measurement}")

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
        h_meas = self._measured_state(h)
        if self.readout_mode == "linear":
            return self.readout(h_meas)
        if self.readout_mode == "proto":
            return self._proto_logits_from_state(h_meas)
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

        a = self.alpha
        if self.dynamics_mode == "unistochastic":
            # 2) Unistochastic connection H_eff (real)
            H = self._current_H(device=x.device, dtype=J.dtype)  # [N, N]
            Ht = H.transpose(0, 1)

            # 3) Relaxation to the fixed point h* (real)
            if state is None:
                h = torch.zeros(x.size(0), self.hidden_dim, device=x.device, dtype=J.dtype)
            else:
                if state.h.shape != (x.size(0), self.hidden_dim):
                    raise ValueError(f"state.h must be [B, {self.hidden_dim}], got {tuple(state.h.shape)}")
                h = state.h.to(device=x.device, dtype=J.dtype)

            for _ in range(self.relaxation_steps):
                h = self._state_act((1.0 - a) * (h @ Ht) + a * J)
        else:
            # 2) Complex unitary dynamics: propagate with U (phase participates in inference).
            U_total = self._unitary_total().to(device=x.device)
            U_H = U_total.conj().transpose(0, 1)
            Jc = torch.complex(J, torch.zeros_like(J)).to(dtype=U_total.dtype)

            if state is None:
                h = torch.zeros(x.size(0), self.hidden_dim, device=x.device, dtype=U_total.dtype)
            else:
                if state.h.shape != (x.size(0), self.hidden_dim):
                    raise ValueError(f"state.h must be [B, {self.hidden_dim}], got {tuple(state.h.shape)}")
                h = state.h.to(device=x.device, dtype=U_total.dtype)

            for _ in range(self.relaxation_steps):
                h = (1.0 - a) * (h @ U_H) + a * Jc

        # 4) Readout (linear local sampling OR prototype-distance logits)
        y = self._logits_from_state(h)

        if return_state and return_H:
            if self.dynamics_mode == "unistochastic":
                return y, RingState(h=h), H
            # For unitary dynamics, return the induced |U|^2 (unistochastic) for diagnostics.
            return y, RingState(h=h), self._current_H_base(device=x.device, dtype=J.dtype)
        if return_state:
            return y, RingState(h=h)
        if return_H:
            if self.dynamics_mode == "unistochastic":
                return y, H
            return y, self._current_H_base(device=x.device, dtype=J.dtype)
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

        # Map output gradient to a measured-state gradient source on the sampled subspace S.
        h_meas = self._measured_state(h_star)
        if self.readout_mode == "linear":
            grad_hs = grad_y @ self.readout.readout.weight  # [B, |S|]
        elif self.readout_mode == "proto":
            if self.state_targets is None:
                raise RuntimeError("readout_mode='proto' requires learnable state_targets")
            hs = self._sampled_state(h_meas)  # [B, |S|]
            P = self.state_targets.index_select(dim=1, index=self.readout.sample_indices)  # [C, |S|]
            sum_g = grad_y.sum(dim=1, keepdim=True)  # [B, 1]
            grad_hs = -((hs * sum_g) - (grad_y @ P)) / self.proto_tau  # [B, |S|]
        else:
            raise RuntimeError(f"Unknown readout_mode: {self.readout_mode}")

        grad_h_meas = torch.zeros_like(h_meas)
        grad_h_meas.index_copy_(1, self.readout.sample_indices, grad_hs)
        grad_h = self._pullback_measured_grad(h_star, grad_h_meas)

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

        a = self.alpha
        if self.dynamics_mode == "unistochastic":
            H = self._current_H(device=h_star.device, dtype=h_star.dtype)  # [N, N]
            act_prime = self._state_act_prime_from_h(h_star)  # [B, N]
            h_dag = torch.zeros_like(h_star)
            for _ in range(steps):
                # Row-vector form (with optional nonlinearity σ):
                #   h^† = (1-α) (h^† ⊙ σ'(h^*)) H + α ∇_{h*} L
                h_dag = (1.0 - a) * ((h_dag * act_prime) @ H) + a * grad_h
            return h_dag

        # Complex unitary dynamics: forward uses U^H, so adjoint uses U.
        U_total = self._unitary_total().to(device=h_star.device, dtype=h_star.dtype)
        h_dag = torch.zeros_like(h_star, dtype=U_total.dtype)
        grad_hc = grad_h.to(dtype=U_total.dtype)
        for _ in range(steps):
            h_dag = (1.0 - a) * (h_dag @ U_total) + a * grad_hc
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

        act_prime = self._state_act_prime_from_h(h_star)  # [B, N]
        grad_H = (h_dag * act_prime).transpose(0, 1) @ h_star  # [N, N]
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
        # Optional learnable H-mixing beta update
        h_mix_beta_lr_ratio: float = 1.0,
        # Optional: return gradient w.r.t input x for training an external encoder (e.g., patch embedding)
        return_grad_x: bool = False,
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
            beta_val = float(self._h_mix_beta_value(device=h_star.device, dtype=torch.float32).detach().cpu().item())
            return {
                "loss": 0.0,
                "loss_cls": 0.0,
                "loss_state": 0.0,
                "logits": logits,
                "h_star": h_star,
                "h_dag": torch.zeros_like(h_star),
                "unitary_error": unitary_error,
                "h_mix_beta": beta_val,
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

        # Work in the *measured* state space for readout and state-level losses.
        h_meas = self._measured_state(h_star)

        # Convert output gradient to a full measured-state gradient source ∇_{h_meas} L (from readout).
        grad_state_targets_cls = None
        if self.readout_mode == "linear":
            grad_hs_out = grad_y @ self.readout.readout.weight  # [B, |S|]
        elif self.readout_mode == "proto":
            # logits_c = -||h_S - P_c||^2 / (2*tau)
            hs = self._sampled_state(h_meas)  # [B, |S|]
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

        grad_h_meas = torch.zeros_like(h_meas)
        grad_h_meas.index_copy_(1, self.readout.sample_indices, grad_hs_out)

        # Optional: add a GT equilibrium-state matching loss in hidden space.
        loss_state = torch.tensor(0.0, device=h_meas.device, dtype=h_meas.dtype)
        grad_state_targets = None
        if state_target_weight > 0:
            if self.state_targets is None:
                raise ValueError("state_target_weight > 0 requires learnable_state_targets=True at init")

            B = h_meas.size(0)
            N = h_meas.size(1)
            denom = float(B * N)

            if target.dim() == 1:
                proto = self.state_targets.index_select(0, target)  # [B, N]
                err = h_meas - proto
                loss_state = 0.5 * err.pow(2).mean()
                grad_h_meas = grad_h_meas + (state_target_weight * (err / denom))

                # d/d proto: (proto - h) / (B*N)
                grad_proto_batch = (proto - h_meas) / denom  # [B, N]
                grad_state_targets = torch.zeros_like(self.state_targets)
                grad_state_targets.index_add_(0, target, grad_proto_batch)
            else:
                # Soft label target: proto = target @ P
                proto = target @ self.state_targets  # [B, N]
                err = h_meas - proto
                loss_state = 0.5 * err.pow(2).mean()
                grad_h_meas = grad_h_meas + (state_target_weight * (err / denom))

                grad_state_targets = target.transpose(0, 1) @ ((proto - h_meas) / denom)  # [C, N]

        # Pull back to the internal-state gradient ∇_{h*} L.
        grad_h = self._pullback_measured_grad(h_star, grad_h_meas)

        # ----------------------------
        # 3) Adjoint fixed point (+ optional dL/dH for unistochastic mode)
        # ----------------------------
        h_dag = self.compute_adjoint_state_from_grad_h(h_star, grad_h, steps=adjoint_steps)
        beta = self._h_mix_beta_value(device=h_meas.device, dtype=h_meas.dtype)

        grad_H = None
        grad_H_base = None
        if self.dynamics_mode == "unistochastic":
            act_prime = self._state_act_prime_from_h(h_star)
            h_eff = h_dag * act_prime
            grad_H = self.approx_grad_H(h_star, h_dag, normalize=normalize_grad_H)  # [N, N]
            beta_H = self._h_mix_beta_value(device=h_star.device, dtype=grad_H.dtype)
            grad_H_base = grad_H * beta_H  # dL/dH = beta * dL/dH_eff
        else:
            h_eff = h_dag  # complex adjoint (used for unitary updates/injection as needed)

        # ----------------------------
        # 4) Parameter updates (Readout / Injection / Unitary manifold)
        # ----------------------------
        # 4.1 Readout update: y = W_readout · h*_S
        lr_readout = lr * readout_lr_ratio
        if lr_readout > 0 and self.readout_mode == "linear":
            hs = h_meas.index_select(1, self.readout.sample_indices)  # [B, |S|]
            grad_W_readout = grad_y.transpose(0, 1) @ hs  # [C, |S|]
            self.readout.readout.weight.data.add_(grad_W_readout, alpha=-lr_readout)

        # 4.2 Injection update: J(x) = W_up W_down x
        lr_inj = lr * injection_lr_ratio
        grad_x = None
        if lr_inj > 0 or return_grad_x:
            h_inj = h_eff.real if torch.is_complex(h_eff) else h_eff
            pre = self.injection.down(x)  # [B, r]
            z = self.injection._act(pre)  # [B, r]
            grad_W_up = h_inj.transpose(0, 1) @ z  # [N, r]
            dz = h_inj @ self.injection.up.weight  # [B, r]
            dz = dz * self.injection._act_prime(pre=pre, act=z)  # [B, r]
            grad_W_down = dz.transpose(0, 1) @ x  # [r, d]
            if return_grad_x:
                # pre = x @ W_down^T  => dL/dx = dz @ W_down
                grad_x = dz @ self.injection.down.weight  # [B, d]

            if lr_inj > 0:
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

            if self.dynamics_mode == "unistochastic":
                H_u = U_total.abs().pow(2)  # [N, N] real

                # HTML: ΔA ∝ skew( U^† · ( (∂L/∂H) ⊙ U ⊙ \bar U ) )
                if grad_H_base is None or grad_H is None:
                    raise RuntimeError("grad_H is required for unistochastic EQProp update")
                inner_total = (grad_H_base * H_u).to(dtype=U_policy.dtype)  # cast real -> complex
                # Pullback through right-multiplication by constant U_base: dU_total = dU_policy U_base
                inner_policy = inner_total if U_base is None else (inner_total @ U_base.conj().transpose(-2, -1))
                M = U_policy.conj().transpose(-2, -1) @ inner_policy
                delta_A = 0.5 * (M - M.conj().transpose(-2, -1))  # skew-Hermitian

                # Map ΔA back to the stored parameters:
                # A = 0.5[(R - R^T) + i(I + I^T)], so updating R by ΔRe(A) (skew),
                # and I by ΔIm(A) (sym) yields an exact ΔA at the A-level.
                self.unitary_param.A_real.data.add_(delta_A.real, alpha=-lr_u)
                self.unitary_param.A_imag.data.add_(delta_A.imag, alpha=-lr_u)

                # Optional: learnable self-retention beta update
                lr_beta = lr * float(h_mix_beta_lr_ratio)
                if lr_beta > 0 and self.h_mix_beta_param is not None:
                    I = torch.eye(self.hidden_dim, device=H_u.device, dtype=H_u.dtype)
                    # Scale by N (not N^2): the diagonal term dominates the dot-product and otherwise
                    # beta updates become numerically negligible for typical N (e.g. 384).
                    grad_beta = (grad_H * (H_u - I)).sum() / float(self.hidden_dim)
                    beta_f32 = torch.sigmoid(self.h_mix_beta_param)
                    grad_beta_param = grad_beta.to(dtype=beta_f32.dtype) * beta_f32 * (1.0 - beta_f32)
                    self.h_mix_beta_param.data.add_(grad_beta_param, alpha=-lr_beta)
            else:
                # Complex unitary dynamics: update U_policy directly (phase participates in inference).
                B = max(1, h_star.size(0))
                grad_U_total = (h_eff.conj().transpose(0, 1) @ h_star) / float(B)  # [N, N] complex
                grad_U_policy = grad_U_total if U_base is None else (grad_U_total @ U_base.conj().transpose(-2, -1))

                # Tangent projection on U(N): ΔA ∝ skew(U^H ∇_U L)
                M = U_policy.conj().transpose(-2, -1) @ grad_U_policy
                delta_A = 0.5 * (M - M.conj().transpose(-2, -1))
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
            "h_mix_beta": float(beta.detach().cpu().item()),
            "did_update": True,
            "grad_x": grad_x,
        }

    @torch.no_grad()
    def get_orthogonal_loss(self) -> torch.Tensor:
        """
        Backward-compatible name from the previous codebase.

        Returns a diagnostic unitarity error ||U^H U - I||_F.
        """
        return self.unitary_param.unitary_error_fro().real.to(dtype=torch.float32)


class ViTBackbone(nn.Module):
    """
    A small, standard Vision Transformer encoder for CIFAR-like images.

    This module is intentionally minimal (no external deps) and follows the usual ViT recipe:
      - patchify via conv (kernel=stride=patch_size)
      - add CLS token + positional embedding
      - TransformerEncoder blocks (GELU MLP)
      - output a single vector via CLS or mean pooling
    """

    def __init__(
        self,
        *,
        img_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
        pool: str = "cls",
    ):
        super().__init__()
        if img_size <= 0:
            raise ValueError("img_size must be positive")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if mlp_dim <= 0:
            raise ValueError("mlp_dim must be positive")
        pool = str(pool)
        if pool not in ("cls", "mean"):
            raise ValueError('pool must be "cls" or "mean"')

        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_dim = int(mlp_dim)
        self.dropout = float(dropout)
        self.pool = pool

        grid = img_size // patch_size
        self.n_patches = grid * grid

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=self.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self._init_parameters()

    def _init_parameters(self):
        # Truncated normal init (common ViT init); fall back to normal if unavailable.
        try:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        except AttributeError:
            nn.init.normal_(self.pos_embed, std=0.02)
            nn.init.normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
        if x.size(2) != self.img_size or x.size(3) != self.img_size:
            raise ValueError(f"Expected H=W={self.img_size}, got {(x.size(2), x.size(3))}")

        z = self.patch_embed(x)  # [B, D, H', W']
        z = z.flatten(2).transpose(1, 2)  # [B, P, D]
        cls = self.cls_token.expand(z.size(0), -1, -1)  # [B, 1, D]
        z = torch.cat([cls, z], dim=1)  # [B, 1+P, D]
        z = z + self.pos_embed
        z = self.pos_drop(z)
        z = self.blocks(z)
        z = self.norm(z)
        if self.pool == "cls":
            return z[:, 0]
        return z[:, 1:].mean(dim=1)


class MoebiusQuantumRingImageClassifier(nn.Module):
    """Thin image wrapper for CIFAR-like inputs. Flattens images and feeds MoebiusQuantumRing."""

    def __init__(
        self,
        *,
        img_size: int = 32,
        in_channels: int = 3,
        num_classes: int = 100,
        image_encoder: str = "flatten",
        patch_size: int = 4,
        patch_embed_dim: int = 256,
        patch_pool: str = "mean",
        # ViT encoder (optional, used when image_encoder="vit")
        vit_dim: int = 384,
        vit_depth: int = 6,
        vit_heads: int = 6,
        vit_mlp_dim: int = 1536,
        vit_dropout: float = 0.0,
        vit_pool: str = "cls",
        hidden_dim: int = 384,
        alpha: float = 0.1,
        relaxation_steps: int = 20,
        lora_rank: int = 16,
        inj_activation: str = "none",
        state_activation: str = "none",
        h_mix_beta: float = 1.0,
        learnable_h_mix_beta: bool = False,
        dynamics_mode: str = "unistochastic",
        measurement: str = "identity",
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

        image_encoder = str(image_encoder)
        if image_encoder not in ("flatten", "patch", "vit"):
            raise ValueError('image_encoder must be "flatten", "patch", or "vit"')
        patch_pool = str(patch_pool)
        if patch_pool not in ("mean", "flatten"):
            raise ValueError('patch_pool must be "mean" or "flatten"')
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if patch_embed_dim <= 0:
            raise ValueError("patch_embed_dim must be positive")
        vit_pool = str(vit_pool)
        if vit_pool not in ("cls", "mean"):
            raise ValueError('vit_pool must be "cls" or "mean"')

        self.image_encoder = image_encoder
        self.patch_size = int(patch_size)
        self.patch_embed_dim = int(patch_embed_dim)
        self.patch_pool = patch_pool
        self.vit_dim = int(vit_dim)
        self.vit_depth = int(vit_depth)
        self.vit_heads = int(vit_heads)
        self.vit_mlp_dim = int(vit_mlp_dim)
        self.vit_dropout = float(vit_dropout)
        self.vit_pool = vit_pool

        self.patch_embed: Optional[nn.Module]
        self.patch_norm: Optional[nn.Module]
        self.vit: Optional[nn.Module]
        self._n_patches: int

        if self.image_encoder == "flatten":
            self.patch_embed = None
            self.patch_norm = None
            self.vit = None
            self._n_patches = 0
            input_dim = in_channels * img_size * img_size
        elif self.image_encoder == "patch":
            if img_size % patch_size != 0:
                raise ValueError("img_size must be divisible by patch_size for patch encoder")
            grid = img_size // patch_size
            self._n_patches = grid * grid
            # Simple patch embedding: conv with stride=patch_size (local receptive field + weight sharing).
            self.patch_embed = nn.Conv2d(
                in_channels,
                patch_embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            )
            # Normalize per patch token.
            self.patch_norm = nn.LayerNorm(patch_embed_dim)
            self.vit = None
            input_dim = patch_embed_dim if patch_pool == "mean" else (self._n_patches * patch_embed_dim)
        else:
            # A standard Vision Transformer encoder used as a strong, mature backbone.
            # The ring becomes a drop-in replacement for the usual linear classification head.
            if img_size % patch_size != 0:
                raise ValueError("img_size must be divisible by patch_size for vit encoder")
            self.patch_embed = None
            self.patch_norm = None
            self._n_patches = (img_size // patch_size) * (img_size // patch_size)
            self.vit = ViTBackbone(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=self.vit_dim,
                depth=self.vit_depth,
                num_heads=self.vit_heads,
                mlp_dim=self.vit_mlp_dim,
                dropout=self.vit_dropout,
                pool=self.vit_pool,
            )
            input_dim = self.vit_dim
        self.ring = MoebiusQuantumRing(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            alpha=alpha,
            relaxation_steps=relaxation_steps,
            lora_rank=lora_rank,
            inj_activation=inj_activation,
            state_activation=state_activation,
            h_mix_beta=h_mix_beta,
            learnable_h_mix_beta=learnable_h_mix_beta,
            dynamics_mode=dynamics_mode,
            measurement=measurement,
            readout_dim=readout_dim,
            readout_mode=readout_mode,
            proto_tau=proto_tau,
            base_unitary_init=base_unitary_init,
            base_unitary_scale=base_unitary_scale,
            base_unitary_seed=base_unitary_seed,
            learnable_state_targets=learnable_state_targets,
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an image batch into a vector input for the ring."""
        if x.dim() != 4:
            raise ValueError(f"Expected image tensor [B,C,H,W], got {tuple(x.shape)}")
        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected C={self.in_channels}, got {x.size(1)}")
        if x.size(2) != self.img_size or x.size(3) != self.img_size:
            raise ValueError(f"Expected H=W={self.img_size}, got {(x.size(2), x.size(3))}")

        if self.image_encoder == "flatten":
            return x.flatten(1)  # [B, C*H*W]

        if self.image_encoder == "vit":
            assert self.vit is not None
            return self.vit(x)  # [B, vit_dim]

        assert self.patch_embed is not None and self.patch_norm is not None
        z = self.patch_embed(x)  # [B, D, H', W']
        z = z.flatten(2).transpose(1, 2)  # [B, P, D]
        z = self.patch_norm(z)
        if self.patch_pool == "mean":
            return z.mean(dim=1)  # [B, D]
        return z.reshape(z.size(0), -1)  # [B, P*D]

    def forward(self, x: torch.Tensor):
        x_vec = self._encode(x)
        return self.ring(x_vec)

    @torch.no_grad()
    def get_orthogonal_loss(self) -> torch.Tensor:
        """Expose the same helper used by legacy training scripts."""
        return self.ring.get_orthogonal_loss()

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
        h_mix_beta_lr_ratio: float = 1.0,
        encoder_lr_ratio: float = 1.0,
        encoder_optimizer: Optional[object] = None,
    ) -> dict:
        """Image wrapper for `MoebiusQuantumRing.eqprop_update_step`."""
        enc_lr = float(encoder_lr_ratio)
        if self.image_encoder in ("patch", "vit") and enc_lr > 0:
            # Build an autograd graph for the encoder only.
            x_vec = self._encode(images)
            enc_params = []
            if self.image_encoder == "patch":
                assert self.patch_embed is not None and self.patch_norm is not None
                enc_params.extend(list(self.patch_embed.parameters()))
                enc_params.extend(list(self.patch_norm.parameters()))
            else:
                assert self.vit is not None
                enc_params.extend(list(self.vit.parameters()))

            if encoder_optimizer is not None:
                # Ensure the optimizer uses the current step LR (cosine schedule is handled outside).
                try:
                    for g in encoder_optimizer.param_groups:
                        g["lr"] = float(lr) * enc_lr
                except Exception:
                    pass
                try:
                    encoder_optimizer.zero_grad(set_to_none=True)
                except TypeError:
                    encoder_optimizer.zero_grad()
            else:
                for p in enc_params:
                    p.grad = None

            info = self.ring.eqprop_update_step(
                x_vec.detach(),
                target,
                lr=lr,
                unitary_lr_ratio=unitary_lr_ratio,
                injection_lr_ratio=injection_lr_ratio,
                readout_lr_ratio=readout_lr_ratio,
                adjoint_steps=adjoint_steps,
                normalize_grad_H=normalize_grad_H,
                state_target_weight=state_target_weight,
                state_target_lr_ratio=state_target_lr_ratio,
                h_mix_beta_lr_ratio=h_mix_beta_lr_ratio,
                return_grad_x=True,
            )

            grad_x = info.get("grad_x", None)
            if grad_x is not None:
                x_vec.backward(grad_x)
                if encoder_optimizer is not None:
                    encoder_optimizer.step()
                else:
                    lr_enc = lr * enc_lr
                    with torch.no_grad():
                        for p in enc_params:
                            if p.grad is None:
                                continue
                            p.data.add_(p.grad, alpha=-lr_enc)
                            p.grad = None
            info.pop("grad_x", None)
            return info

        # No encoder params (flatten) or encoder_lr_ratio==0: pure ring update.
        x_vec = self._encode(images)
        info = self.ring.eqprop_update_step(
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
            h_mix_beta_lr_ratio=h_mix_beta_lr_ratio,
        )
        info.pop("grad_x", None)
        return info


def create_mobius_model(
    num_classes: int = 100,
    img_size: int = 32,
    in_channels: int = 3,
    *,
    image_encoder: str = "flatten",
    patch_size: int = 4,
    patch_embed_dim: int = 256,
    patch_pool: str = "mean",
    vit_dim: int = 384,
    vit_depth: int = 6,
    vit_heads: int = 6,
    vit_mlp_dim: int = 1536,
    vit_dropout: float = 0.0,
    vit_pool: str = "cls",
    embed_dim: int = 384,
    depth: int = 20,
    alpha: float = 0.1,
    lora_rank: int = 16,
    inj_activation: str = "none",
    state_activation: str = "none",
    h_mix_beta: float = 1.0,
    learnable_h_mix_beta: bool = False,
    dynamics_mode: str = "unistochastic",
    measurement: str = "identity",
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
        image_encoder=image_encoder,
        patch_size=patch_size,
        patch_embed_dim=patch_embed_dim,
        patch_pool=patch_pool,
        vit_dim=vit_dim,
        vit_depth=vit_depth,
        vit_heads=vit_heads,
        vit_mlp_dim=vit_mlp_dim,
        vit_dropout=vit_dropout,
        vit_pool=vit_pool,
        hidden_dim=embed_dim,
        alpha=alpha,
        relaxation_steps=depth,
        lora_rank=lora_rank,
        inj_activation=inj_activation,
        state_activation=state_activation,
        h_mix_beta=h_mix_beta,
        learnable_h_mix_beta=learnable_h_mix_beta,
        dynamics_mode=dynamics_mode,
        measurement=measurement,
        readout_dim=readout_dim,
        readout_mode=readout_mode,
        proto_tau=proto_tau,
        base_unitary_init=base_unitary_init,
        base_unitary_scale=base_unitary_scale,
        base_unitary_seed=base_unitary_seed,
        learnable_state_targets=learnable_state_targets,
    )

