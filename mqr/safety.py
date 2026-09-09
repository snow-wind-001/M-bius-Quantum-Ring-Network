"""Safety contracts for causal online sidecar updates.

The helpers in this module deliberately certify local numerical behavior only.
They do not turn a structural MQR invariant into a guarantee about a frozen
backbone's task performance; that requires explicit constancy probes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SidecarSafetyLimits:
    """Optional rejection limits for one candidate online update.

    ``max_policy_kl`` is the mean categorical
    :math:`D_{KL}(p_{before}\Vert p_{after})`.  The two infinity-norm limits
    apply to local output and recurrent-state drift.  Transition drift is the
    Frobenius norm of the change in :math:`H=|U|^2`.  A ``None`` value disables
    that rejection criterion.  ``ogd_retained_warning_threshold`` emits an
    audit warning only; it never forces an unsafe update through OGD.
    """

    max_policy_kl: float | None = None
    max_output_linf_drift: float | None = None
    max_state_linf_drift: float | None = None
    max_transition_fro_drift: float | None = None
    ogd_retained_warning_threshold: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "max_policy_kl",
            "max_output_linf_drift",
            "max_state_linf_drift",
            "max_transition_fro_drift",
        ):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(float(value)) or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative or None")
        threshold = float(self.ogd_retained_warning_threshold)
        if not math.isfinite(threshold) or not (0.0 <= threshold <= 1.0):
            raise ValueError("ogd_retained_warning_threshold must lie in [0, 1]")


def tensor_linf_drift(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Return ``max(abs(candidate-reference))`` after strict shape checks."""

    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate tensors must have the same shape")
    if torch.is_complex(reference) or torch.is_complex(candidate):
        raise TypeError("constancy tensors must be real")
    if not reference.is_floating_point() or not candidate.is_floating_point():
        raise TypeError("constancy tensors must be floating point")
    if not bool(torch.isfinite(reference).all()) or not bool(
        torch.isfinite(candidate).all()
    ):
        return math.inf
    return float((candidate - reference).abs().amax().item())


def categorical_policy_kl(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> float:
    """Return mean ``KL(reference || candidate)`` over the last dimension."""

    if reference_logits.shape != candidate_logits.shape:
        raise ValueError("reference and candidate logits must have the same shape")
    if reference_logits.dim() == 0 or reference_logits.size(-1) < 2:
        raise ValueError("policy logits must have at least two classes")
    if not reference_logits.is_floating_point() or not candidate_logits.is_floating_point():
        raise TypeError("policy logits must be floating point")
    if torch.is_complex(reference_logits) or torch.is_complex(candidate_logits):
        raise TypeError("policy logits must be real")
    if not bool(torch.isfinite(reference_logits).all()) or not bool(
        torch.isfinite(candidate_logits).all()
    ):
        return math.inf
    reference_log_prob = F.log_softmax(reference_logits, dim=-1)
    candidate_log_prob = F.log_softmax(candidate_logits, dim=-1)
    reference_prob = reference_log_prob.exp()
    value = (
        reference_prob * (reference_log_prob - candidate_log_prob)
    ).sum(dim=-1).mean()
    # Exact equality can produce a tiny negative round-off residue.
    return max(0.0, float(value.item()))


def residual_output_diagnostics(
    native: torch.Tensor,
    residual: torch.Tensor,
) -> Dict[str, float]:
    """Measure a residual sidecar without assuming a non-negative state."""

    if native.shape != residual.shape:
        raise ValueError("native and residual tensors must have the same shape")
    if native.dim() < 2:
        raise ValueError("native and residual tensors must include a batch dimension")
    if not native.is_floating_point() or not residual.is_floating_point():
        raise TypeError("native and residual tensors must be floating point")
    if torch.is_complex(native) or torch.is_complex(residual):
        raise TypeError("native and residual tensors must be real")
    if not bool(torch.isfinite(native).all()) or not bool(torch.isfinite(residual).all()):
        return {
            "native_l2_max": math.inf,
            "residual_l2_max": math.inf,
            "residual_linf": math.inf,
            "residual_to_native_l2_ratio_max": math.inf,
        }
    native_flat = native.reshape(native.size(0), -1)
    residual_flat = residual.reshape(residual.size(0), -1)
    native_norm = torch.linalg.vector_norm(native_flat, dim=1)
    residual_norm = torch.linalg.vector_norm(residual_flat, dim=1)
    eps = torch.finfo(native.dtype).eps
    ratio = residual_norm / native_norm.clamp_min(eps)
    return {
        "native_l2_max": float(native_norm.max().item()),
        "residual_l2_max": float(residual_norm.max().item()),
        "residual_linf": float(residual.abs().amax().item()),
        "residual_to_native_l2_ratio_max": float(ratio.max().item()),
    }
