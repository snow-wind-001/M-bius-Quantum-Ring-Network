"""Numerical checks for signed propagation and local online protection."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mqr import CyclicGivensUnistochasticParam, OrthogonalGradientMemory


def verify() -> dict:
    torch.manual_seed(909)
    parameter = CyclicGivensUnistochasticParam(8, base_init="identity").double()
    x, cotangent = torch.randn(2, 8, dtype=torch.float64), torch.randn(2, 8, dtype=torch.float64)
    signed_loss = (parameter.apply_orthogonal(x) * cotangent).sum()
    squared_loss = ((x @ parameter.unistochastic().T) * cotangent).sum()
    signed_gradient = torch.autograd.grad(signed_loss, parameter.angles)[0]
    squared_gradient = torch.autograd.grad(squared_loss, parameter.angles)[0]
    with torch.no_grad():
        parameter.angles.normal_(std=0.3)
    rotation = parameter.unitary().real.detach()
    identity = torch.eye(8, dtype=torch.float64)
    rho = 0.97
    kappa = math.sqrt(1.0 - rho**2)
    covariance_next = rho**2 * rotation @ identity @ rotation.T + kappa**2 * identity
    injection = torch.linalg.qr(torch.randn(8, 4, dtype=torch.float64)).Q
    shift = torch.roll(identity, shifts=1, dims=0)
    twisted_shift = shift.clone()
    twisted_shift[0].neg_()
    reachability = {}
    for name, operator in (
        ("identity", identity), ("unistochastic", rotation.square()),
        ("signed_givens", rotation), ("cyclic_shift", shift),
        ("mobius_twisted_shift", twisted_shift),
    ):
        term = injection
        terms = []
        for _ in range(32):
            terms.append(term)
            term = rho * operator @ term
        matrix = torch.cat(terms, dim=1)
        singular = torch.linalg.svdvals(matrix)
        reachability[name] = {
            "rank": int(torch.linalg.matrix_rank(matrix)),
            "normalized_smallest_singular_value": float(singular[-1] / singular[0]),
            "state_dimension": 8, "injection_rank": 4, "horizon": 32,
        }
    assert reachability["identity"]["rank"] == 4
    assert reachability["signed_givens"]["rank"] == 8
    assert torch.allclose(torch.linalg.matrix_power(twisted_shift, 8), -identity)
    propagated = x.clone()
    for _ in range(32):
        propagated = rho * parameter.apply_orthogonal(propagated)
    memory = OrthogonalGradientMemory(4)
    remembered = []
    for _ in range(4):
        left, right = torch.randn(7, dtype=torch.float64), torch.randn(5, dtype=torch.float64)
        remembered.append((left, right))
        memory.observe([("left", left, 0.1), ("right", right, 0.025)])
    projected, stats = memory.project_preconditioned([
        ("left", torch.randn(7, dtype=torch.float64), 0.1),
        ("right", torch.randn(5, dtype=torch.float64), 0.025),
    ])
    max_overlap = max(abs(float(-0.1 * left @ projected["left"] - 0.025 * right @ projected["right"]))
                      for left, right in remembered)
    result = {
        "seed": 909, "dtype": "float64",
        "signed_gradient_norm_at_identity": float(signed_gradient.norm()),
        "squared_gradient_norm_at_identity": float(squared_gradient.norm()),
        "orthogonality_error": float((rotation.T @ rotation - identity).norm()),
        "isotropic_stationary_covariance_error": float((covariance_next - identity).abs().max()),
        "damped_norm_error_32_steps": float((propagated.detach().norm(dim=1) - x.norm(dim=1) * rho**32).abs().max()),
        "ogd_first_order_overlap": max_overlap,
        "ogd_basis_orthogonality_error": memory.orthogonality_error(),
        "ogd_retained_norm": stats["retained_norm"],
        "low_rank_injection_reachability": reachability,
        "stability_assumptions": "orthogonal carry; exogenous bounded writes; no state-dependent write feedback",
        "covariance_assumptions": "zero-mean isotropic forcing, independent of previous state",
        "performance_or_novelty_claim": False,
    }
    assert result["signed_gradient_norm_at_identity"] > 1e-3
    assert result["squared_gradient_norm_at_identity"] == 0.0
    for key in ("orthogonality_error", "isotropic_stationary_covariance_error",
                "damped_norm_error_32_steps", "ogd_first_order_overlap", "ogd_basis_orthogonality_error"):
        assert result[key] < 1e-12, (key, result[key])
    return result


if __name__ == "__main__":
    result = verify()
    output = ROOT / "analysis/results/orthogonal_ring_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
