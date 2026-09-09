"""Finite-solver certification study for nonlinear MQR implicit gradients.

This offline float64 diagnostic uses no downloaded data.  It separates
forward fixed-point error from adjoint iteration error and writes every raw
configuration before printing aggregate trends.  It is a numerical mechanism
check, not evidence that MQR outperforms another learning algorithm.
"""

from __future__ import annotations

import copy
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mqr import MoebiusQuantumRing


RESULT_PATH = Path(__file__).resolve().parent / "results" / "mqr_solver_certification.json"


def relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = max(float(torch.linalg.vector_norm(reference).item()), 1e-30)
    return float(torch.linalg.vector_norm(actual - reference).item()) / denominator


def cosine(actual: torch.Tensor, reference: torch.Tensor) -> float:
    actual_real = torch.view_as_real(actual).reshape(-1)
    reference_real = torch.view_as_real(reference).reshape(-1)
    denominator = float(
        torch.linalg.vector_norm(actual_real).item()
        * torch.linalg.vector_norm(reference_real).item()
    )
    if denominator == 0.0:
        return 1.0 if torch.equal(actual_real, reference_real) else 0.0
    return float(torch.dot(actual_real, reference_real).item()) / denominator


@torch.no_grad()
def loss_sources(
    model: MoebiusQuantumRing,
    logits: torch.Tensor,
    h: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grad_logits = torch.softmax(logits, dim=1)
    grad_logits[torch.arange(target.numel()), target] -= 1.0
    grad_logits /= target.numel()
    grad_h = grad_logits @ model.readout.readout.weight
    return grad_logits, grad_h


@torch.no_grad()
def cayley_gradient(
    model: MoebiusQuantumRing,
    h: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    grad_H_eff = model.approx_grad_H(h, p, normalize=False)
    beta = model._h_mix_beta_value(device=h.device, dtype=h.dtype)
    U = model._unitary_total()
    grad_U = 2.0 * (beta * grad_H_eff).to(dtype=U.dtype) * U
    return model.unitary_param.cayley_pullback(grad_U)


@torch.no_grad()
def reference_solution(
    base: MoebiusQuantumRing,
    x: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, object]:
    model = copy.deepcopy(base)
    model.relaxation_steps = 3000
    model.relaxation_min_steps = 2
    model.relaxation_tol = 1e-13
    logits, state = model(x, return_state=True)
    _, grad_h = loss_sources(model, logits, state.h, target)
    p, adjoint_info = model.solve_adjoint_state_from_grad_h(
        state.h, grad_h, return_info=True
    )
    return {
        "model": model,
        "state": state,
        "gradient": cayley_gradient(model, state.h, p),
        "adjoint": p,
        "adjoint_info": adjoint_info,
    }


@torch.no_grad()
def finite_record(
    reference: Dict[str, object],
    x: torch.Tensor,
    target: torch.Tensor,
    *,
    steps: int,
    activation: str,
    alpha: float,
    initialization_multiplier: float,
) -> Dict[str, object]:
    reference_model = reference["model"]
    assert isinstance(reference_model, MoebiusQuantumRing)
    model = copy.deepcopy(reference_model)
    model.relaxation_steps = int(steps)
    model.relaxation_min_steps = 1
    model.relaxation_tol = None

    start = time.perf_counter()
    logits, state = model(x, return_state=True)
    _, grad_h = loss_sources(model, logits, state.h, target)
    p_iter, adjoint_info = model.compute_adjoint_state_from_grad_h(
        state.h,
        grad_h,
        steps=steps,
        return_info=True,
    )
    p_direct, direct_info = model.solve_adjoint_state_from_grad_h(
        state.h, grad_h, return_info=True
    )
    gradient_iter = cayley_gradient(model, state.h, p_iter)
    gradient_direct_at_finite_state = cayley_gradient(model, state.h, p_direct)
    elapsed_ms = 1000.0 * (time.perf_counter() - start)

    reference_state = reference["state"]
    reference_gradient = reference["gradient"]
    assert hasattr(reference_state, "h")
    assert isinstance(reference_gradient, torch.Tensor)
    state_error = float((state.h - reference_state.h).abs().amax().item())
    adjoint_error = float((p_iter - p_direct).abs().amax().item())
    forward_bound = float(state.error_bound)
    adjoint_bound = float(adjoint_info["error_bound"])
    certified_at_1e6 = bool(
        float(state.relative_residual) <= 1e-6
        and float(adjoint_info["relative_residual"]) <= 1e-6
    )
    return {
        "activation": activation,
        "alpha": alpha,
        "steps": int(steps),
        "initialization_multiplier": initialization_multiplier,
        "forward_residual": float(state.residual),
        "forward_relative_residual": float(state.relative_residual),
        "forward_error_bound": forward_bound,
        "forward_state_error": state_error,
        "forward_bound_holds": state_error <= forward_bound + 1e-12,
        "adjoint_residual": float(adjoint_info["residual"]),
        "adjoint_relative_residual": float(adjoint_info["relative_residual"]),
        "adjoint_error_bound": adjoint_bound,
        "adjoint_state_error": adjoint_error,
        "adjoint_bound_holds": adjoint_error <= adjoint_bound + 1e-12,
        "direct_adjoint_residual": float(direct_info["residual"]),
        "transition_gradient_relative_error": relative_error(
            gradient_iter, reference_gradient
        ),
        "transition_gradient_cosine": cosine(gradient_iter, reference_gradient),
        "forward_only_gradient_relative_error": relative_error(
            gradient_direct_at_finite_state, reference_gradient
        ),
        "adjoint_only_gradient_relative_error": relative_error(
            gradient_iter, gradient_direct_at_finite_state
        ),
        "certified_at_1e6": certified_at_1e6,
        "elapsed_ms": elapsed_ms,
    }


def aggregate(records: list[Dict[str, object]]) -> list[Dict[str, object]]:
    rows = []
    keys = sorted({(float(row["alpha"]), int(row["steps"])) for row in records})
    for alpha, steps in keys:
        selected = [
            row
            for row in records
            if float(row["alpha"]) == alpha and int(row["steps"]) == steps
        ]
        errors = [float(row["transition_gradient_relative_error"]) for row in selected]
        rows.append(
            {
                "alpha": alpha,
                "steps": steps,
                "cases": len(selected),
                "certified_fraction_at_1e6": statistics.fmean(
                    float(bool(row["certified_at_1e6"])) for row in selected
                ),
                "mean_transition_gradient_relative_error": statistics.fmean(errors),
                "max_transition_gradient_relative_error": max(errors),
                "mean_forward_relative_residual": statistics.fmean(
                    float(row["forward_relative_residual"]) for row in selected
                ),
                "mean_adjoint_relative_residual": statistics.fmean(
                    float(row["adjoint_relative_residual"]) for row in selected
                ),
            }
        )
    return rows


def main() -> None:
    torch.set_default_dtype(torch.float64)
    records: list[Dict[str, object]] = []
    reference_metadata = []
    for activation in ("none", "tanh"):
        for alpha in (0.30, 0.10, 0.03):
            for multiplier in (1.0, 20.0):
                torch.manual_seed(101)
                base = MoebiusQuantumRing(
                    input_dim=5,
                    hidden_dim=6,
                    output_dim=3,
                    alpha=alpha,
                    relaxation_steps=20,
                    lora_rank=4,
                    readout_dim=6,
                    state_activation=activation,
                    h_mix_beta=0.7,
                ).double()
                with torch.no_grad():
                    base.unitary_param.A_real.mul_(multiplier)
                    base.unitary_param.A_imag.mul_(multiplier)
                x = torch.randn(4, 5, dtype=torch.float64)
                target = torch.tensor([0, 1, 2, 1])
                reference = reference_solution(base, x, target)
                reference_state = reference["state"]
                reference_metadata.append(
                    {
                        "activation": activation,
                        "alpha": alpha,
                        "initialization_multiplier": multiplier,
                        "iterations": int(reference_state.iterations),
                        "relative_residual": float(reference_state.relative_residual),
                        "converged": bool(reference_state.converged),
                    }
                )
                for steps in (20, 60, 200):
                    records.append(
                        finite_record(
                            reference,
                            x,
                            target,
                            steps=steps,
                            activation=activation,
                            alpha=alpha,
                            initialization_multiplier=multiplier,
                        )
                    )

    if not all(bool(row["forward_bound_holds"]) for row in records):
        raise AssertionError("A forward residual certificate was violated")
    if not all(bool(row["adjoint_bound_holds"]) for row in records):
        raise AssertionError("An adjoint residual certificate was violated")

    aggregates = aggregate(records)
    payload = {
        "schema_version": 1,
        "purpose": "finite fixed-point/adjoint mechanism certification; not algorithm superiority",
        "dtype": "float64",
        "seed": 101,
        "reference_tolerance": 1e-13,
        "certification_reporting_tolerance": 1e-6,
        "reference_runs": reference_metadata,
        "records": records,
        "aggregate_by_alpha_steps": aggregates,
        "all_forward_bounds_hold": True,
        "all_adjoint_bounds_hold": True,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("alpha  steps  certified  mean_grad_relerr  max_grad_relerr")
    for row in aggregates:
        print(
            f"{float(row['alpha']):.2f}   {int(row['steps']):4d}   "
            f"{float(row['certified_fraction_at_1e6']):8.3f}   "
            f"{float(row['mean_transition_gradient_relative_error']):16.6e}   "
            f"{float(row['max_transition_gradient_relative_error']):15.6e}"
        )
    print(f"wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
