"""Fair local comparison of Cayley and Sinkhorn transition gradients.

Both methods receive ``N^2`` stored parameters and the same nonlinear
fixed-point loss.  Cayley uses the exact modulus-square/Cayley pullback;
Sinkhorn uses a converged implicit matrix-scaling pullback rather than a
straight-through estimator.  Timings are diagnostics for this CPU/runtime,
not universal complexity claims or task-quality results.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mqr import CayleyUnistochasticParam, SinkhornDoublyStochasticParam


RESULT_PATH = Path(__file__).resolve().parent / "results" / "mqr_sinkhorn_fair_compare.json"


def relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = max(float(torch.linalg.vector_norm(reference).item()), 1e-30)
    return float(torch.linalg.vector_norm(actual - reference).item()) / denominator


def solve_state(
    H: torch.Tensor,
    injection: torch.Tensor,
    *,
    alpha: float,
    steps: int,
) -> torch.Tensor:
    h = torch.zeros_like(injection)
    for _ in range(steps):
        h = torch.tanh((1.0 - alpha) * (h @ H.transpose(0, 1)) + alpha * injection)
    return h


@torch.no_grad()
def direct_adjoint_and_grad_H(
    H: torch.Tensor,
    h: torch.Tensor,
    grad_h: torch.Tensor,
    *,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    derivative = 1.0 - h.square()
    n = H.size(0)
    identity = torch.eye(n, device=H.device, dtype=H.dtype).expand(h.size(0), -1, -1)
    system = identity - (1.0 - alpha) * (
        H.transpose(0, 1).unsqueeze(0) * derivative.unsqueeze(1)
    )
    p = torch.linalg.solve(system, alpha * grad_h.unsqueeze(-1)).squeeze(-1)
    grad_H = ((1.0 - alpha) / alpha) * ((p * derivative).transpose(0, 1) @ h)
    residual = (
        (1.0 - alpha) * ((p * derivative) @ H) + alpha * grad_h - p
    ).abs().amax()
    return p, grad_H, float(residual.item())


def downstream_gradient_checks() -> Dict[str, object]:
    torch.manual_seed(211)
    n = 6
    batch = 4
    classes = 3
    alpha = 0.2
    steps = 240
    injection = torch.randn(batch, n, dtype=torch.float64)
    readout = torch.randn(classes, n, dtype=torch.float64)
    target = torch.tensor([0, 1, 2, 1])

    cayley = CayleyUnistochasticParam(n, coordinate_mode="minimal").double()
    with torch.no_grad():
        cayley.A_real.mul_(25.0)
        cayley.A_imag.mul_(25.0)
    H_cayley = cayley.unistochastic()
    h_cayley = solve_state(H_cayley, injection, alpha=alpha, steps=steps)
    logits_cayley = h_cayley @ readout.transpose(0, 1)
    loss_cayley = F.cross_entropy(logits_cayley, target)
    loss_cayley.backward()

    with torch.no_grad():
        grad_logits = torch.softmax(logits_cayley, dim=1)
        grad_logits[torch.arange(batch), target] -= 1.0
        grad_logits /= batch
        grad_h = grad_logits @ readout
        _, grad_H, adjoint_residual = direct_adjoint_and_grad_H(
            H_cayley.detach(), h_cayley.detach(), grad_h, alpha=alpha
        )
        U = cayley.unitary()
        grad_A = cayley.cayley_pullback(2.0 * grad_H.to(U.dtype) * U)
        coordinate_gradients = cayley.coordinate_gradients(grad_A)
        cayley_error = max(
            relative_error(coordinate_gradients["A_real"], cayley.A_real.grad),
            relative_error(coordinate_gradients["A_imag"], cayley.A_imag.grad),
        )

    sinkhorn = SinkhornDoublyStochasticParam(
        n, iterations=240, temperature=0.9, init_scale=0.5
    ).double()
    H_sinkhorn = sinkhorn.doubly_stochastic()
    h_sinkhorn = solve_state(H_sinkhorn, injection, alpha=alpha, steps=steps)
    logits_sinkhorn = h_sinkhorn @ readout.transpose(0, 1)
    loss_sinkhorn = F.cross_entropy(logits_sinkhorn, target)
    loss_sinkhorn.backward()

    with torch.no_grad():
        grad_logits = torch.softmax(logits_sinkhorn, dim=1)
        grad_logits[torch.arange(batch), target] -= 1.0
        grad_logits /= batch
        grad_h = grad_logits @ readout
        _, grad_H, sinkhorn_adjoint_residual = direct_adjoint_and_grad_H(
            H_sinkhorn.detach(), h_sinkhorn.detach(), grad_h, alpha=alpha
        )
        implicit_logits = sinkhorn.implicit_logit_pullback(
            grad_H, H=H_sinkhorn.detach()
        )
        sinkhorn_error = relative_error(implicit_logits, sinkhorn.logits.grad)
        sinkhorn_row_error, sinkhorn_column_error = sinkhorn.doubly_stochastic_errors(
            H=H_sinkhorn.detach()
        )
        cayley_row_error, cayley_column_error = cayley.doubly_stochastic_errors()

    if max(cayley_error, sinkhorn_error) >= 1e-8:
        raise AssertionError("A transition pullback failed the autograd comparison")
    return {
        "dimension": n,
        "batch": batch,
        "fixed_point_steps": steps,
        "alpha": alpha,
        "cayley": {
            "stored_parameters": cayley.raw_parameter_count,
            "transition_effective_dof_max": (n - 1) ** 2,
            "gradient_relative_error": cayley_error,
            "adjoint_residual": adjoint_residual,
            "row_error": float(cayley_row_error.item()),
            "column_error": float(cayley_column_error.item()),
        },
        "sinkhorn": {
            "stored_parameters": int(sinkhorn.logits.numel()),
            "transition_effective_dof": sinkhorn.effective_dof,
            "normalization_iterations": sinkhorn.iterations,
            "gradient_relative_error": sinkhorn_error,
            "adjoint_residual": sinkhorn_adjoint_residual,
            "row_error": float(sinkhorn_row_error.item()),
            "column_error": float(sinkhorn_column_error.item()),
        },
    }


def median_milliseconds(operation: Callable[[], object], *, repeats: int) -> float:
    for _ in range(2):
        operation()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        timings.append(1000.0 * (time.perf_counter() - start))
    return statistics.median(timings)


@torch.no_grad()
def timing_rows() -> list[Dict[str, object]]:
    rows = []
    for n in (8, 16, 32, 64):
        torch.manual_seed(300 + n)
        cayley = CayleyUnistochasticParam(n, coordinate_mode="minimal").double()
        sinkhorn = SinkhornDoublyStochasticParam(
            n, iterations=20, temperature=1.0, init_scale=0.2
        ).double()
        grad_H = torch.randn(n, n, dtype=torch.float64)
        U = cayley.unitary()
        grad_U = 2.0 * grad_H.to(U.dtype) * U
        H_sinkhorn = sinkhorn.doubly_stochastic()
        repeats = 15 if n <= 32 else 8
        cayley_forward_ms = median_milliseconds(cayley.unistochastic, repeats=repeats)
        sinkhorn_forward_ms = median_milliseconds(
            sinkhorn.doubly_stochastic, repeats=repeats
        )
        cayley_pullback_ms = median_milliseconds(
            lambda: cayley.cayley_pullback(grad_U), repeats=repeats
        )
        sinkhorn_pullback_ms = median_milliseconds(
            lambda: sinkhorn.implicit_logit_pullback(grad_H, H=H_sinkhorn),
            repeats=repeats,
        )
        c_row, c_col = cayley.doubly_stochastic_errors()
        s_row, s_col = sinkhorn.doubly_stochastic_errors(H=H_sinkhorn)
        rows.append(
            {
                "dimension": n,
                "repeats": repeats,
                "cayley_forward_median_ms": cayley_forward_ms,
                "sinkhorn_20_forward_median_ms": sinkhorn_forward_ms,
                "cayley_pullback_median_ms": cayley_pullback_ms,
                "sinkhorn_implicit_pullback_median_ms": sinkhorn_pullback_ms,
                "cayley_constraint_error": max(float(c_row.item()), float(c_col.item())),
                "sinkhorn_constraint_error": max(float(s_row.item()), float(s_col.item())),
            }
        )
    return rows


def main() -> None:
    torch.set_default_dtype(torch.float64)
    gradient_checks = downstream_gradient_checks()
    timings = timing_rows()
    payload = {
        "schema_version": 1,
        "purpose": "matched transition-gradient correctness and local CPU cost; not task superiority",
        "dtype": "float64",
        "gradient_checks": gradient_checks,
        "timings": timings,
        "timing_warning": "runtime-specific medians without accelerator synchronization",
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        "pullback relative errors:",
        {
            "cayley": gradient_checks["cayley"]["gradient_relative_error"],
            "sinkhorn": gradient_checks["sinkhorn"]["gradient_relative_error"],
        },
    )
    print("N  Cayley-fwd  Sinkhorn20-fwd  Cayley-bwd  Sinkhorn-implicit-bwd")
    for row in timings:
        print(
            f"{int(row['dimension']):2d} "
            f"{float(row['cayley_forward_median_ms']):11.4f} "
            f"{float(row['sinkhorn_20_forward_median_ms']):14.4f} "
            f"{float(row['cayley_pullback_median_ms']):11.4f} "
            f"{float(row['sinkhorn_implicit_pullback_median_ms']):21.4f}"
        )
    print(f"wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
