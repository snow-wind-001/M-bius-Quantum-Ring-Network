#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Numerical checks for the formal MQR analysis.

These checks are not substitutes for proofs. They are regression witnesses for
the identities used by ``analysis/mqr_math_proof.md`` and deliberately include
a reconstruction of the historical (incorrect) unitary update for comparison.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mqr.ring import MoebiusQuantumRing  # noqa: E402
from mqr.online import OrthogonalGradientMemory  # noqa: E402
from mqr.temporal import (  # noqa: E402
    MultiTimescaleMQR,
    OnlineTemporalMQRClassifier,
    TemporalMQRState,
)
from mqr.unitary import CayleyUnistochasticParam  # noqa: E402


torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


def relative_error(got: torch.Tensor, expected: torch.Tensor) -> float:
    return float((got - expected).norm() / expected.norm().clamp_min(1e-30))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0))


def structural_checks() -> None:
    param = CayleyUnistochasticParam(8).double()
    U = param.unitary()
    H = U.abs().square()
    eye = torch.eye(8, dtype=U.dtype)
    A = param.skew_hermitian_A()
    a_err = torch.linalg.matrix_norm(A + A.conj().T).item()
    u_err = torch.linalg.matrix_norm(U.conj().T @ U - eye).item()
    row_err = (H.sum(1) - 1).abs().max().item()
    col_err = (H.sum(0) - 1).abs().max().item()
    assert max(a_err, u_err, row_err, col_err) < 1e-12
    print("[structure]", {"A_skew_error": a_err, "U_unitary_error": u_err,
                          "H_row_error": row_err, "H_col_error": col_err})


def cayley_unistochastic_coverage_check() -> None:
    """Witness that Cayley's spectral omission does not omit an H=|U|^2.

    A unitary representative may contain eigenvalue -1 and therefore have no
    finite inverse Cayley coordinate. Multiplication by a generic global phase
    leaves its modulus square unchanged and avoids the finite forbidden phase
    set, after which the inverse Cayley transform is skew-Hermitian.
    """

    torch.manual_seed(3)
    n = 5
    raw = torch.randn(n, n, dtype=torch.cdouble)
    basis, _ = torch.linalg.qr(raw)
    phases = torch.tensor(
        [math.pi, -1.1, -0.2, 0.7, 1.8], dtype=torch.float64
    )
    representative = basis @ torch.diag(torch.exp(1j * phases)) @ basis.conj().T
    eye = torch.eye(n, dtype=torch.cdouble)
    assert torch.linalg.svdvals(eye + representative).min().item() < 1e-12

    parameter = CayleyUnistochasticParam(n, coordinate_mode="minimal").double()
    diagnostics = parameter.set_from_unitary_representative_(representative)
    selected_phase = torch.complex(
        torch.tensor(diagnostics["selected_phase_real"], dtype=torch.double),
        torch.tensor(diagnostics["selected_phase_imag"], dtype=torch.double),
    )
    admissible = selected_phase * representative
    coordinate = parameter.skew_hermitian_A()
    reconstructed = parameter.unitary()
    skew_error = float((coordinate + coordinate.conj().T).abs().max().item())
    unitary_error = float((reconstructed - admissible).abs().max().item())
    transition_error = float(
        (reconstructed.abs().square() - representative.abs().square())
        .abs()
        .max()
        .item()
    )
    assert max(skew_error, unitary_error, transition_error) < 1e-12
    print("[cayley-unistochastic-coverage]", {
        "phase_candidate_count": diagnostics["phase_candidate_count"],
        "selected_margin_sigma_min": diagnostics["selected_margin_sigma_min"],
        "skew_error": skew_error,
        "unitary_reconstruction_error": unitary_error,
        "transition_reconstruction_error": transition_error,
    })


def fixed_point_checks() -> None:
    param = CayleyUnistochasticParam(8).double()
    H = param.unistochastic()
    alpha = 0.3
    q = 1.0 - alpha
    J = torch.randn(3, 8)
    h = torch.zeros_like(J)
    for _ in range(160):
        h = q * (h @ H.T) + alpha * J
    closed = alpha * J @ torch.linalg.inv(torch.eye(8) - q * H.T)
    fp_error = (h - closed).abs().max().item()

    h0_a = torch.randn_like(J)
    h0_b = torch.randn_like(J)
    a, b = h0_a, h0_b
    steps = 12
    for _ in range(steps):
        a = q * (a @ H.T) + alpha * J
        b = q * (b @ H.T) + alpha * J
    memory_ratio = (a - b).abs().max().item() / (h0_a - h0_b).abs().max().item()
    assert fp_error < 1e-12
    assert memory_ratio <= q**steps + 1e-12
    print("[fixed-point]", {"closed_form_max_error": fp_error,
                            "initial_state_ratio": memory_ratio,
                            "theoretical_bound": q**steps})


def bounded_trajectory_jacobian_check() -> None:
    """Verify the temporal credit upper bound and absence of a lower bound."""

    torch.manual_seed(31)
    dim = 5
    steps = 7
    leak = 0.08
    transition = CayleyUnistochasticParam(dim).double().unistochastic()
    state = torch.randn(dim, dtype=torch.double)
    product = torch.eye(dim, dtype=torch.double)
    for _ in range(steps):
        forcing = 0.15 * torch.randn(dim, dtype=torch.double)
        preactivation = (1.0 - leak) * (state @ transition.T) + forcing
        derivative = 1.0 - torch.tanh(preactivation).square()
        jacobian = torch.diag(derivative) @ ((1.0 - leak) * transition)
        product = jacobian @ product
        state = torch.tanh(preactivation)
    observed = float(torch.linalg.matrix_norm(product, ord=2).item())
    upper = (1.0 - leak) ** steps
    assert observed <= upper + 1e-12

    saturated = torch.full((dim,), 100.0, dtype=torch.double)
    derivative = 1.0 - torch.tanh(saturated).square()
    saturated_jacobian = torch.diag(derivative) @ (
        (1.0 - leak) * transition
    )
    zero_witness = float(torch.linalg.matrix_norm(saturated_jacobian, ord=2).item())
    assert zero_witness == 0.0 and upper > 0.0
    print("[bounded-trajectory-credit]", {
        "observed_jacobian_norm": observed,
        "contraction_upper_bound": upper,
        "saturated_zero_gradient_witness": zero_witness,
    })


def implicit_gradient_checks() -> None:
    torch.manual_seed(11)
    model = MoebiusQuantumRing(
        input_dim=5,
        hidden_dim=6,
        output_dim=3,
        alpha=0.3,
        relaxation_steps=180,
        lora_rank=4,
        readout_dim=6,
        h_mix_beta=0.7,
        base_unitary_init="random",
        base_unitary_scale=0.05,
        base_unitary_seed=5,
    ).double()
    with torch.no_grad():
        model.unitary_param.A_real.mul_(12.0)
        model.unitary_param.A_imag.mul_(12.0)

    x = torch.randn(4, 5)
    target = torch.tensor([0, 1, 2, 1])
    logits, state = model(x, return_state=True)
    F.cross_entropy(logits, target).backward()
    true_r = model.unitary_param.A_real.grad.detach()
    true_i = model.unitary_param.A_imag.grad.detach()

    with torch.no_grad():
        grad_y = logits.detach().softmax(1)
        grad_y[torch.arange(target.numel()), target] -= 1.0
        grad_y /= target.numel()
        grad_h = grad_y @ model.readout.readout.weight
        p = model.compute_adjoint_state_from_grad_h(state.h.detach(), grad_h, steps=260)
        grad_H_eff = model.approx_grad_H(state.h.detach(), p, normalize=False)
        beta = model._h_mix_beta_value(device=x.device, dtype=x.dtype)
        U = model._unitary_total()
        grad_U_total = 2.0 * (beta * grad_H_eff).to(U.dtype) * U
        grad_U_policy = grad_U_total @ model.U_base.to(U.dtype).conj().transpose(-2, -1)
        grad_A = model.unitary_param.cayley_pullback(grad_U_policy)

    r_err = relative_error(grad_A.real, true_r)
    i_err = relative_error(grad_A.imag, true_i)
    assert max(r_err, i_err) < 1e-10
    print("[implicit-gradient]", {"A_real_relative_error": r_err,
                                  "A_imag_relative_error": i_err,
                                  "cos_real": cosine(grad_A.real, true_r),
                                  "cos_imag": cosine(grad_A.imag, true_i)})


def legacy_direction_counterexample() -> None:
    """Measure the old HTML update against the true Cayley-coordinate gradient."""
    cosines = []
    for seed in range(30):
        torch.manual_seed(seed)
        model = MoebiusQuantumRing(
            input_dim=7,
            hidden_dim=8,
            output_dim=4,
            alpha=0.3,
            relaxation_steps=120,
            lora_rank=5,
            readout_dim=8,
        ).double()
        with torch.no_grad():
            model.unitary_param.A_real.mul_(15.0)
            model.unitary_param.A_imag.mul_(15.0)
        x = torch.randn(6, 7)
        target = torch.randint(0, 4, (6,))
        logits, state = model(x, return_state=True)
        F.cross_entropy(logits, target).backward()
        true_grad = torch.cat([
            model.unitary_param.A_real.grad.flatten(),
            model.unitary_param.A_imag.grad.flatten(),
        ])

        with torch.no_grad():
            grad_y = logits.detach().softmax(1)
            grad_y[torch.arange(target.numel()), target] -= 1.0
            grad_y /= target.numel()
            grad_h = grad_y @ model.readout.readout.weight
            p = model.compute_adjoint_state_from_grad_h(state.h.detach(), grad_h, steps=220)

            # Historical rule reconstructed exactly: it omitted the Cayley
            # differential and used grad_H * |U|^2 instead of 2 grad_H * U.
            old_grad_H = (p.T @ state.h.detach()) / target.numel()
            U = model.unitary_param.unitary()
            old_inner = (old_grad_H * U.abs().square()).to(U.dtype)
            old_M = U.conj().T @ old_inner
            old_A = 0.5 * (old_M - old_M.conj().T)
            old_coordinate = torch.cat([old_A.real.flatten(), old_A.imag.flatten()])
            cosines.append(cosine(old_coordinate, true_grad))

    negative = sum(value < 0 for value in cosines)
    assert negative >= 25
    print("[legacy-counterexample]", {"mean_cosine_with_true_gradient": sum(cosines) / len(cosines),
                                      "negative_cases": f"{negative}/{len(cosines)}"})


def linear_lora_equivalence() -> None:
    torch.manual_seed(31)
    d, n, rank, out = 13, 9, 4, 11
    alpha = 0.25
    q = 1.0 - alpha
    H = CayleyUnistochasticParam(n).double().unistochastic()
    W_down = torch.randn(rank, d)
    W_up = torch.randn(n, rank)
    W_out = torch.randn(out, n)
    x = torch.randn(d, 7)
    resolvent = alpha * torch.linalg.inv(torch.eye(n) - q * H)
    ring_output = W_out @ resolvent @ W_up @ W_down @ x
    merged_weight = W_out @ resolvent @ W_up @ W_down
    merged_output = merged_weight @ x
    effective_rank = int(torch.linalg.matrix_rank(merged_weight).item())
    error = (ring_output - merged_output).abs().max().item()
    assert error < 1e-12 and effective_rank <= rank
    print("[linear-ring=LoRA]", {"max_error": error,
                                 "effective_rank": effective_rank,
                                 "rank_bound": rank})


def orthogonal_gradient_projection_check() -> None:
    # Verify the implemented heterogeneous-step theorem, not only scalar-lr OGD.
    memory = OrthogonalGradientMemory(max_rank=2, tolerance=1e-12).double()
    old = [
        ("fast", torch.tensor([1.0, 2.0]), 0.04),
        ("slow", torch.tensor([-1.0]), 0.01),
    ]
    new = [
        ("fast", torch.tensor([3.0, -1.0]), 0.04),
        ("slow", torch.tensor([2.0]), 0.01),
    ]
    assert memory.observe(old)
    projected, stats = memory.project_preconditioned(new)
    delta_fast = -0.04 * projected["fast"]
    delta_slow = -0.01 * projected["slow"]
    old_first_order = abs(float(
        torch.dot(old[0][1], delta_fast) + torch.dot(old[1][1], delta_slow)
    ))
    new_change = float(
        torch.dot(new[0][1], delta_fast) + torch.dot(new[1][1], delta_slow)
    )
    new_identity_error = abs(new_change + stats["projected_norm"] ** 2)
    assert max(old_first_order, new_identity_error) < 1e-12
    print("[continual-OGD]", {
        "old_loss_first_order_term": old_first_order,
        "new_descent_identity_error": new_identity_error,
        "retained_norm": stats["retained_norm"],
    })


def blockwise_trust_region_check() -> None:
    """Independent block projections and positive trust scaling stay orthogonal."""

    memory_ring = OrthogonalGradientMemory(max_rank=1, tolerance=1e-12).double()
    memory_lora = OrthogonalGradientMemory(max_rank=1, tolerance=1e-12).double()
    old_ring = [("ring", torch.tensor([1.0, -2.0, 0.5]), 0.04)]
    old_lora = [("lora", torch.tensor([-1.0, 0.25, 2.0]), 0.01)]
    new_ring = [("ring", torch.tensor([2.0, 1.0, -1.0]), 0.04)]
    new_lora = [("lora", torch.tensor([0.5, 3.0, 1.0]), 0.01)]
    assert memory_ring.observe(old_ring)
    assert memory_lora.observe(old_lora)
    projected_ring, ring_stats = memory_ring.project_preconditioned(new_ring)
    projected_lora, lora_stats = memory_lora.project_preconditioned(new_lora)

    # The two blocks may be clipped by different positive trust-region scales.
    ring_scale, lora_scale = 0.37, 0.81
    delta_ring = -ring_scale * 0.04 * projected_ring["ring"]
    delta_lora = -lora_scale * 0.01 * projected_lora["lora"]
    old_first_order = abs(float(
        torch.dot(old_ring[0][1], delta_ring)
        + torch.dot(old_lora[0][1], delta_lora)
    ))
    new_change = float(
        torch.dot(new_ring[0][1], delta_ring)
        + torch.dot(new_lora[0][1], delta_lora)
    )
    expected = -(
        ring_scale * ring_stats["projected_norm"] ** 2
        + lora_scale * lora_stats["projected_norm"] ** 2
    )
    assert old_first_order < 1e-12
    assert abs(new_change - expected) < 1e-12
    print("[blockwise-OGD+trust]", {
        "old_loss_first_order_term": old_first_order,
        "new_descent_identity_error": abs(new_change - expected),
    })


def temporal_gate_perturbation_check() -> None:
    """Numerically witness the gated input-to-state perturbation theorem."""

    torch.manual_seed(73)
    leak = 0.17
    write = 0.8
    core = MultiTimescaleMQR(
        4,
        6,
        3,
        leak_rates=(leak,),
        write_scales=(write,),
        injection_rank=3,
        state_activation="tanh",
        transition_mode="unistochastic",
    ).double()
    left = TemporalMQRState((0.2 * torch.randn(2, 6),))
    right = TemporalMQRState((0.2 * torch.randn(2, 6),))
    bound = float((left.rings[0] - right.rings[0]).abs().max().item())
    actual = bound
    for _ in range(9):
        x_left = torch.randn(2, 4)
        x_right = torch.randn(2, 4)
        gate_left = torch.rand(2)
        gate_right = torch.rand(2)
        with torch.no_grad():
            inject_left = core.input_up[0](core._inject(x_left))
            inject_right = core.input_up[0](core._inject(x_right))
            forcing_error = (
                gate_left[:, None] * inject_left
                - gate_right[:, None] * inject_right
            ).abs().max().item()
        bound = (1.0 - leak) * bound + write * forcing_error
        _logits, left = core.forward_step(
            x_left,
            state=left,
            write_gate=gate_left,
        )
        _logits, right = core.forward_step(
            x_right,
            state=right,
            write_gate=gate_right,
        )
        actual = float((left.rings[0] - right.rings[0]).abs().max().item())
        assert actual <= bound + 1e-12
    print("[temporal-gate-bound]", {
        "actual_final_difference": actual,
        "recursive_upper_bound": bound,
        "slack": bound - actual,
    })


def promotion_contraction_check() -> None:
    """External promotion must cancel from paired state differences."""

    torch.manual_seed(77)
    leak = 0.23
    core = MultiTimescaleMQR(
        3,
        5,
        2,
        leak_rates=(leak,),
        injection_rank=3,
        state_activation="tanh",
        transition_mode="unistochastic",
    ).double()
    x = torch.randn(2, 3)
    promotion = torch.randn(2, 1, 5)
    left = TemporalMQRState((0.2 * torch.randn(2, 5),))
    right = TemporalMQRState((0.2 * torch.randn(2, 5),))
    before = float((left.rings[0] - right.rings[0]).abs().max().item())
    _logits, left_next = core.forward_step(x, state=left, promotion=promotion)
    _logits, right_next = core.forward_step(x, state=right, promotion=promotion)
    after = float(
        (left_next.rings[0] - right_next.rings[0]).abs().max().item()
    )
    bound = (1.0 - leak) * before
    assert after <= bound + 1e-12
    print("[promotion-contraction]", {
        "actual_difference": after,
        "theoretical_bound": bound,
    })


def utility_conditional_mean_check() -> None:
    """Squared utility regression is minimized by conditional advantage means."""

    # Two equiprobable outcomes for each causal feature group.
    advantages = torch.tensor([[-3.0, -1.0], [1.0, 3.0]])
    conditional_means = advantages.mean(dim=1)
    prediction = conditional_means.clone().requires_grad_(True)
    risk = ((prediction[:, None] - advantages) ** 2).mean()
    gradient = torch.autograd.grad(risk, prediction)[0]
    assert float(gradient.abs().max().item()) < 1e-12

    perturbed = conditional_means + torch.tensor([0.7, -0.4])
    perturbed_risk = float(((perturbed[:, None] - advantages) ** 2).mean().item())
    optimal_risk = float(risk.item())
    assert perturbed_risk > optimal_risk
    decisions = conditional_means > 0.0
    assert decisions.tolist() == [False, True]
    print("[utility-conditional-mean]", {
        "conditional_means": conditional_means.tolist(),
        "gradient_max_error": float(gradient.abs().max().item()),
        "optimal_risk": optimal_risk,
        "perturbed_risk": perturbed_risk,
    })


def delayed_readout_ticket_check() -> None:
    """Verify exact reconstruction of the issue-version CE readout gradient."""

    torch.manual_seed(79)
    learner = OnlineTemporalMQRClassifier(
        4,
        5,
        3,
        core_kwargs={
            "leak_rates": (0.3, 0.05),
            "transition_mode": "identity",
        },
        lr=0.025,
        transition_lr_ratio=0.0,
        injection_lr_ratio=0.0,
        readout_lr_ratio=1.0,
        carry_state=False,
        max_update_norm=None,
    ).double()
    issued = learner.infer_step(torch.randn(3, 4))
    target = torch.tensor([0, 2, 1])
    features = torch.cat(issued["state"].rings, dim=1)
    probabilities = issued["logits"].softmax(dim=1)
    desired = F.one_hot(target, num_classes=3).to(probabilities.dtype)
    exact_gradient = ((probabilities - desired) / target.numel()).T @ features
    weight_before = learner.core.readout.weight.detach().clone()
    expected = weight_before - learner.lr * exact_gradient
    result = learner.apply_feedback(issued["ticket_id"], target)
    error = float((learner.core.readout.weight - expected).abs().max().item())
    assert error < 1e-12
    assert result["parameter_staleness"] == 0
    print("[delayed-readout-ticket]", {
        "weight_update_max_error": error,
        "parameter_staleness": result["parameter_staleness"],
    })


def learned_gate_causality_and_derivative_check() -> None:
    """Witness the weighted-BCE derivative and current-label noninterference."""

    torch.manual_seed(83)
    raw = torch.randn(2, 3, requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    temperature = 0.7
    positive_weight = 2.5
    loss = F.binary_cross_entropy_with_logits(
        raw / temperature,
        target,
        pos_weight=raw.new_full((3,), positive_weight),
        reduction="mean",
    )
    actual_gradient = torch.autograd.grad(loss, raw)[0]
    probability = torch.sigmoid(raw.detach() / temperature)
    expected_gradient = (
        positive_weight * target * (probability - 1.0)
        + (1.0 - target) * probability
    ) / (raw.numel() * temperature)
    derivative_error = float(
        (actual_gradient - expected_gradient).abs().max().item()
    )
    assert derivative_error < 1e-12

    base = OnlineTemporalMQRClassifier(
        4,
        5,
        3,
        core_kwargs={"leak_rates": (0.3, 0.05), "transition_mode": "identity"},
        lr=0.01,
        transition_lr_ratio=0.0,
        injection_lr_ratio=0.0,
        readout_lr_ratio=1.0,
        gate_input_dim=1,
        gate_kwargs={"rank": 2, "initial_open_probability": 0.5},
        gate_lr=0.02,
        carry_state=True,
        max_update_norm=None,
    ).double()
    closed = copy.deepcopy(base)
    opened = copy.deepcopy(base)
    x = torch.randn(2, 4)
    event = torch.tensor([[1.0], [0.0]])
    task_target = torch.tensor([0, 2])
    left = closed.online_step(
        x,
        task_target,
        gate_input=event,
        gate_target=torch.zeros(2),
    )
    right = opened.online_step(
        x,
        task_target,
        gate_input=event,
        gate_target=torch.ones(2),
    )
    gate_difference = float(
        (left["effective_write_gate"] - right["effective_write_gate"])
        .abs()
        .max()
        .item()
    )
    logit_difference = float(
        (left["logits"] - right["logits"]).abs().max().item()
    )
    state_difference = max(
        float((a - b).abs().max().item())
        for a, b in zip(left["state"].rings, right["state"].rings)
    )
    future_parameter_difference = max(
        float((a - b).abs().max().item())
        for a, b in zip(
            closed.write_gate_controller.parameters(),
            opened.write_gate_controller.parameters(),
        )
    )
    assert max(gate_difference, logit_difference, state_difference) == 0.0
    assert future_parameter_difference > 0.0
    assert closed.online_parameter_version.item() == 1
    assert opened.online_parameter_version.item() == 1
    print("[learned-gate-causality]", {
        "weighted_bce_derivative_max_error": derivative_error,
        "current_gate_max_difference": gate_difference,
        "current_state_max_difference": state_difference,
        "current_logit_max_difference": logit_difference,
        "future_gate_parameter_difference": future_parameter_difference,
    })


def main() -> None:
    structural_checks()
    cayley_unistochastic_coverage_check()
    fixed_point_checks()
    bounded_trajectory_jacobian_check()
    implicit_gradient_checks()
    legacy_direction_counterexample()
    linear_lora_equivalence()
    orthogonal_gradient_projection_check()
    blockwise_trust_region_check()
    temporal_gate_perturbation_check()
    promotion_contraction_check()
    utility_conditional_mean_check()
    learned_gate_causality_and_derivative_check()
    delayed_readout_ticket_check()
    print("all rigorous numerical checks passed")


if __name__ == "__main__":
    main()
