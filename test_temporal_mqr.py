#!/usr/bin/env python3
"""Strict tests for one-step multi-timescale MQR dynamics and online updates."""

from __future__ import annotations

import copy
import io
import logging
import math

import torch
import torch.nn.functional as F

from mqr import (
    CayleyUnistochasticParam,
    CyclicGivensUnistochasticParam,
    MultiTimescaleMQR,
    OnlineTemporalMQRClassifier,
    TemporalMQRSidecar,
    TemporalMQRState,
    TemporalWriteGate,
)
from experiments.temporal_mqr_delayed_digits import make_sequence
from experiments.temporal_mqr_interference_digits import make_interference_sequence


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _make_online(
    *,
    ogd_rank: int = 0,
    carry_state: bool = True,
    transition_mode: str = "unistochastic",
    max_update_norm: float | None = 0.05,
) -> OnlineTemporalMQRClassifier:
    return OnlineTemporalMQRClassifier(
        4,
        5,
        3,
        core_kwargs={
            "leak_rates": (0.5, 0.1, 0.02),
            "injection_rank": 3,
            "injection_activation": "tanh",
            "state_activation": "tanh",
            "transition_mode": transition_mode,
        },
        lr=0.02,
        transition_lr_ratio=0.1,
        injection_lr_ratio=1.0,
        readout_lr_ratio=1.0,
        ogd_max_rank=ogd_rank,
        carry_state=carry_state,
        max_update_norm=max_update_norm,
    )


def _make_learned_gate_online(
    *,
    gate_lr: float = 0.02,
    gate_ogd_rank: int = 0,
    gate_max_update_norm: float | None = None,
    carry_state: bool = True,
) -> OnlineTemporalMQRClassifier:
    return OnlineTemporalMQRClassifier(
        4,
        5,
        3,
        core_kwargs={
            "leak_rates": (0.4, 0.04),
            "injection_rank": 3,
            "transition_mode": "identity",
        },
        lr=0.02,
        transition_lr_ratio=0.0,
        injection_lr_ratio=0.0,
        readout_lr_ratio=0.0,
        gate_input_dim=4,
        gate_kwargs={
            "rank": 3,
            "activation": "tanh",
            "initial_open_probability": 0.5,
        },
        gate_lr=gate_lr,
        gate_ogd_max_rank=gate_ogd_rank,
        gate_max_update_norm=gate_max_update_norm,
        carry_state=carry_state,
        max_update_norm=None,
    )


def _parameter_snapshot(module: torch.nn.Module):
    return {name: value.detach().clone() for name, value in module.named_parameters()}


def _assert_raises(expected, operation, message: str | None = None) -> Exception:
    try:
        operation()
    except expected as exc:
        if message is not None:
            assert message in str(exc)
        return exc
    raise AssertionError(f"expected {expected.__name__}")


def test_exact_multiscale_decay_and_half_life() -> None:
    """Identity transitions must realize the declared fading-memory kernel exactly."""

    core = MultiTimescaleMQR(
        2,
        3,
        2,
        leak_rates=(0.5, 0.1, 0.02, 0.005),
        injection_rank=2,
        injection_activation="none",
        state_activation="none",
        transition_mode="identity",
    ).double()
    state = TemporalMQRState(
        tuple(torch.ones(2, 3, dtype=torch.double) for _ in range(4))
    )
    zero = torch.zeros(2, 2, dtype=torch.double)
    steps = 32
    for _ in range(steps):
        _logits, state = core.forward_step(zero, state=state)

    for index, leak in enumerate((0.5, 0.1, 0.02, 0.005)):
        expected = torch.full_like(state.rings[index], (1.0 - leak) ** steps)
        torch.testing.assert_close(state.rings[index], expected, rtol=1e-12, atol=1e-14)
    assert state.rings[3].abs().mean() > state.rings[2].abs().mean()
    assert state.rings[2].abs().mean() > state.rings[1].abs().mean()
    assert state.rings[1].abs().mean() > state.rings[0].abs().mean()

    half_lives = core.memory_half_lives()
    expected_half_lives = torch.tensor(
        [math.log(0.5) / math.log(1.0 - leak) for leak in (0.5, 0.1, 0.02, 0.005)],
        dtype=torch.double,
    )
    torch.testing.assert_close(half_lives, expected_half_lives, rtol=1e-12, atol=1e-12)


def test_unistochastic_constraints_and_contraction() -> None:
    """Cayley transitions must remain doubly stochastic and contract state differences."""

    torch.manual_seed(101)
    core = MultiTimescaleMQR(
        3,
        6,
        2,
        leak_rates=(0.25,),
        injection_rank=2,
        state_activation="tanh",
        transition_mode="unistochastic",
    ).double()
    x = torch.randn(4, 3, dtype=torch.double)
    left = TemporalMQRState((0.2 * torch.randn(4, 6, dtype=torch.double),))
    right = TemporalMQRState((0.2 * torch.randn(4, 6, dtype=torch.double),))
    _left_logits, left_next = core.forward_step(x, state=left)
    _right_logits, right_next = core.forward_step(x, state=right)

    before = (left.rings[0] - right.rings[0]).abs().max()
    after = (left_next.rings[0] - right_next.rings[0]).abs().max()
    assert float(after.item()) <= 0.75 * float(before.item()) + 1e-12
    assert core.max_unitary_error() < 1e-12
    assert core.max_stochastic_error() < 1e-12


def test_nonflat_temporal_cayley_base_and_inactive_full_leak_ring() -> None:
    """A frozen generic base must open the H Jacobian and mask a dead fast ring."""

    torch.manual_seed(102)
    legacy = MultiTimescaleMQR(
        3,
        5,
        2,
        leak_rates=(1.0, 0.2),
        injection_rank=2,
        cayley_coordinate_mode="minimal",
    ).double()
    # Default behavior remains the historical identity-base construction and
    # does not add a key that would break strict loading of old checkpoints.
    assert legacy.base_unitary_init == "identity"
    assert "_base_unitaries" not in legacy.state_dict()

    core = MultiTimescaleMQR(
        3,
        5,
        2,
        leak_rates=(1.0, 0.2),
        injection_rank=2,
        cayley_coordinate_mode="minimal",
        base_unitary_init="random",
        base_unitary_scale=0.25,
        base_unitary_seed=17,
    ).double()
    assert core.active_transition_indices == (1,)
    assert sum(parameter.numel() for parameter in core.unitary_params[0].parameters()) == 0
    assert core.unitary_params[1].A_real.requires_grad
    torch.testing.assert_close(
        core.unitary_params[1].skew_hermitian_A(),
        torch.zeros(5, 5, dtype=torch.complex128),
        rtol=0.0,
        atol=0.0,
    )
    diagnostics = core.transition_diagnostics(include_jacobian=True)
    assert diagnostics[0]["active"] is False
    assert diagnostics[0]["carry"] == 0.0
    assert diagnostics[1]["active"] is True
    assert diagnostics[1]["modulus_square_jacobian_rank"] == 16
    assert diagnostics[1]["transition_distance_from_identity_fro"] > 0.1
    assert diagnostics[1]["modulus_square_jacobian_largest_singular_value"] > 0.1
    assert core.max_unitary_error() < 1e-12
    assert core.max_stochastic_error() < 1e-12

    snapshot = core._base_unitaries.detach().clone()
    x = torch.randn(1, 3, dtype=torch.double)
    _logits, state = core.forward_step(x)
    _logits, state = core.forward_step(torch.randn_like(x), state=state)
    loss = state.rings[1].square().sum()
    trainable = tuple(
        (name, parameter)
        for name, parameter in core.named_parameters()
        if parameter.requires_grad
    )
    gradients = torch.autograd.grad(
        loss,
        tuple(parameter for _name, parameter in trainable),
        allow_unused=True,
    )
    assert any(value is not None for value in gradients)
    transition_gradients = [
        gradient
        for (name, _parameter), gradient in zip(trainable, gradients)
        if name.startswith("unitary_params.1")
    ]
    assert transition_gradients
    assert any(
        gradient is not None and float(gradient.norm().item()) > 0.0
        for gradient in transition_gradients
    )
    torch.testing.assert_close(core._base_unitaries, snapshot, rtol=0.0, atol=0.0)

    buffer = io.BytesIO()
    torch.save(core.state_dict(), buffer)
    buffer.seek(0)
    restored = MultiTimescaleMQR(
        3,
        5,
        2,
        leak_rates=(1.0, 0.2),
        injection_rank=2,
        cayley_coordinate_mode="minimal",
        base_unitary_init="random",
        base_unitary_scale=0.25,
        base_unitary_seed=999,
    ).double()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(
        restored._base_unitaries, core._base_unitaries, rtol=0.0, atol=0.0
    )
    for index in range(core.num_timescales):
        torch.testing.assert_close(
            restored.transition_matrix(index),
            core.transition_matrix(index),
            rtol=0.0,
            atol=0.0,
        )


def test_cyclic_givens_is_a_low_cost_literal_ring_with_nonzero_jacobian() -> None:
    """Alternating local rotations must retain exact structure and learnability."""

    parameter = CyclicGivensUnistochasticParam(
        6,
        layers=2,
        base_init="random",
        base_scale=0.25,
        base_seed=29,
    ).double()
    diagnostic = parameter.coordinate_diagnostics(include_jacobian=True)
    assert diagnostic["literal_cycle_adjacency"] is True
    assert diagnostic["effective_dof"] == 6
    assert diagnostic["modulus_square_jacobian_rank"] == 6
    assert diagnostic["modulus_square_jacobian_largest_singular_value"] > 0.0
    assert parameter.raw_parameter_count < CayleyUnistochasticParam(
        6, coordinate_mode="minimal"
    ).raw_parameter_count
    assert float(parameter.unitary_error_fro().item()) < 1e-12
    row_error, column_error = parameter.doubly_stochastic_errors()
    assert float(max(row_error, column_error).item()) < 1e-12
    reference = parameter.angles.detach().clone()
    with torch.no_grad():
        parameter.angles.add_(0.02 * torch.randn_like(parameter.angles))
    drift = parameter.drift_diagnostics(reference)
    assert drift["bounds_certified"] is True
    assert drift["transition_fro_drift"] > 0.0

    core = MultiTimescaleMQR(
        4,
        6,
        3,
        leak_rates=(1.0, 0.1),
        injection_rank=3,
        transition_structure="cyclic_givens",
        cyclic_givens_layers=2,
        base_unitary_init="random",
        base_unitary_scale=0.25,
        base_unitary_seed=31,
    ).double()
    assert core.active_transition_indices == (1,)
    assert sum(parameter.numel() for parameter in core.unitary_params[0].parameters()) == 0
    first = torch.randn(1, 4, dtype=torch.double)
    _logits, state = core.forward_step(first)
    _logits, state = core.forward_step(torch.randn_like(first), state=state)
    gradient = torch.autograd.grad(
        state.rings[1].square().sum(),
        core.unitary_params[1].angles,
    )[0]
    assert float(gradient.norm().item()) > 0.0
    assert core.max_unitary_error() < 1e-12
    assert core.max_stochastic_error() < 1e-12
    with torch.no_grad():
        cached_before = core.transition_matrix(1).clone()
        cached_record = core._inference_transition_cache[1]
        core.transition_matrix(1)
        assert core._inference_transition_cache[1][1].data_ptr() == cached_record[1].data_ptr()
        core.unitary_params[1].angles.add_(0.01)
        cached_after = core.transition_matrix(1)
    assert not torch.equal(cached_before, cached_after)
    assert core._inference_transition_cache[1][0] != cached_record[0]


def test_signed_state_certificate_and_cayley_finite_drift_bound() -> None:
    """Signed temporal states and finite Cayley changes must satisfy their bounds."""

    torch.manual_seed(111)
    for transition_mode in ("identity", "unistochastic"):
        core = MultiTimescaleMQR(
            4,
            5,
            3,
            leak_rates=(0.4, 0.08),
            injection_rank=3,
            injection_activation="tanh",
            state_activation="tanh",
            transition_mode=transition_mode,
        ).double()
        x = torch.randn(2, 4, dtype=torch.double)
        state = TemporalMQRState(
            tuple(torch.randn(2, 5, dtype=torch.double) for _ in range(2))
        )
        promotion = 0.2 * torch.randn(2, 2, 5, dtype=torch.double)
        _logits, next_state, certificate = core.forward_step(
            x,
            state=state,
            write_gate=torch.tensor([[1.0, 0.0], [0.3, 1.0]], dtype=torch.double),
            promotion=promotion,
            return_certificate=True,
        )
        assert certificate["certified"] is True
        assert certificate["requires_nonnegative_state"] is False
        assert certificate["max_bound_violation"] <= certificate["numerical_tolerance"]
        observed = torch.stack(
            [value.abs().amax(dim=1) for value in next_state.rings], dim=1
        )
        torch.testing.assert_close(
            observed,
            certificate["observed_next_state_linf"],
            rtol=0.0,
            atol=0.0,
        )

    parameter = CayleyUnistochasticParam(5, coordinate_mode="minimal").double()
    reference = parameter.skew_hermitian_A().detach().clone()
    with torch.no_grad():
        parameter.A_real.add_(0.03 * torch.randn_like(parameter.A_real))
        parameter.A_imag.add_(0.03 * torch.randn_like(parameter.A_imag))
    drift = parameter.drift_diagnostics(reference)
    assert drift["a_fro_drift"] > 0.0
    assert drift["bounds_certified"] is True
    assert drift["unitary_fro_drift"] <= drift["unitary_fro_bound"] + 1e-12
    assert drift["transition_fro_drift"] <= drift["transition_fro_bound"] + 1e-12


def test_temporal_residual_sidecar_is_initially_exact_noop_and_bounded() -> None:
    """The frozen hidden path starts unchanged and residual clipping is auditable."""

    torch.manual_seed(112)
    sidecar = TemporalMQRSidecar(
        4,
        ring_dim=3,
        residual_clip_l2=0.05,
        core_kwargs={
            "leak_rates": (1.0, 0.1),
            "injection_rank": 3,
            "transition_mode": "unistochastic",
            "cayley_coordinate_mode": "minimal",
        },
    ).double()
    hidden = torch.randn(2, 4, dtype=torch.double)
    adapted, state, diagnostics = sidecar.forward_step(
        hidden,
        return_diagnostics=True,
    )
    torch.testing.assert_close(adapted, hidden, rtol=0.0, atol=0.0)
    assert diagnostics["residual_l2_max"] == 0.0
    assert diagnostics["state_certificate"]["certified"] is True

    loss = adapted.square().mean()
    loss.backward()
    assert sidecar.core.readout.weight.grad is not None
    assert float(sidecar.core.readout.weight.grad.norm().item()) > 0.0
    with torch.no_grad():
        sidecar.core.readout.weight.fill_(10.0)
    clipped, _next_state, clipped_diagnostics = sidecar.forward_step(
        hidden,
        state=state.detached(),
        return_diagnostics=True,
    )
    assert clipped_diagnostics["raw_residual_l2_max"] > 0.05
    assert clipped_diagnostics["residual_l2_max"] <= 0.05 + 1e-12
    assert clipped_diagnostics["residual_clip_scale_min"] < 1.0
    assert bool(torch.isfinite(clipped).all())


def test_write_strength_is_independent_from_decay() -> None:
    """Changing write strength may change input response but not recurrent decay."""

    core = MultiTimescaleMQR(
        2,
        2,
        2,
        leak_rates=(0.1, 0.1),
        write_scales=(0.0, 1.0),
        injection_rank=2,
        injection_activation="none",
        state_activation="none",
        transition_mode="identity",
    ).double()
    with torch.no_grad():
        core.input_down.weight.copy_(torch.eye(2, dtype=torch.double))
        for projection in core.input_up:
            projection.weight.copy_(torch.eye(2, dtype=torch.double))
    x = torch.tensor([[1.0, -2.0]], dtype=torch.double)
    _logits, written = core.forward_step(x)
    torch.testing.assert_close(written.rings[0], torch.zeros_like(x), rtol=0.0, atol=0.0)
    torch.testing.assert_close(written.rings[1], x, rtol=0.0, atol=0.0)

    zero = torch.zeros_like(x)
    _logits, decayed = core.forward_step(zero, state=written)
    torch.testing.assert_close(decayed.rings[1], 0.9 * x, rtol=1e-12, atol=1e-12)

    default = MultiTimescaleMQR(2, 2, 2, leak_rates=(0.1, 0.01), injection_rank=2)
    torch.testing.assert_close(
        default.write_scales,
        torch.ones(2, dtype=torch.double),
        rtol=0.0,
        atol=0.0,
    )


def test_batch_lanes_are_independent() -> None:
    """A batched transition must equal separate transitions for every stream lane."""

    torch.manual_seed(202)
    core = MultiTimescaleMQR(
        3,
        4,
        2,
        leak_rates=(0.4, 0.04),
        injection_rank=2,
        transition_mode="unistochastic",
    )
    x = torch.randn(2, 3)
    state = TemporalMQRState((torch.randn(2, 4), torch.randn(2, 4)))
    batched_logits, batched_state = core.forward_step(x, state=state)

    separate_logits = []
    separate_states = [[], []]
    for lane in range(2):
        lane_state = TemporalMQRState(
            tuple(value[lane : lane + 1] for value in state.rings)
        )
        logits, next_state = core.forward_step(x[lane : lane + 1], state=lane_state)
        separate_logits.append(logits)
        for index, value in enumerate(next_state.rings):
            separate_states[index].append(value)

    torch.testing.assert_close(batched_logits, torch.cat(separate_logits), rtol=1e-6, atol=1e-7)
    for index in range(2):
        torch.testing.assert_close(
            batched_state.rings[index],
            torch.cat(separate_states[index]),
            rtol=1e-6,
            atol=1e-7,
        )


def test_frozen_unistochastic_transition_is_cached_exactly() -> None:
    """A frozen reservoir must reuse its saved H instead of solving Cayley per token."""

    torch.manual_seed(252)
    core = MultiTimescaleMQR(
        3,
        4,
        2,
        leak_rates=(0.4, 0.04),
        injection_rank=2,
        transition_mode="unistochastic",
        learn_transitions=False,
    )
    assert core._fixed_transitions.shape == (2, 4, 4)
    for index, parameter in enumerate(core.unitary_params):
        expected = parameter.unistochastic()
        actual = core.transition_matrix(index)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        assert not actual.requires_grad


def test_topology_ablation_has_matched_shared_initialization() -> None:
    """Identity and Cayley controls must differ only in recurrent topology."""

    kwargs = {
        "leak_rates": (0.4, 0.04),
        "write_scales": (1.0, 1.0),
        "injection_rank": 2,
        "learn_transitions": False,
    }
    torch.manual_seed(272)
    identity = MultiTimescaleMQR(3, 4, 2, transition_mode="identity", **kwargs)
    torch.manual_seed(272)
    stochastic = MultiTimescaleMQR(3, 4, 2, transition_mode="unistochastic", **kwargs)
    torch.testing.assert_close(
        identity.input_down.weight,
        stochastic.input_down.weight,
        rtol=0.0,
        atol=0.0,
    )
    for left, right in zip(identity.input_up, stochastic.input_up):
        torch.testing.assert_close(left.weight, right.weight, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        identity.readout.weight,
        stochastic.readout.weight,
        rtol=0.0,
        atol=0.0,
    )
    # At the first observation h_prev=0, so topology has no contribution.
    x = torch.randn(2, 3)
    identity_logits, _ = identity.forward_step(x)
    stochastic_logits, _ = stochastic.forward_step(x)
    torch.testing.assert_close(identity_logits, stochastic_logits, rtol=0.0, atol=0.0)


def test_write_ablation_is_parameter_matched_and_changes_only_amplitude() -> None:
    """Decoupled and EMA writes must share parameters but retain different cue amplitudes."""

    leaks = (0.5, 0.1, 0.02, 0.005)
    kwargs = {
        "leak_rates": leaks,
        "injection_rank": 3,
        "injection_activation": "tanh",
        "state_activation": "none",
        "transition_mode": "unistochastic",
        "learn_transitions": False,
    }
    torch.manual_seed(282)
    coupled = MultiTimescaleMQR(4, 3, 2, write_scales=leaks, **kwargs)
    torch.manual_seed(282)
    decoupled = MultiTimescaleMQR(4, 3, 2, write_scales=(1.0,) * 4, **kwargs)
    for (left_name, left), (right_name, right) in zip(
        coupled.named_parameters(), decoupled.named_parameters()
    ):
        assert left_name == right_name
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)

    cue = torch.randn(1, 4)
    zero = torch.zeros_like(cue)
    _logits, coupled_state = coupled.forward_step(cue)
    _logits, decoupled_state = decoupled.forward_step(cue)
    for _ in range(8):
        _logits, coupled_state = coupled.forward_step(zero, state=coupled_state)
        _logits, decoupled_state = decoupled.forward_step(zero, state=decoupled_state)
    for index, leak in enumerate(leaks):
        torch.testing.assert_close(
            decoupled_state.rings[index] * leak,
            coupled_state.rings[index],
            rtol=2e-5,
            atol=1e-7,
        )


def test_delayed_digits_query_contains_no_digit_features() -> None:
    """The held-out target cannot be recovered from the non-oracle query frame alone."""

    left = make_sequence(torch.zeros(64), 4, instant=False)
    right = make_sequence(torch.ones(64), 4, instant=False)
    torch.testing.assert_close(left[-1], right[-1], rtol=0.0, atol=0.0)
    assert int(torch.count_nonzero(left[-1, :64]).item()) == 0
    assert float(left[-1, 65].item()) == 1.0
    oracle = make_sequence(torch.ones(64), 4, instant=True)
    assert int(torch.count_nonzero(oracle[-1, :64]).item()) == 64


def test_interference_digits_query_is_identical_and_label_free() -> None:
    """Changing cue and real distractors must never alter the zero query frame."""

    left = make_interference_sequence(torch.zeros(64), torch.ones(4, 64))
    right = make_interference_sequence(torch.ones(64), torch.zeros(4, 64))
    torch.testing.assert_close(left[-1], right[-1], rtol=0.0, atol=0.0)
    assert int(torch.count_nonzero(left[-1]).item()) == 0
    assert not torch.equal(left[0], right[0])
    assert not torch.equal(left[1:-1], right[1:-1])


def test_preview_is_read_only_and_feedback_has_no_label_leakage() -> None:
    """Preview and committed pre-update logits must match for any feedback label."""

    torch.manual_seed(303)
    base = _make_online(ogd_rank=2)
    warm = torch.randn(2, 4)
    base.online_step(warm, None)
    parameters_before = _parameter_snapshot(base)
    state_before = base.current_state()
    observations_before = base.online_observations.clone()
    x = torch.randn(2, 4)
    preview = base.preview_step(x)
    preview_again = base.preview_step(x)
    torch.testing.assert_close(preview["logits"], preview_again["logits"], rtol=0.0, atol=0.0)
    assert base.online_observations.item() == observations_before.item()
    assert base.gradient_memory.rank == 0
    assert state_before is not None and base.current_state() is not None
    for old, current in zip(state_before.rings, base.current_state().rings):
        torch.testing.assert_close(old, current, rtol=0.0, atol=0.0)
    for name, parameter in base.named_parameters():
        torch.testing.assert_close(parameter, parameters_before[name], rtol=0.0, atol=0.0)

    model_a = copy.deepcopy(base)
    model_b = copy.deepcopy(base)
    result_a = model_a.online_step(x, torch.tensor([0, 0]))
    result_b = model_b.online_step(x, torch.tensor([2, 2]))
    torch.testing.assert_close(result_a["logits"], preview["logits"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(result_b["logits"], preview["logits"], rtol=0.0, atol=0.0)
    assert any(
        not torch.equal(left, right)
        for left, right in zip(model_a.parameters(), model_b.parameters())
    )


def test_state_memory_and_parameter_memory_are_separately_observable() -> None:
    """Carry-only and update-only interventions must each change behavior independently."""

    torch.manual_seed(404)
    state_model = _make_online(
        carry_state=True,
        transition_mode="identity",
    )
    cue = torch.randn(1, 4)
    blank = torch.zeros(1, 4)
    state_model.online_step(cue, None, learn=False)
    carried = state_model.preview_step(blank)["logits"]
    state_model.reset_state()
    reset = state_model.preview_step(blank)["logits"]
    assert float((carried - reset).abs().max().item()) > 1e-6
    assert state_model.online_updates.item() == 0

    weight_model = _make_online(carry_state=False)
    x = torch.randn(6, 4)
    target = torch.tensor([0, 1, 2, 0, 1, 2])
    before = weight_model.preview_step(x, carry_state=False)["logits"]
    result = weight_model.online_step(x, target, carry_state=False)
    after = weight_model.preview_step(x, carry_state=False)["logits"]
    assert result["did_update"]
    assert float((after - before).abs().max().item()) > 1e-7
    assert weight_model.current_state() is None


def test_small_online_step_decreases_current_loss_and_preserves_manifold() -> None:
    """A sufficiently small full-vector update must descend and keep U unitary."""

    torch.manual_seed(505)
    model = _make_online(carry_state=False, max_update_norm=None)
    model.lr = 1e-3
    x = torch.randn(8, 4)
    target = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0])
    before_logits = model.preview_step(x, carry_state=False)["logits"]
    before_loss = F.cross_entropy(before_logits, target)
    result = model.online_step(x, target, carry_state=False)
    after_logits = model.preview_step(x, carry_state=False)["logits"]
    after_loss = F.cross_entropy(after_logits, target)

    assert float(after_loss.item()) < float(before_loss.item())
    assert result["ogd_first_order_decrease"] < 0.0
    assert result["max_unitary_error"] < 1e-5
    assert result["max_stochastic_error"] < 1e-5


def test_temporal_update_uses_one_complete_ogd_vector() -> None:
    """Injection, transitions, and readout must be projected as one vector."""

    torch.manual_seed(606)
    model = _make_online(ogd_rank=2, carry_state=False)
    x1 = torch.randn(5, 4)
    y1 = torch.tensor([0, 1, 2, 0, 1])
    first = model.online_step(
        x1,
        y1,
        remember_gradient=True,
        project_with_memory=False,
        carry_state=False,
    )
    expected_dimension = sum(
        parameter.numel()
        for name, parameter in model.core.named_parameters()
        if parameter.requires_grad and model._step_ratio(name) > 0.0
    )
    assert first["ogd_memory_added"]
    assert model.gradient_memory.rank == 1
    assert model.gradient_memory.dimension == expected_dimension

    x2 = torch.randn(5, 4)
    y2 = torch.tensor([2, 0, 1, 2, 0])
    second = model.online_step(x2, y2, carry_state=False)
    assert second["ogd_projection_applied"]
    assert 0.0 <= second["ogd_retained_norm"] <= 1.0 + 1e-6
    assert second["ogd_max_abs_overlap"] < 1e-5
    assert second["ogd_first_order_decrease"] <= 0.0


def test_update_norm_is_atomically_clipped() -> None:
    """The global update, not individual parameter blocks, must obey the norm cap."""

    torch.manual_seed(707)
    cap = 1e-5
    model = _make_online(carry_state=False, max_update_norm=cap)
    model.lr = 10.0
    result = model.online_step(
        4.0 * torch.randn(16, 4),
        torch.tensor([0, 1, 2, 0] * 4),
        carry_state=False,
    )
    assert result["unclipped_update_norm"] > cap
    assert result["update_clip_scale"] < 1.0
    assert result["update_norm"] <= cap * (1.0 + 1e-9)


def test_checkpoint_roundtrip_preserves_state_and_ogd() -> None:
    """Fast state, slow parameters, counters, and gradient memory must all resume exactly."""

    torch.manual_seed(808)
    model = _make_online(ogd_rank=2, carry_state=True)
    x = torch.randn(2, 4)
    model.online_step(
        x,
        torch.tensor([0, 2]),
        remember_gradient=True,
        project_with_memory=False,
    )
    query = torch.randn(2, 4)
    expected = model.preview_step(query)["logits"]

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = _make_online(ogd_rank=2, carry_state=True)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    actual = restored.preview_step(query)["logits"]

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert restored.gradient_memory.rank == model.gradient_memory.rank == 1
    assert restored.gradient_memory.dimension == model.gradient_memory.dimension
    assert restored.online_observations.item() == model.online_observations.item()
    assert restored.online_updates.item() == model.online_updates.item()

    # Strict loading of the previous one-slot checkpoint format must inject
    # only empty defaults for newly introduced runtime buffers/modules.
    legacy = copy.deepcopy(model.state_dict())
    legacy_rings = legacy["_extra_state"]["streams"][0][1]
    legacy["_extra_state"] = {"version": 1, "state": legacy_rings}
    for key in list(legacy):
        if key in {
            "online_parameter_version",
            "online_gate_updates",
            "online_gate_labels",
        } or key.startswith((
            "feedback_gradient_memory.",
            "gate_gradient_memory.",
        )):
            del legacy[key]
    legacy_restored = _make_online(ogd_rank=2, carry_state=True)
    incompatible = legacy_restored.load_state_dict(legacy, strict=True)
    assert incompatible.missing_keys == [] and incompatible.unexpected_keys == []
    legacy_actual = legacy_restored.preview_step(query)["logits"]
    torch.testing.assert_close(legacy_actual, expected, rtol=0.0, atol=0.0)


def test_temporal_write_gate_initialization_range_and_validation() -> None:
    """The declared initial probability and floor must be exact and shape-safe."""

    torch.manual_seed(813)
    controller = TemporalWriteGate(
        4,
        3,
        rank=2,
        activation="gelu",
        temperature=0.7,
        minimum_gate=0.2,
        initial_open_probability=0.65,
    ).double()
    x = torch.randn(5, 4, dtype=torch.double)
    raw = controller.raw_logits(x)
    base = controller.base_probability(raw)
    gate = controller(x)
    expected_base = torch.full_like(base, (0.65 - 0.2) / (1.0 - 0.2))
    # The bias is initialized in the module's construction dtype before the
    # explicit float64 cast, so retain float32 initialization round-off.
    torch.testing.assert_close(base, expected_base, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(
        gate,
        torch.full_like(gate, 0.65),
        rtol=1e-9,
        atol=1e-9,
    )
    assert gate.shape == (5, 3)
    saturated = controller.gate_from_logits(
        torch.tensor(
            [[-1000.0, 0.0, 1000.0]],
            dtype=torch.double,
        )
    )
    torch.testing.assert_close(
        saturated[0, 0],
        torch.tensor(0.2, dtype=torch.double),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        saturated[0, 2],
        torch.tensor(1.0, dtype=torch.double),
        rtol=0.0,
        atol=0.0,
    )
    _assert_raises(
        ValueError,
        lambda: controller(torch.randn(5, 5, dtype=torch.double)),
        "gate_input must be",
    )
    _assert_raises(
        ValueError,
        lambda: controller.base_probability(torch.randn(5, 2, dtype=torch.double)),
        "raw_logits must have shape",
    )


def test_gate_label_is_causal_and_cannot_change_current_prediction() -> None:
    """Different event labels may change theta_(t+1), never gate/state/logits at t."""

    torch.manual_seed(823)
    base = _make_learned_gate_online(carry_state=True)
    base.readout_lr_ratio = 1.0
    closed_label = copy.deepcopy(base)
    open_label = copy.deepcopy(base)
    x = torch.randn(4, 4)
    task_target = torch.tensor([0, 1, 2, 0])
    closed = closed_label.online_step(
        x,
        task_target,
        gate_target=torch.zeros(4),
    )
    opened = open_label.online_step(
        x,
        task_target,
        gate_target=torch.ones(4),
    )

    torch.testing.assert_close(closed["logits"], opened["logits"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        closed["effective_write_gate"],
        opened["effective_write_gate"],
        rtol=0.0,
        atol=0.0,
    )
    for left, right in zip(closed["state"].rings, opened["state"].rings):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    assert closed["write_gate_source"] == opened["write_gate_source"] == "learned"
    assert closed["prediction_before_update"] and opened["prediction_before_update"]
    assert closed["gate_supervision_after_prediction"]
    assert opened["gate_supervision_after_prediction"]
    assert closed["gate_task_gradient_detached"]
    assert closed["gate_did_update"] and opened["gate_did_update"]
    assert closed["core_did_update"] and opened["core_did_update"]
    assert closed_label.online_parameter_version.item() == 1
    assert open_label.online_parameter_version.item() == 1
    assert closed_label.online_updates.item() == 1
    assert open_label.online_updates.item() == 1
    assert any(
        not torch.equal(left, right)
        for left, right in zip(
            closed_label.write_gate_controller.parameters(),
            open_label.write_gate_controller.parameters(),
        )
    )
    for left, right in zip(closed_label.core.parameters(), open_label.core.parameters()):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)


def test_small_auxiliary_gate_step_decreases_gate_bce() -> None:
    """A sufficiently small gate-only update must descend its current BCE."""

    torch.manual_seed(833)
    model = _make_learned_gate_online(
        gate_lr=1e-3,
        carry_state=False,
    )
    controller = model.write_gate_controller
    assert controller is not None
    x = torch.randn(8, 4)
    target = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    expanded = target.unsqueeze(1).expand(-1, model.core.num_timescales)
    before = F.binary_cross_entropy_with_logits(
        controller.raw_logits(x) / controller.temperature,
        expanded,
    )
    result = model.online_step(
        x,
        None,
        learn=False,
        carry_state=False,
        gate_target=target,
    )
    after = F.binary_cross_entropy_with_logits(
        controller.raw_logits(x) / controller.temperature,
        expanded,
    )
    assert float(after.item()) < float(before.item())
    assert result["gate_ogd_first_order_decrease"] < 0.0
    assert result["gate_update_norm"] > 0.0
    assert result["gate_label_count"] == x.size(0) * model.core.num_timescales
    assert model.online_gate_updates.item() == 1
    assert model.online_gate_labels.item() == result["gate_label_count"]


def test_gate_ogd_uses_complete_vector_and_has_independent_clip() -> None:
    """All controller blocks share one OGD vector and one gate-specific norm cap."""

    torch.manual_seed(843)
    cap = 1e-5
    model = _make_learned_gate_online(
        gate_lr=10.0,
        gate_ogd_rank=2,
        gate_max_update_norm=cap,
        carry_state=False,
    )
    first = model.online_step(
        torch.randn(6, 4),
        None,
        learn=False,
        carry_state=False,
        gate_target=torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
        remember_gate_gradient=True,
        project_gate_with_memory=False,
    )
    controller = model.write_gate_controller
    assert controller is not None
    expected_dimension = sum(
        parameter.numel()
        for parameter in controller.parameters()
        if parameter.requires_grad
    )
    assert first["gate_ogd_memory_added"]
    assert model.gate_gradient_memory.dimension == expected_dimension
    assert model.gate_gradient_memory.rank == 1

    second = model.online_step(
        3.0 * torch.randn(6, 4),
        None,
        learn=False,
        carry_state=False,
        gate_target=torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
    )
    assert second["gate_ogd_projection_applied"]
    assert second["gate_ogd_max_abs_overlap"] < 1e-5
    assert second["gate_unclipped_update_norm"] > cap
    assert second["gate_update_clip_scale"] < 1.0
    assert second["gate_update_norm"] <= cap * (1.0 + 1e-9)
    assert second["update_norm"] == 0.0


def test_learned_gate_checkpoint_and_external_override_are_exact() -> None:
    """Gate weights/OGD/counters restore exactly; an external gate controls state."""

    torch.manual_seed(853)
    model = _make_learned_gate_online(
        gate_lr=0.03,
        gate_ogd_rank=2,
        carry_state=True,
    )
    x = torch.randn(2, 4)
    result = model.online_step(
        x,
        None,
        learn=False,
        gate_target=torch.tensor([1.0, 0.0]),
        remember_gate_gradient=True,
        project_gate_with_memory=False,
    )
    assert result["write_gate_source"] == "learned"
    query = torch.randn(2, 4)
    expected = model.preview_step(query)

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = _make_learned_gate_online(
        gate_lr=0.03,
        gate_ogd_rank=2,
        carry_state=True,
    )
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    actual = restored.preview_step(query)
    torch.testing.assert_close(actual["logits"], expected["logits"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual["learned_write_gate"],
        expected["learned_write_gate"],
        rtol=0.0,
        atol=0.0,
    )
    for left, right in zip(actual["state"].rings, expected["state"].rings):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    assert restored.gate_gradient_memory.rank == model.gate_gradient_memory.rank == 1
    assert restored.online_gate_updates.item() == model.online_gate_updates.item() == 1
    assert restored.online_gate_labels.item() == model.online_gate_labels.item() == 4
    assert restored.online_parameter_version.item() == model.online_parameter_version.item()

    restored.reset_all_states()
    external_zero = restored.online_step(
        query,
        None,
        learn=False,
        write_gate=torch.zeros(2),
        gate_target=torch.ones(2),
    )
    assert external_zero["write_gate_source"] == "external"
    assert bool((external_zero["learned_write_gate"] > 0.0).all())
    torch.testing.assert_close(
        external_zero["effective_write_gate"],
        torch.zeros_like(external_zero["effective_write_gate"]),
        rtol=0.0,
        atol=0.0,
    )
    for state in external_zero["state"].rings:
        torch.testing.assert_close(state, torch.zeros_like(state), rtol=0.0, atol=0.0)
    assert external_zero["gate_did_update"]

    # A safety override remains usable when its separate event feature is
    # unavailable; background supervision still requires that feature.
    emergency = OnlineTemporalMQRClassifier(
        4,
        3,
        2,
        core_kwargs={"leak_rates": (0.2,), "transition_mode": "identity"},
        gate_input_dim=2,
        gate_kwargs={"rank": 2},
    )
    override = emergency.preview_step(
        torch.randn(1, 4),
        write_gate=torch.zeros(1),
    )
    assert override["write_gate_source"] == "external"
    assert override["learned_write_gate"] is None
    _assert_raises(
        ValueError,
        lambda: emergency.online_step(
            torch.randn(1, 4),
            None,
            write_gate=torch.zeros(1),
            gate_target=torch.ones(1),
        ),
        "gate_input is required",
    )


def test_external_write_gate_is_exact_and_shape_safe() -> None:
    """A gate may protect whole samples or individual timescales without changing decay."""

    core = MultiTimescaleMQR(
        2,
        2,
        2,
        leak_rates=(0.2, 0.2),
        injection_rank=2,
        injection_activation="none",
        state_activation="none",
        transition_mode="identity",
    ).double()
    with torch.no_grad():
        core.input_down.weight.copy_(torch.eye(2, dtype=torch.double))
        for projection in core.input_up:
            projection.weight.copy_(torch.eye(2, dtype=torch.double))
    x = torch.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=torch.double)
    previous = TemporalMQRState(
        tuple(torch.ones(2, 2, dtype=torch.double) for _ in range(2))
    )
    _logits, sample_gated = core.forward_step(
        x,
        state=previous,
        write_gate=torch.tensor([0.0, 1.0], dtype=torch.double),
    )
    expected = torch.stack((torch.full((2,), 0.8, dtype=torch.double), 0.8 + x[1]))
    for state in sample_gated.rings:
        torch.testing.assert_close(state, expected, rtol=1e-12, atol=1e-12)

    scale_gate = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.double)
    _logits, scale_gated = core.forward_step(
        x,
        state=previous,
        write_gate=scale_gate,
    )
    torch.testing.assert_close(
        scale_gated.rings[0],
        torch.stack((torch.full((2,), 0.8, dtype=torch.double), 0.8 + x[1])),
    )
    torch.testing.assert_close(
        scale_gated.rings[1],
        torch.stack((0.8 + x[0], torch.full((2,), 0.8, dtype=torch.double))),
    )

    _assert_raises(
        ValueError,
        lambda: core.forward_step(x, write_gate=torch.ones(2, 1, dtype=torch.double)),
        "write_gate must have shape",
    )
    _assert_raises(
        ValueError,
        lambda: core.forward_step(
            x,
            write_gate=torch.tensor([0.0, 1.01], dtype=torch.double),
        ),
        "[0, 1]",
    )
    _assert_raises(
        TypeError,
        lambda: core.forward_step(x, write_gate=torch.ones(2, dtype=torch.long)),
        "floating-point",
    )
    nonfinite = x.clone()
    nonfinite[0, 0] = float("nan")
    _assert_raises(
        ValueError,
        lambda: core.forward_step(nonfinite),
        "finite",
    )


def test_gated_contraction_is_independent_of_input_write() -> None:
    """Shared gates and inputs cancel from the state-difference recurrence."""

    torch.manual_seed(818)
    core = MultiTimescaleMQR(
        3,
        5,
        2,
        leak_rates=(0.3, 0.05),
        injection_rank=3,
        state_activation="tanh",
        transition_mode="unistochastic",
    ).double()
    x = torch.randn(4, 3, dtype=torch.double)
    gate = torch.rand(4, 2, dtype=torch.double)
    left = TemporalMQRState(
        tuple(0.2 * torch.randn(4, 5, dtype=torch.double) for _ in range(2))
    )
    right = TemporalMQRState(
        tuple(0.2 * torch.randn(4, 5, dtype=torch.double) for _ in range(2))
    )
    _logits, left_next = core.forward_step(x, state=left, write_gate=gate)
    _logits, right_next = core.forward_step(x, state=right, write_gate=gate)
    for index, leak in enumerate((0.3, 0.05)):
        before = (left.rings[index] - right.rings[index]).abs().max()
        after = (left_next.rings[index] - right_next.rings[index]).abs().max()
        assert float(after.item()) <= (1.0 - leak) * float(before.item()) + 1e-12


def test_keyed_stream_bank_has_zero_state_crosstalk() -> None:
    """Interleaved sessions must reproduce two independently executed state trajectories."""

    torch.manual_seed(828)
    interleaved = _make_online(carry_state=True, transition_mode="identity")
    interleaved.max_streams = 2
    separate_a = copy.deepcopy(interleaved)
    separate_b = copy.deepcopy(interleaved)
    a1, a2 = torch.randn(1, 4), torch.randn(1, 4)
    b1, b2 = torch.randn(1, 4), torch.randn(1, 4)

    interleaved.online_step(a1, None, learn=False, stream_id="a")
    interleaved.online_step(b1, None, learn=False, stream_id="b")
    result_a = interleaved.online_step(a2, None, learn=False, stream_id="a")
    result_b = interleaved.online_step(b2, None, learn=False, stream_id="b")
    separate_a.online_step(a1, None, learn=False)
    expected_a = separate_a.online_step(a2, None, learn=False)
    separate_b.online_step(b1, None, learn=False)
    expected_b = separate_b.online_step(b2, None, learn=False)

    torch.testing.assert_close(result_a["logits"], expected_a["logits"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(result_b["logits"], expected_b["logits"], rtol=0.0, atol=0.0)
    for actual, expected in zip(
        interleaved.current_state(stream_id="a").rings,
        separate_a.current_state().rings,
    ):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    for actual, expected in zip(
        interleaved.current_state(stream_id="b").rings,
        separate_b.current_state().rings,
    ):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert interleaved.active_stream_ids() == ("a", "b")
    interleaved.reset_state("a")
    assert interleaved.current_state(stream_id="a") is None
    assert interleaved.current_state(stream_id="b") is not None


def test_stream_capacity_is_explicit_and_lru_is_auditable() -> None:
    """The default policy must fail closed; opt-in LRU must report the exact eviction."""

    torch.manual_seed(838)
    strict = OnlineTemporalMQRClassifier(
        4,
        5,
        3,
        core_kwargs={"leak_rates": (0.2,), "transition_mode": "identity"},
        max_streams=2,
        stream_overflow_policy="error",
    )
    x = torch.randn(1, 4)
    strict.infer_step(x, stream_id="a", issue_feedback_ticket=False)
    strict.infer_step(x, stream_id="b", issue_feedback_ticket=False)
    observations = int(strict.online_observations.item())
    _assert_raises(
        RuntimeError,
        lambda: strict.infer_step(x, stream_id="c", issue_feedback_ticket=False),
        "state bank is full",
    )
    assert strict.active_stream_ids() == ("a", "b")
    assert int(strict.online_observations.item()) == observations

    lru = copy.deepcopy(strict)
    lru.stream_overflow_policy = "lru"
    lru.infer_step(x, stream_id="a", issue_feedback_ticket=False)
    assert lru.active_stream_ids() == ("b", "a")
    result = lru.infer_step(x, stream_id="c", issue_feedback_ticket=False)
    assert result["state_evicted"]
    assert result["evicted_stream_id"] == "b"
    assert lru.active_stream_ids() == ("a", "c")


def test_delayed_feedback_reconstructs_exact_issue_time_readout_gradient() -> None:
    """A stale ticket must use cached old logits/features and touch only the readout."""

    torch.manual_seed(848)
    model = OnlineTemporalMQRClassifier(
        4,
        5,
        3,
        core_kwargs={
            "leak_rates": (0.3, 0.05),
            "transition_mode": "identity",
        },
        lr=0.03,
        transition_lr_ratio=0.0,
        injection_lr_ratio=0.0,
        readout_lr_ratio=1.0,
        carry_state=False,
        max_update_norm=None,
        max_pending_feedback=4,
    )
    non_readout_before = {
        name: parameter.detach().clone()
        for name, parameter in model.core.named_parameters()
        if not name.startswith("readout.")
    }
    first = model.infer_step(torch.randn(2, 4))
    second = model.infer_step(torch.randn(2, 4))
    y1 = torch.tensor([0, 2])
    y2 = torch.tensor([1, 0])
    model.apply_feedback(first["ticket_id"], y1)
    weight_before_second = model.core.readout.weight.detach().clone()
    features = torch.cat(second["state"].rings, dim=1)
    probabilities = second["logits"].softmax(dim=1)
    desired = F.one_hot(y2, num_classes=3).to(dtype=probabilities.dtype)
    expected_gradient = ((probabilities - desired) / 2).T @ features
    expected_weight = weight_before_second - model.lr * expected_gradient
    result = model.apply_feedback(second["ticket_id"], y2)

    torch.testing.assert_close(model.core.readout.weight, expected_weight, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(result["logits"], second["logits"], rtol=0.0, atol=0.0)
    assert result["parameter_staleness"] == 1
    assert not result["state_mutated"]
    for name, parameter in model.core.named_parameters():
        if name in non_readout_before:
            torch.testing.assert_close(parameter, non_readout_before[name], rtol=0.0, atol=0.0)
    _assert_raises(
        RuntimeError,
        lambda: model.apply_feedback(second["ticket_id"], y2),
        "already consumed",
    )

    # The delayed path must also match the declared weighted soft-target loss,
    # even when rows are not normalized probability distributions.
    third = model.infer_step(torch.randn(2, 4))
    soft = torch.tensor([[0.2, 0.3, 0.1], [0.0, 0.5, 0.25]])
    features = torch.cat(third["state"].rings, dim=1)
    probabilities = third["logits"].softmax(dim=1)
    error = probabilities * soft.sum(dim=1, keepdim=True) - soft
    expected_gradient = (error / 2).T @ features
    weight_before_soft = model.core.readout.weight.detach().clone()
    model.apply_feedback(third["ticket_id"], soft)
    torch.testing.assert_close(
        model.core.readout.weight,
        weight_before_soft - model.lr * expected_gradient,
        rtol=1e-6,
        atol=1e-7,
    )


def test_feedback_queue_capacity_expiry_and_unknown_ids_fail_closed() -> None:
    """Bounded tickets must never be silently reused after overflow or expiry."""

    torch.manual_seed(858)
    x = torch.randn(1, 4)
    strict = OnlineTemporalMQRClassifier(
        4,
        4,
        3,
        core_kwargs={"leak_rates": (0.2,)},
        carry_state=False,
        max_pending_feedback=1,
    )
    first = strict.infer_step(x)
    observations = int(strict.online_observations.item())
    _assert_raises(RuntimeError, lambda: strict.infer_step(x), "feedback queue is full")
    assert int(strict.online_observations.item()) == observations
    _assert_raises(KeyError, lambda: strict.apply_feedback(999, torch.tensor([0])))

    evicting = OnlineTemporalMQRClassifier(
        4,
        4,
        3,
        core_kwargs={"leak_rates": (0.2,)},
        carry_state=False,
        max_pending_feedback=1,
        feedback_overflow_policy="oldest",
    )
    old = evicting.infer_step(x)["ticket_id"]
    new = evicting.infer_step(-x)["ticket_id"]
    assert evicting.pending_feedback_ids() == (new,)
    _assert_raises(
        RuntimeError,
        lambda: evicting.apply_feedback(old, torch.tensor([0])),
        "evicted",
    )

    expiring = OnlineTemporalMQRClassifier(
        4,
        4,
        3,
        core_kwargs={"leak_rates": (0.2,)},
        carry_state=False,
        max_pending_feedback=2,
        feedback_ttl_observations=1,
    )
    ticket = expiring.infer_step(x)["ticket_id"]
    expiring.infer_step(x, issue_feedback_ticket=False)
    assert expiring.pending_feedback_ids() == (ticket,)
    expiring.infer_step(x, issue_feedback_ticket=False)
    assert expiring.pending_feedback_count == 0
    _assert_raises(
        RuntimeError,
        lambda: expiring.apply_feedback(ticket, torch.tensor([0])),
        "expired",
    )
    assert first["ticket_id"] == 1


def test_runtime_checkpoint_preserves_stream_lru_and_pending_feedback() -> None:
    """Operational state, ticket payloads, versions, and delayed OGD must resume exactly."""

    torch.manual_seed(868)
    model = OnlineTemporalMQRClassifier(
        4,
        5,
        3,
        core_kwargs={
            "leak_rates": (0.4, 0.04),
            "transition_mode": "identity",
        },
        lr=0.02,
        transition_lr_ratio=0.0,
        injection_lr_ratio=0.0,
        readout_lr_ratio=1.0,
        ogd_max_rank=2,
        max_streams=2,
        stream_overflow_policy="lru",
        max_pending_feedback=4,
    )
    first = model.infer_step(torch.randn(1, 4), stream_id="a")
    second = model.infer_step(torch.randn(1, 4), stream_id="b")
    model.apply_feedback(
        first["ticket_id"],
        torch.tensor([1]),
        remember_gradient=True,
        project_with_memory=False,
    )
    model.infer_step(
        torch.randn(1, 4),
        stream_id="a",
        issue_feedback_ticket=False,
    )
    assert model.active_stream_ids() == ("b", "a")

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = OnlineTemporalMQRClassifier(
        4,
        5,
        3,
        core_kwargs={
            "leak_rates": (0.4, 0.04),
            "transition_mode": "identity",
        },
        lr=0.02,
        transition_lr_ratio=0.0,
        injection_lr_ratio=0.0,
        readout_lr_ratio=1.0,
        ogd_max_rank=2,
        max_streams=2,
        stream_overflow_policy="lru",
        max_pending_feedback=4,
    )
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    assert restored.active_stream_ids() == model.active_stream_ids() == ("b", "a")
    assert restored.stream_statistics() == model.stream_statistics()
    assert restored.pending_feedback_ids() == model.pending_feedback_ids() == (
        second["ticket_id"],
    )
    assert restored.feedback_gradient_memory.rank == model.feedback_gradient_memory.rank == 1
    assert restored.online_parameter_version.item() == model.online_parameter_version.item()
    for stream_id in ("a", "b"):
        left = restored.current_state(stream_id=stream_id)
        right = model.current_state(stream_id=stream_id)
        assert left is not None and right is not None
        for actual, expected in zip(left.rings, right.rings):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    restored_result = restored.apply_feedback(second["ticket_id"], torch.tensor([2]))
    model_result = model.apply_feedback(second["ticket_id"], torch.tensor([2]))
    torch.testing.assert_close(
        restored_result["logits"],
        model_result["logits"],
        rtol=0.0,
        atol=0.0,
    )
    for actual, expected in zip(restored.parameters(), model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def run_all_tests() -> None:
    tests = [
        test_exact_multiscale_decay_and_half_life,
        test_unistochastic_constraints_and_contraction,
        test_nonflat_temporal_cayley_base_and_inactive_full_leak_ring,
        test_cyclic_givens_is_a_low_cost_literal_ring_with_nonzero_jacobian,
        test_signed_state_certificate_and_cayley_finite_drift_bound,
        test_temporal_residual_sidecar_is_initially_exact_noop_and_bounded,
        test_write_strength_is_independent_from_decay,
        test_batch_lanes_are_independent,
        test_frozen_unistochastic_transition_is_cached_exactly,
        test_topology_ablation_has_matched_shared_initialization,
        test_write_ablation_is_parameter_matched_and_changes_only_amplitude,
        test_delayed_digits_query_contains_no_digit_features,
        test_interference_digits_query_is_identical_and_label_free,
        test_preview_is_read_only_and_feedback_has_no_label_leakage,
        test_state_memory_and_parameter_memory_are_separately_observable,
        test_small_online_step_decreases_current_loss_and_preserves_manifold,
        test_temporal_update_uses_one_complete_ogd_vector,
        test_update_norm_is_atomically_clipped,
        test_checkpoint_roundtrip_preserves_state_and_ogd,
        test_temporal_write_gate_initialization_range_and_validation,
        test_gate_label_is_causal_and_cannot_change_current_prediction,
        test_small_auxiliary_gate_step_decreases_gate_bce,
        test_gate_ogd_uses_complete_vector_and_has_independent_clip,
        test_learned_gate_checkpoint_and_external_override_are_exact,
        test_external_write_gate_is_exact_and_shape_safe,
        test_gated_contraction_is_independent_of_input_write,
        test_keyed_stream_bank_has_zero_state_crosstalk,
        test_stream_capacity_is_explicit_and_lru_is_auditable,
        test_delayed_feedback_reconstructs_exact_issue_time_readout_gradient,
        test_feedback_queue_capacity_expiry_and_unknown_ids_fail_closed,
        test_runtime_checkpoint_preserves_stream_lru_and_pending_feedback,
    ]
    for test in tests:
        logging.info("Running %s...", test.__name__)
        test()
        logging.info("✓ %s", test.__name__)
    logging.info("Temporal MQR test results: %d passed, 0 failed", len(tests))


if __name__ == "__main__":
    run_all_tests()
