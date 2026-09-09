#!/usr/bin/env python3
"""Focused tests for prequential multi-ring online learning and OGD."""

from __future__ import annotations

import copy
import io
import logging

import torch

from mqr import OnlineMultiRingClassifier, OrthogonalGradientMemory


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _make_online(*, num_rings: int = 2, carry_state: bool = False, ogd_max_rank: int = 4):
    return OnlineMultiRingClassifier(
        input_dim=5,
        hidden_dim=8,
        output_dim=3,
        num_rings=num_rings,
        lr=0.04,
        unitary_lr_ratio=0.5,
        injection_lr_ratio=1.0,
        readout_lr_ratio=1.0,
        adjoint_steps=40,
        ogd_max_rank=ogd_max_rank,
        carry_state=carry_state,
        ring_kwargs={
            "alpha": 0.3,
            "relaxation_steps": 40,
            "lora_rank": 4,
            "readout_dim": 8,
        },
    )


def _snapshot(module: torch.nn.Module):
    return {name: value.detach().clone() for name, value in module.named_parameters()}


def _assert_snapshot_equal(module: torch.nn.Module, snapshot) -> None:
    current = dict(module.named_parameters())
    assert current.keys() == snapshot.keys()
    for name, expected in snapshot.items():
        torch.testing.assert_close(current[name], expected, rtol=0.0, atol=0.0)


def test_preconditioned_ogd_theorem() -> None:
    """Heterogeneous learning rates must preserve old gradients to first order."""

    memory = OrthogonalGradientMemory(max_rank=3, tolerance=1e-12).double()
    old = [
        ("fast", torch.tensor([1.0, 2.0], dtype=torch.double), 0.04),
        ("slow", torch.tensor([-1.0], dtype=torch.double), 0.01),
    ]
    new = [
        ("fast", torch.tensor([3.0, -1.0], dtype=torch.double), 0.04),
        ("slow", torch.tensor([2.0], dtype=torch.double), 0.01),
    ]
    assert memory.observe(old)
    projected, stats = memory.project_preconditioned(new)

    delta_fast = -0.04 * projected["fast"]
    delta_slow = -0.01 * projected["slow"]
    old_first_order = (old[0][1] * delta_fast).sum() + (old[1][1] * delta_slow).sum()
    new_first_order = (new[0][1] * delta_fast).sum() + (new[1][1] * delta_slow).sum()

    assert abs(float(old_first_order.item())) < 1e-12
    expected_descent = -(stats["projected_norm"] ** 2)
    assert abs(float(new_first_order.item()) - expected_descent) < 1e-12
    assert expected_descent < 0
    assert memory.observe(new)
    assert memory.rank == 2
    assert memory.orthogonality_error() < 1e-12


def test_constrained_ogd_projection_preserves_slow_parameter_mask() -> None:
    """A slow block must remain zero while the allowed fast update stays orthogonal."""

    memory = OrthogonalGradientMemory(max_rank=2, tolerance=1e-12).double()
    old = [
        ("fast", torch.tensor([1.0, 2.0], dtype=torch.double), 0.04),
        ("slow", torch.tensor([3.0, -1.0], dtype=torch.double), 0.01),
    ]
    new = [
        ("fast", torch.tensor([-2.0, 1.0], dtype=torch.double), 0.04),
        ("slow", torch.tensor([5.0, 7.0], dtype=torch.double), 0.01),
    ]
    assert memory.observe(old)
    snapshot = memory.snapshot()
    projected, stats = memory.project_preconditioned(
        new,
        allowed_names=("fast",),
    )
    torch.testing.assert_close(
        projected["slow"], torch.zeros_like(projected["slow"]), rtol=0.0, atol=0.0
    )
    delta_fast = -0.04 * projected["fast"]
    delta_slow = -0.01 * projected["slow"]
    old_first_order = (old[0][1] * delta_fast).sum() + (
        old[1][1] * delta_slow
    ).sum()
    assert abs(float(old_first_order.item())) < 1e-12
    assert stats["projected_norm"] > 0.0

    assert memory.observe(new)
    assert memory.rank == 2
    memory.restore(snapshot)
    assert memory.rank == 1
    restored, _ = memory.project_preconditioned(new, allowed_names=("fast",))
    torch.testing.assert_close(restored["fast"], projected["fast"], rtol=0.0, atol=0.0)


def test_prequential_prediction_has_no_label_leakage() -> None:
    """Changing the feedback label cannot change the prediction returned at that step."""

    torch.manual_seed(101)
    base = _make_online(num_rings=1)
    model_a = copy.deepcopy(base)
    model_b = copy.deepcopy(base)
    x = torch.randn(4, 5)
    target_a = torch.tensor([0, 0, 0, 0])
    target_b = torch.tensor([2, 2, 2, 2])

    before = base.rings[0](x)
    info_a = model_a.online_step(x, target_a, context_id="shared", carry_state=False)
    info_b = model_b.online_step(x, target_b, context_id="shared", carry_state=False)

    torch.testing.assert_close(info_a["logits"], before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(info_b["logits"], before, rtol=0.0, atol=0.0)
    assert info_a["prediction_before_update"] and info_b["prediction_before_update"]
    assert any(
        not torch.equal(pa, pb)
        for pa, pb in zip(model_a.rings[0].parameters(), model_b.rings[0].parameters())
    )


def test_explicit_multiring_isolation_and_capacity() -> None:
    """An update may mutate only the ring selected by a stable explicit context."""

    torch.manual_seed(202)
    model = _make_online(num_rings=2)
    x = torch.randn(5, 5)
    target = torch.tensor([0, 1, 2, 1, 0])

    ring1_before = _snapshot(model.rings[1])
    first = model.online_step(x, target, context_id="context-a", carry_state=False)
    assert first["ring_index"] == 0
    _assert_snapshot_equal(model.rings[1], ring1_before)

    ring0_after_a = _snapshot(model.rings[0])
    second = model.online_step(-x, target, context_id="context-b", carry_state=False)
    assert second["ring_index"] == 1
    _assert_snapshot_equal(model.rings[0], ring0_after_a)
    assert model.route_context("context-a", allocate=False) == 0
    assert model.route_context("context-b", allocate=False) == 1
    usage_before_eval = model.ring_usage.clone()
    logits = model(x, context_id="context-a")
    assert logits.shape == (x.size(0), 3)
    torch.testing.assert_close(model.ring_usage, usage_before_eval, rtol=0.0, atol=0.0)

    try:
        model.route_context("context-c")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Capacity exhaustion must not silently reuse a protected ring")


def test_state_is_carried_between_inference_and_learning_steps() -> None:
    """One-step relaxation must use the prior online state as its warm start."""

    torch.manual_seed(303)
    model = OnlineMultiRingClassifier(
        5,
        7,
        3,
        num_rings=1,
        lr=0.02,
        carry_state=True,
        ring_kwargs={
            "alpha": 0.2,
            "relaxation_steps": 1,
            "lora_rank": 3,
            "readout_dim": 7,
        },
    )
    x = torch.randn(1, 5)
    first = model.online_step(x, None, context_id="stream")
    second = model.online_step(x, None, context_id="stream")

    ring = model.rings[0]
    h1 = first["h_star"]
    injection = ring.injection(x)
    transition = ring._current_H(device=x.device, dtype=x.dtype)
    expected_h2 = (1.0 - ring.alpha) * (h1 @ transition.T) + ring.alpha * injection
    torch.testing.assert_close(second["h_star"], expected_h2, rtol=1e-6, atol=1e-7)
    assert not torch.equal(first["h_star"], second["h_star"])


def test_ring_update_uses_one_complete_ogd_vector() -> None:
    """Readout, injection, and Cayley coordinates must share one memory vector."""

    torch.manual_seed(404)
    model = _make_online(num_rings=1, ogd_max_rank=3)
    x1 = torch.randn(6, 5)
    y1 = torch.tensor([0, 1, 2, 0, 1, 2])
    first = model.online_step(
        x1, y1, context_id="shared", remember_gradient=True, carry_state=False
    )
    memory = model.gradient_memories[0]
    expected_dimension = sum(parameter.numel() for parameter in model.rings[0].parameters())
    assert memory.rank == 1
    assert memory.dimension == expected_dimension
    assert first["ogd_memory_added"]

    x2 = torch.randn(6, 5)
    y2 = torch.tensor([2, 2, 1, 1, 0, 0])
    second = model.online_step(x2, y2, context_id="shared", carry_state=False)
    assert 0.0 <= second["ogd_retained_norm"] <= 1.0 + 1e-6
    assert second["ogd_max_abs_overlap"] < 1e-5
    assert second["ogd_first_order_decrease"] <= 0.0
    assert second["unitary_error"] < 1e-5


def test_novelty_router_and_checkpoint_roundtrip() -> None:
    """Experimental novelty routing and online memory must survive state_dict I/O."""

    torch.manual_seed(505)
    model = OnlineMultiRingClassifier(
        5,
        7,
        3,
        num_rings=2,
        lr=0.02,
        novelty_threshold=0.5,
        ogd_max_rank=2,
        ring_kwargs={
            "alpha": 0.3,
            "relaxation_steps": 8,
            "lora_rank": 3,
            "readout_dim": 7,
        },
    )
    x = torch.ones(1, 5)
    y = torch.tensor([1])
    first = model.online_step(x, y, remember_gradient=True)
    second = model.online_step(-x, None)
    third = model.online_step(x, None)
    assert (first["ring_index"], second["ring_index"], third["ring_index"]) == (0, 1, 0)
    model.route_context("saved-context")

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = OnlineMultiRingClassifier(
        5,
        7,
        3,
        num_rings=2,
        lr=0.02,
        novelty_threshold=0.5,
        ogd_max_rank=2,
        ring_kwargs={
            "alpha": 0.3,
            "relaxation_steps": 8,
            "lora_rank": 3,
            "readout_dim": 7,
        },
    )
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    assert restored.route_context("saved-context", allocate=False) == 0
    assert restored.gradient_memories[0].rank == 1
    assert restored.gradient_memories[0].dimension == model.gradient_memories[0].dimension


def run_all_tests() -> None:
    tests = [
        test_preconditioned_ogd_theorem,
        test_constrained_ogd_projection_preserves_slow_parameter_mask,
        test_prequential_prediction_has_no_label_leakage,
        test_explicit_multiring_isolation_and_capacity,
        test_state_is_carried_between_inference_and_learning_steps,
        test_ring_update_uses_one_complete_ogd_vector,
        test_novelty_router_and_checkpoint_roundtrip,
    ]
    passed = 0
    for test in tests:
        logging.info("Running %s...", test.__name__)
        test()
        passed += 1
        logging.info("✓ %s", test.__name__)
    logging.info("Online test results: %d passed, 0 failed", passed)


if __name__ == "__main__":
    run_all_tests()
