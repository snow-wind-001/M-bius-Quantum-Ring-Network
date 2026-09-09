#!/usr/bin/env python3
"""Strict tests for delayed future-utility learning in Temporal MQR."""

from __future__ import annotations

import copy
import io
import logging

import torch

from mqr import MultiTimescaleMQR, TemporalMQRState, UtilityDrivenMQR


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _assert_raises(expected, operation, message: str | None = None) -> Exception:
    try:
        operation()
    except expected as exc:
        if message is not None:
            assert message in str(exc)
        return exc
    raise AssertionError(f"expected {expected.__name__}")


def _make_utility(
    *,
    max_pending_candidates: int = 16,
    candidate_overflow_policy: str = "error",
    utility_ogd_max_rank: int = 0,
    utility_max_update_norm: float | None = 0.05,
    readout_ogd_max_rank: int = 0,
    readout_max_update_norm: float | None = 0.25,
    context_trace: bool = False,
    context_experts: bool = False,
) -> UtilityDrivenMQR:
    torch.manual_seed(41)
    model = UtilityDrivenMQR(
        2,
        2,
        2,
        core_kwargs={
            "leak_rates": (1.0, 0.1),
            "write_scales": (1.0, 1.0),
            "injection_rank": 2,
            "injection_activation": "none",
            "state_activation": "none",
            "transition_mode": "identity",
            "readout_bias": False,
        },
        gate_rank=3,
        utility_lr=0.2,
        readout_lr=0.05,
        utility_ogd_max_rank=utility_ogd_max_rank,
        readout_ogd_max_rank=readout_ogd_max_rank,
        utility_max_update_norm=utility_max_update_norm,
        readout_max_update_norm=readout_max_update_norm,
        max_pending_candidates=max_pending_candidates,
        candidate_overflow_policy=candidate_overflow_policy,
        initial_advantage=-0.1,
        context_volatility_decay=0.9 if context_trace else None,
        utility_experts=2 if context_experts else 1,
        utility_router_feature="context_volatility" if context_experts else None,
        utility_router_threshold=0.3,
        promotion_max_norm=4.0,
    ).double()
    with torch.no_grad():
        model.core.input_down.weight.copy_(torch.eye(2, dtype=torch.double))
        for projection in model.core.input_up:
            projection.weight.copy_(torch.eye(2, dtype=torch.double))
        model.core.readout.weight.zero_()
        # Only the slow ring drives the deterministic two-class readout.
        model.core.readout.weight[:, 2:].copy_(
            3.0 * torch.eye(2, dtype=torch.double)
        )
    return model


def _query(model: UtilityDrivenMQR, *, stream_id=None):
    return model.observe(
        torch.zeros(1, 2, dtype=torch.double),
        stream_id=stream_id,
        issue_candidate=False,
        external_write=False,
    )


def test_external_promotion_preserves_contraction() -> None:
    """Promotion changes forcing, not the homogeneous contraction factor."""

    torch.manual_seed(10)
    core = MultiTimescaleMQR(
        3,
        4,
        2,
        leak_rates=(0.25,),
        injection_rank=2,
        state_activation="tanh",
        transition_mode="unistochastic",
    ).double()
    x = torch.randn(2, 3, dtype=torch.double)
    promotion = torch.randn(2, 1, 4, dtype=torch.double)
    left = TemporalMQRState((0.1 * torch.randn(2, 4, dtype=torch.double),))
    right = TemporalMQRState((0.1 * torch.randn(2, 4, dtype=torch.double),))
    _left_logits, left_next = core.forward_step(
        x,
        state=left,
        promotion=promotion,
    )
    _right_logits, right_next = core.forward_step(
        x,
        state=right,
        promotion=promotion,
    )
    before = (left.rings[0] - right.rings[0]).abs().max()
    after = (left_next.rings[0] - right_next.rings[0]).abs().max()
    assert float(after.item()) <= 0.75 * float(before.item()) + 1e-12
    assert core.max_unitary_error() < 1e-12
    assert core.max_stochastic_error() < 1e-12
    _assert_raises(
        ValueError,
        lambda: core.forward_step(x, promotion=torch.zeros(2, 4, 1)),
        "promotion must have shape",
    )


def test_marker_free_features_are_strictly_causal() -> None:
    """Novelty is computed from past observations, with no event marker or label."""

    model = _make_utility()
    background = torch.tensor([[0.0, 1.0]], dtype=torch.double)
    target = torch.tensor([[1.0, 0.0]], dtype=torch.double)
    first = model.observe(background, external_write=False)
    second = model.observe(background, external_write=False)
    third = model.observe(target, external_write=False)
    assert first["feature_names"] == (
        "novelty",
        "uncertainty",
        "past_surprise",
        "slow_saturation",
        "context_change",
    )
    assert float(first["features"][0, 0].item()) == 0.0
    assert abs(float(second["features"][0, 0].item())) < 1e-12
    assert float(third["features"][0, 0].item()) > 0.49
    assert float(third["features"][0, 4].item()) > 0.49
    assert model.core.input_dim == 2
    model.resolve_utility(
        first["candidate_id"],
        torch.tensor([0]),
        learn=False,
        promote_missed_positive=False,
    )
    after_feedback = model.observe(
        background,
        issue_candidate=False,
        external_write=False,
    )
    assert float(after_feedback["features"][0, 2].item()) > 0.0


def test_future_target_cannot_change_issue_time_action_or_state() -> None:
    """A future utility label may train the gate but cannot leak into its action."""

    base = _make_utility()
    positive = copy.deepcopy(base)
    negative = copy.deepcopy(base)
    x = torch.tensor([[1.0, 0.0]], dtype=torch.double)
    issued_positive = positive.observe(x)
    issued_negative = negative.observe(x)
    assert issued_positive["effective_write"] is False
    assert issued_negative["effective_write"] is False
    torch.testing.assert_close(
        issued_positive["features"], issued_negative["features"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        issued_positive["logits"], issued_negative["logits"], rtol=0.0, atol=0.0
    )
    for left, right in zip(
        issued_positive["state"].rings,
        issued_negative["state"].rings,
    ):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)

    _query(positive)
    _query(negative)
    result_positive = positive.resolve_utility(
        issued_positive["candidate_id"], torch.tensor([0])
    )
    result_negative = negative.resolve_utility(
        issued_negative["candidate_id"], torch.tensor([1])
    )
    assert result_positive["write_advantage"] > 0.0
    assert result_negative["write_advantage"] < 0.0
    assert result_positive["feedback_after_action"]
    assert result_positive["causal_feature_replay"]
    assert any(
        not torch.equal(left, right)
        for left, right in zip(
            positive.utility_gate.parameters(), negative.utility_gate.parameters()
        )
    )


def test_context_trace_is_strictly_past_and_experts_are_isolated() -> None:
    """A past-only volatility trace routes experts without future-label leakage."""

    model = _make_utility(context_trace=True, context_experts=True)
    background = torch.tensor([[0.0, 1.0]], dtype=torch.double)
    changed = torch.tensor([[1.0, 0.0]], dtype=torch.double)
    first = model.observe(background, issue_candidate=False, external_write=False)
    second = model.observe(changed, issue_candidate=False, external_write=False)
    assert first["feature_names"][-1] == "context_volatility"
    assert float(first["features"][0, -1].item()) == 0.0
    # The current change cannot enter its own decision feature.
    assert float(second["features"][0, -1].item()) == 0.0

    issued = model.observe(changed, external_write=False)
    assert float(issued["features"][0, -1].item()) > 0.49
    assert issued["utility_expert"] == 1
    expert_zero_before = [
        parameter.detach().clone()
        for parameter in model.utility_gate.experts[0].parameters()
    ]
    expert_one_before = [
        parameter.detach().clone()
        for parameter in model.utility_gate.experts[1].parameters()
    ]
    _query(model)
    feedback = model.resolve_utility(issued["candidate_id"], torch.tensor([0]))
    assert feedback["utility_expert_at_issue"] == 1
    for before, after in zip(
        expert_zero_before, model.utility_gate.experts[0].parameters()
    ):
        torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            expert_one_before, model.utility_gate.experts[1].parameters()
        )
    )


def test_hard_routed_expert_isolation_survives_ogd_projection() -> None:
    """Singleton hard routes keep the OGD basis block-supported by expert."""

    model = _make_utility(
        context_trace=True,
        context_experts=True,
        utility_ogd_max_rank=2,
    )
    background = torch.tensor([[0.0, 1.0]], dtype=torch.double)
    changed = torch.tensor([[1.0, 0.0]], dtype=torch.double)

    first = model.observe(background, external_write=False)
    assert first["utility_expert"] == 0
    model.observe(background, issue_candidate=False, external_write=False)
    first_feedback = model.resolve_utility(
        first["candidate_id"],
        torch.tensor([0]),
        remember_gradient=True,
    )
    assert first_feedback["ogd_memory_added"]
    assert model.utility_gradient_memory.rank == 1

    # Sustained alternation raises the past-only EMA above the 0.30 threshold;
    # the change on any frame still affects routing no earlier than a later one.
    routed = []
    for index in range(12):
        output = model.observe(
            changed if index % 2 == 0 else background,
            issue_candidate=False,
            external_write=False,
        )
        routed.append(output["utility_expert"])
    assert routed[0] == 0 and routed[-1] == 1
    second = model.observe(changed, external_write=False)
    assert second["utility_expert"] == 1
    expert_zero_before = [
        parameter.detach().clone()
        for parameter in model.utility_gate.experts[0].parameters()
    ]
    expert_one_before = [
        parameter.detach().clone()
        for parameter in model.utility_gate.experts[1].parameters()
    ]
    _query(model)
    second_feedback = model.resolve_utility(
        second["candidate_id"],
        torch.tensor([0]),
        project_with_memory=True,
    )
    assert second_feedback["ogd_projection_applied"]
    for before, after in zip(
        expert_zero_before, model.utility_gate.experts[0].parameters()
    ):
        torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            expert_one_before, model.utility_gate.experts[1].parameters()
        )
    )


def test_candidate_capacity_and_single_consumption_fail_closed() -> None:
    """Bounded tickets cannot be overwritten silently or consumed twice."""

    model = _make_utility(max_pending_candidates=1)
    first = model.observe(torch.tensor([[1.0, 0.0]], dtype=torch.double))
    observation_count = int(model.online_observations.item())
    state_before = model.current_state()
    assert state_before is not None
    _assert_raises(
        RuntimeError,
        lambda: model.observe(torch.tensor([[0.0, 1.0]], dtype=torch.double)),
        "candidate queue is full",
    )
    assert int(model.online_observations.item()) == observation_count
    state_after = model.current_state()
    assert state_after is not None
    for left, right in zip(state_before.rings, state_after.rings):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    result = model.resolve_utility(first["candidate_id"], torch.tensor([0]), learn=False)
    assert result["pending_candidates"] == 0
    _assert_raises(
        RuntimeError,
        lambda: model.resolve_utility(first["candidate_id"], torch.tensor([0])),
        "consumed",
    )


def test_positive_missed_candidate_is_promoted_on_next_transition() -> None:
    """Delayed benefit queues the cached payload without rewriting the past."""

    model = _make_utility()
    candidate = torch.tensor([[1.0, 0.0]], dtype=torch.double)
    issued = model.observe(candidate)
    assert issued["effective_write"] is False
    _query(model)
    before_feedback = model.current_state()
    assert before_feedback is not None
    torch.testing.assert_close(
        before_feedback.rings[1],
        torch.zeros_like(before_feedback.rings[1]),
        rtol=0.0,
        atol=0.0,
    )
    feedback = model.resolve_utility(
        issued["candidate_id"],
        torch.tensor([0]),
        learn=False,
    )
    assert feedback["write_advantage"] > 0.0
    assert feedback["missed_positive"]
    assert feedback["promotion_queued"]
    after_feedback = model.current_state()
    assert after_feedback is not None
    torch.testing.assert_close(
        after_feedback.rings[1], before_feedback.rings[1], rtol=0.0, atol=0.0
    )

    promoted = _query(model)
    assert promoted["promotion_applied"]
    promoted_state = model.current_state()
    assert promoted_state is not None
    torch.testing.assert_close(
        promoted_state.rings[1], candidate, rtol=1e-12, atol=1e-12
    )


def test_harmful_false_positive_is_reported_and_not_retroactively_erased() -> None:
    """A bad factual write remains in state; v5 makes this limitation auditable."""

    model = _make_utility()
    wrong = torch.tensor([[0.0, 1.0]], dtype=torch.double)
    issued = model.observe(wrong, external_write=True)
    _query(model)
    state_before = model.current_state()
    assert state_before is not None
    feedback = model.resolve_utility(
        issued["candidate_id"],
        torch.tensor([0]),
        learn=False,
    )
    assert feedback["write_advantage"] < 0.0
    assert feedback["irreversible_false_positive"]
    assert not feedback["promotion_queued"]
    state_after = model.current_state()
    assert state_after is not None
    for left, right in zip(state_before.rings, state_after.rings):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)


def test_utility_ogd_uses_complete_vector_and_atomic_clip() -> None:
    """Gate replay uses one full parameter vector with independent OGD/clipping."""

    model = _make_utility(
        utility_ogd_max_rank=2,
        utility_max_update_norm=1e-4,
    )
    first = model.observe(torch.tensor([[1.0, 0.0]], dtype=torch.double))
    _query(model)
    first_feedback = model.resolve_utility(
        first["candidate_id"],
        torch.tensor([0]),
        remember_gradient=True,
    )
    expected_dimension = sum(
        parameter.numel() for parameter in model.utility_gate.parameters()
    )
    assert model.utility_gradient_memory.dimension == expected_dimension
    assert model.utility_gradient_memory.rank == 1
    assert first_feedback["update_norm"] <= 1e-4 + 1e-10
    assert first_feedback["update_clip_scale"] < 1.0

    model.reset_state()
    second = model.observe(torch.tensor([[0.0, 1.0]], dtype=torch.double))
    _query(model)
    second_feedback = model.resolve_utility(
        second["candidate_id"],
        torch.tensor([0]),
        project_with_memory=True,
    )
    assert second_feedback["ogd_projection_applied"]
    assert second_feedback["ogd_max_abs_overlap"] < 1e-10
    assert second_feedback["update_norm"] <= 1e-4 + 1e-10


def test_readout_feedback_is_prequential_and_clipped() -> None:
    """Task adaptation is separate from utility learning and uses cached logits."""

    model = _make_utility(
        readout_ogd_max_rank=1,
        readout_max_update_norm=1e-4,
    )
    model.observe(
        torch.tensor([[1.0, 0.0]], dtype=torch.double),
        issue_candidate=False,
        external_write=True,
    )
    query = _query(model)
    before = model.core.readout.weight.detach().clone()
    feedback = model.learn_current(
        torch.tensor([0]),
        remember_gradient=True,
    )
    torch.testing.assert_close(feedback["logits"], query["logits"], rtol=0.0, atol=0.0)
    assert feedback["prediction_before_update"]
    assert feedback["update_norm"] <= 1e-4 + 1e-10
    assert model.readout_gradient_memory.rank == 1
    assert not torch.equal(before, model.core.readout.weight)
    _assert_raises(
        RuntimeError,
        lambda: model.learn_current(torch.tensor([0])),
        "already consumed",
    )


def test_runtime_checkpoint_restores_shadow_ticket_and_ogd() -> None:
    """Shadow trajectories, feature replay, counters, and OGD survive recovery."""

    model = _make_utility(utility_ogd_max_rank=2)
    remembered = model.observe(
        torch.tensor([[1.0, 0.0]], dtype=torch.double),
        stream_id="old",
    )
    _query(model, stream_id="old")
    model.resolve_utility(
        remembered["candidate_id"],
        torch.tensor([0]),
        remember_gradient=True,
    )
    pending = model.observe(
        torch.tensor([[0.0, 1.0]], dtype=torch.double),
        stream_id="live",
    )

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = _make_utility(utility_ogd_max_rank=2)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    assert restored.pending_candidate_ids(stream_id="live") == (
        pending["candidate_id"],
    )
    assert restored.utility_gradient_memory.rank == model.utility_gradient_memory.rank == 1
    assert restored.online_parameter_version.item() == model.online_parameter_version.item()
    for stream_id in ("old", "live"):
        left = restored.current_state(stream_id=stream_id)
        right = model.current_state(stream_id=stream_id)
        assert left is not None and right is not None
        for actual, expected in zip(left.rings, right.rings):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    restored_query = _query(restored, stream_id="live")
    model_query = _query(model, stream_id="live")
    torch.testing.assert_close(
        restored_query["logits"], model_query["logits"], rtol=0.0, atol=0.0
    )
    restored_feedback = restored.resolve_utility(
        pending["candidate_id"], torch.tensor([0])
    )
    model_feedback = model.resolve_utility(pending["candidate_id"], torch.tensor([0]))
    assert restored_feedback["write_advantage"] == model_feedback["write_advantage"]
    for actual, expected in zip(restored.parameters(), model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def run_all_tests() -> None:
    tests = [
        test_external_promotion_preserves_contraction,
        test_marker_free_features_are_strictly_causal,
        test_future_target_cannot_change_issue_time_action_or_state,
        test_context_trace_is_strictly_past_and_experts_are_isolated,
        test_hard_routed_expert_isolation_survives_ogd_projection,
        test_candidate_capacity_and_single_consumption_fail_closed,
        test_positive_missed_candidate_is_promoted_on_next_transition,
        test_harmful_false_positive_is_reported_and_not_retroactively_erased,
        test_utility_ogd_uses_complete_vector_and_atomic_clip,
        test_readout_feedback_is_prequential_and_clipped,
        test_runtime_checkpoint_restores_shadow_ticket_and_ogd,
    ]
    for test in tests:
        logging.info("Running %s...", test.__name__)
        test()
        logging.info("✓ %s", test.__name__)
    logging.info("Utility MQR test results: %d passed, 0 failed", len(tests))


if __name__ == "__main__":
    run_all_tests()
