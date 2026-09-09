"""Deterministic tests for the unified Temporal Utility MQR agent."""

from __future__ import annotations

import io
import logging
import sys

import torch

from mqr.agent import (
    GoLossWeights,
    SlowLoRAConsolidator,
    SlowLoRAConsolidationSchedule,
    TemporalUtilityMQRAgent,
    generalized_advantage_estimate,
)
from mqr.agent_baselines import (
    MultiTimescaleFastWeightCore,
    MultiTimescaleGRUCore,
    audit_agent_resources,
    matched_resource_gate,
)
from mqr.safety import SidecarSafetyLimits
from mqr.go_agent import (
    GoVectorEncoder,
    generate_go_agent_trajectories,
    generate_ko_history_pairs,
    twin_write_returns,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _snapshot_state(agent: TemporalUtilityMQRAgent, stream: str):
    return tuple(value.detach().clone() for value in agent._stream_states[stream].rings)


def _agent() -> TemporalUtilityMQRAgent:
    torch.manual_seed(20260827)
    return TemporalUtilityMQRAgent(
        8,
        board_size=3,
        ring_dim=6,
        latent_dim=7,
        leak_rates=(1.0, 0.2, 0.05),
        injection_rank=4,
        task_lr=0.02,
        utility_lr=0.1,
        ogd_max_rank=2,
        utility_ogd_max_rank=2,
        max_update_norm=0.1,
        utility_max_update_norm=0.05,
    )


def test_preview_commit_and_four_heads() -> bool:
    logging.info("Running test_preview_commit_and_four_heads...")
    agent = _agent()
    x = torch.randn(1, 8)
    preview = agent.preview_step(x, stream_id="game")
    assert float(preview["features"][0, 0].item()) == 1.0
    assert float(preview["features"][0, 4].item()) == 1.0
    assert agent.pending_ticket_count == 0
    assert "game" not in agent._stream_states
    output = preview["output"]
    assert output.placement_logits.shape == (1, 9)
    assert output.legality_logits.shape == (1, 9)
    assert output.pass_logit.shape == (1,)
    assert output.value.shape == (1,)
    assert output.policy_logits.shape == (1, 10)
    probabilities = output.policy_logits.exp()
    torch.testing.assert_close(
        probabilities.sum(dim=1), torch.ones(1), rtol=1e-6, atol=1e-7
    )
    torch.testing.assert_close(
        probabilities[:, -1], torch.full((1,), 0.1), rtol=1e-6, atol=1e-7
    )

    committed = agent.commit_step(x, stream_id="game")
    torch.testing.assert_close(
        preview["policy_logits"], committed["policy_logits"], rtol=0.0, atol=0.0
    )
    assert committed["prediction_before_update"] is True
    assert committed["ticket_id"] == 1
    assert agent.pending_ticket_count == 1
    repeated = agent.preview_step(x, stream_id="game")
    assert float(repeated["features"][0, 0].item()) == 0.0
    assert float(repeated["features"][0, 4].item()) == 0.0
    no_state, write_state = agent.shadow_states(1)
    torch.testing.assert_close(no_state.rings[0], write_state.rings[0], rtol=0.0, atol=0.0)
    assert any(
        not torch.equal(left, right)
        for left, right in zip(no_state.rings[1:], write_state.rings[1:])
    )
    logging.info("✓ preview/commit causality and four-head shapes")
    return True


def test_task_feedback_and_dual_return_critic() -> bool:
    logging.info("Running test_task_feedback_and_dual_return_critic...")
    agent = _agent()
    x = torch.randn(1, 8)
    committed = agent.commit_step(x, stream_id="game")
    ticket = int(committed["ticket_id"])
    state_before = _snapshot_state(agent, "game")
    feature = agent._pending_tickets[ticket]["causal_features"].detach().clone()
    utility_before = float(agent.advantage_scale * agent.utility_gate(feature).item())

    legality = torch.tensor([[1, 1, 0, 1, 0, 1, 1, 0, 1]], dtype=torch.float32)
    feedback = agent.apply_feedback(
        ticket,
        torch.tensor([4]),
        legality_target=legality,
        value_target=torch.tensor([0.75]),
        learn=True,
        remember_gradient=True,
        return_grad_features=True,
    )
    assert feedback["did_update"] is True
    assert feedback["grad_features"].shape == x.shape
    assert feedback["state_mutated"] is False
    assert feedback["ogd_rank"] == 1
    for before, after in zip(state_before, _snapshot_state(agent, "game")):
        torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)
    assert feedback["max_unitary_error"] < 1e-5
    assert feedback["max_stochastic_error"] < 1e-5
    assert agent.pending_ticket_count == 1

    utility = agent.calibrate_write_critic(
        ticket,
        no_write_return=-0.5,
        write_return=1.0,
        learn=True,
        remember_gradient=True,
    )
    utility_after = float(agent.advantage_scale * agent.utility_gate(feature).item())
    assert utility["write_advantage"] == 1.5
    assert utility_after > utility_before
    assert utility["feedback_after_action"] is True
    assert agent.pending_ticket_count == 0
    critic_features = torch.tensor(
        [[1.0, 0.2, 0.0, 0.1, 1.0], [0.0, 0.8, 0.5, 0.2, 0.0]]
    )
    critic_advantages = torch.tensor([1.0, -1.0])
    with torch.no_grad():
        critic_before = torch.nn.functional.mse_loss(
            agent.utility_gate(critic_features), critic_advantages.unsqueeze(1)
        )
    batch_update = agent.fit_write_critic_batch(
        critic_features,
        critic_advantages,
    )
    with torch.no_grad():
        critic_after = torch.nn.functional.mse_loss(
            agent.utility_gate(critic_features), critic_advantages.unsqueeze(1)
        )
    assert batch_update["did_update"] is True
    assert critic_after < critic_before
    unbounded = agent.fit_write_critic_batch(
        critic_features[:1].expand(6, -1),
        torch.tensor([10.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
        learn=False,
    )
    assert unbounded["target_max_abs"] == 10.0
    assert unbounded["target_mean"] > 0.0
    logging.info("✓ multi-head feedback, state immutability, and twin-return critic")
    return True


def test_task_update_guard_rolls_back_parameters_ogd_and_external_gradient() -> bool:
    logging.info("Running test_task_update_guard_rolls_back_parameters_ogd_and_external_gradient...")
    torch.manual_seed(20260828)
    agent = TemporalUtilityMQRAgent(
        8,
        board_size=3,
        ring_dim=6,
        latent_dim=7,
        leak_rates=(1.0, 0.2, 0.05),
        injection_rank=4,
        task_lr=0.2,
        ogd_max_rank=2,
        safety_limits=SidecarSafetyLimits(max_output_linf_drift=0.0),
    )
    committed = agent.commit_step(torch.randn(1, 8), stream_id="guard")
    ticket = int(committed["ticket_id"])
    parameters_before = {
        name: parameter.detach().clone() for name, parameter in agent.named_parameters()
    }
    state_before = _snapshot_state(agent, "guard")
    version_before = int(agent.online_parameter_version.item())
    feedback = agent.apply_feedback(
        ticket,
        torch.tensor([3]),
        legality_target=torch.ones(1, 9),
        learn=True,
        remember_gradient=True,
        return_grad_features=True,
    )
    assert feedback["rolled_back"] is True
    assert feedback["safety_passed"] is False
    assert "output_linf_drift" in feedback["safety_violations"]
    assert feedback["candidate_update_norm"] > 0.0
    assert feedback["update_norm"] == 0.0
    assert feedback["did_update"] is False
    assert feedback["grad_features"] is None
    assert feedback["external_gradient_authorized"] is False
    assert agent.task_gradient_memory.rank == 0
    assert int(agent.online_parameter_version.item()) == version_before
    for name, parameter in agent.named_parameters():
        torch.testing.assert_close(
            parameter, parameters_before[name], rtol=0.0, atol=0.0
        )
    for before, after in zip(state_before, _snapshot_state(agent, "guard")):
        torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)
    agent.cancel_ticket_branch(ticket, "utility")
    logging.info("✓ unsafe candidate update is atomically rejected")
    return True


def test_cayley_updates_follow_explicit_slow_schedule_with_ogd() -> bool:
    logging.info("Running test_cayley_updates_follow_explicit_slow_schedule_with_ogd...")
    torch.manual_seed(20260829)
    agent = TemporalUtilityMQRAgent(
        8,
        board_size=3,
        ring_dim=6,
        latent_dim=7,
        leak_rates=(1.0, 0.2, 0.05),
        injection_rank=4,
        task_lr=0.03,
        ogd_max_rank=3,
        transition_update_interval=3,
    )
    transition_before = {
        index: agent.core.unitary_params[index].skew_hermitian_A().detach().clone()
        for index in agent.core.active_transition_indices
    }
    head_before = agent.placement_head.weight.detach().clone()
    due = []
    refresh = []
    for index in range(3):
        committed = agent.commit_step(
            torch.randn(1, 8), stream_id="schedule", external_write=True
        )
        ticket = int(committed["ticket_id"])
        feedback = agent.apply_feedback(
            ticket,
            torch.tensor([(index + 1) % 9]),
            legality_target=torch.ones(1, 9),
            remember_gradient=index < 2,
        )
        due.append(feedback["transition_update_due"])
        refresh.append(feedback["transition_refresh_required"])
        agent.cancel_ticket_branch(ticket, "utility")
        if index < 2:
            for transition_index, expected in transition_before.items():
                parameter = agent.core.unitary_params[transition_index]
                torch.testing.assert_close(
                    parameter.skew_hermitian_A(), expected, rtol=0.0, atol=0.0
                )
    assert due == [False, False, True]
    assert refresh == [False, False, True]
    assert not torch.equal(agent.placement_head.weight, head_before)
    assert any(
        not torch.equal(
            agent.core.unitary_params[index].skew_hermitian_A(), expected
        )
        for index, expected in transition_before.items()
    )
    assert agent.task_gradient_memory.rank == 2

    identity = TemporalUtilityMQRAgent(
        8,
        board_size=3,
        ring_dim=6,
        latent_dim=7,
        transition_mode="identity",
        learn_transitions=False,
        transition_update_interval=1,
    )
    committed = identity.commit_step(
        torch.randn(1, 8), stream_id="identity-schedule", external_write=True
    )
    identity_feedback = identity.apply_feedback(
        int(committed["ticket_id"]), torch.tensor([1])
    )
    identity.cancel_ticket_branch(int(committed["ticket_id"]), "utility")
    assert identity_feedback["transition_update_due"] is True
    assert identity_feedback["transition_refresh_required"] is False
    logging.info("✓ fast task blocks update while Cayley follows its slow tick")
    return True


def test_trajectory_future_credit_is_causal_and_reaches_memory_content() -> bool:
    logging.info("Running test_trajectory_future_credit_is_causal_and_reaches_memory_content...")
    torch.manual_seed(20260830)
    agent = TemporalUtilityMQRAgent(
        8,
        board_size=3,
        ring_dim=6,
        latent_dim=7,
        leak_rates=(1.0, 0.12),
        injection_rank=4,
        cayley_coordinate_mode="minimal",
        base_unitary_init="random",
        base_unitary_scale=0.25,
        base_unitary_seed=13,
        utility_content_dim=3,
        utility_content_seed=19,
        write_warmup_observations=8,
        task_lr=0.03,
        ogd_max_rank=2,
        max_update_norm=0.05,
        max_trace_horizon=4,
    )
    assert agent.core.active_transition_indices == (1,)
    gate_before = {
        name: parameter.detach().clone()
        for name, parameter in agent.utility_gate.named_parameters()
    }
    committed_logits = []
    ticket_ids = []
    for _index in range(3):
        committed = agent.commit_step(torch.randn(1, 8), stream_id="trace")
        committed_logits.append(committed["policy_logits"].clone())
        ticket_ids.append(int(committed["ticket_id"]))
        assert committed["write_source"] == "warmup"
        assert committed["features"].shape == (1, 8)
    state_before = _snapshot_state(agent, "trace")
    result = agent.apply_trajectory_feedback(
        ticket_ids,
        [
            {"action": torch.tensor([1])},
            {"action": torch.tensor([2])},
            {"action": torch.tensor([3])},
        ],
        loss_scales=(0.0, 0.0, 1.0),
        remember_gradient=True,
        return_grad_features=True,
    )
    assert result["did_update"] is True
    assert result["prediction_before_update"] is True
    assert result["all_predictions_committed_before_update"] is True
    assert result["future_labels_used_by_gate"] is False
    assert result["trace_horizon"] == 3
    assert result["earliest_input_future_gradient_norm"] > 0.0
    assert result["future_block_gradient_norms"]["injection"] > 0.0
    assert result["future_block_gradient_norms"]["transition"] > 0.0
    assert result["update_norm"] <= 0.05 + 1e-8
    assert result["ogd_rank"] == 1
    assert result["grad_features"] is not None
    assert len(result["grad_features"]) == 3
    for before, after in zip(state_before, _snapshot_state(agent, "trace")):
        torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)
    for ticket_id, expected_logits in zip(ticket_ids, committed_logits):
        record = agent._pending_tickets[ticket_id]
        torch.testing.assert_close(
            record["actual_output"].policy_logits,
            expected_logits,
            rtol=0.0,
            atol=0.0,
        )
        assert record["task_resolved"] is True
        agent.cancel_ticket_branch(ticket_id, "utility")
    for name, parameter in agent.utility_gate.named_parameters():
        torch.testing.assert_close(parameter, gate_before[name], rtol=0.0, atol=0.0)
    assert agent.pending_ticket_count == 0
    logging.info("✓ future-only loss reaches early injection/transition without label leakage")
    return True


def test_trajectory_guard_rolls_back_parameters_and_ogd() -> bool:
    logging.info("Running test_trajectory_guard_rolls_back_parameters_and_ogd...")
    torch.manual_seed(20260831)
    agent = TemporalUtilityMQRAgent(
        8,
        board_size=3,
        ring_dim=5,
        latent_dim=7,
        leak_rates=(1.0, 0.1),
        injection_rank=4,
        base_unitary_init="random",
        base_unitary_seed=23,
        write_warmup_observations=4,
        task_lr=0.2,
        ogd_max_rank=2,
        safety_limits=SidecarSafetyLimits(max_output_linf_drift=0.0),
    )
    tickets = []
    for _index in range(2):
        tickets.append(
            int(
                agent.commit_step(
                    torch.randn(1, 8), stream_id="trace-guard"
                )["ticket_id"]
            )
        )
    parameters_before = {
        name: parameter.detach().clone() for name, parameter in agent.named_parameters()
    }
    state_before = _snapshot_state(agent, "trace-guard")
    version_before = int(agent.online_parameter_version.item())
    result = agent.apply_trajectory_feedback(
        tickets,
        [
            {"action": torch.tensor([1])},
            {"action": torch.tensor([4])},
        ],
        loss_scales=(0.0, 1.0),
        remember_gradient=True,
        return_grad_features=True,
    )
    assert result["rolled_back"] is True
    assert result["did_update"] is False
    assert result["update_norm"] == 0.0
    assert result["grad_features"] is None
    assert result["external_gradient_authorized"] is False
    assert "output_linf_drift" in result["safety_violations"]
    assert agent.task_gradient_memory.rank == 0
    assert int(agent.online_parameter_version.item()) == version_before
    for name, parameter in agent.named_parameters():
        torch.testing.assert_close(
            parameter, parameters_before[name], rtol=0.0, atol=0.0
        )
    for before, after in zip(state_before, _snapshot_state(agent, "trace-guard")):
        torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)
    for ticket in tickets:
        agent.cancel_ticket_branch(ticket, "utility")
    logging.info("✓ unsafe trace update atomically restores parameters and OGD")
    return True


def test_nonflat_gate_and_pending_trace_checkpoint_roundtrip() -> bool:
    logging.info("Running test_nonflat_gate_and_pending_trace_checkpoint_roundtrip...")
    torch.manual_seed(20260901)
    kwargs = {
        "board_size": 3,
        "ring_dim": 5,
        "latent_dim": 7,
        "leak_rates": (1.0, 0.1),
        "injection_rank": 4,
        "cayley_coordinate_mode": "minimal",
        "base_unitary_init": "random",
        "base_unitary_scale": 0.25,
        "utility_content_dim": 3,
        "write_warmup_observations": 4,
        "task_lr": 0.02,
        "max_trace_horizon": 4,
    }
    agent = TemporalUtilityMQRAgent(
        8,
        base_unitary_seed=31,
        utility_content_seed=37,
        **kwargs,
    )
    tickets = []
    for _index in range(2):
        tickets.append(
            int(
                agent.commit_step(
                    torch.randn(1, 8), stream_id="resume"
                )["ticket_id"]
            )
        )
    buffer = io.BytesIO()
    torch.save(agent.state_dict(), buffer)
    buffer.seek(0)
    restored = TemporalUtilityMQRAgent(
        8,
        base_unitary_seed=999,
        utility_content_seed=998,
        **kwargs,
    )
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    assert restored.pending_ticket_count == agent.pending_ticket_count == 2
    assert list(restored._pending_tickets) == list(agent._pending_tickets) == tickets
    torch.testing.assert_close(
        restored.core._base_unitaries,
        agent.core._base_unitaries,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        restored.utility_content_projection,
        agent.utility_content_projection,
        rtol=0.0,
        atol=0.0,
    )
    for left, right in zip(
        restored._stream_states["resume"].rings,
        agent._stream_states["resume"].rings,
    ):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    feedback = [
        {"action": torch.tensor([2])},
        {"action": torch.tensor([5])},
    ]
    left_result = agent.apply_trajectory_feedback(
        tickets, feedback, loss_scales=(0.0, 1.0)
    )
    right_result = restored.apply_trajectory_feedback(
        tickets, feedback, loss_scales=(0.0, 1.0)
    )
    assert left_result["loss"] == right_result["loss"]
    for left, right in zip(agent.parameters(), restored.parameters()):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    for ticket in tickets:
        agent.cancel_ticket_branch(ticket, "utility")
        restored.cancel_ticket_branch(ticket, "utility")
    logging.info("✓ non-flat base, content gate, and pending trace resume exactly")
    return True


def test_awr_ppo_and_gae_objectives() -> bool:
    logging.info("Running test_awr_ppo_and_gae_objectives...")
    agent = _agent()
    agent.legality_policy_scale = 1.0
    x = torch.randn(1, 8)
    state = agent.core.zero_state(1, device=x.device, dtype=x.dtype)
    output, _ = agent._transition(x, state, slow_write=True)
    action = torch.tensor([2])
    old_log_prob = torch.log_softmax(output.policy_logits.detach(), dim=1)[:, 2]
    losses = agent.compute_go_loss(
        output,
        action,
        awr_advantage=torch.tensor([1.0]),
        ppo_advantage=torch.tensor([-0.5]),
        old_log_prob=old_log_prob,
        legality_target=torch.tensor(
            [[1, 1, 1, 0, 0, 1, 1, 0, 1]], dtype=torch.float32
        ),
        reference_log_probs=torch.log_softmax(output.policy_logits.detach(), dim=1),
        weights=GoLossWeights(
            placement=0.0,
            legality=0.0,
            pass_decision=0.0,
            value=0.0,
            awr=1.0,
            ppo=1.0,
            entropy=0.01,
            reference_kl=0.1,
            illegal_mass=0.5,
        ),
    )
    assert torch.isfinite(losses["total"])
    assert 0.0 <= float(losses["illegal_mass"].item()) <= 1.0
    gradients = torch.autograd.grad(
        losses["total"],
        tuple(parameter for parameter in agent.parameters() if parameter.requires_grad),
        allow_unused=True,
        retain_graph=True,
    )
    assert any(gradient is not None and bool((gradient != 0).any()) for gradient in gradients)
    legality_gradients = torch.autograd.grad(
        output.policy_logits.sum(),
        tuple(agent.legality_head.parameters()),
        allow_unused=False,
    )
    assert all(bool((gradient != 0).any()) for gradient in legality_gradients)

    rewards = torch.tensor([[0.0], [0.0], [1.0]])
    values = torch.zeros_like(rewards)
    dones = torch.tensor([[0.0], [0.0], [1.0]])
    advantage, returns = generalized_advantage_estimate(
        rewards, values, dones, gamma=1.0, gae_lambda=1.0
    )
    torch.testing.assert_close(advantage, torch.ones_like(advantage))
    torch.testing.assert_close(returns, torch.ones_like(returns))
    logging.info("✓ AWR/PPO gradients and exact GAE return")
    return True


def test_slow_lora_schedule() -> bool:
    logging.info("Running test_slow_lora_schedule...")
    schedule = SlowLoRAConsolidationSchedule(
        warmup_updates=4,
        interval=3,
        minimum_abs_advantage=0.2,
    )
    assert not schedule.should_consolidate(3, write_advantage=1.0)
    assert not schedule.should_consolidate(4, write_advantage=0.1)
    assert schedule.should_consolidate(4, write_advantage=-0.3)
    assert not schedule.should_consolidate(5, write_advantage=1.0)
    assert schedule.should_consolidate(7, write_advantage=0.2)

    class MockEncoder:
        def __init__(self) -> None:
            self.calls = 0

        def step_from_external_gradient(self, features, grad_features, **kwargs):
            self.calls += 1
            assert features.requires_grad
            assert kwargs["lr"] == 0.01
            assert kwargs["orthogonal_memory"] is not None
            assert kwargs["max_grad_norm"] == 0.5
            return {
                "update_norm": float(grad_features.norm().item()),
                "ogd_rank": kwargs["orthogonal_memory"].rank,
            }

    encoder = MockEncoder()
    consolidator = SlowLoRAConsolidator(
        schedule,
        lr=0.01,
        ogd_max_rank=2,
        max_grad_norm=0.5,
    )
    features = torch.randn(1, 8, requires_grad=True)
    gradient = torch.full_like(features, 0.25)
    skipped = consolidator.maybe_step(
        encoder,
        features,
        gradient,
        update_index=3,
        write_advantage=1.0,
    )
    assert skipped["scheduled"] is False
    assert skipped["did_update"] is False
    assert encoder.calls == 0
    applied = consolidator.maybe_step(
        encoder,
        features,
        gradient,
        update_index=4,
        write_advantage=0.3,
    )
    assert applied["scheduled"] is True
    assert applied["did_update"] is True
    assert applied["updates"] == 1
    assert encoder.calls == 1
    assert int(consolidator.opportunities.item()) == 2
    logging.info("✓ slow LoRA schedule and external-gradient transaction")
    return True


def test_shared_agent_contract_for_non_mqr_cores() -> bool:
    logging.info("Running test_shared_agent_contract_for_non_mqr_cores...")
    cores = {
        "gru": MultiTimescaleGRUCore(8, 4, 7, leak_rates=(1.0, 0.2, 0.05)),
        "fast_weight": MultiTimescaleFastWeightCore(
            8, 2, 7, leak_rates=(1.0, 0.2, 0.05)
        ),
    }
    audits = []
    for index, (name, core) in enumerate(cores.items()):
        torch.manual_seed(100 + index)
        agent = TemporalUtilityMQRAgent(
            8,
            board_size=3,
            latent_dim=7,
            task_lr=0.01,
            utility_lr=0.05,
            core=core,
        )
        x = torch.randn(1, 8)
        result = agent.commit_step(x, stream_id="shared")
        ticket = int(result["ticket_id"])
        feedback = agent.apply_feedback(
            ticket,
            torch.tensor([1]),
            legality_target=torch.ones(1, 9),
        )
        agent.cancel_ticket_branch(ticket, "utility")
        assert feedback["did_update"] is True
        assert result["policy_logits"].shape == (1, 10)
        audits.append(audit_agent_resources(agent, name))
    gate = matched_resource_gate(audits, parameter_ratio_limit=10.0, state_ratio_limit=10.0, mac_ratio_limit=10.0)
    assert gate["all_resources_matched"] is True
    assert all(item.utility_parameters == audits[0].utility_parameters for item in audits)
    logging.info("✓ GRU/fast-weight share routing, heads, losses, and resource audit")
    return True


def test_go_simulator_twin_rollout_contract() -> bool:
    logging.info("Running test_go_simulator_twin_rollout_contract...")
    trajectories = generate_go_agent_trajectories(
        2,
        size=3,
        seed=91,
        recorded_moves=6,
        random_move_probability=0.5,
    )
    assert all(
        example.board.is_legal(example.target_action)
        for trajectory in trajectories
        for example in trajectory.examples
    )
    encoder = GoVectorEncoder(3, 8, seed=17)
    assert encoder.base_dim == 33
    agent = _agent()
    first = trajectories[0].examples[0]
    x = encoder.encode_board(first.board)
    committed = agent.commit_step(x, stream_id="twin")
    ticket = int(committed["ticket_id"])
    state_before = _snapshot_state(agent, "twin")
    no_return, write_return = twin_write_returns(
        agent,
        ticket,
        trajectories[0].examples[1:],
        encoder,
        horizon=3,
    )
    assert torch.isfinite(torch.tensor([no_return, write_return])).all()
    for before, after in zip(state_before, _snapshot_state(agent, "twin")):
        torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)
    result = agent.calibrate_write_critic(
        ticket,
        no_write_return=no_return,
        write_return=write_return,
    )
    assert result["write_advantage"] == write_return - no_return
    agent.cancel_ticket_branch(ticket, "task")
    assert agent.pending_ticket_count == 0

    ko_pair = generate_ko_history_pairs(1, seed=3)
    ko_focus = ko_pair[0].examples[-1]
    fresh_focus = ko_pair[1].examples[-1]
    assert ko_focus.focus_point == fresh_focus.focus_point
    assert not ko_focus.board.is_legal(int(ko_focus.focus_point))
    assert fresh_focus.board.is_legal(int(fresh_focus.focus_point))
    assert ko_focus.target_action != fresh_focus.target_action
    torch.testing.assert_close(
        encoder.exact_features(ko_focus.board),
        encoder.exact_features(fresh_focus.board),
        rtol=0.0,
        atol=0.0,
    )

    large_pair = generate_ko_history_pairs(1, size=10, seed=5)
    large_ko = large_pair[0].examples[-1]
    large_fresh = large_pair[1].examples[-1]
    large_encoder = GoVectorEncoder(
        10,
        3 * 10 * 10 + 6,
        seed=19,
        projection_mode="identity",
    )
    assert large_encoder.base_dim == large_encoder.output_dim == 306
    assert not large_ko.board.is_legal(int(large_ko.focus_point))
    assert large_fresh.board.is_legal(int(large_fresh.focus_point))
    assert large_ko.target_action != large_fresh.target_action
    torch.testing.assert_close(
        large_encoder.encode_board(large_ko.board),
        large_encoder.encode_board(large_fresh.board),
        rtol=0.0,
        atol=0.0,
    )
    logging.info("✓ real-rule trajectories and isolated twin-rollout utility target")
    return True


def test_mechanism_repeated_query_accounting_and_student_t_ci() -> bool:
    logging.info("Running repeated-query accounting and CI regression test...")
    from experiments.mqr_mechanism_qualification import (
        _paired_episodes,
        _query_indices,
        _summary,
        build_agents,
        evaluate,
    )

    sequence_length = 12
    pairs = [
        _paired_episodes(
            9000 + index,
            length=sequence_length,
            context=0,
            reversed_mapping=False,
        )
        for index in range(2)
    ]
    agent = build_agents(
        ring_dim=6,
        seed=23,
        task_lr=0.03,
        utility_lr=0.08,
        max_trace_horizon=sequence_length,
        transition_update_interval=2,
        methods=("identity_trace",),
    )["identity_trace"]
    episodes = [episode for pair in pairs for episode in pair]
    metrics = evaluate(agent, episodes)
    query_count = len(_query_indices(sequence_length))
    assert metrics["event_observations"] == len(episodes)
    assert metrics["query_observations"] == len(episodes) * query_count
    assert metrics["distractor_observations"] == len(episodes) * (
        sequence_length - 1 - query_count
    )

    summary = _summary([0.0, 1.0] * 5)
    normal_radius = 1.96 / 6.0
    assert summary["ci_method"] == "two_sided_student_t_95"
    assert float(summary["ci95_high"]) - float(summary["mean"]) > normal_radius
    logging.info("✓ every repeated query is classified and seed CI uses Student-t")
    return True


def run_all_tests() -> int:
    tests = [
        test_preview_commit_and_four_heads,
        test_task_feedback_and_dual_return_critic,
        test_task_update_guard_rolls_back_parameters_ogd_and_external_gradient,
        test_cayley_updates_follow_explicit_slow_schedule_with_ogd,
        test_trajectory_future_credit_is_causal_and_reaches_memory_content,
        test_trajectory_guard_rolls_back_parameters_and_ogd,
        test_nonflat_gate_and_pending_trace_checkpoint_roundtrip,
        test_awr_ppo_and_gae_objectives,
        test_slow_lora_schedule,
        test_shared_agent_contract_for_non_mqr_cores,
        test_go_simulator_twin_rollout_contract,
        test_mechanism_repeated_query_accounting_and_student_t_ci,
    ]
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception:
            logging.exception("✗ %s failed", test.__name__)
    logging.info("Unified-agent test results: %d passed, %d failed", passed, len(tests) - passed)
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
