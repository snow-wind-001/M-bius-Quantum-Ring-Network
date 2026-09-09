"""Deterministic contract tests for the formal competitive Go benchmark."""

from __future__ import annotations

import logging
import sys

import torch

from experiments.unified_temporal_mqr_go_competitive import (
    LORA_FAMILY,
    METHODS,
    ByteCappedReplay,
    build_agents,
    initial_equivalence,
    input_dim,
    resource_audits,
)
from mqr.baselines import SinkhornDoublyStochasticParam


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def test_implicit_sinkhorn_gradient_matches_converged_unroll() -> bool:
    logging.info("Running implicit Sinkhorn gradient check...")
    torch.manual_seed(20260828)
    parameter = SinkhornDoublyStochasticParam(
        4,
        iterations=300,
        temperature=0.7,
        init_scale=0.2,
    ).double()
    cotangent = torch.randn(4, 4, dtype=torch.float64)
    implicit = parameter.doubly_stochastic_implicit()
    (implicit * cotangent).sum().backward()
    implicit_gradient = parameter.logits.grad.detach().clone()
    parameter.logits.grad = None
    unrolled = parameter.doubly_stochastic(iterations=300)
    (unrolled * cotangent).sum().backward()
    unrolled_gradient = parameter.logits.grad.detach().clone()
    torch.testing.assert_close(implicit, unrolled, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        implicit_gradient,
        unrolled_gradient,
        rtol=1e-10,
        atol=1e-11,
    )
    assert float(implicit_gradient.sum(dim=0).abs().max().item()) < 1e-10
    assert float(implicit_gradient.sum(dim=1).abs().max().item()) < 1e-10
    logging.info("✓ implicit Sinkhorn pullback matches converged autograd")
    return True


def test_all_competitive_cores_share_online_agent_contract() -> bool:
    logging.info("Running all-core online feedback contract...")
    agents = build_agents(10, seed=41, ogd_rank=4)
    features = torch.linspace(-0.5, 0.5, input_dim(10)).reshape(1, -1)
    for method, agent in agents.items():
        result = agent.commit_step(features, stream_id=method)
        ticket = int(result["ticket_id"])
        feedback = agent.apply_feedback(
            ticket,
            torch.tensor([44]),
            legality_target=torch.ones(1, 100),
            value_target=torch.tensor([0.25]),
            remember_gradient=method not in LORA_FAMILY or method == "ogd_lora",
            project_with_memory=False,
        )
        agent.cancel_ticket_branch(ticket, "utility")
        assert result["policy_logits"].shape == (1, 101)
        assert feedback["prediction_before_update"] is True
        assert feedback["did_update"] is True
        assert agent.pending_ticket_count == 0
        assert len(agent._stream_states[method].rings) == 3
        assert all(value.shape == (1, 4) for value in agent._stream_states[method].rings)
        if method == "sinkhorn_ogd":
            assert feedback["max_stochastic_error"] < 1e-5
        if method == "mqr_unistochastic_ogd":
            assert feedback["max_unitary_error"] < 1e-5
    logging.info("✓ all nine methods execute the same prequential feedback API")
    return True


def test_formal_resource_gate_and_initialization() -> bool:
    logging.info("Running 10x10/13x13 resource and initialization gate...")
    for board_size in (10, 13):
        agents = build_agents(board_size, seed=7, ogd_rank=4)
        audits, capacity, gate = resource_audits(agents, ogd_rank=4)
        assert tuple(item.method for item in audits) == METHODS
        assert all(item.persistent_state_scalars == 12 for item in audits)
        assert all(item.persistent_state_bytes == 48 for item in audits)
        assert len({item.frozen_parameters for item in audits}) == 1
        assert len({item.allocated_continual_memory_bytes for item in audits}) == 1
        assert capacity > 0
        assert gate["all_resources_matched"] is True
        assert gate["trainable_parameter_ratio"] <= 1.05
        assert gate["agent_forward_mac_ratio"] <= 1.05
        equivalence = initial_equivalence(agents, board_size)
        assert equivalence["all_base_hashes_equal"] is True
        assert equivalence["all_head_hashes_equal"] is True
        assert equivalence["lora_family_hashes_equal"] is True
        assert equivalence["all_adaptive_residuals_zero"] is True
        assert equivalence["initial_policies_exact"] is True
        first = agents[METHODS[0]]
        expected_trainable_task = sum(
            parameter.numel()
            for _name, parameter in first._named_task_parameters()
            if parameter.requires_grad
        )
        assert first.task_parameter_count == expected_trainable_task
        assert first.task_parameter_count < sum(
            parameter.numel()
            for _name, parameter in first._named_task_parameters()
        )
    logging.info("✓ exact 48-byte state and <=1.05 parameter/MAC gates")
    return True


def test_replay_uses_real_bytes_without_padding() -> bool:
    logging.info("Running byte-capped replay check...")
    buffer = ByteCappedReplay(1800)
    features = torch.zeros(1, input_dim(10))
    action = torch.tensor([3])
    legality = torch.ones(1, 100)
    value = torch.tensor([0.5])
    for _ in range(4):
        buffer.append(features, action, legality, value)
        assert buffer.storage_bytes <= buffer.capacity_bytes
    assert buffer.evictions > 0
    assert 0 < buffer.cyclic(0).storage_bytes <= buffer.capacity_bytes
    assert buffer.storage_bytes == sum(item.storage_bytes for item in buffer.items)
    logging.info("✓ replay reports occupied tensor bytes and never pads capacity")
    return True


def run_all_tests() -> int:
    tests = (
        test_implicit_sinkhorn_gradient_matches_converged_unroll,
        test_all_competitive_cores_share_online_agent_contract,
        test_formal_resource_gate_and_initialization,
        test_replay_uses_real_bytes_without_padding,
    )
    passed = 0
    for test in tests:
        try:
            passed += int(test())
        except Exception:
            logging.exception("✗ %s failed", test.__name__)
    logging.info("Competitive Go tests: %d passed, %d failed", passed, len(tests) - passed)
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
