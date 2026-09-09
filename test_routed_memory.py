"""Deterministic tests for residual address-routing MQR memory."""

from __future__ import annotations

import logging
import math

import torch

from mqr.routing import (
    BalancedAdvantageReplay,
    BudgetedMultiTimescaleUtilityGate,
    ContextualAddressRouterBank,
    ResidualAddressRouter,
    RoutedMemoryState,
    RoutedSlotMemory,
    utility_calibration_metrics,
)
from experiments.mqr_contextual_topology_qualification import (
    build_memories,
    counterfactual_candidate_advantages,
    generate_context_topologies,
    generate_matched_batch,
    marginal_audit,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def test_residual_transition_is_doubly_stochastic() -> bool:
    logging.info("Running residual transition invariant test...")
    torch.manual_seed(20260901)
    configurations = (
        {"route_family": "identity"},
        {
            "route_family": "cyclic_givens",
            "givens_layers": 3,
            "givens_base_angle": 0.31,
        },
        {"route_family": "block_cayley", "block_size": 2},
        {"route_family": "dense_cayley"},
        {"route_family": "local_permutation", "permutation_shift": 1},
    )
    for configuration in configurations:
        router = ResidualAddressRouter(4, epsilon=0.73, **configuration).double()
        transition = router.transition_matrix(dtype=torch.float64)
        torch.testing.assert_close(
            transition.sum(dim=1), torch.ones(4, dtype=torch.float64), rtol=1e-12, atol=1e-12
        )
        torch.testing.assert_close(
            transition.sum(dim=0), torch.ones(4, dtype=torch.float64), rtol=1e-12, atol=1e-12
        )
        assert float(transition.amin().item()) >= -1e-14
        diagnostics = router.stochastic_diagnostics()
        assert diagnostics["identity_channel_weight"] == 1.0 - diagnostics["epsilon"]
        assert diagnostics["row_sum_max_error"] < 1e-12
        assert diagnostics["column_sum_max_error"] < 1e-12
    logging.info("✓ residual routes are non-negative and doubly stochastic")
    return True


def test_residual_formula_and_learned_epsilon_gradient() -> bool:
    logging.info("Running residual formula and epsilon-gradient test...")
    router = ResidualAddressRouter(
        4,
        route_family="local_permutation",
        epsilon=0.25,
    ).double()
    address = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    expected = torch.tensor([[0.75, 0.25, 0.0, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(router(address), expected, rtol=0.0, atol=0.0)

    learned = ResidualAddressRouter(
        4,
        route_family="local_permutation",
        epsilon=0.6,
        learn_epsilon=True,
        epsilon_floor=0.05,
    ).double()
    loss = learned(address)[0, 1]
    loss.backward()
    assert learned.epsilon_logit is not None
    assert learned.epsilon_logit.grad is not None
    assert float(learned.epsilon_logit.grad.abs().item()) > 0.0
    assert 0.05 < float(learned.epsilon().item()) < 0.95
    logging.info("✓ T_epsilon formula and bounded epsilon gradient")
    return True


def test_content_address_separation_and_identity_ablation() -> bool:
    logging.info("Running content/address separation test...")
    router = ResidualAddressRouter(
        4,
        route_family="local_permutation",
        epsilon=1.0,
        permutation_shift=1,
    )
    memory = RoutedSlotMemory(4, 2, router=router, read_mode="hard")
    content = torch.tensor(
        [[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]]
    )
    address = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    state = RoutedMemoryState(content.clone(), address)
    original = state.content.clone()

    routed = memory.route(state)
    assert torch.equal(routed.content, original)
    torch.testing.assert_close(memory.read(routed), content[:, 1], rtol=0.0, atol=0.0)
    restored = memory.route(routed, transpose=True)
    torch.testing.assert_close(restored.address, address, rtol=0.0, atol=0.0)
    torch.testing.assert_close(memory.read(restored), content[:, 0], rtol=0.0, atol=0.0)

    replaced = memory.route(state, identity_override=True)
    torch.testing.assert_close(memory.read(replaced), content[:, 0], rtol=0.0, atol=0.0)
    assert not torch.equal(memory.read(routed), memory.read(replaced))

    current = state
    for _ in range(97):
        current = memory.route(current)
    assert torch.equal(current.content, original)
    logging.info("✓ routes change addresses only; identity replacement changes behavior")
    return True


def test_projected_residual_address_does_not_diffuse() -> bool:
    logging.info("Running projected long-horizon address test...")
    router = ResidualAddressRouter(
        8,
        route_family="local_permutation",
        epsilon=0.75,
        permutation_shift=1,
    ).double()
    memory = RoutedSlotMemory(
        8,
        1,
        router=router,
        route_projection="straight_through_top1",
    )
    state = memory.zero_state(
        1, device=torch.device("cpu"), dtype=torch.float64
    )
    state = RoutedMemoryState(
        state.content,
        state.address.detach().requires_grad_(True),
    )
    initial_address = state.address
    for step in range(1, 513):
        state = memory.route(state)
        assert int(state.address.argmax(dim=1).item()) == step % 8
    state.address[0, 0].backward()
    assert initial_address.grad is not None
    assert bool(torch.isfinite(initial_address.grad).all())
    logging.info("✓ nonlinear address commit preserves the route for 512 steps")
    return True


def test_hard_write_preserves_unaddressed_slots_and_keeps_gradient() -> bool:
    logging.info("Running hard slot-write test...")
    router = ResidualAddressRouter(4, route_family="local_permutation", epsilon=1.0)
    memory = RoutedSlotMemory(4, 2, router=router)
    initial = memory.zero_state(1, device=torch.device("cpu"), dtype=torch.float64)
    soft_address = torch.tensor(
        [[0.1, 0.2, 0.6, 0.1]], dtype=torch.float64, requires_grad=True
    )
    state = RoutedMemoryState(initial.content, soft_address)
    value = torch.tensor([[7.0, -3.0]], dtype=torch.float64)
    written = memory.write(state, value)
    torch.testing.assert_close(written.content[:, 2], value, rtol=0.0, atol=0.0)
    assert torch.equal(written.content[:, 0], initial.content[:, 0])
    assert torch.equal(written.content[:, 1], initial.content[:, 1])
    assert torch.equal(written.content[:, 3], initial.content[:, 3])
    written.content.square().sum().backward()
    assert soft_address.grad is not None
    assert float(soft_address.grad.abs().sum().item()) > 0.0
    logging.info("✓ forward writes are discrete while route credit remains connected")
    return True


def test_sparse_and_block_costs_remove_dense_cubic_peak() -> bool:
    logging.info("Running routing resource audit test...")
    dense = ResidualAddressRouter(32, route_family="dense_cayley")
    block = ResidualAddressRouter(32, route_family="block_cayley", block_size=4)
    sparse = ResidualAddressRouter(32, route_family="cyclic_givens", givens_layers=2)
    dense_cost = dense.cost_profile(content_dim=8, refresh_interval=16)
    block_cost = block.cost_profile(content_dim=8, refresh_interval=16)
    sparse_cost = sparse.cost_profile(content_dim=8, refresh_interval=16)
    assert block_cost.refresh_macs < dense_cost.refresh_macs
    assert sparse_cost.refresh_macs < block_cost.refresh_macs
    assert block_cost.peak_workspace_bytes < dense_cost.peak_workspace_bytes
    assert sparse_cost.peak_workspace_bytes < dense_cost.peak_workspace_bytes
    assert dense_cost.total_state_bytes == block_cost.total_state_bytes
    assert dense_cost.total_state_bytes == sparse_cost.total_state_bytes
    assert math.isclose(
        sparse_cost.amortized_refresh_macs,
        sparse_cost.refresh_macs / 16.0,
    )
    logging.info("✓ sparse/block variants report lower refresh and peak costs")
    return True


def test_givens_and_sparse_permutation_transition_equivalence() -> bool:
    logging.info("Running Givens/direct sparse-route equivalence test...")
    angles = torch.tensor(
        [[0.21, 0.47, 0.82, 1.03], [0.37, 0.64, 0.91, 1.17]],
        dtype=torch.float64,
    )
    givens = ResidualAddressRouter(
        8,
        route_family="cyclic_givens",
        epsilon=0.73,
        givens_layers=2,
    ).double()
    sparse = ResidualAddressRouter(
        8,
        route_family="learnable_sparse_permutation",
        epsilon=0.73,
        givens_layers=2,
    ).double()
    with torch.no_grad():
        givens.givens_angles.copy_(angles)
        probability = angles.sin().square()
        sparse.sparse_permutation_logits.copy_(torch.logit(probability))
    for transpose in (False, True):
        torch.testing.assert_close(
            givens.transition_matrix(dtype=torch.float64, transpose=transpose),
            sparse.transition_matrix(dtype=torch.float64, transpose=transpose),
            rtol=1e-12,
            atol=1e-12,
        )
    givens_cost = givens.cost_profile(content_dim=1, refresh_interval=8)
    sparse_cost = sparse.cost_profile(content_dim=1, refresh_interval=8)
    assert givens_cost.trainable_parameters == sparse_cost.trainable_parameters
    assert givens_cost.forward_macs == sparse_cost.forward_macs
    assert givens_cost.refresh_macs == sparse_cost.refresh_macs
    assert givens_cost.peak_workspace_bytes == sparse_cost.peak_workspace_bytes
    logging.info("✓ local route families match exactly after p=sin(theta)^2")
    return True


def test_contextual_route_bank_isolates_inactive_context() -> bool:
    logging.info("Running contextual route-bank isolation test...")
    torch.manual_seed(20260904)
    bank = ContextualAddressRouterBank(
        2,
        8,
        route_family="cyclic_givens",
        epsilon=0.75,
        givens_layers=1,
    ).double()
    inactive_before = {
        name: value.detach().clone()
        for name, value in bank.routers[1].state_dict().items()
    }
    address = torch.eye(8, dtype=torch.float64)
    context_ids = torch.zeros(8, dtype=torch.long)
    target = torch.roll(address, shifts=1, dims=1)
    optimizer = torch.optim.SGD(bank.parameters(), lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    loss = (bank(address, context_ids) - target).square().sum()
    loss.backward()
    assert bank.routers[0].givens_angles.grad is not None
    assert float(bank.routers[0].givens_angles.grad.abs().sum().item()) > 0.0
    assert bank.routers[1].givens_angles.grad is None
    optimizer.step()
    for name, value in bank.routers[1].state_dict().items():
        torch.testing.assert_close(value, inactive_before[name], rtol=0.0, atol=0.0)
    mixed = bank(
        address,
        torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long),
    )
    assert mixed.shape == address.shape
    profile = bank.cost_profile(content_dim=1, refresh_interval=8)
    assert profile["contexts"] == 2
    assert profile["persistent_trainable_parameters"] == 8
    logging.info("✓ active context updates; inactive route is bitwise unchanged")
    return True


def test_contextual_topology_batch_matches_marginals_and_future_utility() -> bool:
    logging.info("Running matched-marginal contextual-topology generator test...")
    topologies, swap_bits = generate_context_topologies(seed=113, slots=8)
    assert not torch.equal(topologies[0], topologies[1])
    assert bool((swap_bits.sum(dim=1) == 2).all())
    for topology in topologies:
        assert torch.equal(torch.sort(topology).values, torch.arange(8))
    batch = generate_matched_batch(
        seed=991,
        batch_size=8,
        slots=8,
        query_cycles=3,
        context_id=0,
        topology=topologies[0],
    )
    audit = marginal_audit(batch, 8)
    assert audit["joint_marginals_exactly_matched"]
    assert audit["joint_histogram_l1_difference"] == 0
    assert audit["key_histogram_l1_difference"] == 0
    assert audit["positive_value_count_difference"] == 0
    assert audit["event_position_count_difference"] == 0
    advantages = counterfactual_candidate_advantages(batch, 3)
    event_advantage = advantages[batch.event_mask]
    distractor_advantage = advantages[~batch.event_mask]
    assert bool((event_advantage > distractor_advantage).all())
    oracle_decisions = batch.event_mask
    content, _ = build_memories(batch, oracle_decisions, 8)
    torch.testing.assert_close(content, batch.values_by_source, rtol=0.0, atol=0.0)
    wrong_content, _ = build_memories(batch, ~oracle_decisions, 8)
    assert bool((wrong_content != batch.values_by_source).any())
    logging.info("✓ identical marginals, relational utility, and nontrivial contexts")
    return True


def test_budget_is_hard_and_prefix_valid() -> bool:
    logging.info("Running hard write-budget test...")
    torch.manual_seed(20260902)
    gate = BudgetedMultiTimescaleUtilityGate(
        3,
        horizons=(1, 4, 16),
        hidden_dim=4,
        write_budget=0.25,
    )
    with torch.no_grad():
        for parameter in gate.parameters():
            parameter.zero_()
        gate.network[-1].bias.fill_(4.0)
    state = gate.zero_budget_state(2, device=torch.device("cpu"))
    features = torch.zeros(2, 3)
    decisions = []
    for step in range(1, 17):
        decision, state, diagnostics = gate.decide(features, state)
        decisions.append(decision)
        allowance = math.ceil(0.25 * step)
        assert bool((state.writes <= allowance).all())
        assert bool((diagnostics["budget_slack"] >= 0).all())
    stacked = torch.stack(decisions)
    assert int(stacked[:, 0].sum().item()) == 4
    assert int(stacked[:, 1].sum().item()) == 4
    logging.info("✓ every prefix respects the explicit cumulative write budget")
    return True


def test_balanced_multiscale_loss_replay_and_calibration() -> bool:
    logging.info("Running utility balance/calibration test...")
    torch.manual_seed(20260903)
    gate = BudgetedMultiTimescaleUtilityGate(
        3,
        horizons=(1, 4),
        hidden_dim=6,
        write_budget=0.5,
    ).double()
    features = torch.randn(6, 3, dtype=torch.float64)
    advantages = torch.tensor(
        [[2.0, 1.0], [1.0, 0.5], [-1.0, -0.5], [-2.0, -1.0], [-3.0, -2.0], [-4.0, -3.0]],
        dtype=torch.float64,
    )
    before = gate.balanced_loss(features, advantages)
    before["loss"].backward()
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in gate.parameters()
    )
    assert 0.0 < float(before["positive_fraction"].item()) < 1.0

    replay = BalancedAdvantageReplay(3, 2, capacity_per_sign=8)
    replay.add(features.float(), advantages.float())
    sampled_features, sampled_targets = replay.sample(
        6, generator=torch.Generator().manual_seed(7)
    )
    assert sampled_features.shape == (6, 3)
    assert sampled_targets.shape == (6, 2)
    assert int((sampled_targets.mean(dim=1) > 0.0).sum().item()) == 3
    assert replay.counts == {"positive": 2, "non_positive": 4}

    probabilities = torch.tensor([0.95, 0.8, 0.3, 0.1])
    scalar_advantages = torch.tensor([2.0, 1.0, -1.0, -2.0])
    decisions = torch.tensor([True, True, False, False])
    event_mask = torch.tensor([True, True, False, False])
    metrics = utility_calibration_metrics(
        probabilities,
        scalar_advantages,
        decisions=decisions,
        event_mask=event_mask,
        write_budget=0.5,
        bins=4,
    )
    assert metrics["auprc"] == 1.0
    assert metrics["event_distractor_write_gap"] == 1.0
    assert metrics["budget_violation"] == 0.0
    assert metrics["brier"] < 0.1
    logging.info("✓ balanced replay, multi-horizon loss, AUPRC/Brier/ECE/write gap")
    return True


def run_all_tests() -> bool:
    tests = [
        test_residual_transition_is_doubly_stochastic,
        test_residual_formula_and_learned_epsilon_gradient,
        test_content_address_separation_and_identity_ablation,
        test_projected_residual_address_does_not_diffuse,
        test_hard_write_preserves_unaddressed_slots_and_keeps_gradient,
        test_sparse_and_block_costs_remove_dense_cubic_peak,
        test_givens_and_sparse_permutation_transition_equivalence,
        test_contextual_route_bank_isolates_inactive_context,
        test_contextual_topology_batch_matches_marginals_and_future_utility,
        test_budget_is_hard_and_prefix_valid,
        test_balanced_multiscale_loss_replay_and_calibration,
    ]
    for test in tests:
        test()
    logging.info("All %d routed-memory tests passed", len(tests))
    return True


if __name__ == "__main__":
    run_all_tests()
