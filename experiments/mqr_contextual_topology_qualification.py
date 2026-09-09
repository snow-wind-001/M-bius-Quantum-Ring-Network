"""Phase-IV qualification on unknown context-dependent memory topology.

This experiment removes the two shortcuts in the Phase-III routed-memory
study.  First, each opaque context owns an unknown, seed-dependent sparse
topology that must be learned from query loss; no route direction or topology
label reaches the learner.  Second, event and distractor candidate tokens are
exact permutations of the same ``(key, value)`` multiset.  Their one-step
marginals are therefore identical, and write relevance is determined only by
the relation between a previous cue and the current key.

The primary MQR is a residual sparse-Givens route.  A direct transition-space
sparse-permutation learner uses the same factor graph, parameter count,
feedback, optimizer steps, and analytic route resources.  Block-Cayley,
identity, identity replacement, a direct ring buffer, and an oracle are also
reported.  Formal evidence requires at least ten seeds and 64 held-out
episodes per seed.  Smaller runs are smoke tests and cannot support claims.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mqr.routing import (
    ContextualAddressRouterBank,
    ResidualAddressRouter,
    utility_calibration_metrics,
)


TRAINABLE_METHODS = (
    "mqr_sparse_givens",
    "mqr_block_cayley",
    "learnable_sparse_permutation",
)
EVALUATED_METHODS = (
    *TRAINABLE_METHODS,
    "identity",
    "mqr_identity_replacement",
    "direct_ring_buffer",
    "oracle_topology",
)


@dataclass(frozen=True)
class MatchedTopologyBatch:
    """One context batch with exactly matched candidate marginals."""

    context_id: int
    cue_keys: torch.Tensor
    candidate_keys: torch.Tensor
    candidate_values: torch.Tensor
    event_mask: torch.Tensor
    values_by_source: torch.Tensor
    query_destinations: torch.Tensor
    query_sources: torch.Tensor
    query_targets: torch.Tensor


class RelationalWriteGate(nn.Module):
    """Causal gate that receives raw previous-cue/current-candidate pairs."""

    def __init__(self, slots: int, hidden_dim: int) -> None:
        super().__init__()
        self.slots = int(slots)
        self.feature_dim = 2 * self.slots + 1
        self.network = nn.Sequential(
            nn.Linear(self.feature_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 2 or features.size(1) != self.feature_dim:
            raise ValueError(
                f"features must have shape [batch, {self.feature_dim}]"
            )
        return self.network(features).squeeze(1)


def _paired_swap_permutation(swap_bits: torch.Tensor, slots: int) -> torch.Tensor:
    if swap_bits.shape != (slots // 2,):
        raise ValueError("swap_bits have an incompatible shape")
    permutation = torch.arange(slots, dtype=torch.long)
    for block, enabled in enumerate(swap_bits.tolist()):
        if bool(enabled):
            left = 2 * block
            permutation[left], permutation[left + 1] = (
                permutation[left + 1].clone(),
                permutation[left].clone(),
            )
    return permutation


def generate_context_topologies(
    *, seed: int, slots: int, contexts: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return balanced, distinct sparse topologies unknown to the learners."""

    if slots < 4 or slots % 4:
        raise ValueError("slots must be a positive multiple of four")
    if contexts != 2:
        raise ValueError("the A -> B -> A protocol currently requires two contexts")
    blocks = slots // 2
    generator = torch.Generator().manual_seed(int(seed) * 104729 + 17)
    selected = torch.randperm(blocks, generator=generator)[: blocks // 2]
    bits_a = torch.zeros(blocks, dtype=torch.bool)
    bits_a[selected] = True
    bits_b = ~bits_a
    bits = torch.stack([bits_a, bits_b])
    topologies = torch.stack(
        [_paired_swap_permutation(row, slots) for row in bits]
    )
    return topologies, bits


def generate_matched_batch(
    *,
    seed: int,
    batch_size: int,
    slots: int,
    query_cycles: int,
    context_id: int,
    topology: torch.Tensor,
) -> MatchedTopologyBatch:
    """Generate candidate events/distractors with identical joint marginals.

    In every lane, event pairs are ``(source, value[source])``.  Distractors
    are a nontrivial cyclic permutation of exactly those same pairs.  Candidate
    order is balanced.  The event is the candidate whose key equals the cue,
    so utility depends on a causal cross-token relation rather than a marker.
    """

    if batch_size <= 0 or slots <= 0 or slots % 2 or query_cycles <= 0:
        raise ValueError("invalid matched-batch dimensions")
    if topology.shape != (slots,) or topology.dtype != torch.long:
        raise ValueError("topology must be a long permutation vector")
    generator = torch.Generator().manual_seed(int(seed))
    cue_keys = torch.empty(batch_size, slots, dtype=torch.long)
    values_by_source = torch.empty(batch_size, slots)
    candidate_keys = torch.empty(batch_size, slots, 2, dtype=torch.long)
    candidate_values = torch.empty(batch_size, slots, 2)
    event_mask = torch.zeros(batch_size, slots, 2, dtype=torch.bool)
    query_sources = torch.empty(
        batch_size, slots * query_cycles, dtype=torch.long
    )
    for lane in range(batch_size):
        cue_order = torch.randperm(slots, generator=generator)
        cue_keys[lane] = cue_order
        # Unique signed levels make every wrong relational pairing incur a
        # nonzero future MSE while retaining exactly balanced signs.
        magnitudes = torch.linspace(0.25, 1.0, slots // 2)
        balanced_values = torch.cat([-magnitudes.flip(0), magnitudes])
        balanced_values = balanced_values[
            torch.randperm(slots, generator=generator)
        ]
        values_by_source[lane] = balanced_values
        distractor_shift = int(
            torch.randint(1, slots, (1,), generator=generator).item()
        )
        distractor_sources = torch.roll(cue_order, shifts=distractor_shift)
        positions = torch.tensor([0, 1] * (slots // 2), dtype=torch.long)
        positions = positions[torch.randperm(slots, generator=generator)]
        for block in range(slots):
            event_position = int(positions[block].item())
            distractor_position = 1 - event_position
            source = int(cue_order[block].item())
            distractor_source = int(distractor_sources[block].item())
            candidate_keys[lane, block, event_position] = source
            candidate_values[lane, block, event_position] = balanced_values[source]
            candidate_keys[lane, block, distractor_position] = distractor_source
            candidate_values[lane, block, distractor_position] = balanced_values[
                distractor_source
            ]
            event_mask[lane, block, event_position] = True
        cycles = [torch.randperm(slots, generator=generator) for _ in range(query_cycles)]
        query_sources[lane] = torch.cat(cycles)
    query_destinations = topology[query_sources]
    query_targets = values_by_source.gather(1, query_sources)
    return MatchedTopologyBatch(
        context_id=int(context_id),
        cue_keys=cue_keys,
        candidate_keys=candidate_keys,
        candidate_values=candidate_values,
        event_mask=event_mask,
        values_by_source=values_by_source,
        query_destinations=query_destinations,
        query_sources=query_sources,
        query_targets=query_targets,
    )


def marginal_audit(batch: MatchedTopologyBatch, slots: int) -> Dict[str, Any]:
    """Recompute exact empirical marginal equality from candidate tensors."""

    event_keys = batch.candidate_keys[batch.event_mask]
    distractor_keys = batch.candidate_keys[~batch.event_mask]
    event_values = batch.candidate_values[batch.event_mask]
    distractor_values = batch.candidate_values[~batch.event_mask]
    event_joint = 2 * event_keys + (event_values > 0.0).to(torch.long)
    distractor_joint = 2 * distractor_keys + (
        distractor_values > 0.0
    ).to(torch.long)
    event_joint_hist = torch.bincount(event_joint, minlength=2 * slots)
    distractor_joint_hist = torch.bincount(
        distractor_joint, minlength=2 * slots
    )
    event_key_hist = torch.bincount(event_keys, minlength=slots)
    distractor_key_hist = torch.bincount(distractor_keys, minlength=slots)
    event_positive = int((event_values > 0.0).sum().item())
    distractor_positive = int((distractor_values > 0.0).sum().item())
    joint_difference = int(
        (event_joint_hist - distractor_joint_hist).abs().sum().item()
    )
    key_difference = int(
        (event_key_hist - distractor_key_hist).abs().sum().item()
    )
    position_counts = batch.event_mask.to(torch.long).sum(dim=(0, 1))
    digest = hashlib.sha256(
        torch.stack([event_joint_hist, distractor_joint_hist])
        .numpy()
        .tobytes()
    ).hexdigest()
    return {
        "candidate_count_per_class": int(event_keys.numel()),
        "joint_histogram_l1_difference": joint_difference,
        "key_histogram_l1_difference": key_difference,
        "positive_value_count_difference": abs(
            event_positive - distractor_positive
        ),
        "absolute_magnitude_mean_difference": abs(
            float(event_values.abs().mean().item())
            - float(distractor_values.abs().mean().item())
        ),
        "event_position_count_difference": int(
            abs(int(position_counts[0].item()) - int(position_counts[1].item()))
        ),
        "joint_marginals_exactly_matched": joint_difference == 0,
        "marginal_digest": digest,
    }


def gate_features(batch: MatchedTopologyBatch, slots: int) -> torch.Tensor:
    cue = F.one_hot(batch.cue_keys, num_classes=slots).to(torch.float32)
    cue = cue.unsqueeze(2).expand(-1, -1, 2, -1)
    candidate = F.one_hot(
        batch.candidate_keys, num_classes=slots
    ).to(torch.float32)
    value = batch.candidate_values.unsqueeze(3)
    return torch.cat([cue, candidate, value], dim=3)


def counterfactual_candidate_advantages(
    batch: MatchedTopologyBatch, query_cycles: int
) -> torch.Tensor:
    """Exact future query-loss gain from writing each candidate.

    A candidate is written to the slot named by the previous cue.  The no-write
    branch leaves that slot at zero.  Both branches are then queried
    ``query_cycles`` times with identical futures.  Event labels are not used
    in this target.
    """

    correct = batch.values_by_source.gather(1, batch.cue_keys).unsqueeze(2)
    no_write_loss = correct.square()
    write_loss = (batch.candidate_values - correct).square()
    return float(query_cycles) * (no_write_loss - write_loss)


def fit_write_gate(
    *, seed: int, args: argparse.Namespace
) -> Tuple[RelationalWriteGate, Dict[str, Any]]:
    torch.manual_seed(int(seed) + 1000)
    gate = RelationalWriteGate(args.slots, args.gate_hidden_dim)
    optimizer = torch.optim.Adam(gate.parameters(), lr=args.gate_lr)
    topologies, _ = generate_context_topologies(
        seed=seed, slots=args.slots
    )
    losses = []
    for step in range(args.gate_steps):
        context_id = step % 2
        batch = generate_matched_batch(
            seed=seed * 10_000_000 + 10_000 + step,
            batch_size=args.gate_batch_size,
            slots=args.slots,
            query_cycles=args.query_cycles,
            context_id=context_id,
            topology=topologies[context_id],
        )
        features = gate_features(batch, args.slots).reshape(
            -1, gate.feature_dim
        )
        targets = counterfactual_candidate_advantages(
            batch, args.query_cycles
        ).reshape(-1)
        optimizer.zero_grad(set_to_none=True)
        predicted_advantage = gate(features)
        positive = targets > 0.0
        row_weight = torch.ones_like(targets)
        if bool(positive.any()) and bool((~positive).any()):
            row_weight[positive] = 0.5 / positive.sum().to(targets.dtype)
            row_weight[~positive] = 0.5 / (~positive).sum().to(targets.dtype)
            row_weight = row_weight * row_weight.numel()
        regression = F.smooth_l1_loss(
            predicted_advantage, targets, reduction="none"
        )
        calibration = F.binary_cross_entropy_with_logits(
            predicted_advantage,
            positive.to(predicted_advantage.dtype),
            reduction="none",
        )
        loss = (row_weight * (regression + 0.25 * calibration)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().item()))
    quarter = max(1, len(losses) // 4)
    return gate.eval(), {
        "train_steps": args.gate_steps,
        "first_quarter_loss": _mean(losses[:quarter]),
        "last_quarter_loss": _mean(losses[-quarter:]),
        "trainable_parameters": sum(
            parameter.numel() for parameter in gate.parameters()
        ),
    }


@torch.no_grad()
def gate_outputs(
    gate: RelationalWriteGate,
    batch: MatchedTopologyBatch,
    slots: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    features = gate_features(batch, slots)
    logits = gate(features.reshape(-1, gate.feature_dim)).reshape(
        batch.cue_keys.size(0), slots, 2
    )
    probabilities = torch.sigmoid(logits)
    selected = logits.argmax(dim=2)
    decisions = F.one_hot(selected, num_classes=2).bool()
    return decisions, probabilities


@torch.no_grad()
def build_memories(
    batch: MatchedTopologyBatch,
    decisions: torch.Tensor,
    slots: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build keyed slot contents and an arrival-ordered direct ring."""

    if decisions.shape != batch.event_mask.shape:
        raise ValueError("decisions must align with candidate pairs")
    selected = decisions.to(torch.long).argmax(dim=2, keepdim=True)
    values = batch.candidate_values.gather(2, selected).squeeze(2)
    content = torch.zeros(batch.cue_keys.size(0), slots)
    for block in range(slots):
        content.scatter_(
            1,
            batch.cue_keys[:, block : block + 1],
            values[:, block : block + 1],
        )
    return content, values


def _initial_move_probability(angle: float) -> float:
    return math.sin(float(angle)) ** 2


def make_route_bank(
    method: str,
    *,
    seed: int,
    args: argparse.Namespace,
) -> ContextualAddressRouterBank:
    torch.manual_seed(int(seed) + 2000)
    common = {
        "epsilon": args.epsilon,
        "givens_layers": 1,
    }
    if method == "mqr_sparse_givens":
        bank = ContextualAddressRouterBank(
            2,
            args.slots,
            route_family="cyclic_givens",
            givens_base_angle=args.route_base_angle,
            **common,
        )
    elif method == "learnable_sparse_permutation":
        bank = ContextualAddressRouterBank(
            2,
            args.slots,
            route_family="learnable_sparse_permutation",
            sparse_permutation_probability=_initial_move_probability(
                args.route_base_angle
            ),
            **common,
        )
    elif method == "mqr_block_cayley":
        bank = ContextualAddressRouterBank(
            2,
            args.slots,
            route_family="block_cayley",
            block_size=2,
            cayley_coordinate_mode="minimal",
            **common,
        )
        # Open the modulus-square Jacobian at the same initial move
        # probability as the local Givens and sparse-permutation routes.
        cayley_scale = math.tan(args.route_base_angle / 2.0)
        real = torch.tensor(
            [[0.0, -cayley_scale], [cayley_scale, 0.0]],
            dtype=torch.double,
        )
        skew = torch.complex(real, torch.zeros_like(real))
        with torch.no_grad():
            for router in bank.routers:
                for block in router.cayley_blocks:
                    block.set_from_skew_hermitian_(skew)
    else:
        raise ValueError(f"unknown trainable method: {method}")
    return bank


def _route_weights(
    bank: ContextualAddressRouterBank,
    destinations: torch.Tensor,
    context_id: int,
    slots: int,
    *,
    hard_forward: bool,
    identity_override: bool = False,
) -> torch.Tensor:
    one_hot = F.one_hot(destinations, num_classes=slots).to(torch.float32)
    flat = one_hot.reshape(-1, slots)
    contexts = torch.full(
        (flat.size(0),), int(context_id), dtype=torch.long
    )
    weights = bank(
        flat,
        contexts,
        transpose=True,
        identity_override=identity_override,
    ).reshape(*destinations.shape, slots)
    if hard_forward:
        hard = F.one_hot(weights.argmax(dim=2), num_classes=slots).to(weights)
        weights = hard + weights - weights.detach()
    return weights


def _predict_route(
    bank: ContextualAddressRouterBank,
    batch: MatchedTopologyBatch,
    content: torch.Tensor,
    slots: int,
    *,
    hard_forward: bool,
    identity_override: bool = False,
) -> torch.Tensor:
    weights = _route_weights(
        bank,
        batch.query_destinations,
        batch.context_id,
        slots,
        hard_forward=hard_forward,
        identity_override=identity_override,
    )
    return torch.einsum("bqs,bs->bq", weights, content)


def _binary_accuracy(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(((prediction >= 0.0) == (target > 0.0)).float().mean().item())


def _parameter_vector(
    bank: ContextualAddressRouterBank, context_id: int
) -> torch.Tensor:
    values = [
        parameter.detach().reshape(-1).cpu()
        for parameter in bank.routers[int(context_id)].parameters()
    ]
    return torch.cat(values) if values else torch.empty(0)


@torch.no_grad()
def route_recovery(
    bank: ContextualAddressRouterBank,
    topologies: torch.Tensor,
) -> Dict[str, float]:
    per_context = []
    stochastic_errors = []
    for context_id in range(topologies.size(0)):
        transition = bank.transition_matrix(context_id, dtype=torch.float64)
        predicted = transition.argmax(dim=1)
        per_context.append(
            float((predicted == topologies[context_id]).double().mean().item())
        )
        stochastic_errors.append(
            max(
                float((transition.sum(dim=0) - 1.0).abs().amax().item()),
                float((transition.sum(dim=1) - 1.0).abs().amax().item()),
            )
        )
    return {
        "mean": _mean(per_context),
        "context_a": per_context[0],
        "context_b": per_context[1],
        "doubly_stochastic_max_error": max(stochastic_errors),
    }


def train_context(
    bank: ContextualAddressRouterBank,
    gate: RelationalWriteGate,
    topology: torch.Tensor,
    *,
    context_id: int,
    seed: int,
    steps: int,
    learning_rate: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    optimizer = torch.optim.Adam(bank.parameters(), lr=float(learning_rate))
    losses = []
    prequential_accuracy = []
    for step in range(steps):
        batch = generate_matched_batch(
            seed=seed * 10_000_000
            + 100_000 * (context_id + 1)
            + step,
            batch_size=args.route_batch_size,
            slots=args.slots,
            query_cycles=args.query_cycles,
            context_id=context_id,
            topology=topology,
        )
        with torch.no_grad():
            decisions, _ = gate_outputs(gate, batch, args.slots)
            content, _ = build_memories(batch, decisions, args.slots)
        optimizer.zero_grad(set_to_none=True)
        prediction = _predict_route(
            bank,
            batch,
            content,
            args.slots,
            hard_forward=False,
        )
        prequential_accuracy.append(
            _binary_accuracy(prediction.detach(), batch.query_targets)
        )
        loss = F.mse_loss(prediction, batch.query_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bank.parameters(), args.route_grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().item()))
    quarter = max(1, len(losses) // 4)
    return {
        "steps": steps,
        "learning_rate": float(learning_rate),
        "first_quarter_loss": _mean(losses[:quarter]),
        "last_quarter_loss": _mean(losses[-quarter:]),
        "prequential_accuracy_curve": prequential_accuracy,
        "prequential_regret": _mean([1.0 - value for value in prequential_accuracy]),
        "last_quarter_prequential_accuracy": _mean(
            prequential_accuracy[-quarter:]
        ),
    }


@torch.no_grad()
def evaluate_method(
    method: str,
    bank: ContextualAddressRouterBank | None,
    batch: MatchedTopologyBatch,
    gate: RelationalWriteGate,
    topology: torch.Tensor,
    *,
    args: argparse.Namespace,
) -> Dict[str, float]:
    decisions, _ = gate_outputs(gate, batch, args.slots)
    content, ring_values = build_memories(batch, decisions, args.slots)
    started = time.perf_counter()
    if method in TRAINABLE_METHODS:
        if bank is None:
            raise ValueError("trainable route requires a bank")
        prediction = _predict_route(
            bank,
            batch,
            content,
            args.slots,
            hard_forward=True,
        )
    elif method in {"identity", "mqr_identity_replacement"}:
        prediction = content.gather(1, batch.query_destinations)
    elif method == "direct_ring_buffer":
        indices = torch.arange(batch.query_destinations.size(1)) % args.slots
        prediction = ring_values[:, indices]
    elif method == "oracle_topology":
        inverse = torch.empty_like(topology)
        inverse[topology] = torch.arange(args.slots)
        source = inverse[batch.query_destinations]
        prediction = content.gather(1, source)
    else:
        raise ValueError(f"unknown evaluated method: {method}")
    elapsed = time.perf_counter() - started
    first_cycle = slice(0, args.slots)
    repeated = slice(args.slots, None)
    return {
        "accuracy": _binary_accuracy(prediction, batch.query_targets),
        "mse": float(F.mse_loss(prediction, batch.query_targets).item()),
        "first_query_cycle_accuracy": _binary_accuracy(
            prediction[:, first_cycle], batch.query_targets[:, first_cycle]
        ),
        "repeated_query_accuracy": _binary_accuracy(
            prediction[:, repeated], batch.query_targets[:, repeated]
        )
        if args.query_cycles > 1
        else _binary_accuracy(prediction, batch.query_targets),
        "runtime_ms_per_episode": 1000.0 * elapsed / batch.cue_keys.size(0),
    }


def _route_learning_rate(method: str, args: argparse.Namespace) -> float:
    if method == "mqr_sparse_givens":
        return args.givens_lr
    if method == "mqr_block_cayley":
        return args.block_cayley_lr
    if method == "learnable_sparse_permutation":
        return args.sparse_permutation_lr
    raise ValueError("method has no route learning rate")


def resource_audit(args: argparse.Namespace) -> Dict[str, Any]:
    families = {
        "mqr_sparse_givens": {
            "route_family": "cyclic_givens",
            "givens_layers": 1,
        },
        "mqr_block_cayley": {
            "route_family": "block_cayley",
            "block_size": 2,
        },
        "learnable_sparse_permutation": {
            "route_family": "learnable_sparse_permutation",
            "givens_layers": 1,
            "sparse_permutation_probability": _initial_move_probability(
                args.route_base_angle
            ),
        },
    }
    methods: Dict[str, Dict[str, float | int]] = {}
    for method, configuration in families.items():
        router = ResidualAddressRouter(
            args.slots,
            epsilon=args.epsilon,
            **configuration,
        )
        cost = router.cost_profile(
            content_dim=1,
            refresh_interval=1,
        )
        persistent_parameters = 2 * cost.trainable_parameters
        methods[method] = {
            "persistent_trainable_parameters": persistent_parameters,
            "optimizer_state_bytes": 8 * persistent_parameters,
            "memory_state_bytes": 4 * args.slots,
            "active_forward_macs_per_query": cost.forward_macs + args.slots,
            "active_update_macs_per_query": 3 * cost.forward_macs
            + cost.refresh_macs,
            "peak_workspace_bytes": cost.peak_workspace_bytes
            + 4 * args.slots,
        }
    methods["identity"] = {
        "persistent_trainable_parameters": 0,
        "optimizer_state_bytes": 0,
        "memory_state_bytes": 4 * args.slots,
        "active_forward_macs_per_query": 0,
        "active_update_macs_per_query": 0,
        "peak_workspace_bytes": 4 * args.slots,
    }
    methods["direct_ring_buffer"] = {
        "persistent_trainable_parameters": 0,
        "optimizer_state_bytes": 0,
        "memory_state_bytes": 4 * args.slots + 8,
        "active_forward_macs_per_query": 0,
        "active_update_macs_per_query": 0,
        "peak_workspace_bytes": 4 * args.slots + 8,
    }
    fields = (
        "persistent_trainable_parameters",
        "optimizer_state_bytes",
        "memory_state_bytes",
        "active_forward_macs_per_query",
        "active_update_macs_per_query",
        "peak_workspace_bytes",
    )
    parity = {
        field: float(methods["mqr_sparse_givens"][field])
        / max(float(methods["learnable_sparse_permutation"][field]), 1.0)
        for field in fields
    }
    return {
        "methods": methods,
        "mqr_givens_to_sparse_permutation_ratios": parity,
        "same_factor_graph_parameters_and_resources_within_1_05": all(
            1.0 / 1.05 <= value <= 1.05 for value in parity.values()
        ),
        "direct_ring_is_deliberately_resource_superior_diagnostic": True,
        "shared_write_gate_excluded_from_route_core_ratios": True,
        "ratio_semantics": (
            "primary Givens route divided by direct transition-space sparse "
            "route; no shared-module dilution"
        ),
    }


def run_seed(args: argparse.Namespace, seed: int) -> Dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    topologies, swap_bits = generate_context_topologies(
        seed=seed, slots=args.slots
    )
    gate, gate_training = fit_write_gate(seed=seed, args=args)
    evaluation_batches = [
        generate_matched_batch(
            seed=seed * 10_000_000 + 9_000_000 + context_id,
            batch_size=args.eval_episodes,
            slots=args.slots,
            query_cycles=args.query_cycles,
            context_id=context_id,
            topology=topologies[context_id],
        )
        for context_id in range(2)
    ]
    audits = [marginal_audit(batch, args.slots) for batch in evaluation_batches]
    all_probabilities = []
    all_advantages = []
    all_decisions = []
    all_events = []
    for batch in evaluation_batches:
        decisions, probabilities = gate_outputs(gate, batch, args.slots)
        all_probabilities.append(probabilities.reshape(-1))
        all_advantages.append(
            counterfactual_candidate_advantages(
                batch, args.query_cycles
            ).reshape(-1)
        )
        all_decisions.append(decisions.reshape(-1))
        all_events.append(batch.event_mask.reshape(-1))
    gate_metrics = utility_calibration_metrics(
        torch.cat(all_probabilities),
        torch.cat(all_advantages),
        decisions=torch.cat(all_decisions),
        event_mask=torch.cat(all_events),
        write_budget=0.5,
    )

    banks: Dict[str, ContextualAddressRouterBank] = {}
    training: Dict[str, Any] = {}
    phase_metrics: Dict[str, Any] = {}
    for method in TRAINABLE_METHODS:
        bank = make_route_bank(method, seed=seed, args=args)
        banks[method] = bank
        initial_a = _parameter_vector(bank, 0)
        phase_a = train_context(
            bank,
            gate,
            topologies[0],
            context_id=0,
            seed=seed,
            steps=args.phase_a_steps,
            learning_rate=_route_learning_rate(method, args),
            args=args,
        )
        a_before = evaluate_method(
            method,
            bank,
            evaluation_batches[0],
            gate,
            topologies[0],
            args=args,
        )
        a_snapshot = _parameter_vector(bank, 0)
        b_zero_shot = evaluate_method(
            method,
            bank,
            evaluation_batches[1],
            gate,
            topologies[1],
            args=args,
        )
        phase_b = train_context(
            bank,
            gate,
            topologies[1],
            context_id=1,
            seed=seed,
            steps=args.phase_b_steps,
            learning_rate=_route_learning_rate(method, args),
            args=args,
        )
        b_after = evaluate_method(
            method,
            bank,
            evaluation_batches[1],
            gate,
            topologies[1],
            args=args,
        )
        a_after = evaluate_method(
            method,
            bank,
            evaluation_batches[0],
            gate,
            topologies[0],
            args=args,
        )
        a_after_vector = _parameter_vector(bank, 0)
        training[method] = {
            "context_a": phase_a,
            "context_b": phase_b,
            "context_a_parameter_drift_from_initial_linf": float(
                (a_snapshot - initial_a).abs().amax().item()
            ),
            "context_a_parameter_drift_during_b_linf": float(
                (a_after_vector - a_snapshot).abs().amax().item()
            ),
        }
        phase_metrics[method] = {
            "context_a_before_switch": a_before,
            "context_b_zero_shot": b_zero_shot,
            "context_b_after_adaptation": b_after,
            "context_a_after_return": a_after,
            "context_a_forgetting": a_before["accuracy"] - a_after["accuracy"],
            "final_mean_accuracy": 0.5
            * (b_after["accuracy"] + a_after["accuracy"]),
            "final_mean_repeated_query_accuracy": 0.5
            * (
                b_after["repeated_query_accuracy"]
                + a_after["repeated_query_accuracy"]
            ),
            "route_recovery": route_recovery(bank, topologies),
        }

    for method in ("identity", "direct_ring_buffer", "oracle_topology"):
        a = evaluate_method(
            method,
            None,
            evaluation_batches[0],
            gate,
            topologies[0],
            args=args,
        )
        b = evaluate_method(
            method,
            None,
            evaluation_batches[1],
            gate,
            topologies[1],
            args=args,
        )
        phase_metrics[method] = {
            "context_a_before_switch": a,
            "context_b_zero_shot": b,
            "context_b_after_adaptation": b,
            "context_a_after_return": a,
            "context_a_forgetting": 0.0,
            "final_mean_accuracy": 0.5 * (a["accuracy"] + b["accuracy"]),
            "final_mean_repeated_query_accuracy": 0.5
            * (a["repeated_query_accuracy"] + b["repeated_query_accuracy"]),
        }
    identity_replacement_a = evaluate_method(
        "mqr_identity_replacement",
        banks["mqr_sparse_givens"],
        evaluation_batches[0],
        gate,
        topologies[0],
        args=args,
    )
    identity_replacement_b = evaluate_method(
        "mqr_identity_replacement",
        banks["mqr_sparse_givens"],
        evaluation_batches[1],
        gate,
        topologies[1],
        args=args,
    )
    phase_metrics["mqr_identity_replacement"] = {
        "context_a_before_switch": identity_replacement_a,
        "context_b_zero_shot": identity_replacement_b,
        "context_b_after_adaptation": identity_replacement_b,
        "context_a_after_return": identity_replacement_a,
        "context_a_forgetting": 0.0,
        "final_mean_accuracy": 0.5
        * (identity_replacement_a["accuracy"] + identity_replacement_b["accuracy"]),
        "final_mean_repeated_query_accuracy": 0.5
        * (
            identity_replacement_a["repeated_query_accuracy"]
            + identity_replacement_b["repeated_query_accuracy"]
        ),
    }
    return {
        "seed": seed,
        "topologies": topologies.tolist(),
        "swap_bits": swap_bits.to(torch.long).tolist(),
        "marginal_audits": audits,
        "gate_training": gate_training,
        "gate_metrics": gate_metrics,
        "training": training,
        "methods": phase_metrics,
        "resource_audit": resource_audit(args),
    }


def _run_seed_entry(payload: Tuple[argparse.Namespace, int]) -> Dict[str, Any]:
    args, seed = payload
    torch.set_num_threads(1)
    return run_seed(args, seed)


def _mean(values: Iterable[float]) -> float:
    parsed = [float(value) for value in values]
    return sum(parsed) / len(parsed)


def _student_t_critical_95(sample_count: int) -> float:
    table = {
        1: float("inf"),
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
        11: 2.228,
        12: 2.201,
        13: 2.179,
        14: 2.160,
        15: 2.145,
        16: 2.131,
        17: 2.120,
        18: 2.110,
        19: 2.101,
        20: 2.093,
    }
    return table.get(sample_count, 1.96)


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    tensor = torch.tensor(tuple(float(value) for value in values), dtype=torch.double)
    count = int(tensor.numel())
    mean = float(tensor.mean().item())
    if count <= 1:
        return {
            "count": count,
            "mean": mean,
            "std": 0.0,
            "ci95_low": mean,
            "ci95_high": mean,
            "interval": "singleton",
        }
    std = float(tensor.std(unbiased=True).item())
    radius = _student_t_critical_95(count) * std / math.sqrt(count)
    return {
        "count": count,
        "mean": mean,
        "std": std,
        "ci95_low": mean - radius,
        "ci95_high": mean + radius,
        "interval": "two_sided_student_t_95",
    }


def aggregate(
    args: argparse.Namespace, runs: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    method_summaries: Dict[str, Any] = {}
    for method in EVALUATED_METHODS:
        method_summaries[method] = {
            metric: _summary([run["methods"][method][metric] for run in runs])
            for metric in (
                "final_mean_accuracy",
                "final_mean_repeated_query_accuracy",
                "context_a_forgetting",
            )
        }
        if method in TRAINABLE_METHODS:
            method_summaries[method]["route_recovery"] = _summary(
                [
                    run["methods"][method]["route_recovery"]["mean"]
                    for run in runs
                ]
            )
            method_summaries[method]["context_b_prequential_regret"] = _summary(
                [
                    run["training"][method]["context_b"]["prequential_regret"]
                    for run in runs
                ]
            )
    primary = "mqr_sparse_givens"
    controls = (
        "identity",
        "learnable_sparse_permutation",
        "direct_ring_buffer",
        "mqr_identity_replacement",
    )
    paired = {
        control: _summary(
            [
                run["methods"][primary]["final_mean_accuracy"]
                - run["methods"][control]["final_mean_accuracy"]
                for run in runs
            ]
        )
        for control in controls
    }
    gate_summaries = {
        metric: _summary([run["gate_metrics"][metric] for run in runs])
        for metric in (
            "auprc",
            "brier",
            "ece",
            "event_write_rate",
            "distractor_write_rate",
            "event_distractor_write_gap",
            "write_rate",
            "budget_violation",
        )
    }
    matched = all(
        bool(audit["joint_marginals_exactly_matched"])
        and audit["joint_histogram_l1_difference"] == 0
        and audit["key_histogram_l1_difference"] == 0
        and audit["positive_value_count_difference"] == 0
        and audit["absolute_magnitude_mean_difference"] == 0.0
        for run in runs
        for audit in run["marginal_audits"]
    )
    resource_parity = all(
        bool(
            run["resource_audit"][
                "same_factor_graph_parameters_and_resources_within_1_05"
            ]
        )
        for run in runs
    )
    primary_summary = method_summaries[primary]
    block_summary = method_summaries["mqr_block_cayley"]
    mechanism_gates = {
        "formal_seed_count_at_least_10": len(runs) >= 10,
        "held_out_episodes_at_least_64": args.eval_episodes >= 64,
        "opaque_context_topologies_are_distinct": all(
            run["topologies"][0] != run["topologies"][1] for run in runs
        ),
        "topology_not_provided_to_route_learner": True,
        "predict_before_route_update": True,
        "event_distractor_joint_marginals_exactly_matched": matched,
        "write_gap_ci_positive": gate_summaries[
            "event_distractor_write_gap"
        ]["ci95_low"]
        > 0.0,
        "hard_pairwise_write_budget_never_violated": gate_summaries[
            "budget_violation"
        ]["ci95_high"]
        <= 1e-12,
        "givens_route_recovery_ci_above_chance": primary_summary[
            "route_recovery"
        ]["ci95_low"]
        > 0.5,
        "block_cayley_route_recovery_ci_above_chance": block_summary[
            "route_recovery"
        ]["ci95_low"]
        > 0.5,
        "primary_accuracy_ci_above_0_80": primary_summary[
            "final_mean_accuracy"
        ]["ci95_low"]
        > 0.8,
        "beats_identity": paired["identity"]["ci95_low"] > 0.0,
        "identity_replacement_hurts": paired[
            "mqr_identity_replacement"
        ]["ci95_low"]
        > 0.0,
        "context_a_forgetting_ci_not_above_0_01": primary_summary[
            "context_a_forgetting"
        ]["ci95_high"]
        <= 0.01,
        "equal_route_resource_gate_vs_sparse_permutation": resource_parity,
    }
    mechanism_gates["contextual_route_mechanism_qualified"] = all(
        mechanism_gates.values()
    )
    independent_gates = {
        "contextual_route_mechanism_qualified": mechanism_gates[
            "contextual_route_mechanism_qualified"
        ],
        "beats_learnable_sparse_permutation": paired[
            "learnable_sparse_permutation"
        ]["ci95_low"]
        > 0.0,
        "beats_direct_ring_buffer": paired["direct_ring_buffer"]["ci95_low"]
        > 0.0,
    }
    independent_gates["mqr_independent_algorithm_advantage"] = all(
        independent_gates.values()
    )
    return {
        "method_summaries": method_summaries,
        "paired_primary_mqr_minus_control_accuracy": paired,
        "gate_summaries": gate_summaries,
        "mechanism_gates": mechanism_gates,
        "contextual_route_mechanism_qualified": bool(
            mechanism_gates["contextual_route_mechanism_qualified"]
        ),
        "independent_advantage_gates": independent_gates,
        "mqr_independent_algorithm_advantage": bool(
            independent_gates["mqr_independent_algorithm_advantage"]
        ),
        "claim_boundary": (
            "Passing the independent gate would support only this matched-"
            "marginal contextual routing benchmark. Go, MiniCPM, policy RL, "
            "general intelligence, and animal-like learning remain out of scope."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=101)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--query-cycles", type=int, default=3)
    parser.add_argument("--delay-tokens", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=0.75)
    parser.add_argument("--route-base-angle", type=float, default=0.35)
    parser.add_argument("--gate-hidden-dim", type=int, default=32)
    parser.add_argument("--gate-steps", type=int, default=160)
    parser.add_argument("--gate-batch-size", type=int, default=32)
    parser.add_argument("--gate-lr", type=float, default=0.01)
    parser.add_argument("--phase-a-steps", type=int, default=96)
    parser.add_argument("--phase-b-steps", type=int, default=96)
    parser.add_argument("--route-batch-size", type=int, default=32)
    parser.add_argument("--givens-lr", type=float, default=0.03)
    parser.add_argument("--block-cayley-lr", type=float, default=0.03)
    parser.add_argument("--sparse-permutation-lr", type=float, default=0.1)
    parser.add_argument("--route-grad-clip", type=float, default=5.0)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "analysis/results/mqr_contextual_topology_qualification_smoke.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds <= 0 or args.workers <= 0 or args.eval_episodes <= 0:
        raise ValueError("seeds, workers, and eval_episodes must be positive")
    if args.slots < 4 or args.slots % 4:
        raise ValueError("slots must be a multiple of four")
    if args.query_cycles < 2:
        raise ValueError("formal repeated-query evaluation requires at least two cycles")
    if args.delay_tokens < 4 * args.slots:
        raise ValueError("delay_tokens must represent at least four slot cycles")
    if not 0.5 < args.epsilon < 1.0:
        raise ValueError("hard residual route selection requires epsilon in (0.5, 1)")
    torch.set_num_threads(1)
    seeds = [args.seed_start + index for index in range(args.seeds)]
    if args.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(args.workers, args.seeds)
        ) as executor:
            runs = list(
                executor.map(
                    _run_seed_entry,
                    ((args, seed) for seed in seeds),
                )
            )
    else:
        runs = [run_seed(args, seed) for seed in seeds]
    formal = (
        args.seeds >= 10
        and args.eval_episodes >= 64
        and args.phase_a_steps >= 64
        and args.phase_b_steps >= 64
    )
    result = {
        "schema_version": 1,
        "status": "formal" if formal else "smoke_or_pilot",
        "protocol": {
            "phase": "IV unknown context-dependent topology",
            "primary_method": "mqr_sparse_givens",
            "contexts": 2,
            "context_schedule": "A train -> B online adapt -> A retained probe",
            "context_identifier_semantics": "opaque route-bank index",
            "topology_family": "unknown balanced disjoint local swaps",
            "topology_supervision_to_learner": False,
            "route_feedback": "held-out query value MSE only",
            "predict_before_update": True,
            "event_definition": "candidate key equals previous cue key",
            "candidate_feature_input": "raw previous cue one-hot, current key one-hot, value",
            "explicit_event_marker_in_gate": False,
            "event_distractor_marginal_contract": (
                "exact per-lane permutation of the same (key,value) multiset"
            ),
            "write_critic_target": (
                "exact finite future query-loss gain of candidate write versus no-write"
            ),
            "write_budget": "exactly one of two candidates per cue",
            "residual_transition": "(1-epsilon) I + epsilon T",
            "inference_projection": "straight_through_top1 during training; hard top1 at evaluation",
            "strong_route_control": "learnable sparse permutation on identical factor graph",
            "diagnostic_control": "direct arrival-ordered ring buffer",
            "route_resource_gate": (
                "Givens/sparse-permutation ratio within [1/1.05,1.05] on six axes; shared gate excluded"
            ),
            "go_minicpm_policy_rl_frozen_until_gate": True,
            "slots": args.slots,
            "query_cycles": args.query_cycles,
            "delay_tokens": args.delay_tokens,
            "epsilon": args.epsilon,
            "route_base_angle": args.route_base_angle,
            "seeds": args.seeds,
            "seed_start": args.seed_start,
            "gate_steps": args.gate_steps,
            "phase_a_steps": args.phase_a_steps,
            "phase_b_steps": args.phase_b_steps,
            "route_batch_size": args.route_batch_size,
            "eval_episodes": args.eval_episodes,
            "learning_rates": {
                "mqr_sparse_givens": args.givens_lr,
                "mqr_block_cayley": args.block_cayley_lr,
                "learnable_sparse_permutation": args.sparse_permutation_lr,
            },
        },
        "runs": runs,
        "aggregate": aggregate(args, runs),
        "preserved_phase_iii_evidence": (
            "analysis/results/mqr_routed_memory_qualification_formal_10seed.json"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
