"""Marker-free qualification for residual MQR address routing.

The benchmark is intentionally narrower than Go or language modelling.  It
asks whether a content-preserving cyclic address route is behaviorally used on
long-delay, repeated-query, and context-reversed retrieval.  The utility gate
never receives an event flag: it sees only causal magnitude, novelty, and
prediction-error proxies.  Counterfactual write/no-write returns are used only
as delayed critic targets.

Formal evidence requires at least ten seeds.  Smaller runs are smoke/pilot
diagnostics and must not replace the historical negative Go evidence.
"""

from __future__ import annotations

import argparse
import concurrent.futures
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
    BalancedAdvantageReplay,
    BudgetedMultiTimescaleUtilityGate,
    ResidualAddressRouter,
    RoutedMemoryState,
    RoutedSlotMemory,
    utility_calibration_metrics,
)


FEATURE_DIM = 5
TOKEN_DIM = 4
UTILITY_HORIZONS = (16, 64, 128)


@dataclass(frozen=True)
class SequenceBatch:
    tokens: torch.Tensor
    observations: torch.Tensor
    values: torch.Tensor
    contexts: torch.Tensor
    event_mask: torch.Tensor
    query_mask: torch.Tensor
    repeated_query_mask: torch.Tensor
    targets: torch.Tensor
    query_start: int


class GRUMemoryControl(nn.Module):
    """Standard GRU with exactly the routed-memory recurrent-state bytes."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.cell = nn.GRUCell(TOKEN_DIM, self.hidden_dim)
        self.readout = nn.Linear(self.hidden_dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = tokens.new_zeros(tokens.size(0), self.hidden_dim)
        outputs = []
        for step in range(tokens.size(1)):
            hidden = self.cell(tokens[:, step], hidden)
            outputs.append(self.readout(hidden).squeeze(1))
        return torch.stack(outputs, dim=1)


class FastWeightMemoryControl(nn.Module):
    """Rank-four online key/value matrix with the same 16-float state budget."""

    def __init__(self, memory_dim: int = 4) -> None:
        super().__init__()
        self.memory_dim = int(memory_dim)
        self.key = nn.Linear(TOKEN_DIM, self.memory_dim)
        self.value = nn.Linear(TOKEN_DIM, self.memory_dim)
        self.query = nn.Linear(TOKEN_DIM, self.memory_dim)
        self.readout = nn.Linear(self.memory_dim, 1)
        self.logit_write_strength = nn.Parameter(torch.tensor(1.0))

    def forward(self, tokens: torch.Tensor, decisions: torch.Tensor) -> torch.Tensor:
        batch = tokens.size(0)
        memory = tokens.new_zeros(batch, self.memory_dim, self.memory_dim)
        outputs = []
        strength = torch.sigmoid(self.logit_write_strength)
        for step in range(tokens.size(1)):
            token = tokens[:, step]
            query = F.normalize(self.query(token), dim=1, eps=1e-6)
            read = torch.einsum("bij,bj->bi", memory, query)
            outputs.append(self.readout(read).squeeze(1))
            key = F.normalize(self.key(token), dim=1, eps=1e-6)
            value = torch.tanh(self.value(token))
            update = torch.einsum("bi,bj->bij", value, key)
            gate = decisions[:, step].to(tokens.dtype).view(batch, 1, 1)
            memory = (1.0 - gate * strength) * memory + gate * strength * update
        return torch.stack(outputs, dim=1)


def _modular_inverse(value: int, modulus: int) -> int:
    for candidate in range(modulus):
        if (value * candidate) % modulus == 1:
            return candidate
    raise ValueError("event spacing and slot count must be coprime")


def generate_batch(
    *,
    seed: int,
    batch_size: int,
    slots: int,
    event_spacing: int,
    delay: int,
    query_cycles: int,
    reverse_probability: float,
) -> SequenceBatch:
    """Generate one marker-free delayed cyclic retrieval batch."""

    if math.gcd(slots, event_spacing) != 1:
        raise ValueError("event_spacing must be coprime with slots")
    if delay % slots:
        raise ValueError("delay must be a multiple of slots")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    values = (
        2.0
        * torch.randint(0, 2, (batch_size, slots), generator=generator).float()
        - 1.0
    )
    reverse = torch.rand(batch_size, generator=generator) < float(reverse_probability)
    contexts = torch.where(reverse, -torch.ones(batch_size), torch.ones(batch_size))
    write_length = slots * event_spacing
    query_start = write_length + delay
    query_count = slots * query_cycles
    sequence_length = query_start + query_count
    observations = 0.05 * torch.randn(
        batch_size, sequence_length, generator=generator
    )
    event_mask = torch.zeros(sequence_length, dtype=torch.bool)
    for index in range(slots):
        step = index * event_spacing
        event_mask[step] = True
        observations[:, step] = values[:, index]
    query_mask = torch.zeros(sequence_length, dtype=torch.bool)
    query_mask[query_start:] = True
    repeated_query_mask = torch.zeros(sequence_length, dtype=torch.bool)
    if query_cycles > 1:
        repeated_query_mask[query_start + slots :] = True

    physical_address = torch.empty(batch_size, sequence_length, dtype=torch.long)
    for step in range(sequence_length):
        if step < query_start:
            physical_address[:, step] = step % slots
        else:
            query_index = step - query_start
            direction = contexts.to(torch.long)
            physical_address[:, step] = (direction * query_index) % slots
    phase = 2.0 * math.pi * physical_address.float() / float(slots)
    tokens = torch.stack(
        [observations, contexts[:, None].expand_as(observations), phase.sin(), phase.cos()],
        dim=2,
    )
    targets = torch.zeros_like(observations)
    inverse_spacing = _modular_inverse(event_spacing, slots)
    for step in range(query_start, sequence_length):
        query_index = step - query_start
        physical = (contexts.to(torch.long) * query_index) % slots
        value_index = (inverse_spacing * physical) % slots
        targets[:, step] = values.gather(1, value_index.unsqueeze(1)).squeeze(1)
    return SequenceBatch(
        tokens=tokens,
        observations=observations,
        values=values,
        contexts=contexts,
        event_mask=event_mask,
        query_mask=query_mask,
        repeated_query_mask=repeated_query_mask,
        targets=targets,
        query_start=query_start,
    )


def causal_gate_features(observations: torch.Tensor) -> torch.Tensor:
    """Build causal features without event/query/phase/context indicators."""

    if observations.dim() != 2:
        raise ValueError("observations must be [batch, time]")
    previous = torch.zeros(observations.size(0), dtype=observations.dtype)
    ema = torch.zeros_like(previous)
    ema_abs = torch.zeros_like(previous)
    result = []
    for step in range(observations.size(1)):
        value = observations[:, step]
        magnitude = value.abs()
        novelty = (value - ema).abs()
        prediction_error = (value - previous).abs()
        uncertainty = torch.exp(-2.0 * magnitude)
        result.append(
            torch.stack(
                [magnitude, novelty, prediction_error, ema_abs, uncertainty], dim=1
            )
        )
        ema = 0.9 * ema + 0.1 * value
        ema_abs = 0.9 * ema_abs + 0.1 * magnitude
        previous = value
    return torch.stack(result, dim=1)


def twin_trajectory_advantages(
    batch: SequenceBatch,
    *,
    slots: int,
    horizons: Sequence[int] = UTILITY_HORIZONS,
) -> torch.Tensor:
    """Exact finite-horizon write/no-write return differences.

    At each candidate tick, two content memories branch from the same oracle
    pre-state.  One writes the current observation and the other does not.
    Both then follow identical future oracle writes.  Utility is the reduction
    in future squared query error.  Event labels define that held-fixed teacher
    policy during critic supervision, but never enter the critic features or
    issue-time write decision.
    """

    maximum_horizon = max(int(value) for value in horizons)
    batch_size, sequence_length = batch.observations.shape
    result = torch.zeros(batch_size, sequence_length, len(horizons))
    oracle_pre_content = []
    oracle_pre_pointer = []
    for lane in range(batch_size):
        content = torch.zeros(slots)
        pointer = 0
        lane_contents = []
        lane_pointers = []
        reverse = bool(batch.contexts[lane] < 0.0)
        for step in range(sequence_length):
            lane_contents.append(content.clone())
            lane_pointers.append(pointer)
            if bool(batch.event_mask[step]):
                content[pointer] = batch.observations[lane, step]
            direction = -1 if reverse and step >= batch.query_start else 1
            pointer = (pointer + direction) % slots
        oracle_pre_content.append(lane_contents)
        oracle_pre_pointer.append(lane_pointers)

    for lane in range(batch_size):
        reverse = bool(batch.contexts[lane] < 0.0)
        for candidate in range(sequence_length):
            no_write = oracle_pre_content[lane][candidate].clone()
            yes_write = no_write.clone()
            pointer = oracle_pre_pointer[lane][candidate]
            yes_write[pointer] = batch.observations[lane, candidate]
            direction = -1 if reverse and candidate >= batch.query_start else 1
            pointer = (pointer + direction) % slots
            accumulated = torch.zeros(len(horizons))
            stop = min(sequence_length, candidate + maximum_horizon + 1)
            for future in range(candidate + 1, stop):
                if bool(batch.query_mask[future]):
                    target = batch.targets[lane, future]
                    no_loss = (no_write[pointer] - target).square()
                    yes_loss = (yes_write[pointer] - target).square()
                    delta = no_loss - yes_loss
                    for index, horizon in enumerate(horizons):
                        if future - candidate <= int(horizon):
                            accumulated[index] += delta
                if bool(batch.event_mask[future]):
                    value = batch.observations[lane, future]
                    no_write[pointer] = value
                    yes_write[pointer] = value
                direction = -1 if reverse and future >= batch.query_start else 1
                pointer = (pointer + direction) % slots
            result[lane, candidate] = accumulated
    return result


def fit_utility_gate(
    *,
    seed: int,
    args: argparse.Namespace,
) -> Tuple[BudgetedMultiTimescaleUtilityGate, Dict[str, Any]]:
    torch.manual_seed(seed + 1000)
    gate = BudgetedMultiTimescaleUtilityGate(
        FEATURE_DIM,
        horizons=UTILITY_HORIZONS,
        hidden_dim=args.gate_hidden_dim,
        write_budget=args.write_budget,
        horizon_weights=(0.2, 0.3, 0.5),
    )
    replay = BalancedAdvantageReplay(
        FEATURE_DIM,
        len(UTILITY_HORIZONS),
        capacity_per_sign=max(256, args.gate_episodes * 32),
    )
    for index in range(args.gate_episodes):
        batch = generate_batch(
            seed=seed * 100000 + index,
            batch_size=1,
            slots=args.slots,
            event_spacing=args.event_spacing,
            delay=args.train_delay,
            query_cycles=args.query_cycles,
            reverse_probability=0.5,
        )
        features = causal_gate_features(batch.observations).reshape(-1, FEATURE_DIM)
        targets = twin_trajectory_advantages(batch, slots=args.slots).reshape(
            -1, len(UTILITY_HORIZONS)
        )
        replay.add(features, targets)
    optimizer = torch.optim.Adam(gate.parameters(), lr=args.gate_lr)
    generator = torch.Generator().manual_seed(seed + 2000)
    losses = []
    for _step in range(args.gate_steps):
        features, targets = replay.sample(
            args.gate_batch_size, generator=generator
        )
        optimizer.zero_grad(set_to_none=True)
        terms = gate.balanced_loss(features, targets)
        terms["loss"].backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(), 5.0)
        optimizer.step()
        gate.update_dual_threshold_(
            float(terms["mean_probability"].detach().item()),
            learning_rate=args.dual_lr,
        )
        losses.append(float(terms["loss"].detach().item()))
    return gate.eval(), {
        "replay_counts": replay.counts,
        "replay_bytes": replay.byte_size,
        "first_quarter_loss": _mean(losses[: max(1, len(losses) // 4)]),
        "last_quarter_loss": _mean(losses[-max(1, len(losses) // 4) :]),
        "dual_threshold": float(gate.dual_threshold.item()),
    }


@torch.no_grad()
def gate_decisions(
    gate: BudgetedMultiTimescaleUtilityGate,
    observations: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    features = causal_gate_features(observations)
    budget = gate.zero_budget_state(observations.size(0), device=observations.device)
    decisions = []
    probabilities = []
    for step in range(observations.size(1)):
        decision, budget, diagnostics = gate.decide(features[:, step], budget)
        decisions.append(decision)
        probabilities.append(diagnostics["probability"])
    return torch.stack(decisions, dim=1), torch.stack(probabilities, dim=1)


def filtered_tokens(batch: SequenceBatch, decisions: torch.Tensor) -> torch.Tensor:
    tokens = batch.tokens.clone()
    tokens[:, :, 0] = tokens[:, :, 0] * decisions.to(tokens.dtype)
    return tokens


def _query_loss(logits: torch.Tensor, batch: SequenceBatch) -> torch.Tensor:
    target = (batch.targets[:, batch.query_mask] > 0.0).to(logits.dtype)
    return F.binary_cross_entropy_with_logits(logits[:, batch.query_mask], target)


def train_control(
    method: str,
    gate: BudgetedMultiTimescaleUtilityGate,
    *,
    seed: int,
    args: argparse.Namespace,
) -> nn.Module:
    torch.manual_seed(seed + (3000 if method == "gru" else 4000))
    state_floats = args.slots * 2
    if method == "gru":
        model: nn.Module = GRUMemoryControl(state_floats)
    elif method == "fast_weight":
        memory_dim = int(round(math.sqrt(state_floats)))
        if memory_dim * memory_dim != state_floats:
            raise ValueError("fast-weight state requires a square state-float budget")
        model = FastWeightMemoryControl(memory_dim)
    else:
        raise ValueError("unknown control method")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.control_lr)
    model.train()
    for step in range(args.control_steps):
        batch = generate_batch(
            seed=seed * 1000000 + 50000 + step,
            batch_size=args.control_batch_size,
            slots=args.slots,
            event_spacing=args.event_spacing,
            delay=args.train_delay,
            query_cycles=args.query_cycles,
            reverse_probability=0.5,
        )
        with torch.no_grad():
            decisions, _ = gate_decisions(gate, batch.observations)
        tokens = filtered_tokens(batch, decisions)
        optimizer.zero_grad(set_to_none=True)
        if method == "gru":
            logits = model(tokens)
        else:
            logits = model(tokens, decisions)
        loss = _query_loss(logits, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return model.eval()


@torch.no_grad()
def run_routed_memory(
    batch: SequenceBatch,
    decisions: torch.Tensor,
    *,
    slots: int,
    epsilon: float,
    identity_override: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    router = ResidualAddressRouter(
        slots,
        route_family="local_permutation",
        epsilon=epsilon,
        permutation_shift=1,
    )
    memory = RoutedSlotMemory(slots, 1, router=router)
    state = memory.zero_state(
        batch.observations.size(0),
        device=batch.observations.device,
        dtype=batch.observations.dtype,
    )
    initial_content = state.content.clone()
    outputs = []
    address_correct = []
    core_started = time.perf_counter()
    for step in range(batch.observations.size(1)):
        outputs.append(memory.read(state).squeeze(1))
        if bool(decisions[:, step].any()):
            state = memory.write(
                state,
                batch.observations[:, step : step + 1],
                gate=decisions[:, step].to(batch.observations.dtype),
            )
        forward_address = router(
            state.address, identity_override=identity_override
        )
        reverse_address = router(
            state.address, transpose=True, identity_override=identity_override
        )
        use_reverse = (
            (batch.contexts < 0.0) & (step >= batch.query_start)
        ).unsqueeze(1)
        route_scores = torch.where(use_reverse, reverse_address, forward_address)
        hard_address = F.one_hot(
            route_scores.argmax(dim=1), num_classes=slots
        ).to(route_scores)
        next_address = hard_address + route_scores - route_scores.detach()
        state = RoutedMemoryState(state.content, next_address)
        if not identity_override:
            expected_step = step + 1
            before_query = expected_step <= batch.query_start
            if before_query:
                expected = torch.full(
                    (batch.observations.size(0),),
                    expected_step % slots,
                    dtype=torch.long,
                )
            else:
                query_offset = expected_step - batch.query_start
                expected = (batch.contexts.to(torch.long) * query_offset) % slots
            address_correct.append(state.address.argmax(dim=1) == expected)
    output = torch.stack(outputs, dim=1)
    core_elapsed = time.perf_counter() - core_started
    # A separate no-write route proves that routing alone cannot alter content.
    no_write_state = RoutedMemoryState(initial_content, state.address)
    for _ in range(2 * batch.observations.size(1)):
        no_write_state = memory.route(
            no_write_state, identity_override=identity_override
        )
    content_drift = float((no_write_state.content - initial_content).abs().amax().item())
    return output, {
        "cyclic_address_accuracy": float(
            torch.cat(address_correct).float().mean().item()
        )
        if address_correct
        else 0.0,
        "no_write_content_drift_linf": content_drift,
        "_core_elapsed_seconds": core_elapsed,
    }


def _accuracy(logits_or_values: torch.Tensor, batch: SequenceBatch, mask: torch.Tensor) -> float:
    predictions = logits_or_values[:, mask] >= 0.0
    targets = batch.targets[:, mask] > 0.0
    return float((predictions == targets).float().mean().item())


@torch.no_grad()
def evaluate_method(
    method: str,
    model: nn.Module | None,
    gate: BudgetedMultiTimescaleUtilityGate,
    batch: SequenceBatch,
    *,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    decisions, probabilities = gate_decisions(gate, batch.observations)
    started = time.perf_counter()
    diagnostics: Dict[str, float] = {}
    if method == "mqr":
        values, diagnostics = run_routed_memory(
            batch,
            decisions,
            slots=args.slots,
            epsilon=args.epsilon,
            identity_override=False,
        )
    elif method in {"identity", "mqr_identity_replacement"}:
        values, diagnostics = run_routed_memory(
            batch,
            decisions,
            slots=args.slots,
            epsilon=args.epsilon,
            identity_override=True,
        )
    elif method == "fixed_ring_buffer":
        values, diagnostics = run_routed_memory(
            batch,
            decisions,
            slots=args.slots,
            epsilon=1.0,
            identity_override=False,
        )
    elif method == "gru":
        assert isinstance(model, GRUMemoryControl)
        values = model(filtered_tokens(batch, decisions))
    elif method == "fast_weight":
        assert isinstance(model, FastWeightMemoryControl)
        values = model(filtered_tokens(batch, decisions), decisions)
    else:
        raise ValueError("unknown method")
    elapsed = float(diagnostics.pop("_core_elapsed_seconds", time.perf_counter() - started))
    normal_lanes = batch.contexts > 0.0
    reverse_lanes = batch.contexts < 0.0

    def lane_accuracy(lanes: torch.Tensor, mask: torch.Tensor) -> float:
        if not bool(lanes.any()):
            return float("nan")
        predictions = values[lanes][:, mask] >= 0.0
        targets = batch.targets[lanes][:, mask] > 0.0
        return float((predictions == targets).float().mean().item())

    result: Dict[str, Any] = {
        "accuracy": _accuracy(values, batch, batch.query_mask),
        "forward_accuracy": lane_accuracy(normal_lanes, batch.query_mask),
        "context_reversal_accuracy": lane_accuracy(reverse_lanes, batch.query_mask),
        "repeated_query_accuracy": _accuracy(values, batch, batch.repeated_query_mask),
        "runtime_ms_per_episode": 1000.0 * elapsed / batch.observations.size(0),
        **diagnostics,
    }
    if method == "mqr":
        advantages = twin_trajectory_advantages(batch, slots=args.slots)
        aggregate_advantage = (
            advantages * gate.horizon_weights.view(1, 1, -1)
        ).sum(dim=2)
        result["gate_calibration"] = utility_calibration_metrics(
            probabilities,
            aggregate_advantage,
            decisions=decisions,
            event_mask=batch.event_mask.unsqueeze(0).expand_as(decisions),
            write_budget=args.write_budget,
        )
    return result


def resource_audit(
    gate: BudgetedMultiTimescaleUtilityGate,
    models: Mapping[str, nn.Module | None],
    *,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    gate_parameters = sum(parameter.numel() for parameter in gate.parameters())
    gate_macs = FEATURE_DIM * args.gate_hidden_dim + args.gate_hidden_dim * len(
        UTILITY_HORIZONS
    )
    state_floats = args.slots * 2
    router = ResidualAddressRouter(
        args.slots,
        route_family="local_permutation",
        epsilon=args.epsilon,
    )
    route_cost = router.cost_profile(
        content_dim=1,
        refresh_interval=args.route_refresh_interval,
    )
    identity_route_cost = ResidualAddressRouter(
        args.slots,
        route_family="identity",
        epsilon=args.epsilon,
    ).cost_profile(
        content_dim=1,
        refresh_interval=args.route_refresh_interval,
    )
    methods: Dict[str, Dict[str, float | int]] = {}
    common_memory_macs = 3 * args.slots
    identity_memory_macs = common_memory_macs + identity_route_cost.forward_macs
    mqr_memory_macs = common_memory_macs + route_cost.forward_macs
    common_peak = args.gate_hidden_dim
    methods["mqr"] = {
        "trainable_parameters": gate_parameters,
        "state_bytes": 4 * state_floats,
        "forward_macs_per_token": gate_macs + mqr_memory_macs,
        "amortized_update_macs_per_token": gate_macs + route_cost.amortized_refresh_macs,
        "peak_workspace_bytes": 4 * (common_peak + args.slots),
    }
    methods["identity"] = {
        "trainable_parameters": gate_parameters,
        "state_bytes": 4 * state_floats,
        "forward_macs_per_token": gate_macs + identity_memory_macs,
        "amortized_update_macs_per_token": gate_macs,
        "peak_workspace_bytes": 4 * (common_peak + args.slots),
    }
    methods["fixed_ring_buffer"] = dict(methods["mqr"])
    gru = models["gru"]
    fast_weight = models["fast_weight"]
    assert isinstance(gru, GRUMemoryControl)
    assert isinstance(fast_weight, FastWeightMemoryControl)
    gru_parameters = sum(parameter.numel() for parameter in gru.parameters())
    gru_core_macs = 3 * (
        TOKEN_DIM * state_floats + state_floats * state_floats
    ) + state_floats
    methods["gru"] = {
        "trainable_parameters": gate_parameters + gru_parameters,
        "state_bytes": 4 * state_floats,
        "forward_macs_per_token": gate_macs + gru_core_macs,
        "amortized_update_macs_per_token": gate_macs + 3 * gru_core_macs,
        "peak_workspace_bytes": 4 * (common_peak + 3 * state_floats),
    }
    fast_parameters = sum(parameter.numel() for parameter in fast_weight.parameters())
    memory_dim = fast_weight.memory_dim
    fast_core_macs = 3 * TOKEN_DIM * memory_dim + 3 * memory_dim**2 + memory_dim
    methods["fast_weight"] = {
        "trainable_parameters": gate_parameters + fast_parameters,
        "state_bytes": 4 * memory_dim**2,
        "forward_macs_per_token": gate_macs + fast_core_macs,
        "amortized_update_macs_per_token": gate_macs + 3 * fast_core_macs,
        "peak_workspace_bytes": 4 * (common_peak + memory_dim**2),
    }
    ratios: Dict[str, Dict[str, float]] = {}
    for control in ("identity", "gru", "fast_weight", "fixed_ring_buffer"):
        ratios[control] = {
            key: float(methods["mqr"][key]) / max(float(methods[control][key]), 1.0)
            for key in (
                "trainable_parameters",
                "state_bytes",
                "forward_macs_per_token",
                "amortized_update_macs_per_token",
                "peak_workspace_bytes",
            )
        }
    gate = all(
        ratio <= 1.05 + 1e-12
        for comparison in ratios.values()
        for ratio in comparison.values()
    )
    alternatives = {
        family: asdict(
            ResidualAddressRouter(
                args.slots,
                route_family=family,
                epsilon=args.epsilon,
                givens_layers=2,
                block_size=4,
            ).cost_profile(
                content_dim=1,
                refresh_interval=args.route_refresh_interval,
            )
        )
        for family in ("cyclic_givens", "block_cayley", "dense_cayley")
    }
    return {
        "methods": methods,
        "mqr_to_control_ratios": ratios,
        "one_sided_all_resources_within_1_05": gate,
        "ratio_semantics": "MQR resource divided by control resource; smaller is conservative",
        "route_cost": asdict(route_cost),
        "alternative_route_costs": alternatives,
    }


def run_seed(args: argparse.Namespace, seed: int) -> Dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    gate, gate_training = fit_utility_gate(seed=seed, args=args)
    gru = train_control("gru", gate, seed=seed, args=args)
    fast_weight = train_control("fast_weight", gate, seed=seed, args=args)
    models: Dict[str, nn.Module | None] = {
        "mqr": None,
        "identity": None,
        "mqr_identity_replacement": None,
        "fixed_ring_buffer": None,
        "gru": gru,
        "fast_weight": fast_weight,
    }
    evaluation = generate_batch(
        seed=seed * 100000 + 90000,
        batch_size=args.eval_episodes,
        slots=args.slots,
        event_spacing=args.event_spacing,
        delay=args.eval_delay,
        query_cycles=args.query_cycles,
        reverse_probability=0.5,
    )
    methods = {
        method: evaluate_method(
            method, model, gate, evaluation, args=args
        )
        for method, model in models.items()
    }
    return {
        "seed": seed,
        "gate_training": gate_training,
        "methods": methods,
        "resource_audit": resource_audit(gate, models, args=args),
    }


def _run_seed_entry(payload: Tuple[argparse.Namespace, int]) -> Dict[str, Any]:
    args, seed = payload
    torch.set_num_threads(1)
    return run_seed(args, seed)


def _mean(values: Iterable[float]) -> float:
    parsed = [float(value) for value in values]
    return sum(parsed) / len(parsed) if parsed else float("nan")


def _student_t_critical_95(sample_count: int) -> float:
    table = {
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
    parsed = [float(value) for value in values]
    mean = _mean(parsed)
    if len(parsed) <= 1:
        radius = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in parsed) / (len(parsed) - 1)
        radius = _student_t_critical_95(len(parsed)) * math.sqrt(variance / len(parsed))
    return {
        "mean": mean,
        "ci95_low": mean - radius,
        "ci95_high": mean + radius,
        "n": len(parsed),
        "ci_method": "two_sided_student_t_95",
    }


def aggregate(args: argparse.Namespace, runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    methods = (
        "mqr",
        "identity",
        "gru",
        "fast_weight",
        "mqr_identity_replacement",
        "fixed_ring_buffer",
    )
    summaries: Dict[str, Any] = {}
    for method in methods:
        summaries[method] = {
            metric: _summary([run["methods"][method][metric] for run in runs])
            for metric in (
                "accuracy",
                "forward_accuracy",
                "context_reversal_accuracy",
                "repeated_query_accuracy",
                "runtime_ms_per_episode",
            )
        }
    mqr = summaries["mqr"]
    paired_differences = {
        control: _summary(
            [
                run["methods"]["mqr"]["accuracy"]
                - run["methods"][control]["accuracy"]
                for run in runs
            ]
        )
        for control in ("identity", "gru", "fast_weight", "fixed_ring_buffer")
    }
    identity_replacement_drop = _summary(
        [
            run["methods"]["mqr"]["accuracy"]
            - run["methods"]["mqr_identity_replacement"]["accuracy"]
            for run in runs
        ]
    )
    gate_metrics = {
        metric: _summary(
            [run["methods"]["mqr"]["gate_calibration"][metric] for run in runs]
        )
        for metric in (
            "auprc",
            "brier",
            "ece",
            "write_rate",
            "event_write_rate",
            "distractor_write_rate",
            "event_distractor_write_gap",
            "budget_violation",
            "positive_rate",
        )
    }
    beats_temporal_control = any(
        paired_differences[control]["ci95_low"] > 0.0
        for control in ("gru", "fast_weight")
    )
    resources_pass = all(
        bool(run["resource_audit"]["one_sided_all_resources_within_1_05"])
        for run in runs
    )
    gates = {
        "formal_seed_count_at_least_10": len(runs) >= 10,
        "formal_episode_count_at_least_64": args.eval_episodes >= 64,
        "long_delay_at_least_four_slot_cycles": args.eval_delay >= 4 * args.slots,
        "residual_identity_channel_present": 0.0 < args.epsilon < 1.0,
        "no_explicit_event_or_query_marker_in_gate": True,
        "mqr_accuracy_ci_above_0_75": mqr["accuracy"]["ci95_low"] > 0.75,
        "long_delay_forward_ci_above_0_75": mqr["forward_accuracy"]["ci95_low"] > 0.75,
        "context_reversal_ci_above_0_75": mqr["context_reversal_accuracy"]["ci95_low"] > 0.75,
        "repeated_query_ci_above_0_75": mqr["repeated_query_accuracy"]["ci95_low"] > 0.75,
        "beats_identity": paired_differences["identity"]["ci95_low"] > 0.0,
        "beats_gru_or_fast_weight": beats_temporal_control,
        "identity_replacement_hurts": identity_replacement_drop["ci95_low"] > 0.0,
        "event_distractor_write_gap_positive": gate_metrics[
            "event_distractor_write_gap"
        ]["ci95_low"]
        > 0.0,
        "write_budget_never_violated": gate_metrics["budget_violation"]["ci95_high"] <= 1e-12,
        "utility_auprc_above_prevalence": gate_metrics["auprc"]["ci95_low"]
        > gate_metrics["positive_rate"]["ci95_high"],
        "cyclic_address_accuracy_above_0_99": _summary(
            [run["methods"]["mqr"]["cyclic_address_accuracy"] for run in runs]
        )["ci95_low"]
        > 0.99,
        "content_zero_drift": all(
            run["methods"]["mqr"]["no_write_content_drift_linf"] == 0.0
            for run in runs
        ),
        "all_peak_and_amortized_resources_within_1_05": resources_pass,
    }
    gates["mqr_route_mechanism_qualified"] = all(gates.values())
    independent_gates = {
        "route_mechanism_qualified": bool(gates["mqr_route_mechanism_qualified"]),
        "beats_direct_fixed_ring_buffer": paired_differences[
            "fixed_ring_buffer"
        ]["ci95_low"]
        > 0.0,
    }
    independent_gates["mqr_independent_algorithm_advantage"] = all(
        independent_gates.values()
    )
    return {
        "method_summaries": summaries,
        "paired_mqr_minus_control_accuracy": paired_differences,
        "identity_replacement_accuracy_drop": identity_replacement_drop,
        "gate_calibration_summaries": gate_metrics,
        "decision_gates": gates,
        "mqr_route_mechanism_qualified": bool(gates["mqr_route_mechanism_qualified"]),
        "independent_advantage_gates": independent_gates,
        "mqr_independent_algorithm_advantage": bool(
            independent_gates["mqr_independent_algorithm_advantage"]
        ),
        "claim_boundary": (
            "Passing qualifies a residual cyclic address-routing mechanism on this "
            "pre-registered synthetic task only; it does not establish Go, LLM, "
            "general continual-learning, or animal-learning advantage."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=71)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--event-spacing", type=int, default=5)
    parser.add_argument("--train-delay", type=int, default=16)
    parser.add_argument("--eval-delay", type=int, default=64)
    parser.add_argument("--query-cycles", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=0.75)
    parser.add_argument("--write-budget", type=float, default=0.2)
    parser.add_argument("--gate-hidden-dim", type=int, default=32)
    parser.add_argument("--gate-episodes", type=int, default=16)
    parser.add_argument("--gate-steps", type=int, default=120)
    parser.add_argument("--gate-batch-size", type=int, default=128)
    parser.add_argument("--gate-lr", type=float, default=0.01)
    parser.add_argument("--dual-lr", type=float, default=0.02)
    parser.add_argument("--control-steps", type=int, default=120)
    parser.add_argument("--control-batch-size", type=int, default=32)
    parser.add_argument("--control-lr", type=float, default=0.01)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--route-refresh-interval", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/results/mqr_routed_memory_qualification_smoke.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds <= 0 or args.eval_episodes <= 0:
        raise ValueError("seeds and eval_episodes must be positive")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.slots != 8:
        raise ValueError("the matched 16-float GRU/fast-weight protocol currently requires 8 slots")
    if math.gcd(args.slots, args.event_spacing) != 1:
        raise ValueError("event_spacing must be coprime with slots")
    if args.train_delay % args.slots or args.eval_delay % args.slots:
        raise ValueError("train/eval delays must be multiples of slots")
    if not 0.0 < args.epsilon < 1.0:
        raise ValueError("formal residual routing requires epsilon strictly in (0, 1)")
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
    result = {
        "schema_version": 1,
        "status": "formal" if args.seeds >= 10 and args.eval_episodes >= 64 else "smoke_or_pilot",
        "protocol": {
            "content_address_separated": True,
            "residual_transition": "(1-epsilon) I + epsilon |P|^2",
            "primary_route": "local cyclic permutation (a monomial unitary)",
            "address_commit_projection": "straight_through_top1",
            "predict_before_write": True,
            "gate_input_features": [
                "magnitude",
                "novelty_vs_ema",
                "prediction_error_vs_previous",
                "ema_magnitude",
                "uncertainty_proxy",
            ],
            "gate_has_explicit_event_marker": False,
            "gate_has_query_or_phase_feature": False,
            "critic_target": "exact paired finite-horizon write/no-write return",
            "utility_horizons": list(UTILITY_HORIZONS),
            "positive_negative_replay_balanced": True,
            "hard_cumulative_write_budget": args.write_budget,
            "same_state_floats_all_cores": args.slots * 2,
            "slots": args.slots,
            "event_spacing": args.event_spacing,
            "write_budget": args.write_budget,
            "resource_gate_semantics": "one-sided MQR/control <= 1.05 on five axes",
            "context_reversal": "forward writes; context-conditioned transpose route during queries",
            "non_mqr_diagnostic_control": "direct fixed cyclic ring buffer",
            "seeds": args.seeds,
            "train_delay": args.train_delay,
            "eval_delay": args.eval_delay,
            "query_cycles": args.query_cycles,
            "eval_episodes": args.eval_episodes,
            "epsilon": args.epsilon,
        },
        "runs": runs,
        "aggregate": aggregate(args, runs),
        "preserved_negative_evidence": [
            "analysis/results/mqr_mechanism_qualification_formal_10seed.json",
            "analysis/results/mqr_cyclic_capacity_sweep_formal_10seed.json",
            "analysis/results/unified_temporal_mqr_go_competitive_10x10_formal_10seed.json",
            "analysis/results/unified_temporal_mqr_go_competitive_13x13_formal_10seed.json",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
