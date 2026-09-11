#!/usr/bin/env python3
"""
Möbius Quantum Ring (MQR) reference implementation (backward-compatible facade).

This repository historically exposed the implementation via a single module file
`mobius_quantum_ring.py`. During the refactor towards an engineering-style layout,
the core implementation is moved to the `mqr/` package, while keeping the original
symbols importable for existing scripts:

  from mobius_quantum_ring import create_mobius_model

Algorithm source of truth:
  `Möbius Quantum Ring.html` (MQR / UHR-Net description)
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import torch

from mqr import (
    CAUSAL_UTILITY_FEATURES,
    CayleyUnistochasticParam,
    CyclicGivensUnistochasticParam,
    FutureUtilityGate,
    GoAgentExample,
    GoAgentTrajectory,
    GoVectorEncoder,
    GoOnlineSession,
    GoHistoryEncoder,
    GoObservationMemory,
    GoResearchCore,
    HistoryGoSession,
    PositionQueryGoAgent,
    GoBehaviorMemory,
    ProtectedGoSession,
    generate_reachable_history_pairs,
    consolidate_task_memory,
    GoLossWeights,
    GoMultiHeadOutput,
    GoSpatialSkipHeads,
    LoRALinear,
    MiniCPMGoBoardEncoder,
    MiniCPMLoRAEncoder,
    AgentResourceAudit,
    AddressRoutingCost,
    BalancedAdvantageReplay,
    BudgetedMultiTimescaleUtilityGate,
    ContextualAddressRouterBank,
    MoebiusQuantumRing,
    MoebiusQuantumRingImageClassifier,
    MultiTimescaleMQR,
    OnlineMultiRingClassifier,
    OnlineTemporalMQRClassifier,
    OrthogonalGradientMemory,
    MultiTimescaleFastWeightCore,
    MultiTimescaleGRUCore,
    MultiTimescaleLoRAResidualCore,
    MultiTimescaleLSTMCore,
    MultiTimescaleSinkhornCore,
    FrozenResidualCoreWrapper,
    RingState,
    ResidualAddressRouter,
    RoutedMemoryState,
    RoutedSlotMemory,
    SlowLoRAConsolidator,
    SlowLoRAConsolidationSchedule,
    SinkhornDoublyStochasticParam,
    TemporalMQRState,
    TemporalWriteGate,
    TemporalUtilityMQRAgent,
    UtilityDrivenMQR,
    WriteBudgetState,
    audit_agent_resources,
    create_mobius_model,
    example_targets,
    generate_go_agent_trajectories,
    generate_ko_history_pairs,
    generalized_advantage_estimate,
    legality_target,
    load_minicpm_awq_encoder,
    matched_resource_gate,
    matched_competitive_resource_gate,
    twin_write_returns,
    utility_calibration_metrics,
)

# Backward-compatible alias (some scripts import the Unicode name).
MöbiusQuantumRing = MoebiusQuantumRingImageClassifier  # noqa: N816


class HamiltonianOptimizer:
    """
    Backward-compatible optimizer wrapper.

    The HTML text discusses Hamiltonian/phase updates on the unitary manifold.
    For now, we keep the existing training script working by providing an optimizer-like
    interface (zero_grad / step) backed by SGD with momentum.
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter], lr: float = 1e-3, momentum: float = 0.9):
        self._opt = torch.optim.SGD(list(parameters), lr=lr, momentum=momentum)

    def zero_grad(self):
        self._opt.zero_grad()

    def step(self, closure: Optional[Any] = None):
        return self._opt.step(closure=closure)


__all__ = [
    "GoHistoryEncoder", "GoObservationMemory", "GoResearchCore", "HistoryGoSession",
    "PositionQueryGoAgent", "GoBehaviorMemory", "ProtectedGoSession", "generate_reachable_history_pairs",
    "CayleyUnistochasticParam",
    "CyclicGivensUnistochasticParam",
    "MoebiusQuantumRing",
    "MoebiusQuantumRingImageClassifier",
    "RingState",
    "OnlineMultiRingClassifier",
    "MultiTimescaleMQR",
    "OnlineTemporalMQRClassifier",
    "TemporalMQRState",
    "TemporalWriteGate",
    "CAUSAL_UTILITY_FEATURES",
    "FutureUtilityGate",
    "GoAgentExample",
    "GoAgentTrajectory",
    "GoVectorEncoder",
    "GoOnlineSession",
    "consolidate_task_memory",
    "GoLossWeights",
    "GoMultiHeadOutput",
    "GoSpatialSkipHeads",
    "LoRALinear",
    "MiniCPMGoBoardEncoder",
    "MiniCPMLoRAEncoder",
    "AgentResourceAudit",
    "AddressRoutingCost",
    "BalancedAdvantageReplay",
    "BudgetedMultiTimescaleUtilityGate",
    "ContextualAddressRouterBank",
    "UtilityDrivenMQR",
    "ResidualAddressRouter",
    "RoutedMemoryState",
    "RoutedSlotMemory",
    "WriteBudgetState",
    "MultiTimescaleFastWeightCore",
    "MultiTimescaleGRUCore",
    "MultiTimescaleLoRAResidualCore",
    "MultiTimescaleLSTMCore",
    "MultiTimescaleSinkhornCore",
    "FrozenResidualCoreWrapper",
    "SlowLoRAConsolidator",
    "SlowLoRAConsolidationSchedule",
    "TemporalUtilityMQRAgent",
    "generalized_advantage_estimate",
    "audit_agent_resources",
    "example_targets",
    "generate_go_agent_trajectories",
    "generate_ko_history_pairs",
    "legality_target",
    "load_minicpm_awq_encoder",
    "matched_resource_gate",
    "matched_competitive_resource_gate",
    "twin_write_returns",
    "utility_calibration_metrics",
    "OrthogonalGradientMemory",
    "MöbiusQuantumRing",
    "HamiltonianOptimizer",
    "create_mobius_model",
]


if __name__ == "__main__":
    # Minimal smoke test.
    model = create_mobius_model(num_classes=100, img_size=32, embed_dim=128, depth=10, lora_rank=8, readout_dim=16)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print("OK:", x.shape, "->", y.shape)
