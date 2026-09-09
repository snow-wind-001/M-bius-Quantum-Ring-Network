#!/usr/bin/env python3
"""Deterministic numerical certification for the Phase-II sidecar contract."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mqr.agent import TemporalUtilityMQRAgent
from mqr.minicpm import MiniCPMLoRAEncoder
from mqr.online import OrthogonalGradientMemory
from mqr.safety import SidecarSafetyLimits
from mqr.temporal import MultiTimescaleMQR, TemporalMQRSidecar, TemporalMQRState
from mqr.unitary import CayleyUnistochasticParam


def signed_state_scan() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for mode in ("identity", "unistochastic"):
        for seed in range(10):
            torch.manual_seed(1000 + seed)
            core = MultiTimescaleMQR(
                5,
                7,
                3,
                leak_rates=(0.5, 0.1, 0.02),
                injection_rank=4,
                injection_activation="tanh",
                state_activation="tanh",
                transition_mode=mode,
            ).double()
            x = torch.randn(4, 5, dtype=torch.double)
            state = TemporalMQRState(
                tuple(torch.randn(4, 7, dtype=torch.double) for _ in range(3))
            )
            promotion = 0.25 * torch.randn(4, 3, 7, dtype=torch.double)
            gates = torch.rand(4, 3, dtype=torch.double)
            _output, _next_state, certificate = core.forward_step(
                x,
                state=state,
                write_gate=gates,
                promotion=promotion,
                return_certificate=True,
            )
            rows.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "certified": bool(certificate["certified"]),
                    "max_bound_violation": certificate["max_bound_violation"],
                    "numerical_tolerance": certificate["numerical_tolerance"],
                    "max_transition_linf_gain": float(
                        certificate["transition_linf_gain"].max().item()
                    ),
                }
            )
    return {
        "cases": len(rows),
        "certified_cases": sum(bool(row["certified"]) for row in rows),
        "max_bound_violation": max(float(row["max_bound_violation"]) for row in rows),
        "max_numerical_tolerance": max(
            float(row["numerical_tolerance"]) for row in rows
        ),
        "max_transition_linf_gain": max(
            float(row["max_transition_linf_gain"]) for row in rows
        ),
        "passed": all(bool(row["certified"]) for row in rows),
    }


def cayley_drift_scan() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for scale in (1e-4, 1e-3, 1e-2, 1e-1):
        for seed in range(10):
            torch.manual_seed(2000 + seed)
            parameter = CayleyUnistochasticParam(6, coordinate_mode="minimal").double()
            reference = parameter.skew_hermitian_A().detach().clone()
            with torch.no_grad():
                parameter.A_real.add_(scale * torch.randn_like(parameter.A_real))
                parameter.A_imag.add_(scale * torch.randn_like(parameter.A_imag))
            row = parameter.drift_diagnostics(reference)
            rows.append({"scale": scale, "seed": seed, **row})
    return {
        "cases": len(rows),
        "certified_cases": sum(bool(row["bounds_certified"]) for row in rows),
        "max_unitary_bound_ratio": max(
            float(row["unitary_fro_drift"])
            / max(float(row["unitary_fro_bound"]), 1e-300)
            for row in rows
        ),
        "max_transition_bound_ratio": max(
            float(row["transition_fro_drift"])
            / max(float(row["transition_fro_bound"]), 1e-300)
            for row in rows
        ),
        "passed": all(bool(row["bounds_certified"]) for row in rows),
    }


def constrained_ogd_check() -> Dict[str, Any]:
    memory = OrthogonalGradientMemory(max_rank=2, tolerance=1e-12).double()
    old = [
        ("fast", torch.tensor([1.0, 2.0], dtype=torch.double), 0.04),
        ("slow", torch.tensor([3.0, -1.0], dtype=torch.double), 0.01),
    ]
    new = [
        ("fast", torch.tensor([-2.0, 1.0], dtype=torch.double), 0.04),
        ("slow", torch.tensor([5.0, 7.0], dtype=torch.double), 0.01),
    ]
    memory.observe(old)
    projected, stats = memory.project_preconditioned(new, allowed_names=("fast",))
    delta_fast = -0.04 * projected["fast"]
    delta_slow = -0.01 * projected["slow"]
    overlap = float(
        (
            (old[0][1] * delta_fast).sum()
            + (old[1][1] * delta_slow).sum()
        ).abs().item()
    )
    forbidden_norm = float(projected["slow"].norm().item())
    return {
        "old_gradient_first_order_overlap": overlap,
        "forbidden_slow_gradient_norm": forbidden_norm,
        "retained_norm": float(stats["retained_norm"]),
        "passed": overlap < 1e-12 and forbidden_norm == 0.0,
    }


def temporal_sidecar_check() -> Dict[str, Any]:
    torch.manual_seed(3000)
    sidecar = TemporalMQRSidecar(
        8,
        ring_dim=5,
        residual_clip_l2=0.1,
        core_kwargs={
            "leak_rates": (1.0, 0.1, 0.02),
            "injection_rank": 4,
            "transition_mode": "unistochastic",
            "cayley_coordinate_mode": "minimal",
        },
    ).double()
    hidden = torch.randn(3, 8, dtype=torch.double)
    adapted, state, initial = sidecar.forward_step(hidden, return_diagnostics=True)
    noop_error = float((adapted - hidden).abs().max().item())
    with torch.no_grad():
        sidecar.core.readout.weight.fill_(8.0)
    _adapted, _state, clipped = sidecar.forward_step(
        hidden,
        state=state.detached(),
        return_diagnostics=True,
    )
    return {
        "initial_noop_linf_error": noop_error,
        "initial_residual_l2_max": initial["residual_l2_max"],
        "clipped_raw_residual_l2_max": clipped["raw_residual_l2_max"],
        "clipped_residual_l2_max": clipped["residual_l2_max"],
        "state_bound_certified": bool(clipped["state_certificate"]["certified"]),
        "passed": bool(
            noop_error == 0.0
            and clipped["residual_l2_max"] <= 0.1 + 1e-12
            and clipped["state_certificate"]["certified"]
        ),
    }


def agent_atomic_rollback_check() -> Dict[str, Any]:
    torch.manual_seed(4000)
    agent = TemporalUtilityMQRAgent(
        8,
        board_size=3,
        ring_dim=6,
        latent_dim=7,
        task_lr=0.2,
        ogd_max_rank=2,
        safety_limits=SidecarSafetyLimits(max_output_linf_drift=0.0),
    ).double()
    committed = agent.commit_step(torch.randn(1, 8, dtype=torch.double), stream_id="audit")
    ticket = int(committed["ticket_id"])
    before = {
        name: parameter.detach().clone() for name, parameter in agent.named_parameters()
    }
    version = int(agent.online_parameter_version.item())
    info = agent.apply_feedback(
        ticket,
        torch.tensor([2]),
        legality_target=torch.ones(1, 9, dtype=torch.double),
        remember_gradient=True,
        return_grad_features=True,
    )
    parameter_drift = max(
        float((parameter - before[name]).abs().max().item())
        for name, parameter in agent.named_parameters()
    )
    return {
        "candidate_update_norm": info["candidate_update_norm"],
        "committed_update_norm": info["update_norm"],
        "parameter_linf_drift_after_rollback": parameter_drift,
        "ogd_rank_after_rollback": agent.task_gradient_memory.rank,
        "parameter_version_change": int(agent.online_parameter_version.item()) - version,
        "external_gradient_authorized": info["external_gradient_authorized"],
        "violations": info["safety_violations"],
        "passed": bool(
            info["rolled_back"]
            and parameter_drift == 0.0
            and agent.task_gradient_memory.rank == 0
            and int(agent.online_parameter_version.item()) == version
            and info["grad_features"] is None
        ),
    }


class _TinyAttention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)


class _TinyLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.self_attn = _TinyAttention(width)


class _TinyBackbone(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_TinyLayer(width)])
        self.config = SimpleNamespace(hidden_size=width)


def lora_atomic_rollback_check() -> Dict[str, Any]:
    torch.manual_seed(5000)
    encoder = MiniCPMLoRAEncoder(
        _TinyBackbone(6),
        tokenizer=None,
        base_model_path=ROOT,
        layer_index=0,
        target_modules=("q_proj", "v_proj"),
        lora_rank=2,
        lora_alpha=4.0,
    )
    attention = encoder.backbone.layers[0].self_attn
    x = torch.randn(3, 6)

    def probe() -> torch.Tensor:
        return attention.q_proj(x) + attention.v_proj(x)

    features = probe()
    before = {
        name: parameter.detach().clone()
        for name, parameter in encoder.named_lora_parameters()
    }
    memory = OrthogonalGradientMemory(max_rank=2)
    info = encoder.step_from_external_gradient(
        features,
        torch.ones_like(features),
        lr=0.2,
        orthogonal_memory=memory,
        remember_gradient=True,
        max_update_norm=0.05,
        constancy_closure=probe,
        safety_limits=SidecarSafetyLimits(max_output_linf_drift=0.0),
    )
    parameter_drift = max(
        float((parameter - before[name]).abs().max().item())
        for name, parameter in encoder.named_lora_parameters()
    )
    return {
        "candidate_update_norm": info["candidate_update_norm"],
        "committed_update_norm": info["update_norm"],
        "parameter_linf_drift_after_rollback": parameter_drift,
        "ogd_rank_after_rollback": memory.rank,
        "constancy_output_linf_drift": info["constancy_output_linf_drift"],
        "violations": info["safety_violations"],
        "passed": bool(
            info["rolled_back"]
            and parameter_drift == 0.0
            and memory.rank == 0
            and info["update_norm"] == 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "analysis/results/mqr_sidecar_safety_certification.json",
    )
    args = parser.parse_args()
    sections = {
        "signed_state": signed_state_scan(),
        "cayley_finite_drift": cayley_drift_scan(),
        "constrained_ogd": constrained_ogd_check(),
        "temporal_residual_sidecar": temporal_sidecar_check(),
        "agent_atomic_rollback": agent_atomic_rollback_check(),
        "lora_atomic_rollback": lora_atomic_rollback_check(),
    }
    payload = {
        "schema_version": 1,
        "purpose": "phase_ii_sidecar_numerical_and_transaction_certification",
        "evidence_scope": "deterministic mechanism certification; not task-performance evidence",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "dtype": "float64 for mathematical scans; float32 for tiny LoRA transaction",
            "network_used": False,
        },
        **sections,
        "all_passed": all(bool(section["passed"]) for section in sections.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
