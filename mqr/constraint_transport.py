"""Explicit behavior covectors through finite orthogonal Go histories.

The state adjoint is analytic; local parameter VJPs include the injection and
input-conditioned rotations. Contributions to shared parameters are summed.
This is a finite-sequence adjoint, not an infinite-history online derivative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from .agent import GoMultiHeadOutput
from .go_memory import GoResearchCore, PositionQueryGoAgent
from .temporal import TemporalMQRState


@dataclass
class TerminalGradient:
    """Detached objective values [M] and parameter Jacobian [M, P]."""

    values: torch.Tensor
    jacobian: torch.Tensor
    parameter_names: Tuple[str, ...]
    diagnostics: Dict[str, Any]


def _validate_core(agent: PositionQueryGoAgent) -> None:
    # Subclasses may change either the state Jacobian or the readout contract.
    # Such extensions need their own derivation instead of a silent fallback.
    if type(agent) is not PositionQueryGoAgent or type(agent.core) is not GoResearchCore:
        raise TypeError("constraint transport requires PositionQueryGoAgent with GoResearchCore")
    if agent.core.state_activation != "none":
        raise ValueError("analytic transport requires state_activation='none'")
    if agent.core.transition_mode not in ("orthogonal", "identity"):
        raise ValueError("analytic transport requires signed orthogonal or identity dynamics")


@torch.no_grad()
def pullback_ring_covectors(
    core: GoResearchCore,
    x: torch.Tensor,
    covectors: Tuple[torch.Tensor, ...],
) -> Tuple[Tuple[torch.Tensor, ...], float]:
    """Pull back [M, B, N] covectors per ring at fixed exogenous x [B, D].

    The returned row covectors are rho * q @ R(x), using inverse Givens order.
    The diagnostic measures |norm(q_previous) - rho * norm(q_next)|.
    """

    if type(core) is not GoResearchCore or core.state_activation != "none":
        raise ValueError("unsupported state Jacobian for analytic covector transport")
    if core.transition_mode not in ("orthogonal", "identity"):
        raise ValueError("nonnegative squared rotations are not orthogonal transport")
    if len(covectors) != core.num_timescales:
        raise ValueError("one covector tensor is required per ring")
    result, error = [], 0.0
    for index, q in enumerate(covectors):
        if q.ndim != 3 or q.shape[1:] != (x.size(0), core.ring_dim):
            raise ValueError("covectors must be [objectives, batch, ring_dim]")
        rho = 1.0 - core.leak_rates[index].to(q)
        if float(rho) == 0.0:
            previous = torch.zeros_like(q)
        elif core.transition_mode == "identity":
            previous = rho * q
        else:
            previous = rho * core.unitary_params[index].apply_orthogonal(
                q, transpose=True, angle_offsets=core.orthogonal_offsets(index, x),
            )
        discrepancy = (previous.norm(dim=-1) - rho * q.norm(dim=-1)).abs().max()
        error = max(error, float(discrepancy))
        result.append(previous)
    return tuple(result), error


@torch.enable_grad()
def terminal_gradients(
    agent: PositionQueryGoAgent,
    features: torch.Tensor,
    objective: Callable[[GoMultiHeadOutput], torch.Tensor],
    *,
    backend: str = "transport",
    credit_horizon: Optional[int] = None,
) -> TerminalGradient:
    """Differentiate terminal objectives through a fixed-parameter history.

    features: [T, D] for one stream, or [T, B, D] for independent batch lanes.
    objective: differentiable terminal scalar or vector [M].
    Both backends replay from zero with all writes enabled, matching behavioral
    anchors. credit_horizon uses the same periodic detach boundaries as the
    online agent. No parameter .grad, live state, ticket, or counter is changed.

    transport stores detached prefix states and rebuilds one local graph at a
    time. autograd is the complete-graph reference for the identical function.
    """

    _validate_core(agent)
    if backend not in ("autograd", "transport"):
        raise ValueError("backend must be 'autograd' or 'transport'")
    if credit_horizon is not None and (
        isinstance(credit_horizon, bool) or not isinstance(credit_horizon, int) or credit_horizon < 1
    ):
        raise ValueError("credit_horizon must be a positive integer or None")
    if features.ndim == 2:
        features = features[:, None, :]
    if (features.ndim != 3 or features.size(0) < 1 or features.size(1) < 1
            or features.size(2) != agent.input_dim):
        raise ValueError("features must be nonempty [T, B, input_dim] or [T, input_dim]")
    if not features.is_floating_point() or features.is_complex() or not bool(torch.isfinite(features).all()):
        raise ValueError("features must be finite real floating-point observations")
    active = [(name, p) for name, p in agent._named_task_parameters() if p.requires_grad]
    if not active:
        raise ValueError("terminal gradients require trainable task parameters")
    if len({id(p) for _, p in active}) != len(active):
        raise ValueError("shared parameters must occur once in the parameter layout")
    reference = active[0][1]
    xs = features.detach().to(reference)
    count = xs.size(0)
    start = 0 if credit_horizon is None else (count - 1) // credit_horizon * credit_horizon
    version = int(agent.online_parameter_version)
    parameters = [p for _, p in active]
    widths = [p.numel() for p in parameters]
    offsets = [0]
    for width in widths:
        offsets.append(offsets[-1] + width)
    state = agent.core.zero_state(xs.size(1), device=xs.device, dtype=xs.dtype)
    history = []
    if backend == "transport":
        with torch.no_grad():
            for index, x in enumerate(xs):
                if index >= start:
                    history.append(state.detached())
                _, state = agent.core.forward_step(x, state=state)
        final_state = TemporalMQRState(tuple(h.detach().requires_grad_(True) for h in state.rings))
    else:
        for index, x in enumerate(xs):
            if credit_horizon is not None and index and index % credit_horizon == 0:
                state = state.detached()
            _, state = agent.core.forward_step(x, state=state)
        final_state = state

    output = agent.readout_state(final_state, features=xs[-1])
    values = objective(output)
    if not isinstance(values, torch.Tensor) or values.ndim > 1 or values.numel() < 1:
        raise ValueError("objective must return a nonempty scalar or vector")
    values = values.reshape(-1)
    if not values.is_floating_point() or values.is_complex() or not bool(torch.isfinite(values).all()):
        raise ValueError("objective values must be finite and real")
    detached_values = values.detach().clone()
    rows = values.numel()
    jacobian = reference.new_zeros(rows, offsets[-1])
    cotangents = [reference.new_zeros(rows, *h.shape) for h in final_state.rings]
    inputs = parameters + (list(final_state.rings) if backend == "transport" else [])
    for row in range(rows):
        gradients = (
            torch.autograd.grad(values[row], inputs, allow_unused=True, retain_graph=row + 1 < rows)
            if values[row].requires_grad else (None,) * len(inputs)
        )
        for index, gradient in enumerate(gradients[:len(parameters)]):
            if gradient is not None:
                jacobian[row, offsets[index]:offsets[index + 1]] = gradient.detach().reshape(-1)
        if backend == "transport":
            for index, gradient in enumerate(gradients[len(parameters):]):
                if gradient is not None:
                    cotangents[index][row] = gradient.detach()
    trace_bytes = (
        sum(h.numel() * h.element_size() for s in history for h in s.rings)
        if backend == "transport" else 0
    )
    del output, values, final_state, state, inputs, gradients
    max_error, local_steps = 0.0, 0
    if backend == "transport":
        core_indices = [i for i, (name, _) in enumerate(active) if name.startswith("core.")]
        core_parameters = [parameters[i] for i in core_indices]
        q = tuple(cotangents)
        for index in range(count - 1, start - 1, -1):
            if core_parameters:
                latent, next_state = agent.core.forward_step(xs[index], state=history[index - start])
                del latent
                differentiable = [r for r, h in enumerate(next_state.rings) if h.requires_grad]
                if differentiable:
                    local_steps += 1
                    for row in range(rows):
                        gradients = torch.autograd.grad(
                            [next_state.rings[r] for r in differentiable],
                            core_parameters,
                            grad_outputs=[q[r][row] for r in differentiable],
                            allow_unused=True, retain_graph=row + 1 < rows,
                        )
                        for parameter_index, gradient in zip(core_indices, gradients):
                            if gradient is not None:
                                jacobian[row, offsets[parameter_index]:offsets[parameter_index + 1]].add_(
                                    gradient.detach().reshape(-1)
                                )
                    del gradients
                del next_state
            q, error = pullback_ring_covectors(agent.core, xs[index], q)
            max_error = max(max_error, error)
    if int(agent.online_parameter_version) != version:
        raise RuntimeError("parameters changed during constraint construction")
    if not bool(torch.isfinite(jacobian).all()):
        raise FloatingPointError("nonfinite terminal gradient")
    return TerminalGradient(
        detached_values, jacobian.detach(), tuple(name for name, _ in active),
        {"backend": backend, "history_steps": count, "credit_steps": count - start,
         "objectives": rows, "batch_size": xs.size(1), "state_trace_bytes": trace_bytes,
         "local_parameter_vjp_steps": local_steps, "max_adjoint_norm_error": max_error,
         "parameter_version": version},
    )
