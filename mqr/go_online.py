"""Causal Go sessions and current-parameter OGD consolidation.

The session reuses the unified agent's bounded trajectory transactions. A
teacher supplies labels only after a prediction is committed. Legal masking
is an environment action constraint; raw model legality is reported separately.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .agent import GoMultiHeadOutput, TemporalUtilityMQRAgent
from .go import GoBoard
from .go_agent import GoVectorEncoder, legality_target
from .online import OrthogonalGradientMemory
from .temporal import TemporalMQRState


class SpatialRingGoAgent(TemporalUtilityMQRAgent):
    """Use a temporal ring to modulate the frozen spatial feature channels.

    A zero-initialized affine modulation gives an exact initial no-op. Unlike
    a global vector of point biases, the correction reuses local board features
    at every point. Orthogonal state evolution and OGD use the inherited online
    transactions unchanged. Modulation itself is not an orthogonal operation.
    """

    def __init__(self, input_dim: int, **kwargs: Any) -> None:
        super().__init__(input_dim, **kwargs)
        if self.spatial_skip_heads is None:
            raise ValueError("spatial modulation requires spatial_skip_channels > 0")
        self.ring_modulation = torch.nn.Linear(
            self.latent_dim, 2 * self.spatial_skip_channels, bias=False,
        )
        torch.nn.init.zeros_(self.ring_modulation.weight)
        # Predictions are supplied by the spatial base and its modulation.
        # Disable unused global point heads; resource counts use requires_grad.
        for head in (self.placement_head, self.legality_head, self.pass_head, self.value_head):
            for parameter in head.parameters():
                parameter.requires_grad_(False)

    def _head_output(
        self, latent: torch.Tensor, x: Optional[torch.Tensor] = None,
    ) -> GoMultiHeadOutput:
        if x is None:
            raise ValueError("spatially modulated readout requires the current board features")
        base = self.spatial_skip_heads
        latent = torch.tanh(latent)
        planes = x[:, :3 * self.points].reshape(-1, 3, self.board_size, self.board_size)
        hidden = torch.tanh(base.local(planes))
        gamma, beta = self.ring_modulation(latent).chunk(2, dim=1)
        hidden = hidden * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        pooled = hidden.mean(dim=(2, 3))
        if base.extra_dim:
            pooled = torch.cat((pooled, x[:, 3 * self.points:]), dim=1)
        return GoMultiHeadOutput(
            placement_logits=base.placement(hidden).flatten(1),
            legality_logits=base.legality(hidden).flatten(1),
            pass_logit=base.pass_decision(pooled).squeeze(1),
            value=base.value(pooled).squeeze(1).tanh(),
            latent=latent, legality_policy_scale=self.legality_policy_scale,
        )

    def _named_task_parameters(self) -> List[Tuple[str, torch.nn.Parameter]]:
        return super()._named_task_parameters() + [
            (f"ring_modulation.{name}", parameter)
            for name, parameter in self.ring_modulation.named_parameters()
        ]

    def _causal_state_output(
        self, state: TemporalMQRState, previous_input: Optional[torch.Tensor]
    ) -> GoMultiHeadOutput:
        reference = state.rings[0]
        features = (
            reference.new_zeros(reference.size(0), self.input_dim)
            if previous_input is None else previous_input.to(reference)
        )
        return self._head_output(self.core.readout_state(state), features)


class GoOnlineSession:
    """Single game stream with feedback delayed by at most ``update_every`` plies.

    Call ``observe`` before querying a teacher, then ``feedback``. ``step``
    combines these in that order. Call ``flush`` at a game boundary; state and
    pending feedback are included in ``state_dict`` for exact continuation.
    The session owns the agent's updates while a window is open.
    """

    def __init__(
        self,
        agent: TemporalUtilityMQRAgent,
        encoder: GoVectorEncoder,
        *,
        update_every: int = 8,
        stream_id: str = "online-go",
        project_with_memory: bool = True,
    ) -> None:
        if not 1 <= update_every <= agent.max_trace_horizon:
            raise ValueError("update_every must be within the agent trace horizon")
        if agent.board_size != encoder.board_size or agent.input_dim != encoder.output_dim:
            raise ValueError("agent and encoder dimensions must match")
        self.agent = agent
        self.encoder = encoder
        self.update_every = int(update_every)
        self.stream_id = str(stream_id)
        self.project_with_memory = bool(project_with_memory)
        self._pending: List[Dict[str, Any]] = []

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def online_tensor_bytes(self) -> int:
        """Occupied runtime tensor storage, excluding parameters and autograd."""
        storages: Dict[Tuple[str, int], int] = {}

        def visit(value: Any) -> None:
            if isinstance(value, torch.Tensor):
                storage = value.untyped_storage()
                storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
            elif dataclasses.is_dataclass(value):
                for field in dataclasses.fields(value):
                    visit(getattr(value, field.name))
            elif isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child)

        for value in (
            self.agent._stream_states, self.agent._feature_history,
            self.agent._pending_tickets, self.agent.task_gradient_memory._basis,
            self.agent.utility_gradient_memory._basis, self._pending,
        ):
            visit(value)
        return sum(storages.values())

    def observe(self, board: GoBoard, *, learn: bool = True) -> Dict[str, Any]:
        """Commit a label-free prediction, then constrain the executed action."""
        if board.game_over:
            raise ValueError("cannot observe a terminated game")
        if board.size != self.encoder.board_size:
            raise ValueError("board size does not match the session")
        if len(self._pending) >= self.update_every:
            raise RuntimeError("the feedback window is full; provide feedback and flush")
        if not learn and self._pending:
            raise RuntimeError("flush training feedback before evaluating")
        reference = next(self.agent.parameters())
        features = self.encoder.encode_board(board).to(reference)
        result = self.agent.commit_step(
            features, stream_id=self.stream_id, external_write=True,
            issue_ticket=learn,
        )
        logits = result["policy_logits"]
        raw_action = int(logits.argmax(dim=1).item())
        legal = torch.tensor(board.legal_moves(), device=logits.device, dtype=torch.long)
        action = int(legal[logits[0, legal].argmax()].item())
        if learn:
            ticket = int(result["ticket_id"])
            # This experiment studies recurrence and OGD with an open write
            # gate. No unused counterfactual-utility labels are manufactured.
            self.agent.cancel_ticket_branch(ticket, "utility")
            self._pending.append({
                "ticket_id": ticket,
                "legality": legality_target(board, device=reference.device),
                "feedback": None,
            })
        return {
            **result, "raw_action": raw_action, "action": action,
            "raw_legal": board.is_legal(raw_action),
            "action_mask_applied": True,
        }

    def feedback(
        self, ticket_id: int, target_action: int, *, value_target: Optional[float] = None
    ) -> None:
        """Attach a teacher action and optional already revealed game outcome."""
        matches = [item for item in self._pending if item["ticket_id"] == ticket_id]
        if not matches:
            raise ValueError("ticket does not belong to this pending window")
        item = matches[0]
        if item["feedback"] is not None:
            raise RuntimeError("feedback was already supplied")
        if not 0 <= target_action < self.agent.action_size:
            raise ValueError("target action is out of range")
        if target_action < self.agent.points and not bool(item["legality"][0, target_action]):
            raise ValueError("teacher action must be legal in the observed position")
        reference = next(self.agent.parameters())
        if value_target is not None and not -1.0 <= value_target <= 1.0:
            raise ValueError("value_target must be finite and within [-1, 1]")
        values: Dict[str, torch.Tensor] = {
            "action": torch.tensor([target_action], device=reference.device),
            "legality_target": item["legality"],
        }
        if value_target is not None:
            values["value_target"] = reference.new_tensor([value_target])
        item["feedback"] = values

    def flush(self) -> Optional[Dict[str, Any]]:
        """Update once from the committed window; keep live state unchanged."""
        if not self._pending:
            return None
        if any(item["feedback"] is None for item in self._pending):
            raise RuntimeError("every prediction in the window needs feedback")
        ids = [item["ticket_id"] for item in self._pending]
        feedback = [item["feedback"] for item in self._pending]
        if len(ids) == 1:
            result = self.agent.apply_feedback(
                ids[0], **feedback[0], project_with_memory=self.project_with_memory,
            )
        else:
            result = self.agent.apply_trajectory_feedback(
                ids, feedback, project_with_memory=self.project_with_memory,
            )
        self._pending.clear()
        return result

    def step(self, board: GoBoard, teacher: Any, *, learn: bool = True) -> Dict[str, Any]:
        """Predict, query ``teacher.select_move(board)``, then possibly update."""
        result = self.observe(board, learn=learn)
        target = int(teacher.select_move(board.copy()))
        if not board.is_legal(target):
            raise ValueError("teacher returned an illegal action")
        update = None
        if learn:
            self.feedback(int(result["ticket_id"]), target)
            if len(self._pending) == self.update_every:
                update = self.flush()
        return {**result, "teacher_action": target, "update": update}

    def reset_game(self) -> None:
        """Start an independent game after consuming the previous game's labels."""
        if self._pending:
            raise RuntimeError("flush pending feedback before resetting a game")
        self.agent.reset_state(self.stream_id)

    def state_dict(self) -> Dict[str, Any]:
        """Snapshot weights, OGD, committed state, and pending feedback tensors."""
        return copy.deepcopy({
            "version": 1, "update_every": self.update_every,
            "stream_id": self.stream_id, "project_with_memory": self.project_with_memory,
            "agent": self.agent.state_dict(), "encoder": self.encoder.state_dict(),
            "pending": self._pending,
        })

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore into a session with the same architecture and update protocol."""
        if state.get("version") != 1 or state.get("update_every") != self.update_every:
            raise ValueError("incompatible session version or feedback window")
        if state.get("stream_id") != self.stream_id:
            raise ValueError("checkpoint stream_id does not match the session")
        if bool(state.get("project_with_memory")) != self.project_with_memory:
            raise ValueError("checkpoint OGD policy does not match the session")
        self.agent.load_state_dict(state["agent"])
        self.encoder.load_state_dict(state["encoder"])
        reference = next(self.agent.parameters())
        self._pending = copy.deepcopy(state["pending"])
        for item in self._pending:
            item["legality"] = item["legality"].to(reference)
            if item["feedback"] is not None:
                for key, value in item["feedback"].items():
                    item["feedback"][key] = value.to(
                        device=reference.device,
                        dtype=torch.long if key == "action" else reference.dtype,
                    )


@torch.enable_grad()
def consolidate_task_memory(
    agent: TemporalUtilityMQRAgent,
    anchors: Sequence[Sequence[Tuple[torch.Tensor, int]]],
) -> Dict[str, Any]:
    """Replace OGD with current output gradients on previously seen traces.

    One anchor contributes the gradient of its final target log-probability,
    replaying its label-free inputs from zero state. No parameter, live state,
    or counter is changed. The caller must use training data, not test probes.
    Only the resulting basis persists; historical loss gradients are replaced.
    Orthogonality is a first-order local constraint, not a no-forgetting theorem.
    """
    if agent.pending_ticket_count:
        raise RuntimeError("consume pending feedback before consolidating OGD")
    active = [(name, parameter) for name, parameter in agent._named_task_parameters()
              if parameter.requires_grad]
    if not active or agent.task_gradient_memory.max_rank == 0:
        raise ValueError("consolidation needs trainable parameters and OGD capacity")
    reference = active[0][1]
    memory = OrthogonalGradientMemory(
        agent.task_gradient_memory.max_rank,
        tolerance=agent.task_gradient_memory.tolerance,
    ).to(reference)
    evaluated = 0
    for trace in anchors:
        if memory.is_full:
            break
        if not trace or len(trace) > agent.max_trace_horizon:
            raise ValueError("anchor length must be within the trace horizon")
        state = agent.core.zero_state(1, device=reference.device, dtype=reference.dtype)
        for features, action in trace:
            if features.shape != (1, agent.input_dim):
                raise ValueError("anchor features must have shape [1, input_dim]")
            if not 0 <= action < agent.action_size:
                raise ValueError("anchor action is out of range")
            output, state = agent._transition(features.detach().to(reference), state, slow_write=True)
        score = output.policy_logits[0, action]
        gradients = torch.autograd.grad(score, [p for _, p in active], allow_unused=True)
        memory.observe([
            (name, torch.zeros_like(p) if gradient is None else gradient, agent.task_lr)
            for (name, p), gradient in zip(active, gradients)
        ])
        evaluated += 1
    if memory.rank == 0:
        raise ValueError("anchors produced no nonzero output gradients")
    agent.task_gradient_memory.restore(memory.snapshot())
    return {
        "rank": memory.rank, "anchors_evaluated": evaluated,
        "storage_bytes": memory.storage_bytes,
        "orthogonality_error": memory.orthogonality_error(),
        "parameter_version": int(agent.online_parameter_version),
        "score": "current_target_log_probability", "state_mutated": False,
    }
