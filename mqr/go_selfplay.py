"""Shared-policy outcome learning with complete, unsupervised opening context."""
from __future__ import annotations

import copy
from typing import Any, Dict

import torch

from .go_cone import TaskMarginMemory
from .go_outcome import SearchOutcomeGoSession, _board_from_record, _board_record


class SharedPolicyGoSession(SearchOutcomeGoSession):
    """Both colors must supply pre-feedback policy targets in shared-policy mode.

    External matches use learn=False and never train the self-play value head.
    Opening context is replayed without labels, so forced-prefix return targets
    do not masquerade as shared-policy rollouts.
    """

    def __init__(self, *args: Any, shared_policy: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.shared_policy = bool(shared_policy)
        if isinstance(self.behavior_memory, TaskMarginMemory) and (
            self.project_with_memory or self.max_anchor_kl is not None or self.norm_matched_sgd
        ):
            raise ValueError("margin halfspaces use their own projection and no probability-KL guard")
        self.context = []
        self.context_board = None

    def observe_context(self, board) -> Dict[str, Any]:
        if self._episode or self._pending or len(self.context) >= self.episode_capacity:
            raise RuntimeError("opening context must precede the learning episode within capacity")
        self._check_context_successor(board)
        result = super().observe(board, learn=False)
        self.context.append(self._encode_observation(board).detach().cpu().clone())
        self.context_board = _board_record(board)
        return result

    def _check_context_successor(self, board) -> None:
        if self.context_board is not None:
            previous = _board_from_record(self.context_board)
            if len(board.move_history) != len(previous.move_history) + 1:
                raise ValueError("opening context must observe consecutive actual plies")
            previous.play(board.move_history[-1])
            if _board_record(previous) != _board_record(board):
                raise ValueError("opening context and episode are not legal successors")

    def observe(self, board, *, learn: bool = True) -> Dict[str, Any]:
        if learn and not self._episode:
            self._check_context_successor(board)
        return super().observe(board, learn=learn)

    def feedback(self, ticket_id: int, target_action: int, *, policy_target=None, value_target=None) -> None:
        if self.shared_policy and policy_target is None:
            raise ValueError("shared-policy returns require search targets on both colors' turns")
        super().feedback(ticket_id, target_action, policy_target=policy_target, value_target=value_target)

    def _prime_replay(self, stream_id: str) -> None:
        reference = next(self.agent.parameters())
        for features in self.context:
            self.agent.commit_step(features.to(reference), stream_id=stream_id,
                                   external_write=True, issue_ticket=False)

    def finish_game(self, board) -> Dict[str, Any]:
        context_count = len(self.context)
        result = super().finish_game(board)
        result.update(value_semantics="shared_policy" if self.shared_policy else "mixed_behavior",
                      replayed_context=context_count)
        self.context.clear()
        self.context_board = None
        return result

    def abort_game(self) -> Dict[str, Any]:
        result = super().abort_game()
        self.context.clear()
        self.context_board = None
        return result

    def reset_game(self) -> None:
        super().reset_game()
        self.context.clear()
        self.context_board = None

    @property
    def online_tensor_bytes(self) -> int:
        return super().online_tensor_bytes + sum(x.numel() * x.element_size() for x in self.context)

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["shared_policy"] = copy.deepcopy({"version": 1, "enabled": self.shared_policy,
                                               "context": self.context, "board": self.context_board})
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        config = state.get("shared_policy", {})
        if config.get("version") != 1 or config.get("enabled") != self.shared_policy:
            raise ValueError("shared-policy value semantics differ")
        context = config["context"]
        if (len(context) > self.episode_capacity or bool(context) != (config["board"] is not None)
                or any(x.shape != (1, self.agent.input_dim) or not bool(torch.isfinite(x).all()) for x in context)):
            raise ValueError("invalid saved opening context")
        raw = state["protection"]["memory"]
        candidate = None
        if raw is not None and "task_margin" in raw:
            if not isinstance(self.behavior_memory, TaskMarginMemory):
                raise ValueError("configured protection is not task-margin projection")
            candidate = TaskMarginMemory(raw["capacity"], strata=raw["strata"], reliable_only=True,
                margin_fraction=raw["policy_selection"]["margin_fraction"],
                competitors=raw["task_margin"]["competitors"],
                backend=raw["constraint_transport"]["backend"],
                selection_tolerance=raw["constraint_transport"]["selection_tolerance"])
            candidate.load_state_dict(raw)
        elif isinstance(self.behavior_memory, TaskMarginMemory):
            raise ValueError("saved protection is missing margin halfspaces")
        super().load_state_dict(state)
        if candidate is not None:
            self.behavior_memory = candidate
        self.context, self.context_board = copy.deepcopy(context), copy.deepcopy(config["board"])
