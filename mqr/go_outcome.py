"""Reliable reference selection and bounded, real-terminal Go replay."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, Dict, List, Optional

import torch

from .go import GoBoard
from .go_constraints import ConstraintGoBehaviorMemory, ConstraintGoSession
from .go_search import policy_value_search


class PolicyBehaviorMemory(ConstraintGoBehaviorMemory):
    """Optionally protect only references whose argmax matches the teacher.

    Agreement is a limited train-time reliability criterion, not a claim of
    optimal Go behavior. Selection never uses held-out games or win rates.
    """

    def __init__(self, *args: Any, reliable_only: bool = False,
                 margin_fraction: Optional[float] = None, **kwargs: Any) -> None:
        if margin_fraction is not None and (not reliable_only or not 0 < margin_fraction <= 1):
            raise ValueError("margin protection requires reliable anchors and a fraction in (0, 1]")
        super().__init__(*args, **kwargs)
        self.reliable_only = bool(reliable_only)
        self.margin_fraction = margin_fraction
        self.candidate_count = 0
        self.agreement_count = 0

    @torch.no_grad()
    def freeze(self, agent) -> None:
        if self.frozen or agent.pending_ticket_count or not self.anchors:
            raise RuntimeError("freeze requires fresh, nonempty anchors and resolved feedback")
        groups, agreements = [], 0
        for group in self.records:
            selected = []
            for record in group:
                reference = self._output(agent, record).detach().cpu()
                correct = int(reference.argmax()) == record["action"]
                agreements += int(correct)
                if correct or not self.reliable_only:
                    saved = {**record, "reference_log_probs": reference}
                    if self.margin_fraction is not None:
                        alternatives = reference.clone()
                        alternatives[record["action"]] = -torch.inf
                        margin = float(reference[record["action"]] - alternatives.max())
                        if margin <= 0:
                            continue
                        saved["margin_floor"] = self.margin_fraction * margin
                    selected.append(saved)
            groups.append(selected)
        self.candidate_count, self.agreement_count = len(self.anchors), agreements
        self.records = groups
        self.frozen = True

    def refresh(self, agent) -> Dict[str, Any]:
        if self.anchors:
            return super().refresh(agent)
        if not self.frozen or agent.pending_ticket_count:
            raise RuntimeError("refresh requires frozen anchors and no pending feedback")
        agent.task_gradient_memory.clear()
        self.last_refresh_version = int(agent.online_parameter_version)
        self.refresh_count += 1
        self.last_diagnostics = {
            "backend": self.backend, "rank": 0, "gradient_directions": 0,
            "selected_candidates": [], "mean_gradient_energy_coverage": 0.0,
            "min_gradient_energy_coverage": 0.0, "orthogonality_error": 0.0,
            "parameter_version": self.last_refresh_version,
            "max_adjoint_norm_error": 0.0, "peak_state_trace_bytes": 0,
            "max_credit_steps": 0, "refresh_seconds": 0.0,
        }
        return self.last_diagnostics

    def drift(self, agent) -> Dict[str, float]:
        if self.anchors or not self.frozen:
            result = super().drift(agent)
            if self.margin_fraction is not None:
                violations = []
                for record in self.anchors:
                    with torch.no_grad():
                        scores = self._output(agent, record)
                        target = scores[record["action"]].clone()
                        alternatives = scores.clone()
                        alternatives[record["action"]] = -torch.inf
                        violations.append(max(0.0, record["margin_floor"] - float(target - alternatives.max())))
                result["max_margin_violation"] = max(violations, default=0.0)
            return result
        return {"max_policy_kl": 0.0, "mean_policy_kl": 0.0,
                "max_target_log_probability_drift": 0.0, "max_margin_violation": 0.0}

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["policy_selection"] = {
            "version": 1, "reliable_only": self.reliable_only,
            "candidate_count": self.candidate_count, "agreement_count": self.agreement_count,
        }
        if self.margin_fraction is not None:
            state["policy_selection"]["margin_fraction"] = self.margin_fraction
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        config = state.get("policy_selection", {})
        if (config.get("version") != 1 or config.get("reliable_only") != self.reliable_only
                or config.get("margin_fraction") != self.margin_fraction):
            raise ValueError("reference selection configuration differs")
        candidates, agreements = config["candidate_count"], config["agreement_count"]
        if (not isinstance(candidates, int) or not isinstance(agreements, int)
                or not 0 <= agreements <= candidates <= self.capacity):
            raise ValueError("invalid reference selection counts")
        super().load_state_dict(state)
        self.candidate_count, self.agreement_count = candidates, agreements


def _board_record(board: GoBoard) -> Dict[str, Any]:
    return {"size": board.size, "komi": board.komi, "board": list(board.board),
            "to_play": board.to_play, "consecutive_passes": board.consecutive_passes,
            "game_over": board.game_over, "move_history": list(board.move_history),
            "position_history": sorted(board.position_history)}


def _board_from_record(record: Dict[str, Any]) -> GoBoard:
    board = GoBoard(record["size"], komi=record["komi"])
    for key in ("board", "to_play", "consecutive_passes", "game_over", "move_history"):
        setattr(board, key, copy.deepcopy(record[key]))
    board.position_history = set(record["position_history"])
    return board


class OutcomeGoSession(ConstraintGoSession):
    """Causal imitation plus optional terminal policy/value replay.

    replay_mode='policy' is the update-count control for 'outcome'. Both
    re-predict recorded observations with current parameters and fresh tickets,
    retaining the ordinary OGD/KL guard. Only outcome mode supplies Monte Carlo
    winner * side_to_play targets. This is auxiliary value learning under the
    observed behavior, not a search-policy target or policy-gradient algorithm.
    """

    def __init__(
        self, *args: Any, replay_mode: str = "none", episode_capacity: int = 100,
        outcome_weight: float = 1.0, **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if replay_mode not in ("none", "policy", "outcome"):
            raise ValueError("unknown terminal replay mode")
        if isinstance(episode_capacity, bool) or not isinstance(episode_capacity, int) or episode_capacity < 1:
            raise ValueError("episode_capacity must be a positive integer")
        if not 0 < outcome_weight < float("inf"):
            raise ValueError("outcome_weight must be positive and finite")
        self.replay_mode, self.episode_capacity = replay_mode, episode_capacity
        self.outcome_weight = float(outcome_weight)
        self._episode: List[Dict[str, Any]] = []
        self._last_board: Optional[Dict[str, Any]] = None
        self.finished_games = self.aborted_games = self.replayed_positions = 0

    def _check_successor(self, board: GoBoard) -> None:
        if self._last_board is None:
            return
        previous = _board_from_record(self._last_board)
        if (len(board.move_history) != len(previous.move_history) + 1
                or board.move_history[:-1] != previous.move_history):
            raise ValueError("episode must observe each actual ply of the same game")
        previous.play(board.move_history[-1])
        if _board_record(previous) != _board_record(board):
            raise ValueError("board is not the legal successor of the recorded game")

    def observe(self, board: GoBoard, *, learn: bool = True) -> Dict[str, Any]:
        if learn:
            if len(self._episode) >= self.episode_capacity:
                raise RuntimeError("episode capacity reached; finish or explicitly abort this game")
            self._check_successor(board)
            features = self._encode_observation(board).detach().cpu().clone()
        elif self._episode:
            raise RuntimeError("finish or abort the training game before evaluation")
        result = super().observe(board, learn=learn)
        if learn:
            self._episode.append({
                "features": features, "ticket_id": int(result["ticket_id"]), "to_play": board.to_play,
                "value_before_feedback": float(result["output"].value[0]), "feedback": None,
            })
            self._last_board = _board_record(board)
        return result

    def feedback(self, ticket_id: int, target_action: int, *, value_target: Optional[float] = None) -> None:
        if value_target is not None:
            raise ValueError("terminal labels must be supplied by finish_game on a real terminal board")
        super().feedback(ticket_id, target_action)
        record = next(item for item in self._episode if item["ticket_id"] == ticket_id)
        pending = next(item for item in self._pending if item["ticket_id"] == ticket_id)
        record["feedback"] = {name: tensor.detach().cpu().clone()
                              for name, tensor in pending["feedback"].items()}

    def step(
        self, board: GoBoard, teacher: Any, *, learn: bool = True,
        simulations: int = 0, search_depth: int = 8,
    ) -> Dict[str, Any]:
        if (isinstance(simulations, bool) or not isinstance(simulations, int) or simulations < 0
                or isinstance(search_depth, bool) or not isinstance(search_depth, int) or search_depth < 1):
            raise ValueError("invalid search budget")
        if simulations and (self.encoder.history_slots or self.agent.history_slots):
            raise ValueError("search branch memory is not implemented; history_slots must be zero")
        result = self.observe(board, learn=learn)
        search = None
        if simulations:
            # Search is part of the pre-feedback decision, even at a full window.
            search = policy_value_search(self.agent, self.encoder, board, result["state"],
                                         result["policy_logits"], simulations=simulations,
                                         max_depth=search_depth)
            result["action"] = search["action"]
        target = int(teacher.select_move(board.copy()))
        if not board.is_legal(target):
            raise ValueError("teacher returned an illegal action")
        update = None
        if learn:
            self.feedback(int(result["ticket_id"]), target)
            if self.pending_count == self.update_every:
                update = self.flush()
        return {**result, "teacher_action": target, "update": update, "search": search}

    def finish_game(self, board: GoBoard) -> Dict[str, Any]:
        """Consume a verified terminal episode exactly once; return update audits."""
        if not board.game_over or not self._episode:
            raise ValueError("a recorded, truly terminal game is required")
        self._check_successor(board)
        if any(record["feedback"] is None for record in self._episode):
            raise RuntimeError("all observed positions require action feedback before terminal replay")
        update = self.flush()
        winner = board.winner()
        targets = [float(winner * record["to_play"]) for record in self._episode]
        mse = sum((record["value_before_feedback"] - target) ** 2
                  for record, target in zip(self._episode, targets)) / len(targets)
        replay_updates = []
        if self.replay_mode != "none":
            replay_stream = self.stream_id + "-terminal-replay"
            replay = ConstraintGoSession(
                self.agent, self.encoder, update_every=self.update_every, stream_id=replay_stream,
                credit_horizon=self.credit_horizon, project_with_memory=self.project_with_memory,
                behavior_memory=self.behavior_memory, refresh_every=self.refresh_every,
                max_anchor_kl=self.max_anchor_kl, norm_matched_sgd=self.norm_matched_sgd,
            )
            self.agent.reset_state(replay_stream)
            original_weights = self.agent.loss_weights
            reference = next(self.agent.parameters())
            try:
                self.agent.loss_weights = replace(
                    original_weights, value=self.outcome_weight if self.replay_mode == "outcome" else 0.0,
                )
                for record, target in zip(self._episode, targets):
                    memory = self.behavior_memory
                    if not replay.pending_count and memory is not None and memory.frozen:
                        last = memory.last_refresh_version
                        if last is None or int(self.agent.online_parameter_version) - last >= self.refresh_every:
                            replay.refresh_protection()
                    prediction = self.agent.commit_step(record["features"].to(reference),
                                                        stream_id=replay_stream, external_write=True)
                    ticket = int(prediction["ticket_id"])
                    self.agent.cancel_ticket_branch(ticket, "utility")
                    feedback = {key: value.to(device=reference.device,
                                              dtype=torch.long if key == "action" else reference.dtype)
                                for key, value in record["feedback"].items()}
                    if self.replay_mode == "outcome":
                        feedback["value_target"] = reference.new_tensor([target])
                    replay._pending.append({"ticket_id": ticket, "legality": feedback["legality_target"],
                                            "feedback": feedback})
                    if replay.pending_count == replay.update_every:
                        replay_updates.append(replay.flush())
                if replay.pending_count:
                    replay_updates.append(replay.flush())
            finally:
                self.agent.loss_weights = original_weights
            self.agent.reset_state(replay_stream)
            self.replayed_positions += len(self._episode)
        result = {
            "terminal": True, "winner": winner, "positions": len(self._episode),
            "prequential_value_mse": mse, "value_targets": targets if self.replay_mode == "outcome" else [],
            "replayed_positions": len(self._episode) if self.replay_mode != "none" else 0,
            "online_update": update, "replay_updates": replay_updates,
        }
        self.finished_games += 1
        self._episode.clear()
        self._last_board = None
        return result

    def abort_game(self) -> Dict[str, Any]:
        """Close a truncated episode without manufacturing terminal value labels."""
        update = self.flush()
        count = len(self._episode)
        self._episode.clear()
        self._last_board = None
        self.aborted_games += int(count > 0)
        return {"terminal": False, "positions": count, "replayed_positions": 0,
                "value_targets": [], "online_update": update, "replay_updates": []}

    def reset_game(self) -> None:
        if self._episode:
            raise RuntimeError("finish or explicitly abort the recorded game before resetting")
        super().reset_game()

    @property
    def online_tensor_bytes(self) -> int:
        tensors = [record["features"] for record in self._episode]
        tensors += [value for record in self._episode if record["feedback"] is not None
                    for value in record["feedback"].values()]
        return super().online_tensor_bytes + sum(x.numel() * x.element_size() for x in tensors)

    def _outcome_config(self) -> Dict[str, Any]:
        return {"version": 1, "replay_mode": self.replay_mode, "episode_capacity": self.episode_capacity,
                "outcome_weight": self.outcome_weight}

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["outcome"] = copy.deepcopy({
            "config": self._outcome_config(), "episode": self._episode, "last_board": self._last_board,
            "finished_games": self.finished_games, "aborted_games": self.aborted_games,
            "replayed_positions": self.replayed_positions,
        })
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        outcome = state.get("outcome", {})
        if outcome.get("config") != self._outcome_config():
            raise ValueError("terminal replay checkpoint configuration differs")
        episode = outcome["episode"]
        if len(episode) > self.episode_capacity or any(
            item["features"].shape != (1, self.agent.input_dim)
            or not bool(torch.isfinite(item["features"]).all()) or item["to_play"] not in (-1, 1)
            for item in episode
        ):
            raise ValueError("invalid terminal replay observations")
        if bool(episode) != (outcome["last_board"] is not None):
            raise ValueError("terminal replay board and observation count disagree")
        raw = state["protection"]["memory"]
        candidate = None
        if raw is not None:
            selection, transport = raw["policy_selection"], raw["constraint_transport"]
            candidate = PolicyBehaviorMemory(
                raw["capacity"], strata=raw["strata"], reliable_only=selection["reliable_only"],
                margin_fraction=selection.get("margin_fraction"),
                backend=transport["backend"], selection_tolerance=transport["selection_tolerance"],
            )
            candidate.load_state_dict(raw)
            if self.behavior_memory is not None and (
                not isinstance(self.behavior_memory, PolicyBehaviorMemory)
                or self.behavior_memory.reliable_only != candidate.reliable_only
                or self.behavior_memory.margin_fraction != candidate.margin_fraction
            ):
                raise ValueError("restored reference selection differs from configured memory")
        for name in ("finished_games", "aborted_games", "replayed_positions"):
            if not isinstance(outcome[name], int) or outcome[name] < 0:
                raise ValueError("invalid terminal replay counters")
        super().load_state_dict(state)
        self.behavior_memory = candidate
        self._episode, self._last_board = copy.deepcopy(episode), copy.deepcopy(outcome["last_board"])
        for name in ("finished_games", "aborted_games", "replayed_positions"):
            setattr(self, name, outcome[name])


class SearchOutcomeGoSession(OutcomeGoSession):
    """Train from pre-feedback search policies and verified terminal outcomes.

    feedback() records an actually selected legal action; no expert action is
    required. A missing policy_target omits policy distillation for that ply.
    Raw rules may still supervise legality. Terminal replay uses fresh tickets
    and the original detached search targets, never a fabricated terminal.
    """

    def feedback(self, ticket_id: int, target_action: int, *,
                 policy_target: Optional[torch.Tensor] = None,
                 value_target: Optional[float] = None) -> None:
        if policy_target is not None:
            item = next((item for item in self._pending if item["ticket_id"] == ticket_id), None)
            if item is None:
                raise ValueError("ticket does not belong to this pending window")
            target = policy_target.detach().to(item["legality"])
            if (target.shape != (1, self.agent.action_size)
                    or not bool(torch.isfinite(target).all()) or bool((target < 0).any())
                    or abs(float(target.sum()) - 1.0) > 1e-5
                    or bool((target[:, :self.agent.points] * (1 - item["legality"]) > 0).any())):
                raise ValueError("search target must be normalized, finite and supported on legal actions")
        super().feedback(ticket_id, target_action, value_target=value_target)
        if policy_target is not None:
            pending = next(item for item in self._pending if item["ticket_id"] == ticket_id)
            pending["feedback"]["policy_target"] = target.clone()
            record = next(item for item in self._episode if item["ticket_id"] == ticket_id)
            record["feedback"]["policy_target"] = target.cpu().clone()

    @staticmethod
    def search_target(search: Dict[str, Any], action_size: int) -> torch.Tensor:
        """Normalize positive root visit counts into a detached [1,A] target."""
        target = torch.zeros(1, action_size)
        for action, visits in search["visits"].items():
            if not 0 <= int(action) < action_size or visits < 0:
                raise ValueError("invalid search visits")
            target[0, int(action)] = visits
        if not bool(torch.isfinite(target).all()) or float(target.sum()) <= 0:
            raise ValueError("search distillation requires positive finite visits")
        return target / target.sum()
