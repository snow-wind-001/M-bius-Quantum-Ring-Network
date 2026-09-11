"""Read-only, bounded policy/value PUCT for the research Go agents."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict

import torch

from .go import GoBoard
from .go_memory import GoHistoryEncoder, PositionQueryGoAgent
from .temporal import TemporalMQRState


@dataclass
class _Node:
    board: GoBoard
    state: TemporalMQRState
    priors: Dict[int, float]
    value: float
    visits: Dict[int, int] = field(default_factory=dict)
    totals: Dict[int, float] = field(default_factory=dict)
    children: Dict[int, "_Node"] = field(default_factory=dict)


@torch.no_grad()
def policy_value_search(
    agent: PositionQueryGoAgent, encoder: GoHistoryEncoder, board: GoBoard,
    root_state: TemporalMQRState, root_log_probs: torch.Tensor, *,
    simulations: int = 16, max_depth: int = 8, exploration: float = 1.5,
) -> Dict[str, Any]:
    """Search from the state AFTER observing board; never commit branch states.

    Leaf values use the side-to-play convention and flip sign on every edge.
    Terminal values use the actual two-pass outcome. There are no teacher
    queries, training updates, rollout heuristics or transposition merging.
    """
    if (isinstance(simulations, bool) or not isinstance(simulations, int) or simulations < 0
            or isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1
            or not math.isfinite(exploration) or exploration < 0):
        raise ValueError("invalid search budget")
    if board.game_over or board.size != agent.board_size or encoder.output_dim != agent.input_dim:
        raise ValueError("search requires a live matching board and encoder")
    if encoder.history_slots or agent.history_slots:
        raise ValueError("search branch observation memory is not implemented; history_slots must be zero")
    if root_log_probs.shape != (1, board.action_size) or not bool(torch.isfinite(root_log_probs).all()):
        raise ValueError("root_log_probs must be finite [1, action_size]")

    def priors(position: GoBoard, log_probs: torch.Tensor) -> Dict[int, float]:
        legal = position.legal_moves()
        probabilities = log_probs[0, legal].softmax(0).tolist()
        return dict(zip(legal, probabilities))

    root = _Node(board.copy(), root_state.detached(), priors(board, root_log_probs), 0.0)
    evaluated = 0
    reference = next(agent.parameters())
    for _ in range(simulations):
        node, path = root, []
        for depth in range(max_depth):
            total = sum(node.visits.values())
            action = max(node.priors, key=lambda a: (
                node.totals.get(a, 0.0) / max(1, node.visits.get(a, 0))
                + exploration * node.priors[a] * math.sqrt(total + 1) / (1 + node.visits.get(a, 0)),
                node.priors[a], -a,
            ))
            path.append((node, action))
            fresh = action not in node.children
            if fresh:
                child_board = node.board.copy()
                child_board.play(action)
                if child_board.game_over:
                    child = _Node(child_board, node.state, {}, float(child_board.winner() * child_board.to_play))
                else:
                    features = encoder.encode_board(child_board).to(reference)
                    output, state = agent._transition(features, node.state, slow_write=True)
                    child = _Node(child_board, state.detached(), priors(child_board, output.policy_logits),
                                  float(output.value[0]))
                    evaluated += 1
                node.children[action] = child
            node = node.children[action]
            if fresh or node.board.game_over or depth + 1 == max_depth:
                value = node.value
                break
        for parent, action in reversed(path):
            value = -value
            parent.visits[action] = parent.visits.get(action, 0) + 1
            parent.totals[action] = parent.totals.get(action, 0.0) + value
    selected = max(root.priors, key=lambda a: (
        root.visits.get(a, 0),
        root.totals.get(a, 0.0) / max(1, root.visits.get(a, 0)),
        root.priors[a], -a,
    ))
    return {
        "action": selected, "simulations": simulations, "network_evaluations": evaluated,
        "visits": dict(root.visits),
        "action_values": {a: root.totals[a] / n for a, n in root.visits.items()},
    }
