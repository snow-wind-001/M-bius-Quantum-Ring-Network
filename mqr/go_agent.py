"""Deterministic Go trajectories and counterfactual utility targets.

This module is the simulator-facing part of the unified agent.  It deliberately
uses a frozen board encoder so core comparisons do not inherit language-model
or prompt-parsing confounders.  MiniCPM features can replace the encoder through
the same ``encode_board`` contract in larger experiments.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .agent import GoLossWeights, TemporalUtilityMQRAgent
from .go import BLACK, EMPTY, GoBoard, HeuristicGoTeacher, format_go_prompt


@dataclass(frozen=True)
class GoAgentExample:
    """One causal board observation and its delayed teacher targets."""

    board: GoBoard
    target_action: int
    value_target: float
    episode_index: int
    move_index: int
    focus_point: Optional[int] = None
    history_condition: str = ""


@dataclass(frozen=True)
class GoAgentTrajectory:
    """A fixed sequence from one simulated game."""

    examples: Tuple[GoAgentExample, ...]
    winner: int
    task: str


class GoVectorEncoder(nn.Module):
    """Frozen rule-neutral encoder with no explicit legal-move mask.

    The exact representation contains own stones, opponent stones, the previous
    placement, and six causal scalars.  ``projection_mode="random"`` preserves
    the historical seeded dense projection.  ``projection_mode="identity"``
    requires ``output_dim == base_dim`` and returns the exact representation,
    avoiding the irreversible 306-to-214 compression on a 10x10 board.
    Legality must still be learned from board structure rather than copied from
    an input mask.
    """

    def __init__(
        self,
        board_size: int,
        output_dim: int,
        *,
        seed: int = 0,
        projection_mode: str = "random",
    ) -> None:
        super().__init__()
        if board_size < 2 or output_dim <= 0:
            raise ValueError("board_size must be at least two and output_dim positive")
        projection_mode = str(projection_mode)
        if projection_mode not in ("random", "identity"):
            raise ValueError('projection_mode must be "random" or "identity"')
        self.board_size = int(board_size)
        self.points = self.board_size * self.board_size
        self.base_dim = 3 * self.points + 6
        self.output_dim = int(output_dim)
        self.projection_mode = projection_mode
        if self.projection_mode == "identity":
            if self.output_dim != self.base_dim:
                raise ValueError(
                    "identity projection requires output_dim == 3*board_size**2 + 6"
                )
            projection = torch.eye(self.base_dim, dtype=torch.float32)
        else:
            generator = torch.Generator().manual_seed(int(seed))
            projection = torch.randn(
                self.base_dim,
                self.output_dim,
                generator=generator,
                dtype=torch.float32,
            ) / math.sqrt(self.base_dim)
        self.register_buffer("projection", projection, persistent=True)

    def exact_features(self, board: GoBoard) -> torch.Tensor:
        if board.size != self.board_size:
            raise ValueError("board size does not match the encoder")
        own = [float(value == board.to_play) for value in board.board]
        opponent = [float(value == -board.to_play) for value in board.board]
        previous = [0.0] * self.points
        previous_was_pass = 0.0
        if board.move_history:
            action = int(board.move_history[-1])
            if action == board.pass_action:
                previous_was_pass = 1.0
            else:
                previous[action] = 1.0
        occupancy = 1.0 - board.board.count(EMPTY) / float(self.points)
        move_fraction = min(1.0, len(board.move_history) / float(2 * self.points))
        scalars = [
            float(board.to_play == BLACK),
            occupancy,
            min(1.0, board.consecutive_passes / 2.0),
            move_fraction,
            previous_was_pass,
            1.0,
        ]
        return torch.tensor(
            [own + opponent + previous + scalars],
            device=self.projection.device,
            dtype=self.projection.dtype,
        )

    def encode_board(
        self,
        board: GoBoard,
        *,
        require_encoder_grad: bool = False,
    ) -> torch.Tensor:
        if require_encoder_grad:
            raise ValueError("the frozen vector encoder has no adapter gradient")
        exact = self.exact_features(board)
        if self.projection_mode == "identity":
            return exact
        projected = exact @ self.projection
        return F.layer_norm(torch.tanh(projected), (self.output_dim,))


class MiniCPMGoBoardEncoder(nn.Module):
    """Adapt a :class:`MiniCPMLoRAEncoder` to the board-encoder contract."""

    def __init__(
        self,
        encoder: Any,
        *,
        max_length: int = 192,
        prompt_mode: str = "rules",
    ) -> None:
        super().__init__()
        if not hasattr(encoder, "encode_prompts") or not hasattr(encoder, "hidden_size"):
            raise TypeError("encoder must expose encode_prompts and hidden_size")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.encoder = encoder
        self.output_dim = int(encoder.hidden_size)
        self.max_length = int(max_length)
        self.prompt_mode = str(prompt_mode)

    def encode_board(
        self,
        board: GoBoard,
        *,
        require_encoder_grad: bool = False,
    ) -> torch.Tensor:
        features = self.encoder.encode_prompts(
            format_go_prompt(board, prompt_mode=self.prompt_mode),
            max_length=self.max_length,
            require_lora_grad=bool(require_encoder_grad),
        )
        return F.layer_norm(features, (features.size(-1),))


def legality_target(board: GoBoard, *, device: Optional[torch.device] = None) -> torch.Tensor:
    """Return point-only legality labels with shape ``[1, size**2]``."""

    return torch.tensor(
        [[float(board.is_legal(point)) for point in range(board.pass_action)]],
        device=device,
        dtype=torch.float32,
    )


def example_targets(
    example: GoAgentExample,
    *,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([example.target_action], device=device, dtype=torch.long),
        legality_target(example.board, device=device),
        torch.tensor([example.value_target], device=device, dtype=torch.float32),
    )


def generate_go_agent_trajectories(
    episode_count: int,
    *,
    size: int = 3,
    komi: float = 2.5,
    seed: int = 0,
    task: str = "opening",
    start_random_moves: int = 0,
    recorded_moves: int = 8,
    random_move_probability: float = 0.35,
    teacher: Optional[HeuristicGoTeacher] = None,
) -> List[GoAgentTrajectory]:
    """Generate actual legal games and attach final-outcome value targets.

    ``start_random_moves`` creates a distribution shift without an explicit task
    marker.  It is used for the tactical second phase in continual-learning
    tests, whereas the opening phase starts from an empty board.
    """

    if episode_count <= 0 or recorded_moves <= 0 or start_random_moves < 0:
        raise ValueError("trajectory counts must be positive and prefix non-negative")
    if not (0.0 <= random_move_probability <= 1.0):
        raise ValueError("random_move_probability must lie in [0, 1]")
    resolved_teacher = teacher or HeuristicGoTeacher()
    rng = random.Random(int(seed))
    trajectories: List[GoAgentTrajectory] = []
    for episode_index in range(int(episode_count)):
        board = GoBoard(size, komi=komi)
        for _ in range(int(start_random_moves)):
            legal_stones = board.legal_moves(include_pass=False)
            if not legal_stones:
                board.play(board.pass_action)
                if board.game_over:
                    break
            else:
                board.play(rng.choice(legal_stones))

        pending: List[Tuple[GoBoard, int, int]] = []
        for move_index in range(int(recorded_moves)):
            if board.game_over:
                break
            snapshot = board.copy()
            target = int(resolved_teacher.select_move(snapshot))
            pending.append((snapshot, target, move_index))
            legal_stones = board.legal_moves(include_pass=False)
            if legal_stones and rng.random() < float(random_move_probability):
                played = rng.choice(legal_stones)
            else:
                played = target
            board.play(int(played))

        winner = int(board.winner())
        examples = tuple(
            GoAgentExample(
                board=snapshot,
                target_action=target,
                value_target=float(winner * snapshot.to_play),
                episode_index=episode_index,
                move_index=move_index,
            )
            for snapshot, target, move_index in pending
        )
        if examples:
            trajectories.append(GoAgentTrajectory(examples, winner, str(task)))
    if len(trajectories) != episode_count:
        raise RuntimeError("failed to generate the requested number of trajectories")
    return trajectories


def _transform_point(point: int, size: int, symmetry: int) -> int:
    row, column = divmod(int(point), int(size))
    transform = int(symmetry) % 8
    if transform >= 4:
        column = size - 1 - column
        transform -= 4
    for _ in range(transform):
        row, column = column, size - 1 - row
    return row * size + column


def generate_ko_history_pairs(
    pair_count: int,
    *,
    size: int = 3,
    komi: float = 2.5,
    seed: int = 0,
    symmetry_pool: Sequence[int] = tuple(range(8)),
) -> List[GoAgentTrajectory]:
    """Return paired superko/fresh-history trajectories on any size >= 3.

    At each focus observation the stone grid, side to play, move counters, and
    previous-move feature are identical.  In the ``ko`` branch an immediately
    repeated situational position is in ``position_history`` and recapture is
    illegal.  In the ``fresh`` branch the same grid is a setup position and the
    recapture is legal.  Only the preceding observation distinguishes them.
    """

    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    if size < 3:
        raise ValueError("ko-history pairs require size >= 3")
    pool = tuple(int(value) for value in symmetry_pool)
    if not pool or any(value < 0 or value >= 8 for value in pool):
        raise ValueError("symmetry_pool must contain values in [0, 7]")
    legacy_moves = (1, 4, 3, 6, 5, 8, 0, 2, 1, 7, 5)
    legacy_recapture = 2
    teacher = HeuristicGoTeacher()
    order = list(range(int(pair_count)))
    random.Random(int(seed)).shuffle(order)
    trajectories: List[GoAgentTrajectory] = []
    for output_index, source_index in enumerate(order):
        symmetry = pool[source_index % len(pool)]
        if size == 3:
            moves = [
                _transform_point(point, 3, symmetry) for point in legacy_moves
            ]
            recapture = _transform_point(legacy_recapture, 3, symmetry)
            before = GoBoard(3, komi=komi)
            for action in moves[:-1]:
                before.play(action)
            capture = moves[-1]
        else:
            # Interior simple-ko template before Black captures at ``capture``:
            #   . X O .
            #   X O . O
            #   . X O .
            # The white stone at recapture has one liberty (capture), and the
            # new black stone then has exactly one liberty (recapture).
            def point(row: int, column: int) -> int:
                return _transform_point(row * size + column, size, symmetry)

            recapture = point(1, 1)
            capture = point(1, 2)
            black = [point(1, 0), point(0, 1), point(2, 1)]
            white = [
                recapture,
                point(1, 3),
                point(0, 2),
                point(2, 2),
            ]
            before = GoBoard.from_stones(
                size,
                black=black,
                white=white,
                to_play=BLACK,
                komi=komi,
            )
        ko = before.copy()
        ko.play(capture)
        fresh = GoBoard.from_stones(
            size,
            black=[point for point, value in enumerate(ko.board) if value == BLACK],
            white=[point for point, value in enumerate(ko.board) if value == -BLACK],
            to_play=ko.to_play,
            komi=komi,
        )
        # Equalize every encoder-visible temporal scalar.  Only the simulator's
        # hidden position-history set and the explicit prior observation differ.
        fresh.move_history = list(ko.move_history)
        fresh.consecutive_passes = ko.consecutive_passes
        if ko.is_legal(recapture) or not fresh.is_legal(recapture):
            raise RuntimeError("constructed ko pair does not isolate superko history")
        if teacher.select_move(ko) == teacher.select_move(fresh):
            raise RuntimeError("ko pair did not change the history-dependent teacher move")

        before_value = float(before.winner() * before.to_play)
        current_value = float(ko.winner() * ko.to_play)
        ko_examples = (
            GoAgentExample(
                board=before,
                target_action=int(teacher.select_move(before)),
                value_target=before_value,
                episode_index=2 * output_index,
                move_index=0,
            ),
            GoAgentExample(
                board=ko,
                target_action=int(teacher.select_move(ko)),
                value_target=current_value,
                episode_index=2 * output_index,
                move_index=1,
                focus_point=recapture,
                history_condition="ko",
            ),
        )
        fresh_context = GoBoard(size, komi=komi)
        fresh_examples = (
            GoAgentExample(
                board=fresh_context,
                target_action=int(teacher.select_move(fresh_context)),
                value_target=float(fresh_context.winner() * fresh_context.to_play),
                episode_index=2 * output_index + 1,
                move_index=0,
            ),
            GoAgentExample(
                board=fresh,
                target_action=int(teacher.select_move(fresh)),
                value_target=current_value,
                episode_index=2 * output_index + 1,
                move_index=1,
                focus_point=recapture,
                history_condition="fresh",
            ),
        )
        trajectories.append(GoAgentTrajectory(ko_examples, ko.winner(), "ko_history"))
        trajectories.append(
            GoAgentTrajectory(fresh_examples, fresh.winner(), "fresh_history")
        )
    return trajectories


@torch.no_grad()
def twin_write_returns(
    agent: TemporalUtilityMQRAgent,
    ticket_id: int,
    future_examples: Sequence[GoAgentExample],
    encoder: Any,
    *,
    horizon: int = 3,
    gamma: float = 0.95,
    weights: Optional[GoLossWeights] = None,
    normalize_return: bool = False,
) -> Tuple[float, float]:
    """Evaluate one write/no-write intervention on matched future boards.

    Both branches start from the ticket's post-action shadow state.  Subsequent
    slow writes are closed in both branches, while the fast ring still receives
    every board.  Therefore the return difference isolates the current slow
    write instead of comparing two different future routing policies.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not (0.0 <= float(gamma) <= 1.0):
        raise ValueError("gamma must lie in [0, 1]")
    no_state, write_state = agent.shadow_states(int(ticket_id))
    no_return = 0.0
    write_return = 0.0
    normalizer = 0.0
    selected = weights or GoLossWeights(
        placement=1.0,
        legality=0.5,
        pass_decision=0.5,
        value=0.25,
    )
    reference = next(agent.parameters())
    for offset, example in enumerate(future_examples[: int(horizon)]):
        discount = float(gamma) ** offset
        x = encoder.encode_board(example.board).to(
            device=reference.device,
            dtype=reference.dtype,
        )
        no_output, no_state = agent.branch_step(x, no_state, slow_write=False)
        write_output, write_state = agent.branch_step(x, write_state, slow_write=False)
        action, legal, value = example_targets(example, device=reference.device)
        no_loss = agent.compute_go_loss(
            no_output,
            action,
            legality_target=legal,
            value_target=value,
            weights=selected,
        )["total"]
        write_loss = agent.compute_go_loss(
            write_output,
            action,
            legality_target=legal,
            value_target=value,
            weights=selected,
        )["total"]
        no_return -= discount * float(no_loss.item())
        write_return -= discount * float(write_loss.item())
        normalizer += discount
    if normalizer == 0.0:
        return 0.0, 0.0
    if normalize_return:
        return no_return / normalizer, write_return / normalizer
    return no_return, write_return


__all__ = [
    "GoAgentExample",
    "GoAgentTrajectory",
    "GoVectorEncoder",
    "MiniCPMGoBoardEncoder",
    "example_targets",
    "generate_go_agent_trajectories",
    "generate_ko_history_pairs",
    "legality_target",
    "twin_write_returns",
]
