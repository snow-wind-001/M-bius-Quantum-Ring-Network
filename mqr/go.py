"""Small-board Go rules and a deterministic online-learning teacher.

The environment uses Chinese area scoring, situational superko for stone
placements, suicide prohibition, and two consecutive passes for termination.
Passes are exempt from superko so a game can terminate normally.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Sequence, Set, Tuple


EMPTY = 0
BLACK = 1
WHITE = -1
_GTP_COLUMNS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


def color_name(color: int) -> str:
    if color == BLACK:
        return "black"
    if color == WHITE:
        return "white"
    raise ValueError("color must be BLACK or WHITE")


@dataclass(frozen=True)
class GoMoveResult:
    action: int
    player: int
    captures: int
    game_over: bool


@dataclass(frozen=True)
class GoTrainingExample:
    board: "GoBoard"
    target_action: int


class GoBoard:
    """Mutable Go position with an action space of ``size**2 + 1``."""

    def __init__(self, size: int = 5, *, komi: float = 5.5):
        if not (2 <= int(size) <= len(_GTP_COLUMNS)):
            raise ValueError(f"size must be in [2, {len(_GTP_COLUMNS)}]")
        self.size = int(size)
        self.komi = float(komi)
        self.board: List[int] = [EMPTY] * (self.size * self.size)
        self.to_play = BLACK
        self.consecutive_passes = 0
        self.game_over = False
        self.move_history: List[int] = []
        self.position_history: Set[Tuple[Tuple[int, ...], int]] = {self.position_key()}

    @classmethod
    def from_stones(
        cls,
        size: int,
        *,
        black: Iterable[int] = (),
        white: Iterable[int] = (),
        to_play: int = BLACK,
        komi: float = 5.5,
    ) -> "GoBoard":
        """Construct a test/research position from flat point indices."""

        if to_play not in (BLACK, WHITE):
            raise ValueError("to_play must be BLACK or WHITE")
        result = cls(size, komi=komi)
        occupied: Set[int] = set()
        for color, points in ((BLACK, black), (WHITE, white)):
            for point in points:
                point = int(point)
                if not (0 <= point < size * size):
                    raise ValueError(f"stone index out of range: {point}")
                if point in occupied:
                    raise ValueError(f"overlapping setup stone: {point}")
                occupied.add(point)
                result.board[point] = color
        result.to_play = int(to_play)
        result.position_history = {result.position_key()}
        return result

    @property
    def pass_action(self) -> int:
        return self.size * self.size

    @property
    def action_size(self) -> int:
        return self.pass_action + 1

    def position_key(self) -> Tuple[Tuple[int, ...], int]:
        """Situational-superko key: stones plus the player to move."""

        return tuple(self.board), int(self.to_play)

    def copy(self) -> "GoBoard":
        result = GoBoard(self.size, komi=self.komi)
        result.board = list(self.board)
        result.to_play = self.to_play
        result.consecutive_passes = self.consecutive_passes
        result.game_over = self.game_over
        result.move_history = list(self.move_history)
        result.position_history = set(self.position_history)
        return result

    def _neighbors(self, point: int) -> Iterator[int]:
        row, column = divmod(point, self.size)
        if row > 0:
            yield point - self.size
        if row + 1 < self.size:
            yield point + self.size
        if column > 0:
            yield point - 1
        if column + 1 < self.size:
            yield point + 1

    def _group_and_liberties(
        self, stones: Sequence[int], start: int
    ) -> Tuple[Set[int], Set[int]]:
        color = stones[start]
        if color == EMPTY:
            raise ValueError("cannot find a group from an empty point")
        group = {start}
        liberties: Set[int] = set()
        stack = [start]
        while stack:
            point = stack.pop()
            for neighbor in self._neighbors(point):
                value = stones[neighbor]
                if value == EMPTY:
                    liberties.add(neighbor)
                elif value == color and neighbor not in group:
                    group.add(neighbor)
                    stack.append(neighbor)
        return group, liberties

    def _simulate_stone(self, action: int) -> Tuple[List[int], int, int]:
        if not (0 <= action < self.pass_action):
            raise ValueError("stone action out of range")
        if self.board[action] != EMPTY:
            raise ValueError("point is occupied")

        candidate = list(self.board)
        candidate[action] = self.to_play
        captured: Set[int] = set()
        checked: Set[int] = set()
        for neighbor in self._neighbors(action):
            if candidate[neighbor] != -self.to_play or neighbor in checked:
                continue
            group, liberties = self._group_and_liberties(candidate, neighbor)
            checked.update(group)
            if not liberties:
                captured.update(group)
        for point in captured:
            candidate[point] = EMPTY

        _, own_liberties = self._group_and_liberties(candidate, action)
        if not own_liberties:
            raise ValueError("suicide is forbidden")
        return candidate, len(captured), len(own_liberties)

    def is_legal(self, action: int) -> bool:
        if self.game_over or not isinstance(action, int):
            return False
        if action == self.pass_action:
            return True
        if not (0 <= action < self.pass_action) or self.board[action] != EMPTY:
            return False
        try:
            candidate, _, _ = self._simulate_stone(action)
        except ValueError:
            return False
        next_key = (tuple(candidate), -self.to_play)
        return next_key not in self.position_history

    def legal_moves(self, *, include_pass: bool = True) -> List[int]:
        if self.game_over:
            return []
        result = [point for point in range(self.pass_action) if self.is_legal(point)]
        if include_pass:
            result.append(self.pass_action)
        return result

    def play(self, action: int) -> GoMoveResult:
        if not self.is_legal(action):
            raise ValueError(f"illegal move: {action_to_gtp(action, self.size)}")
        player = self.to_play
        captures = 0
        if action == self.pass_action:
            self.consecutive_passes += 1
        else:
            candidate, captures, _ = self._simulate_stone(action)
            self.board = candidate
            self.consecutive_passes = 0

        self.move_history.append(action)
        self.to_play = -self.to_play
        self.position_history.add(self.position_key())
        if self.consecutive_passes >= 2:
            self.game_over = True
        return GoMoveResult(action, player, captures, self.game_over)

    def score(self) -> Dict[str, float]:
        """Return Chinese area score; no dead-stone adjudication is attempted."""

        black = float(sum(value == BLACK for value in self.board))
        white = float(sum(value == WHITE for value in self.board))
        visited: Set[int] = set()
        for start, value in enumerate(self.board):
            if value != EMPTY or start in visited:
                continue
            region = {start}
            boundary: Set[int] = set()
            stack = [start]
            visited.add(start)
            while stack:
                point = stack.pop()
                for neighbor in self._neighbors(point):
                    neighbor_value = self.board[neighbor]
                    if neighbor_value == EMPTY and neighbor not in visited:
                        visited.add(neighbor)
                        region.add(neighbor)
                        stack.append(neighbor)
                    elif neighbor_value != EMPTY:
                        boundary.add(neighbor_value)
            if boundary == {BLACK}:
                black += len(region)
            elif boundary == {WHITE}:
                white += len(region)
        white_with_komi = white + self.komi
        return {
            "black": black,
            "white": white_with_komi,
            "white_without_komi": white,
            "komi": self.komi,
            "margin_black": black - white_with_komi,
        }

    def winner(self) -> int:
        margin = self.score()["margin_black"]
        if margin > 0:
            return BLACK
        if margin < 0:
            return WHITE
        return EMPTY

    def render(self) -> str:
        columns = " ".join(_GTP_COLUMNS[: self.size])
        lines = [f"   {columns}"]
        symbols = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        for row in range(self.size):
            number = self.size - row
            values = " ".join(
                symbols[self.board[row * self.size + column]]
                for column in range(self.size)
            )
            lines.append(f"{number:>2} {values} {number:>2}")
        lines.append(f"   {columns}")
        lines.append(f"to_play={color_name(self.to_play)} passes={self.consecutive_passes}")
        return "\n".join(lines)


def action_to_gtp(action: int, size: int) -> str:
    if action == size * size:
        return "pass"
    if not (0 <= int(action) < size * size):
        raise ValueError("action out of range")
    row, column = divmod(int(action), size)
    return f"{_GTP_COLUMNS[column]}{size - row}"


def gtp_to_action(coordinate: str, size: int) -> int:
    text = coordinate.strip().upper()
    if text == "PASS":
        return size * size
    if len(text) < 2 or text[0] not in _GTP_COLUMNS[:size]:
        raise ValueError(f"invalid GTP coordinate: {coordinate!r}")
    try:
        number = int(text[1:])
    except ValueError as exc:
        raise ValueError(f"invalid GTP coordinate: {coordinate!r}") from exc
    if not (1 <= number <= size):
        raise ValueError(f"invalid GTP coordinate: {coordinate!r}")
    column = _GTP_COLUMNS.index(text[0])
    row = size - number
    return row * size + column


class HeuristicGoTeacher:
    """Deterministic tactical teacher for mechanism tests, not a strong Go AI."""

    def score_move(self, board: GoBoard, action: int) -> float:
        if action == board.pass_action:
            return -1.0
        if not board.is_legal(action):
            return float("-inf")
        candidate, captures, own_liberties = board._simulate_stone(action)

        friendly_groups: Set[int] = set()
        pressure = 0
        for neighbor in board._neighbors(action):
            if board.board[neighbor] == board.to_play:
                group, _ = board._group_and_liberties(board.board, neighbor)
                friendly_groups.add(min(group))
            elif candidate[neighbor] == -board.to_play:
                _, liberties = board._group_and_liberties(candidate, neighbor)
                pressure += max(0, 3 - len(liberties))

        row, column = divmod(action, board.size)
        center = (board.size - 1) / 2.0
        center_bonus = board.size - (abs(row - center) + abs(column - center))
        edge_penalty = float(row in (0, board.size - 1)) + float(
            column in (0, board.size - 1)
        )
        self_atari_penalty = 12.0 if own_liberties == 1 and captures == 0 else 0.0
        return (
            100.0 * captures
            + 7.0 * len(friendly_groups)
            + 3.0 * min(own_liberties, 4)
            + 2.0 * pressure
            + center_bonus
            - edge_penalty
            - self_atari_penalty
        )

    def select_move(self, board: GoBoard) -> int:
        legal_stones = board.legal_moves(include_pass=False)
        if not legal_stones:
            return board.pass_action
        scored = [(self.score_move(board, action), -action, action) for action in legal_stones]
        best_score, _, best_action = max(scored)
        occupancy = 1.0 - board.board.count(EMPTY) / float(board.pass_action)
        if board.consecutive_passes == 1 and occupancy >= 0.65:
            return board.pass_action
        if occupancy >= 0.84 and best_score < 12.0:
            return board.pass_action
        return best_action


class DefensiveGoTeacher(HeuristicGoTeacher):
    """A second local opponent emphasizing rescue and avoiding own-eye filling.

    This is a deterministic workload shift, not a strength-rated Go engine.
    """

    def score_move(self, board: GoBoard, action: int) -> float:
        score = super().score_move(board, action)
        if action == board.pass_action or score == float("-inf"):
            return score
        saved, checked = 0, set()
        neighbors = tuple(board._neighbors(action))
        for neighbor in neighbors:
            if board.board[neighbor] != board.to_play or neighbor in checked:
                continue
            group, liberties = board._group_and_liberties(board.board, neighbor)
            checked.update(group)
            if liberties == {action}:
                saved += len(group)
        own_eye = all(board.board[n] == board.to_play for n in neighbors)
        _, captures, liberties = board._simulate_stone(action)
        return score + 18.0 * saved - 25.0 * own_eye - 10.0 * (liberties == 1 and captures == 0)


def format_go_prompt(board: GoBoard, *, prompt_mode: str = "rules") -> str:
    """Format a label-free prompt, optionally withholding explicit Go rules.

    ``board-only`` keeps the symbol legend, side to move, and exact board while
    removing rule statements.  It is the matched ablation for testing whether
    a constant natural-language rule prefix contributes beyond board features.
    """

    if prompt_mode in ("rules", "wrong-rules"):
        rule_text = (
            "规则：中国数子法，禁止自杀，情境超级劫，连续两次停一手终局。\n"
            if prompt_mode == "rules"
            else "规则：白棋先行，允许自杀，允许立即重复局面，任意一次停一手即终局。\n"
        )
        prefix = (
            f"你正在学习 {board.size}x{board.size} 路围棋。X 是黑棋，O 是白棋，. 是空点。\n"
            f"{rule_text}"
            f"当前轮到：{color_name(board.to_play)}。请根据当前局面判断下一手。\n"
        )
    elif prompt_mode == "board-only":
        prefix = (
            f"{board.size}x{board.size} 棋盘输入。X 是黑棋，O 是白棋，. 是空点。\n"
            f"当前轮到：{color_name(board.to_play)}。\n"
        )
    else:
        raise ValueError(
            'prompt_mode must be "rules", "wrong-rules", or "board-only"'
        )
    return prefix + board.render()


def generate_basic_go_dataset(
    num_positions: int,
    *,
    size: int = 5,
    komi: float = 5.5,
    seed: int = 0,
    random_move_probability: float = 0.25,
) -> List[GoTrainingExample]:
    """Generate an offline, deterministic set of legal positions and teacher moves."""

    if num_positions <= 0:
        raise ValueError("num_positions must be positive")
    if not (0.0 <= random_move_probability <= 1.0):
        raise ValueError("random_move_probability must be in [0, 1]")
    rng = random.Random(int(seed))
    teacher = HeuristicGoTeacher()
    board = GoBoard(size, komi=komi)
    examples: List[GoTrainingExample] = []
    episode_moves = 0
    while len(examples) < num_positions:
        if board.game_over or episode_moves >= 2 * size * size:
            board = GoBoard(size, komi=komi)
            episode_moves = 0
        target = teacher.select_move(board)
        examples.append(GoTrainingExample(board.copy(), target))

        legal_stones = board.legal_moves(include_pass=False)
        if legal_stones and rng.random() < random_move_probability:
            rollout = rng.choice(legal_stones)
        else:
            rollout = target
        board.play(rollout)
        episode_moves += 1
    return examples


__all__ = [
    "BLACK",
    "EMPTY",
    "WHITE",
    "GoBoard",
    "GoMoveResult",
    "GoTrainingExample",
    "HeuristicGoTeacher",
    "action_to_gtp",
    "color_name",
    "format_go_prompt",
    "generate_basic_go_dataset",
    "gtp_to_action",
]
