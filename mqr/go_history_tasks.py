"""Reachable Go histories for a deliberately history-dependent recall probe.

The task asks for the empty antipode of Black's first placement. This is a
memory diagnostic on legal Go trajectories, not a claim about optimal Go play
or superko. The simulator's hidden history is never edited to create examples.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import torch

from .go import GoBoard
from .go_agent import GoVectorEncoder


@dataclass(frozen=True)
class GoHistoryPair:
    histories: Tuple[Tuple[GoBoard, ...], Tuple[GoBoard, ...]]
    targets: Tuple[int, int]
    moves: Tuple[Tuple[int, ...], Tuple[int, ...]]


@dataclass(frozen=True)
class GoKoHistoryPair:
    """Two legal setup-to-capture histories with aliased final observations."""

    histories: Tuple[Tuple[GoBoard, ...], Tuple[GoBoard, ...]]
    recapture: int
    legal: Tuple[bool, bool] = (False, True)


def generate_ko_history_pairs(count: int, *, seed: int = 0, size: int = 10) -> List[GoKoHistoryPair]:
    """Predict recapture legality from history, without editing superko records.

    One setup contains the ko victim; the other has that point empty. The same
    legal placement gives identical final encoded boards. Recapture repeats a
    recorded situational position only in the first history. These are legal
    setup positions, not claimed to be full games starting from an empty board.
    """
    if count < 1 or size < 5:
        raise ValueError("ko pairs require positive count and size >= 5")
    rng, pairs, signatures = random.Random(seed), [], set()
    encoder = GoVectorEncoder(size, 3 * size * size + 6, projection_mode="identity")
    for _ in range(count * 100):
        row, column = rng.randrange(size - 3), rng.randrange(size - 3)
        rotations, mirror = rng.randrange(4), rng.randrange(2)

        def point(r, c):
            r, c = row + r, column + c
            if mirror:
                c = size - 1 - c
            for _ in range(rotations):
                r, c = c, size - 1 - r
            return r * size + c

        capture, recapture = point(1, 1), point(1, 2)
        friendly = {point(0, 2), point(2, 2), point(1, 3)}
        enemy = {point(0, 1), point(2, 1), point(1, 0)}
        occupied = friendly | enemy | {capture, recapture}
        board = GoBoard(size, komi=5.5)
        choices = list(range(size * size))
        rng.shuffle(choices)
        for extra in choices:
            if len(occupied) >= 20:
                break
            if extra in occupied or any(n in occupied for n in board._neighbors(extra)):
                continue
            (friendly if rng.randrange(2) else enemy).add(extra)
            occupied.add(extra)
        color = rng.choice((-1, 1))
        histories = []
        for victim in (True, False):
            enemies = enemy | ({recapture} if victim else set())
            board = GoBoard.from_stones(size, black=friendly if color == 1 else enemies,
                                       white=enemies if color == 1 else friendly,
                                       to_play=color, komi=5.5)
            previous = board.copy()
            if not board.is_legal(capture):
                break
            board.play(capture)
            histories.append((previous, board))
        if len(histories) != 2:
            continue
        final = [history[-1] for history in histories]
        signature = (final[0].position_key(), capture, recapture)
        if signature in signatures:
            continue
        if (not torch.equal(encoder.encode_board(final[0]), encoder.encode_board(final[1]))
                or tuple(board.is_legal(recapture) for board in final) != (False, True)):
            raise RuntimeError("constructed ko histories failed the exact alias contract")
        signatures.add(signature)
        pairs.append(GoKoHistoryPair(tuple(histories), recapture))
        if len(pairs) == count:
            return pairs
    raise RuntimeError("could not generate the requested unique ko pairs")


def generate_reachable_history_pairs(
    count: int, *, seed: int = 0, size: int = 5, moves: int = 12,
) -> List[GoHistoryPair]:
    """Match every encoder-visible final field, while swapping the first move.

    Both branches start empty and use only legal moves. Black's first two
    moves commute; the final board, side, previous move, passes and move count
    match exactly. Held-out splits should use distinct full trajectory hashes.
    """
    if count < 1 or size < 5 or not 4 <= moves <= size ** 2 - 4:
        raise ValueError("need positive count, size >= 5, and 4 <= moves <= points - 4")
    rng = random.Random(seed)
    encoder = GoVectorEncoder(size, 3 * size ** 2 + 6, projection_mode="identity")
    points = size ** 2
    corner_pairs = ((0, size - 1), (0, points - size), (size - 1, points - 1),
                    (points - size, points - 1))
    pairs, signatures = [], set()
    attempts = 0
    while len(pairs) < count and attempts < count * 200:
        attempts += 1
        a, c = rng.choice(corner_pairs)
        forbidden = {a, c, points - 1 - a, points - 1 - c}
        candidates = list(set(range(points)) - forbidden)
        rng.shuffle(candidates)
        b, d = candidates[:2]
        initial = ((a, b, c, d), (c, b, a, d))
        branches = []
        try:
            for prefix in initial:
                board = GoBoard(size, komi=2.5)
                history = [board.copy()]
                for action in prefix:
                    if board.play(action).captures:
                        raise ValueError("opening commutation must not capture")
                    history.append(board.copy())
                branches.append(history)
        except ValueError:
            continue
        if branches[0][-1].board != branches[1][-1].board:
            continue
        for _ in range(moves - 4):
            left, right = branches[0][-1], branches[1][-1]
            choices = [x for x in candidates if left.board[x] == 0 and left.is_legal(x) and right.is_legal(x)]
            rng.shuffle(choices)
            chosen = None
            for action in choices:
                next_left, next_right = left.copy(), right.copy()
                if next_left.play(action).captures or next_right.play(action).captures:
                    continue
                chosen = (next_left, next_right)
                break
            if chosen is None:
                break
            branches[0].append(chosen[0])
            branches[1].append(chosen[1])
        if any(len(branch) != moves + 1 for branch in branches):
            continue
        final = [branch[-1] for branch in branches]
        targets = (points - 1 - a, points - 1 - c)
        if any(not board.is_legal(target) for board in final for target in targets):
            continue
        if not torch.equal(encoder.encode_board(final[0]), encoder.encode_board(final[1])):
            raise RuntimeError("reachable pair failed full-observation equality")
        signature = tuple(tuple(board.move_history) for board in final)
        if signature in signatures:
            continue
        signatures.add(signature)
        pairs.append(GoHistoryPair((tuple(branches[0]), tuple(branches[1])), targets, signature))
    if len(pairs) != count:
        raise RuntimeError("could not generate enough reachable pairs within the bounded attempt budget")
    return pairs
