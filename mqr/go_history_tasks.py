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
