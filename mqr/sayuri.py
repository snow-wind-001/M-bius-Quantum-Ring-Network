"""Subprocess-only integration with the GPLv3 Sayuri Go engine.

No Sayuri source code is copied or imported. The GPLv3 engine remains an
optional, separately built program and communicates with this repository
through the standard Go Text Protocol (GTP).
"""

from __future__ import annotations

import os
import selectors
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from .go import BLACK, GoBoard, action_to_gtp, color_name, gtp_to_action


PathLike = Union[str, Path]


class SayuriError(RuntimeError):
    """Raised when Sayuri cannot start or returns a failed GTP response."""


def discover_sayuri_paths(repository_root: Optional[PathLike] = None) -> Tuple[Path, Path]:
    """Find a local Sayuri executable and weight file without downloading them."""

    root = (
        Path(repository_root).expanduser().resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    # Keep discovery aligned with scripts/setup_sayuri.sh. An in-tree build
    # remains a supported fallback for contributors who build Sayuri there.
    binary_candidates = [
        root / ".external/sayuri-build/sayuri",
        root / "UsedCode/Sayuri/build/sayuri",
        root / ".external/Sayuri/build/sayuri",
    ]
    binary = next((path for path in binary_candidates if path.is_file()), None)
    if binary is None:
        raise FileNotFoundError(
            "Sayuri executable not found; run scripts/setup_sayuri.sh or pass --sayuri-binary"
        )

    weight_roots = [
        root / ".external/sayuri-weights",
        root / ".external/Sayuri/weights",
        root / "UsedCode/Sayuri/weights",
    ]
    weight = None
    for directory in weight_roots:
        if not directory.is_dir():
            continue
        candidates = sorted(path for path in directory.glob("*.bin*") if path.is_file())
        if candidates:
            weight = candidates[-1]
            break
    if weight is None:
        raise FileNotFoundError(
            "Sayuri weight file not found; run scripts/setup_sayuri.sh --download-weights "
            "or pass --sayuri-weights"
        )
    return binary.resolve(), weight.resolve()


class SayuriGTPClient:
    """Synchronous, timeout-aware GTP client for one persistent Sayuri process."""

    def __init__(
        self,
        binary: PathLike,
        weights: Optional[PathLike] = None,
        *,
        board_size: int = 5,
        komi: float = 5.5,
        threads: int = 1,
        playouts: int = 16,
        scoring_rule: str = "area",
        friendly_pass: bool = True,
        timeout: float = 30.0,
    ):
        self.binary = Path(binary).expanduser().resolve()
        self.weights = None if weights is None else Path(weights).expanduser().resolve()
        if not self.binary.is_file():
            raise FileNotFoundError(f"Sayuri executable not found: {self.binary}")
        if self.weights is not None and not self.weights.is_file():
            raise FileNotFoundError(f"Sayuri weights not found: {self.weights}")
        if board_size < 2 or threads <= 0 or playouts <= 0 or timeout <= 0:
            raise ValueError("board_size, threads, playouts, and timeout must be positive")
        if scoring_rule not in ("area", "territory"):
            raise ValueError('scoring_rule must be "area" or "territory"')
        self.board_size = int(board_size)
        self.komi = float(komi)
        self.timeout = float(timeout)
        self._next_id = 1
        self._closed = False
        self._stdout_buffer = bytearray()
        self._stderr = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")

        command = [
            str(self.binary),
            "--quiet",
            "--threads",
            str(int(threads)),
            "--playouts",
            str(int(playouts)),
            "--scoring-rule",
            scoring_rule,
        ]
        if self.weights is not None:
            command.extend(["--weights", str(self.weights)])
        if friendly_pass:
            command.append("--friendly-pass")
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=False,
            bufsize=0,
        )
        if self._process.stdin is None or self._process.stdout is None:
            self.close(force=True)
            raise SayuriError("Failed to open Sayuri GTP pipes")
        self._selector = selectors.DefaultSelector()
        self._selector.register(self._process.stdout, selectors.EVENT_READ)
        try:
            if self.command("protocol_version") != "2":
                raise SayuriError("Sayuri did not report GTP protocol version 2")
            self.engine_name = self.command("name")
            self.engine_version = self.command("version")
            self.command(f"boardsize {self.board_size}")
            self.command(f"komi {self.komi}")
            self.command("clear_board")
        except Exception:
            self.close(force=True)
            raise

    def _diagnostics(self) -> str:
        try:
            self._stderr.flush()
            self._stderr.seek(0)
            return self._stderr.read()[-4000:]
        except Exception:
            return ""

    def _readline(self, deadline: float) -> str:
        while b"\n" not in self._stdout_buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._selector.select(remaining):
                raise TimeoutError(
                    f"Timed out waiting for Sayuri; diagnostics: {self._diagnostics()}"
                )
            chunk = os.read(self._process.stdout.fileno(), 4096)
            if not chunk:
                raise SayuriError(
                    f"Sayuri exited with code {self._process.poll()}; "
                    f"diagnostics: {self._diagnostics()}"
                )
            self._stdout_buffer.extend(chunk)
        end = self._stdout_buffer.index(b"\n")
        line = bytes(self._stdout_buffer[:end])
        del self._stdout_buffer[: end + 1]
        return line.rstrip(b"\r").decode("utf-8", errors="replace")

    def command(self, command: str) -> str:
        if self._closed:
            raise SayuriError("Sayuri client is closed")
        if not command.strip() or "\n" in command or "\r" in command:
            raise ValueError("GTP command must be one non-empty line")
        request_id = self._next_id
        self._next_id += 1
        self._process.stdin.write(f"{request_id} {command}\n".encode("utf-8"))
        self._process.stdin.flush()

        deadline = time.monotonic() + self.timeout
        success_prefix = f"={request_id}"
        failure_prefix = f"?{request_id}"
        while True:
            first = self._readline(deadline)
            if first.startswith(success_prefix) or first.startswith(failure_prefix):
                break
        success = first.startswith(success_prefix)
        prefix = success_prefix if success else failure_prefix
        lines = [first[len(prefix) :].lstrip()]
        while True:
            line = self._readline(deadline)
            if line == "":
                break
            lines.append(line)
        response = "\n".join(lines).strip()
        if not success:
            raise SayuriError(f"GTP command {command!r} failed: {response}")
        return response

    def clear(self) -> None:
        self.command("clear_board")

    def sync_board(self, board: GoBoard) -> None:
        """Replay a normally reached :class:`GoBoard` position into Sayuri."""

        if board.size != self.board_size:
            raise ValueError(
                f"board size mismatch: client={self.board_size}, position={board.size}"
            )
        self.command("clear_board")
        if not board.move_history:
            if any(board.board):
                raise ValueError("arbitrary setup stones cannot be synchronized without move history")
            return
        player = BLACK
        replay = GoBoard(board.size, komi=board.komi)
        for action in board.move_history:
            coordinate = action_to_gtp(action, board.size)
            self.command(f"play {color_name(player)} {coordinate}")
            replay.play(action)
            player = -player
        if replay.board != board.board or replay.to_play != board.to_play:
            raise ValueError("move_history does not reproduce the supplied GoBoard")

    def legal_moves(self, board: GoBoard) -> Set[int]:
        self.sync_board(board)
        response = self.command("gogui-rules_legal_moves")
        if not response:
            return set()
        return {gtp_to_action(token, board.size) for token in response.split()}

    def compare_legal_moves(self, board: GoBoard) -> Dict[str, Set[int]]:
        sayuri = self.legal_moves(board)
        local = set(board.legal_moves())
        return {
            "local_only": local - sayuri,
            "sayuri_only": sayuri - local,
        }

    def raw_policy(self, board: GoBoard) -> List[float]:
        """Return Sayuri's raw neural policy in local row-major action order."""

        if self.weights is None:
            raise SayuriError("sayuri-raw_nn requires a neural-network weight file")
        self.sync_board(board)
        response = self.command("sayuri-raw_nn")
        lines = response.splitlines()
        try:
            start = next(i for i, line in enumerate(lines) if line.strip() == "probabilities:")
            pass_line = next(line for line in lines if line.strip().startswith("pass probabilities:"))
        except StopIteration as exc:
            raise SayuriError("Could not parse sayuri-raw_nn policy output") from exc
        values: List[float] = []
        for line in lines[start + 1 : start + 1 + board.size]:
            values.extend(float(item) for item in line.split())
        if len(values) != board.pass_action:
            raise SayuriError(
                f"Expected {board.pass_action} board probabilities, parsed {len(values)}"
            )
        pass_probability = float(pass_line.split(":", 1)[1].strip())
        values.append(pass_probability)
        return values

    def select_move(self, board: GoBoard, *, mode: str = "policy") -> int:
        """Select a legal move from raw policy or MCTS ``genmove``."""

        if mode == "policy":
            policy = self.raw_policy(board)
            legal = board.legal_moves()
            return max(legal, key=lambda action: (policy[action], -action))
        if mode == "mcts":
            if self.weights is None:
                raise SayuriError("MCTS teacher requires a neural-network weight file")
            self.sync_board(board)
            response = self.command(f"genmove {color_name(board.to_play)}").strip().lower()
            if response == "resign":
                return board.pass_action
            action = gtp_to_action(response, board.size)
            if not board.is_legal(action):
                raise SayuriError(f"Sayuri returned a locally illegal move: {response}")
            return action
        raise ValueError('mode must be "policy" or "mcts"')

    def close(self, *, force: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if not force and self._process.poll() is None:
                request_id = self._next_id
                self._process.stdin.write(f"{request_id} quit\n".encode("utf-8"))
                self._process.stdin.flush()
                self._process.wait(timeout=min(self.timeout, 5.0))
        except Exception:
            force = True
        if force and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)
        try:
            self._selector.close()
        finally:
            self._stderr.close()

    def __enter__(self) -> "SayuriGTPClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(force=exc_type is not None)

    def __del__(self) -> None:
        try:
            self.close(force=True)
        except Exception:
            pass


class SayuriTeacher:
    """Teacher facade matching :class:`mqr.go.HeuristicGoTeacher`."""

    def __init__(self, client: SayuriGTPClient, *, mode: str = "policy"):
        if mode not in ("policy", "mcts"):
            raise ValueError('mode must be "policy" or "mcts"')
        self.client = client
        self.mode = mode

    def select_move(self, board: GoBoard) -> int:
        return self.client.select_move(board, mode=self.mode)


__all__ = [
    "SayuriError",
    "SayuriGTPClient",
    "SayuriTeacher",
    "discover_sayuri_paths",
]
