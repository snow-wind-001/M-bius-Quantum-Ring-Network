"""Old 16-step replay cadence under the otherwise identical Go12 protocol.

This process-local adapter reuses the frozen Go12 runner verbatim. Its two
explicit substitutions select the old replay window and record this adapter
in source provenance. No running Go12 process or repository source is changed.
"""
from pathlib import Path
import hashlib
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import go12_efficiency as study

_original_session = study.make_session
_original_source = study.source_hashes


def make_session(args, agent, encoder):
    session = _original_session(args, agent, encoder)
    session.replay_window = 16
    assert session.replay_credit_horizon == 16 and session.ownership_weight == 0
    return session


def source_hashes():
    result = _original_source()
    result[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def main():
    torch.set_num_threads(1)
    args = study.parse_args()
    args.method = "legacy"
    if Path(args.directory) == ROOT / "checkpoints/go12_v1":
        args.directory = str(ROOT / "checkpoints/go12_legacy")
    destination = Path(args.directory) / f"base-{args.seed}.pt"
    source = ROOT / "checkpoints/go12_v1" / destination.name
    if not source.exists():
        raise RuntimeError("prepare the common Go12 prior before this cadence control")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copyfile(source, destination)
    if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(destination.read_bytes()).digest():
        raise RuntimeError("legacy control must share the exact Go12 pretrained prior")
    study.make_session = make_session
    study.source_hashes = source_hashes
    study.run(args)


if __name__ == "__main__":
    main()
