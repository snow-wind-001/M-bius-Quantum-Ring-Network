"""Post-hoc live-history drift audit at the first observed B-phase checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.go11_continual import SEEDS, build, make_session, source_hashes, save_json
from mqr import GoBoard


@torch.no_grad()
def inspect(raw: bytes, path: Path):
    work = torch.load(io.BytesIO(raw), weights_only=True)
    assert work["source"] == source_hashes()
    config = argparse.Namespace(**work["config"])
    agent, encoder = build(config, config.method, config.seed)
    session = make_session(config, agent, encoder, config.method)
    session.load_state_dict(work["session"])
    game = work["active_game"]
    before = hashlib.sha256(b"".join(p.numpy().tobytes() for p in agent.parameters())).hexdigest()
    board = GoBoard(10, komi=5.5)
    fresh = agent.core.zero_state(1, device="cpu", dtype=torch.float32)
    for action in game["board"]["move_history"]:
        observed = board.copy()
        features = encoder.encode_board(board)
        replay_output, fresh = agent._transition(features, fresh, slow_write=True)
        board.play(action)
    live = agent._stream_states[session.stream_id]
    live_output = agent.readout_state(live, features=features)
    delta = torch.cat([(a-b).flatten() for a,b in zip(live.rings,fresh.rings)])
    original = torch.cat([a.flatten() for a in live.rings])
    p, q = live_output.policy_logits, replay_output.policy_logits
    legal = torch.tensor(observed.legal_moves())
    actions = [int(legal[o.policy_logits[0,legal].argmax()]) for o in (live_output,replay_output)]
    assert before == hashlib.sha256(b"".join(p.numpy().tobytes() for p in agent.parameters())).hexdigest()
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(raw)
    return {"seed":config.seed,"method":config.method,"phase":game["phase"],"game_index":game["index"],
        "observed_ply":len(observed.move_history),"parameter_version":int(agent.online_parameter_version),
        "pending_count":session.pending_count,"state_max_absolute_difference":float(delta.abs().max()),
        "state_relative_l2_difference":float(delta.norm()/original.norm().clamp_min(1e-30)),
        "policy_kl":float((p.exp()*(p-q)).sum().clamp_min(0)),
        "value_absolute_difference":float((live_output.value-replay_output.value).abs().max()),
        "legal_argmax_changed":actions[0]!=actions[1],"actions":actions,
        "checkpoint":str(path.relative_to(ROOT)),"checkpoint_sha256":hashlib.sha256(raw).hexdigest(),
        "parameters_unchanged":True}


def main():
    torch.set_num_threads(1)
    output = ROOT / "analysis/results/go11_state_verify.json"
    if output.exists():
        raise RuntimeError("state sampling is not rerun or replaced after seeing its values")
    rows=[]
    remaining=set(SEEDS)
    while remaining:
        for seed in sorted(remaining):
            path=ROOT/f"checkpoints/go11_v1/work-selfplay_cone6-{seed}.pt"
            if not path.exists():continue
            raw=path.read_bytes()
            work=torch.load(io.BytesIO(raw),weights_only=True)
            game=work["active_game"]
            if game is None or game["phase"]!="b" or game["board"]["game_over"]:continue
            if len(game["board"]["move_history"])<=len(game["opening"]):continue
            saved=ROOT/f"checkpoints/go11_state_audit/cone6-{seed}.pt"
            row=inspect(raw,saved)
            rows.append(row);remaining.remove(seed)
            print(json.dumps(row),flush=True)
        if remaining:time.sleep(15)
    save_json({"verified":True,"scope":"post-hoc first observed live B checkpoint per six-ring seed; current-head live state versus full-history current-parameter replay",
        "rows":rows,"source":source_hashes(),"verifier_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},output)


if __name__=="__main__":
    main()
