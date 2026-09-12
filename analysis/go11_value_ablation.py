"""Complete paired matches with a trained policy and zero learned leaf values."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np  # Initialize the NumPy MKL runtime before torch's OpenMP.
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.go11_continual import (SEEDS, begin_game, build, finish_record, make_session,
    save_json, save_torch, source_hashes)
from mqr.go import DefensiveGoTeacher, HeuristicGoTeacher
from mqr.go_outcome import _board_record, _board_from_record
from mqr.go_search import policy_value_search


def parameter_hash(agent):
    return hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in agent.parameters())).hexdigest()


def run(options):
    result_path=Path(options.result) if getattr(options,"result",None) else ROOT/f"analysis/results/go11_selfplay_cone6_seed{options.seed}.json"
    original=json.loads(result_path.read_text())
    assert original["completed"] and original["source"]==source_hashes()
    args=argparse.Namespace(**original["config"])
    assert args.seed==options.seed and args.method=="selfplay_cone6"
    evaluator_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    agent,encoder=build(args,args.method,args.seed)
    session=make_session(args,agent,encoder,args.method)
    source_checkpoint=Path(original["checkpoint"])
    if not source_checkpoint.is_absolute():source_checkpoint=ROOT/source_checkpoint
    assert hashlib.sha256(source_checkpoint.read_bytes()).hexdigest()==original["checkpoint_sha256"]
    path=Path(args.directory)/f"value-zero-{args.seed}.pt"
    if path.exists():
        work=torch.load(path,weights_only=True)
        assert work["source"]==source_hashes() and work["model_checkpoint_sha256"]==original["checkpoint_sha256"]
        assert work["evaluator_sha256"]==evaluator_hash
        session.load_state_dict(work["session"])
    else:
        session.load_state_dict(torch.load(source_checkpoint,weights_only=True)["session"])
        session.reset_game()
        work={"source":source_hashes(),"model_checkpoint_sha256":original["checkpoint_sha256"],
            "evaluator_sha256":evaluator_hash,
            "config":original["config"],"value_scale":0,"games":[],"active_game":None,"completed":False,
            "parameters_before":parameter_hash(agent),"version_before":int(agent.online_parameter_version)}
    def checkpoint():
        work["session"]=session.state_dict()
        save_torch(work,path)
    consumed=0
    while len(work["games"])<args.eval_games:
        index=len(work["games"])
        if work["active_game"] is None:
            work["active_game"]=begin_game(args,session,"search_eval",index)
        game=work["active_game"];board=_board_from_record(game["board"])
        teacher=DefensiveGoTeacher() if index//2%2 else HeuristicGoTeacher()
        while not board.game_over:
            started=time.perf_counter()
            result=session.observe(board,learn=False)
            student=board.to_play==game["student_color"]
            search=policy_value_search(agent,encoder,board,result["state"],result["policy_logits"],
                simulations=args.eval_simulations,max_depth=args.search_depth,value_scale=0) if student else None
            teacher_action=teacher.select_move(board)
            action=search["action"] if student else teacher_action
            if student:
                captures,atari=0,False
                if action!=board.pass_action:
                    _,captures,liberties=board._simulate_stone(action)
                    atari=liberties==1 and captures==0
                game["decisions"].append({"ply":len(board.move_history),"action":action,
                    "raw_action":result["raw_action"],"raw_legal":result["raw_legal"],
                    "teacher_action":teacher_action,"nll":float(-result["policy_logits"][0,teacher_action]),
                    "captures":captures,"self_atari":bool(atari),"search_evaluations":search["network_evaluations"]})
            game["predictions"].append((board.to_play,float(result["output"].value[0])))
            board.play(action);game["board"]=_board_record(board)
            game["seconds"]+=time.perf_counter()-started
            consumed+=1
            if consumed%64==0 or (options.max_plies and consumed>=options.max_plies):checkpoint()
            if options.max_plies and consumed>=options.max_plies and not board.game_over:return
        work["games"].append(finish_record(game,board,{}));work["active_game"]=None
        checkpoint()
        print(json.dumps({"seed":args.seed,"value_scale":0,"game":index+1,"plies":len(board.move_history),"win":work["games"][-1]["win"]}),flush=True)
        if options.max_plies and consumed>=options.max_plies:return
    work["parameters_after"]=parameter_hash(agent)
    work["version_after"]=int(agent.online_parameter_version)
    assert work["parameters_before"]==work["parameters_after"]
    assert work["version_before"]==work["version_after"]
    work["completed"]=True;checkpoint()
    result={k:v for k,v in work.items() if k not in ("session","active_game")}
    result["checkpoint"]=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    result["checkpoint_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    result["evaluator_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save_json(result,ROOT/f"analysis/results/go11_value_zero_seed{args.seed}.json")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,required=True,choices=SEEDS)
    parser.add_argument("--max-plies",type=int,default=0)
    parser.add_argument("--result",help="optional original result path with its recorded model checkpoint")
    options=parser.parse_args()
    if options.max_plies<0:parser.error("max-plies is a pause budget, not an adjudication limit")
    torch.set_num_threads(1)
    run(options)
