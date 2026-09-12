"""Verify all complete value-ablation matches and paired win-rate differences."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np  # Match the experiment's NumPy-before-torch runtime order.
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.go11_verify import paired_interval
from analysis.go11_value_ablation import parameter_hash
from experiments.go11_continual import SEEDS,build,make_session,save_json,source_hashes
from mqr import GoBoard
from mqr.go import DefensiveGoTeacher,HeuristicGoTeacher
from mqr.go_search import policy_value_search


def verify(seed):
    torch.set_num_threads(1)
    original=json.loads((ROOT/f'analysis/results/go11_selfplay_cone6_seed{seed}.json').read_text())
    zero=json.loads((ROOT/f'analysis/results/go11_value_zero_seed{seed}.json').read_text())
    assert original['completed'] and zero['completed'] and zero['source']==original['source']==source_hashes()
    assert zero['evaluator_sha256']==hashlib.sha256((ROOT/'analysis/go11_value_ablation.py').read_bytes()).hexdigest()
    assert zero['value_scale']==0 and zero['config']['eval_games']==48 and len(zero['games'])==48
    assert zero['model_checkpoint_sha256']==original['checkpoint_sha256']
    assert zero['parameters_before']==zero['parameters_after']==original['evaluation_parameter_checks']['search_eval']['parameters_after']
    assert zero['version_before']==zero['version_after']==original['parameter_version']
    main=[g for g in original['games'] if g['phase']=='search_eval']
    for index,(game,reference) in enumerate(zip(zero['games'],main)):
        assert game['index']==index and game['opening']==reference['opening'] and game['student_color']==reference['student_color']
        assert game['updates']==[] and game['terminal_labels']==game['replayed_positions']==0
        assert game['moves'][:len(game['opening'])]==game['opening']
        board=GoBoard(10,komi=5.5)
        decisions={d['ply']:d for d in game['decisions']}
        teacher=DefensiveGoTeacher() if index//2%2 else HeuristicGoTeacher()
        for ply,action in enumerate(game['moves']):
            assert not board.game_over
            if ply>=len(game['opening']):
                if board.to_play==game['student_color']:
                    d=decisions[ply]
                    assert action==d['action'] and d['raw_legal']==board.is_legal(d['raw_action'])
                    assert d['teacher_action']==teacher.select_move(board)
                    assert 0<=d['search_evaluations']<=64
                    if action!=board.pass_action:
                        _,captures,liberties=board._simulate_stone(action)
                        assert captures==d['captures'] and (liberties==1 and captures==0)==d['self_atari']
                else:assert action==teacher.select_move(board)
            board.play(action)
        assert board.game_over and board.consecutive_passes==2 and game['terminated']
        assert board.score()==game['score'] and board.winner()==game['winner']
        assert game['win']==(board.winner()==game['student_color'])
        assert game['observations']==len(game['moves'])-len(game['opening'])
    checkpoint=Path(zero['checkpoint'])
    if not checkpoint.is_absolute():checkpoint=ROOT/checkpoint
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==zero['checkpoint_sha256']
    args=argparse.Namespace(**zero['config'])
    agent,encoder=build(args,args.method,seed);session=make_session(args,agent,encoder,args.method)
    session.load_state_dict(torch.load(checkpoint,weights_only=True)['session'])
    assert parameter_hash(agent)==zero['parameters_before']
    decisions=0
    for game in zero['games'][:2]:
        session.reset_game();board=GoBoard(10,komi=5.5)
        for ply,action in enumerate(game['moves']):
            prediction=session.observe(board,learn=False)
            if ply>=len(game['opening']) and board.to_play==game['student_color']:
                result=policy_value_search(agent,encoder,board,prediction['state'],prediction['policy_logits'],
                    simulations=64,max_depth=args.search_depth,value_scale=0)
                assert action==result['action'];decisions+=1
            board.play(action)
    return {'seed':seed,'games':48,'normal_wins':sum(g['win'] for g in main),
        'zero_wins':sum(g['win'] for g in zero['games']),'matches_reproduced':2,'decisions_reproduced':decisions,
        'network_evaluations':sum(g['search_evaluations'] for g in zero['games']),
        'plies_min_max':[min(len(g['moves']) for g in zero['games']),max(len(g['moves']) for g in zero['games'])]}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:rows=list(pool.map(verify,SEEDS))
    result={'verified':True,'rows':rows,'games':sum(r['games'] for r in rows),
        'normal_wins':sum(r['normal_wins'] for r in rows),'zero_wins':sum(r['zero_wins'] for r in rows),
        'normal_minus_zero':paired_interval([(r['normal_wins']-r['zero_wins'])/48 for r in rows]),
        'reproduced_matches':sum(r['matches_reproduced'] for r in rows),
        'reproduced_decisions':sum(r['decisions_reproduced'] for r in rows),
        'source':source_hashes(),'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save_json(result,ROOT/'analysis/results/go11_value_summary.json')
    print(json.dumps(result|{'source':'recorded'}))
