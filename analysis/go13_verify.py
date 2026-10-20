"""Independently replay Go13 rules/games and reproduce checkpoint decisions."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.go13_algorithm import (SEEDS, METHODS, source_hashes, sha, build, restore,
    evaluate, materialize, boards, parameter_hash, prior)
from mqr import GoBoard, HeuristicGoTeacher
from mqr.go import DefensiveGoTeacher
from mqr.go_online import GoOnlineSession
from mqr.go_research import history_output, queried_logits, task_parameters
from mqr.go_search import policy_value_search
from analysis.go12_verify import reference_score

OUT = ROOT / 'analysis/results'
DIRECTORY = ROOT / 'checkpoints/go13_v1'


def load(name):
    return json.loads((OUT / (name + '.json')).read_text())


def verify_game(record, *, teacher=None, color=None):
    b = GoBoard(10, komi=5.5)
    decisions = {d['ply']:d for d in record.get('decisions', [])}
    for index, move in enumerate(record['moves']):
        assert not b.game_over and b.is_legal(move)
        if teacher is not None and index >= len(record['opening']):
            if b.to_play == color:
                assert decisions[index]['action'] == move
                assert 0 <= decisions[index]['search_evaluations'] <= 16
            else:
                assert teacher.select_move(b) == move
        b.play(move)
    assert b.game_over and b.consecutive_passes == 2 and record['terminated']
    assert record['moves'][-2:] == [100, 100]
    assert b.score() == record['score'] and b.winner() == record['winner']
    assert reference_score(b) == record['score']['margin_black']
    if color is not None:
        assert record['win'] == int(b.winner() == color)
        assert record['margin'] == reference_score(b) * color
    return b


def audit_seed(seed):
    torch.set_num_threads(1)
    prepared_path = DIRECTORY / f'prepared-{seed}.pt'
    p = torch.load(prepared_path, weights_only=True)
    assert p['source'] == source_hashes() and p['same_initial_parameters']
    assert sha(DIRECTORY / f'base-{seed}.pt') == p['base_sha256']
    base = torch.load(DIRECTORY / f'base-{seed}.pt', weights_only=True)
    assert len(base['training_games']) == 24 and len(base['probe_games']) == 4 and base['epochs'] == 24
    for g in base['training_games'] + base['probe_games']:
        verify_game(g)
    assert not bool(base['base']['value.weight'].count_nonzero())
    counts = {}
    expected = {'a_train':2048, 'a_test':256, 'b_train':1024, 'b_test':256}
    observed_signatures = {}
    for split, records in p['data'].items():
        assert len(records) == expected[split]
        signatures = []
        for r in records:
            hs = boards(r); assert len(hs) == 2
            initial = hs[0]
            for point, color in enumerate(initial.board):
                if color:
                    assert initial._group_and_liberties(initial.board, point)[1], 'dead setup group'
            assert len(initial.position_history) == 1
            assert hs[-1].is_legal(r['query']) == bool(r['label'])
            signatures.append(repr((hs[-1].board, hs[-1].to_play, r['query'])))
        observed_signatures[split] = set(signatures)
        batch = materialize(records)
        assert batch.labels.tolist() == [0., 1.] * (len(records)//2)
        if split.startswith('a'):
            assert torch.equal(batch.features[-1,::2],batch.features[-1,1::2])
            assert not torch.equal(batch.features[0,::2],batch.features[0,1::2])
        counts[split] = len(records)
    assert not (observed_signatures['a_train'] | observed_signatures['b_train']) & (observed_signatures['a_test'] | observed_signatures['b_test'])
    assert len(p['anchor_indices']) <= 16 and len(set(p['anchor_indices'])) == len(p['anchor_indices'])
    for geometry in ('orth', 'similar'):
        agent, _ = build(seed, geometry);agent.load_state_dict(p['models'][geometry])
        train = materialize(p['data']['a_train']); held = materialize(p['data']['a_test'])
        assert evaluate(agent,held) == p['curves'][geometry][-1]['a']
        assert len(p['a_updates'][geometry]) == 128
        assert sum(r['feedback'] for r in p['a_updates'][geometry]) == 2048
        with torch.no_grad():
            anchors = train.subset(p['anchor_indices'])
            margins = (2*anchors.labels-1)*queried_logits(history_output(agent,anchors.features),anchors.queries) if len(anchors.labels) else torch.empty(0)
        assert bool((margins > .02).all())
        torch.testing.assert_close(p['anchor_floors'][geometry],.5*margins,rtol=0,atol=0)
        assert p['optimizers'][geometry]['state']
        assert all(int(s['step']) == 128 for s in p['optimizers'][geometry]['state'].values())
    return {'seed':seed,'verified':True,'base_complete_games':28,'rule_histories':counts,
            'anchors':len(p['anchor_indices']),'prepared_sha256':sha(prepared_path)}


def audit_run(method, seed):
    torch.set_num_threads(1)
    learning_path = OUT / f'go13_learning_{method}_seed{seed}.json'
    games_path = OUT / f'go13_games_{method}_seed{seed}.json'
    run = json.loads(learning_path.read_text()); games = json.loads(games_path.read_text())
    assert run['completed'] and games['completed']
    assert run['source'] == games['source'] == source_hashes()
    assert run['method'] == games['method'] == method and run['seed'] == games['seed'] == seed
    args = argparse.Namespace(**run['config']);args.method=method
    assert (args.size,args.a_pairs,args.b_pairs,args.test_pairs,args.eval_games,args.simulations,args.search_depth)==(10,1024,512,128,24,16,8)
    p, agent, encoder, batches, learner = restore(args,method)
    for task in ('a','b'):
        assert evaluate(agent,batches[task+'_test']) == run['before'][task]
    initial_hash = parameter_hash(agent)
    first_update = learner.step(batches['b_train'].subset(slice(0,16)))
    assert first_update == run['updates'][0]
    assert len(run['updates']) == 64 and sum(r['feedback'] for r in run['updates']) == 1024
    for index, update in enumerate(run['updates']):
        logits = torch.tensor(update['pre_logits']); labels=batches['b_train'].labels[index*16:index*16+16]
        assert len(logits)==16
        assert abs(float(F.binary_cross_entropy_with_logits(logits,labels))-update['pre_loss'])<1e-6
        assert float(((logits>=0)==labels.bool()).float().mean())==update['pre_accuracy']
        assert update['step_norm'] <= update['raw_step_norm'] + 1e-5
        assert update['backtracks'] <= 8
        if method.endswith(('_current','_stale')):assert update['actual_anchor_violations']==0
    path = DIRECTORY / f'learned-{method}-{seed}.pt'
    assert sha(path)==run['checkpoint_sha256']==games['checkpoint_sha256']
    saved=torch.load(path,weights_only=True);assert saved['source']==source_hashes()
    learner.load_state_dict(saved['learner'])
    assert parameter_hash(agent)==run['final_parameter_hash']==games['parameter_hash']
    if method.endswith('_frozen'):assert parameter_hash(agent)==initial_hash
    for task in ('a','b'):
        assert evaluate(agent,batches[task+'_test'])==run['after'][task]
    for intervention in ('reset','swap','no_query'):
        assert evaluate(agent,batches['a_test'],intervention=intervention)==run['interventions'][intervention]
    assert run['interventions']['reset']['accuracy']==.5
    assert len(games['games'])==24
    session=GoOnlineSession(agent,encoder,update_every=16,project_with_memory=False)
    reproduced=0; raw_query_changes=0
    for index, game in enumerate(games['games']):
        assert game['index']==index and game['student_color']==(1 if index%2==0 else -1)
        assert game['opening']==prior.opening(seed+91000+index//2,10,(4,24,60)[(index//2)%3])
        assert game['moves'][:len(game['opening'])]==game['opening']
        teacher=DefensiveGoTeacher() if index//2%2 else HeuristicGoTeacher()
        verify_game(game,teacher=teacher,color=game['student_color'])
        if index<2:
            session.reset_game(); board=GoBoard(10,komi=5.5)
            for ply,action in enumerate(game['moves']):
                result=session.observe(board,learn=False)
                if ply>=len(game['opening']) and board.to_play==game['student_color']:
                    search=policy_value_search(agent,encoder,board,result['state'],result['policy_logits'],simulations=16,max_depth=8)
                    assert search['action']==action
                    with torch.no_grad():
                        x=encoder.encode_board(board); state=result['state']
                        no_query=agent._query_output(agent.core.readout_state(state),state,x,include_query=False)
                        raw_query_changes+=int(int(no_query.policy_logits.argmax())!=result['raw_action'])
                    reproduced+=1
                board.play(action)
    assert parameter_hash(agent)==run['final_parameter_hash']
    def tensor_bytes(value):
        if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
        if isinstance(value,dict):return sum(tensor_bytes(v) for v in value.values())
        if isinstance(value,(tuple,list)):return sum(tensor_bytes(v) for v in value)
        return 0
    audit={'method':method,'seed':seed,'verified':True,'complete_games':24,'matches_reproduced':2,
           'decisions_reproduced':reproduced,'raw_query_action_changes_in_reproduced_decisions':raw_query_changes,
           'constraint_tensor_bytes':tensor_bytes(saved['learner']['constraint']),
           'optimizer_tensor_bytes':tensor_bytes(saved['learner']['optimizer']),
           'learning_sha256':sha(learning_path),'games_sha256':sha(games_path),'verifier_sha256':sha(__file__)}
    prior.save_json(audit,OUT/f'go13_audit_{method}_seed{seed}.json')
    return audit


def paired(values):
    values=np.array(values,dtype=float)
    samples=np.array([values[list(indices)].mean() for indices in itertools.product(range(len(values)),repeat=len(values))])
    return {'mean':float(values.mean()),'interval95':np.quantile(samples,[.025,.975]).tolist(),
            'per_seed':values.tolist(),'method':'descriptive paired training-seed bootstrap; all 3125 resamples'}


def summarize(audits,base_audits):
    table={}; all_runs={}
    for method in METHODS:
        runs=[load(f'go13_learning_{method}_seed{seed}') for seed in SEEDS]
        gs=[load(f'go13_games_{method}_seed{seed}') for seed in SEEDS]
        all_runs[method]=runs
        mean=lambda fn:float(np.mean([fn(r) for r in runs]))
        updates=[u for r in runs for u in r['updates']]
        games=[g for result in gs for g in result['games']]
        table[method]={'a_before_accuracy':mean(lambda r:r['before']['a']['accuracy']),
            'a_after_accuracy':mean(lambda r:r['after']['a']['accuracy']),
            'a_forgetting_loss':mean(lambda r:r['after']['a']['loss']-r['before']['a']['loss']),
            'b_before_accuracy':mean(lambda r:r['before']['b']['accuracy']),
            'b_after_accuracy':mean(lambda r:r['after']['b']['accuracy']),
            'b_learning_gain_loss':mean(lambda r:r['before']['b']['loss']-r['after']['b']['loss']),
            'a_pair_accuracy':mean(lambda r:r['after']['a']['both_correct']/r['after']['a']['pairs']),
            'reset_accuracy':mean(lambda r:r['interventions']['reset']['accuracy']),
            'swap_accuracy':mean(lambda r:r['interventions']['swap']['accuracy']),
            'no_query_accuracy':mean(lambda r:r['interventions']['no_query']['accuracy']),
            'prequential_b_loss':float(np.mean([u['pre_loss'] for u in updates])),
            'attempted_updates':0 if method.endswith('_frozen') else len(updates),
            'accepted_updates':sum(u['accepted'] for u in updates),
            'active_constraint_updates':sum(u['active']>0 for u in updates),
            'raw_anchor_violation_updates':sum(u['raw_anchor_violations']>0 for u in updates),
            'backtracked_updates':sum(u['backtracks']>0 for u in updates),
            'actual_anchor_violations':sum(r['anchor_violations'] for r in runs),
            'anchors':sum(len(r['anchor_indices']) for r in runs),
            'mean_step_norm':float(np.mean([u['step_norm'] for u in updates])),
            'mean_raw_step_norm':float(np.mean([u['raw_step_norm'] for u in updates])),
            'mean_first_order_max_error':float(np.mean([u['first_order_max_error'] for u in updates])),
            'mean_normal_age':float(np.mean([u['normal_age'] for u in updates])),
            'constraint_refreshes':sum(r['constraint_refreshes'] for r in runs),
            'b_training_seconds':sum(r['seconds'] for r in runs),
            'total_parameters':runs[0]['total_parameters'],'trainable_parameters':runs[0]['trainable_parameters'],
            'wins':sum(g['win'] for g in games),'games':len(games),'win_rate':float(np.mean([g['win'] for g in games])),
            'mean_margin':float(np.mean([g['margin'] for g in games])),
            'per_seed':[{ 'seed':r['seed'],'a_before':r['before']['a']['accuracy'],'a_after':r['after']['a']['accuracy'],
                         'b_before':r['before']['b']['accuracy'],'b_after':r['after']['b']['accuracy'],
                         'wins':sum(g['win'] for g in game['games'])} for r,game in zip(runs,gs)]}
    comparisons={}
    for left,right in (('orth_current','orth_free'),('similar_current','similar_free'),('orth_current','similar_current'),
                       ('orth_free','similar_free'),('orth_current','orth_stale'),('orth_current','orth_norm')):
        comparisons[left+'_minus_'+right]={}
        for metric,fn in {
            'a_retention_accuracy_change':lambda r:r['after']['a']['accuracy']-r['before']['a']['accuracy'],
            'b_learning_accuracy_change':lambda r:r['after']['b']['accuracy']-r['before']['b']['accuracy'],
            'a_forgetting_loss':lambda r:r['after']['a']['loss']-r['before']['a']['loss'],
            'b_learning_gain_loss':lambda r:r['before']['b']['loss']-r['after']['b']['loss'],
        }.items():
            comparisons[left+'_minus_'+right][metric]=paired([fn(a)-fn(b) for a,b in zip(all_runs[left],all_runs[right])])
    result={'verified':True,'source':source_hashes(),'verifier_sha256':sha(__file__),'table':table,'comparisons':comparisons,
            'base_audits':base_audits,'run_audits':audits,'complete_games':sum(r['complete_games'] for r in audits)+sum(r['base_complete_games'] for r in base_audits),
            'matches_reproduced':sum(r['matches_reproduced'] for r in audits),'decisions_reproduced':sum(r['decisions_reproduced'] for r in audits)}
    assert result['complete_games']==980
    prior.save_json(result,OUT/'go13_summary.json')
    print(json.dumps({'verified':True,'complete_games':result['complete_games'],'decisions_reproduced':result['decisions_reproduced']}),flush=True)


def main(workers):
    pending={(m,s) for m in METHODS for s in SEEDS};running={};audits=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while pending or running:
            for key in sorted(pending):
                if len(running)>=workers:break
                method,seed=key
                if not (OUT/f'go13_games_{method}_seed{seed}.json').exists():continue
                path=OUT/f'go13_audit_{method}_seed{seed}.json'
                if path.exists():
                    cached=json.loads(path.read_text())
                    if cached['verifier_sha256']==sha(__file__) and cached['games_sha256']==sha(OUT/f'go13_games_{method}_seed{seed}.json') and cached['learning_sha256']==sha(OUT/f'go13_learning_{method}_seed{seed}.json'):
                        audits.append(cached);pending.remove(key);continue
                running[pool.submit(audit_run,*key)]=key;pending.remove(key)
            if running:
                done,_=wait(running,timeout=2,return_when=FIRST_COMPLETED)
                for future in done:
                    row=future.result();audits.append(row);running.pop(future)
                    print(json.dumps({'audited':row['method'],'seed':row['seed'],'decisions':row['decisions_reproduced']}),flush=True)
            else:time.sleep(2)
    base=[audit_seed(seed) for seed in SEEDS]
    summarize(audits,base)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2)
    main(parser.parse_args().workers)
