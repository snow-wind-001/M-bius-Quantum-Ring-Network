"""Follow-up Go13 task order: protect mastered suicide rules, then learn ko."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import sys
import time
import json

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import go13_algorithm as main
from mqr import GoBoard, HeuristicGoTeacher
from mqr.go import DefensiveGoTeacher
from mqr.go_history_tasks import generate_ko_history_pairs
from mqr.go_online import GoOnlineSession
from mqr.go_research import OnlineRuleLearner,RuleConstraint,history_output,queried_logits

DIRECTORY=ROOT/'checkpoints/go13_transfer'
OUT=ROOT/'analysis/results'


def source_hashes():
    return {**main.source_hashes(),'experiments/go13_transfer.py':main.sha(__file__)}


def prepare(seed):
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    target=DIRECTORY/f'prepared-{seed}.pt'
    if target.exists():
        assert torch.load(target,weights_only=True)['source']==source_hashes()
        return
    original=torch.load(ROOT/f'checkpoints/go13_v1/prepared-{seed}.pt',weights_only=True)
    records=[]
    for index,p in enumerate(generate_ko_history_pairs(1152,seed=seed+101000,size=10)):
        records.extend(main.pack_history(h,p.recapture,v,'ko_followup',index) for h,v in zip(p.histories,p.legal))
    data={'old_train':original['data']['b_train'],'old_test':original['data']['b_test'],
          'new_train':records[:2048],'new_test':records[2048:]}
    def keys(rows):return {repr((main.boards(r)[-1].board,main.boards(r)[-1].to_play,r['query'])) for r in rows}
    prior_keys=set().union(*(keys(rows) for rows in original['data'].values()))
    assert not prior_keys & keys(records)
    assert not keys(data['new_train']) & keys(data['new_test'])
    old=main.materialize(data['old_train'])
    save={'source':source_hashes(),'seed':seed,'data':data,'models':{},'optimizers':{},'parent_checkpoints':{},'anchor_floors':{}}
    margins=[]
    for geometry in ('orth','similar'):
        path=ROOT/f'checkpoints/go13_v1/learned-{geometry}_free-{seed}.pt'
        parent=torch.load(path,weights_only=True)
        assert parent['source']==main.source_hashes()
        agent,_=main.build(seed,geometry);agent.load_state_dict(parent['learner']['agent'])
        save['models'][geometry]=copy.deepcopy(agent.state_dict())
        save['optimizers'][geometry]=copy.deepcopy(parent['learner']['optimizer'])
        save['parent_checkpoints'][geometry]=main.sha(path)
        prediction=torch.tensor(main.evaluate(agent,old)['logits'])
        margins.append((2*old.labels-1)*prediction)
    minimum=torch.stack(margins).amin(0).reshape(-1,2).amin(1)
    pairs=[int(i) for i in minimum.argsort(descending=True) if minimum[i]>.02][:8]
    indices=[2*i+j for i in pairs for j in (0,1)]
    save['anchor_indices']=indices
    save['anchor_floors']={g:.5*m[indices] for g,m in zip(('orth','similar'),margins)}
    main.prior.save_torch(save,target)
    main.prior.save_json({k:v for k,v in save.items() if k not in ('models','optimizers','anchor_floors')},OUT/f'go13_transfer_prepared_seed{seed}.json')
    print(json.dumps({'seed':seed,'followup_anchors':len(indices)}),flush=True)


def restore(seed,method):
    saved=torch.load(DIRECTORY/f'prepared-{seed}.pt',weights_only=True)
    assert saved['source']==source_hashes()
    geometry,mode=method.split('_')
    agent,encoder=main.build(seed,geometry);agent.load_state_dict(saved['models'][geometry])
    batches={k:main.materialize(v) for k,v in saved['data'].items()}
    constraint=None
    if mode in ('current','stale','norm'):
        constraint=RuleConstraint(batches['old_train'].subset(saved['anchor_indices']),saved['anchor_floors'][geometry],
            backend='transport' if geometry=='orth' else 'autograd',refresh_every=8 if mode=='stale' else 1)
    learner=OnlineRuleLearner(agent,mode=mode,lr=.003,max_step=.5,constraint=constraint)
    learner.optimizer.load_state_dict(copy.deepcopy(saved['optimizers'][geometry]))
    return saved,agent,encoder,batches,learner


def learn(seed,method):
    target=DIRECTORY/f'learned-{method}-{seed}.pt'
    if target.exists():
        assert torch.load(target,weights_only=True)['source']==source_hashes();return
    saved,agent,encoder,batches,learner=restore(seed,method)
    started=time.perf_counter()
    before={task:main.evaluate(agent,batches[task+'_test']) for task in ('old','new')}
    updates=[];curves=[]
    for start in range(0,2048,16):
        updates.append(learner.step(batches['new_train'].subset(slice(start,start+16))))
        if start+16 in (32,128,512,1024,2048):
            curves.append({'feedback':start+16,**{task:main.evaluate(agent,batches[task+'_test']) for task in ('old','new')}})
    after={task:main.evaluate(agent,batches[task+'_test']) for task in ('old','new')}
    interventions={kind:main.evaluate(agent,batches['new_test'],intervention=kind) for kind in ('reset','swap','no_query')}
    geometry=method.split('_')[0];anchors=batches['old_train'].subset(saved['anchor_indices'])
    with torch.no_grad():
        margins=(2*anchors.labels-1)*queried_logits(history_output(agent,anchors.features),anchors.queries) if len(anchors.labels) else torch.empty(0)
    result={'source':source_hashes(),'method':method,'seed':seed,'before':before,'after':after,'curves':curves,
        'updates':updates,'interventions':interventions,'anchor_indices':saved['anchor_indices'],
        'anchor_floors':saved['anchor_floors'][geometry].tolist(),'anchor_margins_after':margins.tolist(),
        'anchor_violations':int((margins<saved['anchor_floors'][geometry]-1e-5).sum()),
        'constraint_refreshes':0 if learner.constraint is None else learner.constraint.refreshes,
        'parent_checkpoints':saved['parent_checkpoints'],'parameter_hash':main.parameter_hash(agent),
        'seconds':time.perf_counter()-started,'completed':True}
    main.prior.save_torch({'source':source_hashes(),'learner':learner.state_dict(),'result':result},target)
    result['checkpoint_sha256']=main.sha(target)
    main.prior.save_json(result,OUT/f'go13_transfer_learning_{method}_seed{seed}.json')
    print(json.dumps({'method':method,'seed':seed,'old':after['old']['accuracy'],'new':after['new']['accuracy'],
                      'active':sum(u['active']>0 for u in updates),'accepted':sum(u['accepted'] for u in updates)}),flush=True)


def games(seed,method):
    _,agent,encoder,_,learner=restore(seed,method)
    path=DIRECTORY/f'learned-{method}-{seed}.pt';saved=torch.load(path,weights_only=True)
    assert saved['source']==source_hashes();learner.load_state_dict(saved['learner'])
    session=GoOnlineSession(agent,encoder,update_every=16,project_with_memory=False)
    checkpoint=DIRECTORY/f'games-{method}-{seed}.pt'
    result={'source':source_hashes(),'seed':seed,'method':method,'checkpoint_sha256':main.sha(path),
            'parameter_hash':main.parameter_hash(agent),'games':[],'completed':False}
    if checkpoint.exists():
        result=torch.load(checkpoint,weights_only=True);assert result['source']==source_hashes()
    args=argparse.Namespace(seed=seed,search_depth=8)
    for index in range(len(result['games']),24):
        session.reset_game();board=GoBoard(10,komi=5.5)
        opening=main.prior.opening(seed+111000+index//2,10,(4,24,60)[(index//2)%3])
        for action in opening:
            session.observe(board,learn=False);board.play(action)
        color=1 if index%2==0 else -1
        teacher=DefensiveGoTeacher() if index//2%2 else HeuristicGoTeacher()
        decisions=[];started=time.perf_counter()
        while not board.game_over:
            observation=session.observe(board,learn=False)
            if board.to_play==color:
                search=main.prior.search_decision(args,agent,encoder,board,observation,phase='a',index=index,simulations=16,training=False)
                action=search['action']
                decisions.append({'ply':len(board.move_history),'action':action,'raw_legal':observation['raw_legal'],
                                  'search_evaluations':search['network_evaluations']})
            else:action=teacher.select_move(board)
            board.play(action)
        result['games'].append({'index':index,'opening':opening,'student_color':color,'moves':board.move_history,
            'decisions':decisions,'terminated':board.game_over,'consecutive_passes':board.consecutive_passes,
            'score':board.score(),'winner':board.winner(),'win':int(board.winner()==color),
            'margin':board.score()['margin_black']*color,'seconds':time.perf_counter()-started})
        assert main.parameter_hash(agent)==result['parameter_hash']
        main.prior.save_torch(result,checkpoint)
        print(json.dumps({'method':method,'seed':seed,'game':index+1,'plies':len(board.move_history)}),flush=True)
    result['completed']=True;main.prior.save_torch(result,checkpoint)
    main.prior.save_json(result,OUT/f'go13_transfer_games_{method}_seed{seed}.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','learn','games'),required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--method',choices=main.METHODS,default='orth_current')
    args=parser.parse_args();torch.set_num_threads(1)
    prepare(args.seed) if args.stage=='prepare' else {'learn':learn,'games':games}[args.stage](args.seed,args.method)
