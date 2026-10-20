"""Post-hoc frozen-feature probes after failure of original historical learning.

Ridge is fixed at one and trained only on already observed A labels. This is
an offline diagnostic, never a replacement for online model performance.
"""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.go13_algorithm import SEEDS,build,materialize,evaluate,source_hashes,sha,prior,parameter_hash


@torch.no_grad()
def features(agent,batch,reset=False):
    chunks=[];max_error=0.
    for indices in torch.arange(len(batch.labels)).split(64):
        state=agent.core.zero_state(len(indices),device='cpu',dtype=torch.float32)
        for x in batch.features[:,indices]:
            if reset:state=agent.core.zero_state(len(indices),device='cpu',dtype=torch.float32)
            latent,state=agent.core.forward_step(x,state=state)
        hidden,correction,_,memory=agent._query_features(latent,state,x)
        q=batch.queries[indices];rows=torch.arange(len(indices))
        spatial=hidden.flatten(2).transpose(1,2)[rows,q]
        read=memory[rows,q]
        x_probe=torch.cat((spatial,read),1)
        weight=torch.cat((agent.spatial_skip_heads.legality.weight.flatten(),agent.point_correction.weight[1,:agent.query_dim]))
        reconstructed=x_probe@weight+agent.spatial_skip_heads.legality.bias[0]
        actual=agent._query_output(latent,state,x).legality_logits[rows,q]
        max_error=max(max_error,float((reconstructed-actual).abs().max()))
        chunks.append(x_probe.double())
    assert max_error<1e-4
    return torch.cat(chunks),max_error


def ridge(train,labels,held,test_labels):
    mean=train.mean(0);std=train.std(0).clamp_min(1e-8)
    x=(train-mean)/std;test=(held-mean)/std
    y=2*labels.double()-1
    weight=torch.linalg.solve(x.T@x+torch.eye(x.size(1),dtype=x.dtype),x.T@y)
    score=test@weight
    correct=(score>=0)==test_labels.bool()
    return {'accuracy':float(correct.double().mean()),'both_correct':int(correct.reshape(-1,2).all(1).sum()),
            'lambda':1.,'dimension':x.size(1),'weight':weight.tolist(),'mean':mean.tolist(),'std':std.tolist()}


def main():
    torch.set_num_threads(1);rows=[]
    for seed in SEEDS:
        p=torch.load(ROOT/f'checkpoints/go13_v1/prepared-{seed}.pt',weights_only=True)
        train=materialize(p['data']['a_train']);held=materialize(p['data']['a_test'])
        for geometry in ('orth','similar'):
            for stage in ('after_a','after_b'):
                agent,_=build(seed,geometry)
                if stage=='after_a':state=p['models'][geometry]
                else:state=torch.load(ROOT/f'checkpoints/go13_v1/learned-{geometry}_free-{seed}.pt',weights_only=True)['learner']['agent']
                agent.load_state_dict(state);before=parameter_hash(agent)
                fit,error=features(agent,train);test,test_error=features(agent,held)
                reset_fit,_=features(agent,train,reset=True);reset_test,_=features(agent,held,reset=True)
                probe=ridge(fit,train.labels,test,held.labels)
                reset=ridge(reset_fit,train.labels,reset_test,held.labels)
                assert reset['accuracy']==.5
                raw=evaluate(agent,held)
                assert before==parameter_hash(agent)
                rows.append({'seed':seed,'geometry':geometry,'stage':stage,'model_accuracy':raw['accuracy'],
                    'probe':probe,'reset_probe':reset,'exact_readout_reconstruction_error':max(error,test_error),
                    'paired_feature_distance':float((test[::2]-test[1::2]).norm(dim=1).mean()),'parameter_hash':before})
    summary={}
    for geometry in ('orth','similar'):
        for stage in ('after_a','after_b'):
            selected=[r for r in rows if r['geometry']==geometry and r['stage']==stage]
            summary[f'{geometry}_{stage}']={
                'model_accuracy':float(np.mean([r['model_accuracy'] for r in selected])),
                'offline_probe_accuracy':float(np.mean([r['probe']['accuracy'] for r in selected])),
                'reset_probe_accuracy':.5,'paired_feature_distance':float(np.mean([r['paired_feature_distance'] for r in selected]))}
    prior.save_json({'source':source_hashes(),'diagnostic_sha256':sha(__file__),'trigger':'Original A training failed the qualified historical-anchor gate; diagnose information available to the existing raw legality readout.',
        'scope':'Fixed 20-dimensional features already used by the existing linear legality readout; offline train-only ridge. No model update, no held-out tuning and no claim of online improvement.','rows':rows,'summary':summary},ROOT/'analysis/results/go13_readout_diagnostic.json')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
