"""Post-hoc policy drift on fixed held-out game prefixes; no updates or games."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.go13_algorithm import SEEDS, build, source_hashes, sha, prior, parameter_hash
from mqr import GoBoard


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    rows = []
    for seed in SEEDS:
        p = torch.load(ROOT / f'checkpoints/go13_v1/prepared-{seed}.pt', weights_only=True)
        base = torch.load(ROOT / f'checkpoints/go13_v1/base-{seed}.pt', weights_only=True)
        for geometry in ('orth', 'similar'):
            agents = [build(seed, geometry)[0] for _ in range(2)]
            agents[0].load_state_dict(p['models'][geometry])
            after = torch.load(ROOT / f'checkpoints/go13_v1/learned-{geometry}_free-{seed}.pt', weights_only=True)
            agents[1].load_state_dict(after['learner']['agent'])
            hashes = [parameter_hash(agent) for agent in agents]
            encoder = build(seed, geometry)[1]
            values = []
            for game_index, game in enumerate(base['probe_games']):
                board = GoBoard(10, komi=5.5)
                states = [agent.core.zero_state(1, device='cpu', dtype=torch.float32) for agent in agents]
                for ply, action in enumerate(game['moves']):
                    features = encoder.encode_board(board)
                    outputs = []
                    for index, agent in enumerate(agents):
                        _, states[index] = agent.core.forward_step(features, state=states[index])
                        outputs.append(agent.readout_state(states[index], features=features))
                    if ply % 8 == 4:
                        legal = torch.tensor(board.legal_moves())
                        logp = [out.policy_logits[0, legal].log_softmax(0) for out in outputs]
                        values.append({'game': game_index, 'ply': ply,
                                       'kl_before_to_after': float((logp[0].exp() * (logp[0] - logp[1])).sum()),
                                       'greedy_changed': int(logp[0].argmax() != logp[1].argmax()),
                                       'pass_before': float(outputs[0].pass_logit.sigmoid()),
                                       'pass_after': float(outputs[1].pass_logit.sigmoid())})
                    board.play(action)
                assert board.game_over
            assert hashes == [parameter_hash(agent) for agent in agents]
            rows.append({'seed': seed, 'geometry': geometry, 'positions': len(values),
                         'mean_legal_policy_kl': float(np.mean([v['kl_before_to_after'] for v in values])),
                         'greedy_changes': sum(v['greedy_changed'] for v in values),
                         'mean_pass_before': float(np.mean([v['pass_before'] for v in values])),
                         'mean_pass_after': float(np.mean([v['pass_after'] for v in values])),
                         'parameter_hashes': hashes, 'samples': values})
    summary = {}
    for geometry in ('orth', 'similar'):
        selected = [r for r in rows if r['geometry'] == geometry]
        summary[geometry] = {'positions': sum(r['positions'] for r in selected),
                             'greedy_changes': sum(r['greedy_changes'] for r in selected),
                             **{key: float(np.mean([r[key] for r in selected])) for key in
                                ('mean_legal_policy_kl', 'mean_pass_before', 'mean_pass_after')}}
    prior.save_json({'source': source_hashes(), 'diagnostic_sha256': sha(__file__), 'rows': rows,
                    'summary': summary, 'scope': 'Post-hoc checkpoint comparison on identical complete held-out probe trajectories; sample ply modulo 8 equals 4. Legal action policy comparison uses the engine mask only for evaluation. Existing complete games are replayed, not counted again. No policy training or causal attribution among shared features, pass and legality paths.'},
                   ROOT / 'analysis/results/go13_policy_drift.json')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
