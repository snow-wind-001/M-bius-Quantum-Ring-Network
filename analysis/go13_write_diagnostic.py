"""Post-hoc exact two-observation write factorization; no model updates."""
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.go13_algorithm import SEEDS, build, materialize, source_hashes, sha, prior, parameter_hash


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    rows = []
    for seed in SEEDS:
        prepared = torch.load(ROOT / f'checkpoints/go13_v1/prepared-{seed}.pt', weights_only=True)
        batch = materialize(prepared['data']['a_test'])
        x0, x1 = batch.features.double()
        assert torch.equal(x1[::2], x1[1::2])
        for geometry in ('orth', 'similar'):
            agent, _ = build(seed, geometry)
            agent.load_state_dict(prepared['models'][geometry])
            agent.double()
            before = parameter_hash(agent)
            core = agent.core
            u0 = core._inject(x0)
            _, h1 = core.forward_step(x0)
            _, h2 = core.forward_step(x1, state=h1)
            blocks = []
            for index, up in enumerate(core.input_up):
                rotation = core.conditioned_transition(index, x1[::2])
                blocks.append((1 - core.leak_rates[index]) * core.write_scales[index] * (rotation @ up.weight))
            maps = torch.cat(blocks, dim=1)
            delta_u = u0[::2] - u0[1::2]
            expected = (maps @ delta_u[:, :, None]).squeeze(-1)
            state = torch.cat(h2.rings, 1)
            actual = state[::2] - state[1::2]
            error = float((expected - actual).abs().max())
            assert error < 1e-10
            singular = torch.linalg.svdvals(maps)
            ranks = (singular > singular[:, :1] * 1e-10).sum(1)
            assert int(ranks.max()) <= core.injection_rank == 8
            assert before == parameter_hash(agent)
            rows.append({'seed': seed, 'geometry': geometry, 'paired_histories': len(delta_u),
                         'ring_coordinates': core.num_timescales * core.ring_dim,
                         'shared_write_coordinates': core.injection_rank,
                         'factorization_max_error': error, 'conditional_rank_min': int(ranks.min()),
                         'conditional_rank_max': int(ranks.max()),
                         'mean_pair_injected_difference': float(delta_u.norm(dim=1).mean()),
                         'mean_pair_state_difference': float(actual.norm(dim=1).mean()),
                         'tanh_saturation_fraction_abs_gt_095': float((u0.abs() > .95).double().mean()),
                         'parameter_hash': before})
    result = {'source': source_hashes(), 'diagnostic_sha256': sha(__file__), 'verified': True, 'rows': rows,
              'scope': 'For each fixed final observation, all six rings contain an affine image of the same eight first-observation write features. This exact two-observation rank bound does not prove impossibility on the finite ko dataset, or bound all long-history representations by eight.'}
    prior.save_json(result, ROOT / 'analysis/results/go13_write_diagnostic.json')
    print(json.dumps({'configurations': len(rows), 'maximum_error': max(r['factorization_max_error'] for r in rows),
                      'ranks': sorted(set(r['conditional_rank_max'] for r in rows))}), flush=True)


if __name__ == '__main__':
    main()
