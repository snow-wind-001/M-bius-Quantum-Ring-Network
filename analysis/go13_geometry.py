"""Finite-history geometry checks, not a long-history learning experiment."""
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.go13_algorithm import SEEDS,build,source_hashes,sha,prior
from mqr.go_history_tasks import generate_reachable_history_pairs
from mqr.temporal import TemporalMQRState


def main():
    torch.set_num_threads(1);rows=[]
    for seed in SEEDS:
        prepared=torch.load(ROOT/f'checkpoints/go13_v1/prepared-{seed}.pt',weights_only=True)
        pair=generate_reachable_history_pairs(1,seed=seed+121000,size=10,moves=64)[0]
        for geometry in ('orth','similar'):
            agent,encoder=build(seed,geometry);agent.load_state_dict(prepared['models'][geometry]);agent.double()
            features=torch.cat([encoder.encode_board(b) for b in pair.histories[0]]).double()
            for length in (8,16,32,64):
                torch.manual_seed(seed+length)
                h=torch.randn(1,16,dtype=torch.float64,requires_grad=True)
                q=torch.randn(16,dtype=torch.float64)
                state=agent.core.zero_state(1,device='cpu',dtype=torch.float64)
                state=TemporalMQRState(state.rings[:-1]+(h,))
                product=torch.eye(16,dtype=torch.float64)
                for x in features[:length].split(1):
                    with torch.no_grad():product=agent.core.conditioned_transition(5,x)[0]@product
                    _,state=agent.core.forward_step(x,state=state)
                gradient=torch.autograd.grad(state.rings[-1][0]@q,h)[0][0]
                rho=1-float(agent.core.leak_rates[5])
                expected=rho**length*product.T@q
                error=float((gradient-expected).abs().max())
                assert error<1e-9
                singular=torch.linalg.svdvals(product)
                bound=1. if geometry=='orth' else 2.
                assert float(singular.max())<=bound+1e-9
                assert float(singular.min())>=1/bound-1e-9
                basis=torch.linalg.qr(torch.randn(16,3,dtype=torch.float64)).Q
                previous=torch.linalg.qr(product.T@basis).Q
                terminal_projector=torch.eye(16,dtype=torch.float64)-basis@basis.T
                previous_projector=torch.eye(16,dtype=torch.float64)-previous@previous.T
                feasible_error=float((basis.T@product@previous_projector).abs().max())
                commute_error=float((terminal_projector@product-product@previous_projector).norm())
                assert feasible_error<1e-10
                if geometry=='orth':assert commute_error<1e-10
                rows.append({'seed':seed,'geometry':geometry,'history_steps':length,'adjoint_max_error':error,
                    'normalized_product_sigma_min':float(singular.min()),'normalized_product_sigma_max':float(singular.max()),
                    'covector_norm_ratio_to_rho':float(gradient.norm()/(q.norm()*rho**length)),
                    'constraint_pullback_error':feasible_error,'euclidean_projection_commutation_error':commute_error})
    prior.save_json({'source':source_hashes(),'diagnostic_sha256':sha(__file__),'verified':True,'rows':rows,
        'scope':'Fixed-parameter legal prefixes, exact state-adjoint and local equalities only. Similarity controls preserve constraint feasibility, but not Euclidean projection commutation. No additional long-history learning result is claimed.'},ROOT/'analysis/results/go13_geometry.json')
    print('verified',len(rows),'finite-history geometry configurations',flush=True)


if __name__=='__main__':main()
