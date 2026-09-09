# Phase IV: Unknown Contextual Topology Qualification

> Evidence date: 2026-09-01. The primary machine-readable artifact is
> `analysis/results/mqr_contextual_topology_formal_10seed.json`, verified by
> `analysis/mqr_contextual_topology_verify.py`. The learning-rate robustness
> artifact is `mqr_contextual_topology_sparse_lr02_sensitivity_10seed.json`.
> Phase-III evidence is preserved and was not overwritten.

## Decision

The revised MQR can learn an unknown context-dependent route from downstream
query loss. Sparse Givens and block-Cayley routes both recover 100% of the
hidden topology over ten seeds, retain context A after adapting context B, beat
identity, and beat a direct arrival-ordered ring buffer. Event and distractor
candidate tokens have exactly identical empirical `(key,value)` marginals.
The write critic is trained on finite future write/no-write loss gain rather
than event labels.

The independent-algorithm gate nevertheless **fails**. A learnable sparse
permutation on the identical factor graph, with exactly equal route parameters,
state, analytic MACs, update budget, and peak workspace, also reaches 1.000
accuracy and 1.000 route recovery. The paired final-accuracy difference is
exactly `0.00000 [0.00000, 0.00000]`. Go, MiniCPM, and policy RL therefore
remain frozen.

## Task and causal protocol

Each seed samples two opaque contexts, A and B. Each context swaps a different
balanced subset of adjacent slot pairs. The learner receives the context index
but never receives the swap bits or source-to-destination map. It first predicts
queries under A, updates from query-value MSE, switches to an initially unknown B,
adapts online, and finally re-queries A.

For every cue key, two candidates are shown. Across every batch lane, event and
distractor sets are exact permutations of the same `(key,value)` multiset; key
histograms, signs, magnitudes, and candidate positions all match exactly. A
candidate is useful only when its key is relationally consistent with the prior
cue. The critic target is

\[
\Delta^{(j)} = \sum_{q\in\mathcal Q_t}
  \left[\ell_q(C_t)-\ell_q(C_t\leftarrow v^{(j)})\right],
\]

computed from paired futures. The gate sees raw previous-cue one-hot, current-key
one-hot, and value; it never sees the event mask. Exactly one of two candidates
may be written, so the hard write rate is 0.5 by construction.

The recorded 64 `delay_tokens` denote only a no-write hold interval. Because
this benchmark performs no recurrent transition during that interval, it is a
storage-persistence condition, not evidence of long-horizon recurrent credit.

## Why the strong control is decisive

For one local pair, the Givens-induced route is

\[
M_\theta=
\begin{bmatrix}
\cos^2\theta&\sin^2\theta\\
\sin^2\theta&\cos^2\theta
\end{bmatrix}.
\]

The learnable sparse-permutation control is

\[
M_z=
\begin{bmatrix}
1-\sigma(z)&\sigma(z)\\
\sigma(z)&1-\sigma(z)
\end{bmatrix}.
\]

For every interior probability, choose
`z = logit(sin²(theta))`; then `M_z = M_theta`. Composing the same pair schedule
preserves this equality layer by layer. Consequently, the two primary routes
have the same transition family. A consistent accuracy advantage cannot come
from representation capacity; only coordinate conditioning, optimization, or
additional constraints could distinguish them.

The derivatives explain why an untuned speed difference is not sufficient:

\[
\frac{dp}{d\theta}=2\sqrt{p(1-p)},\qquad
\frac{dp}{dz}=p(1-p).
\]

Thus equal nominal learning rates are not equal transition-space learning
rates. The test suite numerically verifies exact forward equivalence after this
coordinate map.

## Formal results

Ten seeds (`101..110`) use 96 predict-before-update steps on A, 96 on B, 64
held-out episodes per context, eight slots, and three query cycles.

| Method | Final accuracy | Route recovery | A forgetting | B prequential regret |
|---|---:|---:|---:|---:|
| residual sparse-Givens MQR | 1.0000 | 1.0000 | 0.0000 | 0.08826 `[0.08683,0.08969]` |
| block-Cayley MQR | 1.0000 | 1.0000 | 0.0000 | 0.07943 `[0.07816,0.08069]` |
| learnable sparse permutation | 1.0000 | 1.0000 | 0.0000 | 0.10564 `[0.10380,0.10747]` |
| identity / identity replacement | 0.7127 `[0.7046,0.7208]` | — | 0.0000 | — |
| direct ring buffer | 0.5029 `[0.4970,0.5089]` | — | 0.0000 | — |
| topology oracle | 1.0000 | 1.0000 | 0.0000 | — |

Paired primary-MQR accuracy gains are `0.28730 [0.27920,0.29541]` over identity
and `0.49707 [0.49112,0.50302]` over the direct ring buffer. Identity replacement
causes the same `0.28730` loss. Against the equally learnable sparse route, the
difference is exactly zero.

The critic obtains AUPRC `0.94876 [0.94501,0.95252]`, Brier score
`0.13603 [0.13189,0.14017]`, ECE `0.17198 [0.16591,0.17805]`, event/distractor
write rates `1.000/0.000`, and zero budget violations. Perfect write separation
is a result on a finite relational one-hot task, not a natural-stream claim.

### Learning-rate sensitivity

In the primary run, sparse-permutation regret minus Givens regret is
`+0.01738 [0.01668,0.01808]`. A post-hoc strong-baseline check changes only the
sparse route learning rate from 0.1 to 0.2. Its regret becomes
`0.07230 [0.07083,0.07377]`, and sparse-minus-Givens reverses to
`-0.01596 [-0.01647,-0.01545]`. Final accuracy and recovery remain tied at 1.0.
The adaptation-speed ranking is therefore coordinate/hyperparameter sensitive
and cannot support an MQR advantage claim.

## Resource audit

Shared write-gate costs are excluded from the route-core ratio, preventing a
large common module from hiding route overhead.

| Route core | Parameters | Optimizer bytes | State bytes | Forward MAC/query | Update MAC/query | Peak bytes |
|---|---:|---:|---:|---:|---:|---:|
| sparse-Givens MQR | 8 | 64 | 32 | 56 | 176 | 192 |
| sparse permutation | 8 | 64 | 32 | 56 | 176 | 192 |
| block Cayley | 32 | 256 | 32 | 40 | 352 | 112 |
| direct ring buffer | 0 | 0 | 40 | 0 | 0 | 40 |

All six Givens/sparse-permutation ratios are exactly 1.0. Block Cayley learns the
task but uses four times as many persistent route parameters and twice the
reported update MACs; it is a secondary mechanism result, not the resource-matched
primary. These are analytic counts. Python CPU timing is implementation-specific
and does not establish deployment efficiency.

## Interpretation and remaining limitations

This phase closes the fixed-ring shortcut and the marginal-magnitude shortcut.
It demonstrates useful progress: MQR route parameters receive behavioral credit,
learn hidden A/B maps, and are causally necessary relative to identity. It also
shows that block Cayley is numerically trainable once initialized away from the
zero-Jacobian identity point.

It does not establish MQR specificity:

1. The strongest equal-resource control has the same transition family and ties.
2. Zero A forgetting follows from context-bank parameter isolation; it is not
   evidence that one shared route resists catastrophic forgetting.
3. The task has eight slots and disjoint pair swaps, not arbitrary sparse graphs,
   unseen-context inference, or changing topology without an explicit context id.
4. The write gate is calibrated before route adaptation; this isolates routing
   but is not joint end-to-end online critic learning.
5. The hard top-1 straight-through gradient remains biased.

The next valid step is not Go or a larger model. It is a topology-generalization
study with held-out swap compositions, shared-context inference, and a
pre-registered transition-space learning-rate/optimizer sweep. MQR must then
beat the best tuned direct route on prequential regret or accuracy while retaining
the equal-resource gate. Until that happens, the defensible artifact is a
functional contextual routing memory, not a uniquely competitive MQR algorithm.

## Reproduction

```bash
python3 test_routed_memory.py
python3 experiments/mqr_contextual_topology_qualification.py \
  --seeds 10 --seed-start 101 --workers 4 \
  --output analysis/results/mqr_contextual_topology_formal_10seed.json
python3 analysis/mqr_contextual_topology_verify.py \
  analysis/results/mqr_contextual_topology_formal_10seed.json

# Post-hoc strong-baseline sensitivity; not a replacement for the primary run.
python3 experiments/mqr_contextual_topology_qualification.py \
  --seeds 10 --seed-start 101 --workers 4 \
  --sparse-permutation-lr 0.2 \
  --output analysis/results/mqr_contextual_topology_sparse_lr02_sensitivity_10seed.json
```
