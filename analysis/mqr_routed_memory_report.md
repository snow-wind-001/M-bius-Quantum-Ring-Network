# Residual Address-Routing MQR: Design and Qualification

> Evidence date: 2026-09-01. The machine-readable authority is
> `analysis/results/mqr_routed_memory_qualification_formal_10seed.json`, checked
> by `analysis/mqr_routed_memory_verify.py`. Historical negative Go and mixed-state
> MQR results are preserved and remain valid.

## 1. Decision

The redesign fixes the previously identified mechanism failure: MQR no longer
repeatedly averages remembered content. Physical content slots remain unchanged
unless an explicit write transaction targets one slot; the residual MQR acts on
addresses only. On the pre-registered synthetic routing task, all requested
mechanism gates pass: MQR beats identity, GRU, and fast-weight, identity replacement
causes a large loss, event and distractor write rates separate, and the five-axis
resource gate passes.

This is **not yet an independent MQR algorithm advantage**. A direct fixed cyclic
ring buffer obtains the same perfect accuracy, so MQR-minus-fixed-ring is exactly
`0.0000 [0.0000, 0.0000]`. The result establishes that the new routing formulation
is functional and behaviorally necessary relative to identity; it does not establish
that Cayley/unistochastic machinery is necessary.

## 2. Architecture and invariants

Let (C_t\in\mathbb R^{S\times d}) contain physical slot contents and let
(a_t\in\Delta^{S-1}) be a logical address. For a doubly stochastic route (T),

\[
T_\varepsilon=(1-\varepsilon)I+\varepsilon T,\qquad 0\le\varepsilon\le1.
\]

Both terms are non-negative and doubly stochastic, so their convex combination is
also doubly stochastic. Hence

\[
T_\varepsilon\mathbf1=\mathbf1,\qquad
\mathbf1^\top T_\varepsilon=\mathbf1^\top,
\]

and its induced (ell_1) and (ell_\infty) norms equal one. Routing proposes
(\tilde a_{t+1}=a_tT_\varepsilon); it never multiplies (C_t). With no write,

\[
C_{t+1}=C_t
\]

exactly, not merely up to a norm bound. A hard write to slot (j) is

\[
C_{t+1,j}=(1-g_t)C_{t,j}+g_tv_t,
\qquad C_{t+1,k}=C_{t,k}\;(k\ne j).
\]

Thus (g_t=1) preserves every unaddressed slot bit-for-bit.

### Discrete address commit

Pure residual routing still diffuses addresses. For a cyclic permutation (P),
the eigenvalues of ((1-\varepsilon)I+\varepsilon P) are

\[
\lambda_k=(1-\varepsilon)+\varepsilon e^{2\pi ik/S}.
\]

For (0<\varepsilon<1) and (k\ne0), (|\lambda_k|<1); repeated soft routing
therefore converges toward the uniform address. The implementation avoids reintroducing
this long-delay failure by committing

\[
a_{t+1}=\operatorname{onehot}\!\left(\arg\max_i\tilde a_{t+1,i}\right)
\]

and using a straight-through surrogate only in the backward pass. If (a_t=e_j),
(T=P), (P(j)\ne j), and (\varepsilon>1/2), the routed coordinate has mass
(\varepsilon) while the identity coordinate has mass (1-\varepsilon). Therefore
the forward projection selects (P(j)) exactly at every step, by induction. The
512-step test verifies this property at (\varepsilon=0.75).

This projection is nonlinear. Claims about the doubly stochastic proposal do not
automatically apply to the committed one-hot dynamics, and the straight-through
gradient is biased rather than the exact derivative of `argmax`.

## 3. Sparse route and cost contract

`mqr/routing.py` implements five route families:

| Route | Forward | Refresh | Peak route workspace | Guarantee |
|---|---:|---:|---:|---|
| identity | (O(1)) | (0) | (O(S)) | exact identity |
| local permutation | (O(S)) | (0) | (O(S)) | monomial-unitary route |
| cyclic Givens | (O(LS)) | (O(LS)) | (O(S)) | product of local unistochastic factors |
| block Cayley, block (b) | (O(Sb)) | (O(Sb^2)) | (O(b^2)) | each block is exactly ‎(|U_b|^2) |
| dense Cayley | (O(S^2)) | (O(S^3)) | (O(S^2)) | exact global ‎(|U|^2) |

The cyclic Givens product is guaranteed doubly stochastic, but a product of
unistochastic matrices need not itself be unistochastic. The code records this
distinction instead of claiming a nonexistent global certificate.

At (S=8), 32-step refresh intervals, the analytic refresh estimates are
`64/1024/4096` MACs for cyclic Givens, block Cayley, and dense Cayley; peak route
workspaces are `160/288/1088` bytes. These are analytic estimates, not measured
accelerator FLOPs.

## 4. Budgeted temporal utility

The critic predicts advantages at horizons 16, 64, and 128 and combines them with
fixed weights. Positive and non-positive examples occupy separate bounded replay
partitions and are sampled equally. For target write rate (ho), the online ledger
uses

\[
A_t=\lceil\rho t\rceil,\quad
w_t=\mathbf1[p_t\ge\tau]\mathbf1[W_{t-1}<A_t],\quad
W_t=W_{t-1}+w_t.
\]

Because (W_{t-1}\le A_{t-1}\le A_t), either decision leaves (W_t\le A_t).
The hard budget therefore holds for every prefix, not only in expectation. Reports
include AUPRC, Brier score, ECE, total/event/distractor write rates, write gap, and
budget violation.

## 5. Formal protocol

- 10 seeds (`71..80`), 64 held-out episodes per seed.
- Eight content slots; every method has exactly 16 float state values (`64 B`).
- Training delay 16, held-out delay 64, and two complete query cycles.
- No event/query/phase/context flag enters the write critic. Its five inputs are
  magnitude, novelty versus EMA, prediction error versus the previous observation,
  EMA magnitude, and an uncertainty proxy.
- Critic supervision is computed by exact finite paired write/no-write rollouts.
- Primary route: (0.25I+0.75P), where (P) is a cyclic monomial unitary, followed
  by straight-through top-1 address commit.
- Controls: identity, identity replacement, state-matched GRU-16, state-matched
  (4\times4) fast weight, and a direct fixed cyclic ring buffer.
- MQR/control ratios must not exceed 1.05 for trainable parameters, state bytes,
  forward MACs, amortized update MACs, and peak analytic workspace.

The task cores receive causal phase and context values needed to identify an address;
the write critic does not. Events have much larger magnitude than distractors, so
the current marker-free task still has an easy distributional cue. It is not a
matched-marginal novelty benchmark.

## 6. Results

| Method | Overall | Forward | Context reversal | Repeated query |
|---|---:|---:|---:|---:|
| residual routed MQR | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| identity | 0.5662 | 0.5557 | 0.5723 | 0.5662 |
| GRU-16 | 0.5075 | 0.4954 | 0.5149 | 0.5074 |
| fast-weight (4\times4) | 0.6482 | 0.6385 | 0.6581 | 0.6482 |
| fixed ring buffer | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Paired MQR accuracy improvements are:

- versus identity: `0.43379 [0.42028, 0.44729]`;
- versus GRU: `0.49248 [0.48207, 0.50289]`;
- versus fast-weight: `0.35176 [0.32264, 0.38087]`;
- versus fixed ring buffer: `0.00000 [0.00000, 0.00000]`.

Identity replacement has the same significant `0.43379` drop. The gate obtains
AUPRC `1.000`, Brier `0.02359 [0.02217,0.02500]`, ECE
`0.14885 [0.14389,0.15381]`, event write rate `1.000`, distractor write rate
`0.000`, and no budget violations. Perfect separation reflects the easy magnitude
cue and should not be extrapolated to natural streams.

The strict analytic resource ratios pass. MQR and identity execute the same residual
interpolation/address-commit kernel; their parameters, state, forward MACs,
amortized updates, and peak workspace ratios are all exactly `1.0` (`291` trainable
parameters, `64 B` state, `296` MAC/token, and `160 B` analytic peak workspace).
Measured Python CPU runtime is not competitive: MQR averages `1.222 ms/episode`,
versus identity `0.889`, GRU `0.259`, and fast-weight `0.678`. Formal deployment
claims therefore remain unsupported.

## 7. What changed—and what did not

The old negative result was caused in part by asking a doubly stochastic operator
to preserve content while repeatedly averaging it. Address/content separation fixes
that mismatch. It also produces a decisive identity ablation, which the old mixed
state failed to do.

However, the success currently comes from a known cyclic permutation plus discrete
slot storage. A conventional ring buffer implements the same computation without
Cayley coordinates, unistochastic learning, or a quantum interpretation. The formal
conclusions are therefore:

1. mathematical route invariants: **passed**;
2. causal/budgeted write mechanism: **passed on an easy synthetic distribution**;
3. requested route-mechanism qualification: **passed**;
4. MQR-specific independent algorithm advantage: **failed**;
5. Go/LLM continual-learning advantage: **unchanged and unproved**.

## 8. Next decisive experiment

The next task must remove the fixed-ring explanation, not merely add more seeds:

1. Sample an unseen sparse permutation graph or block topology per context; a fixed
   ring buffer must no longer implement the answer.
2. Learn cyclic Givens/block-Cayley routes from causal future loss, and compare with
   equally learnable sparse permutation, deque, GRU, fast-weight, and Sinkhorn routes.
3. Match event and distractor marginal magnitude; only relational novelty,
   prediction error, or uncertainty may reveal future utility.
4. Train forward context A, switch online to reversed/permuted context B, and measure
   adaptation regret plus A forgetting. The present experiment supplies the route
   direction and therefore is not online reversal learning.
5. Ablate soft routing, hard projection, straight-through, Gumbel/top-k, and fixed
   routes. Report gradient bias and address entropy.
6. Measure actual P50/P95 latency and peak allocated memory; analytic MACs remain a
   secondary audit.

Only if learned MQR beats the direct learnable sparse-route control under this
protocol should it be integrated into MiniCPM/Go again. Until then, the practical
artifact is a reliable budgeted ring-memory sidecar, not a unique general-learning
algorithm.

## 9. Reproduction

```bash
python3 test_routed_memory.py
python3 experiments/mqr_routed_memory_qualification.py \
  --seeds 10 --seed-start 71 --workers 4 \
  --output analysis/results/mqr_routed_memory_qualification_formal_10seed.json
python3 analysis/mqr_routed_memory_verify.py \
  analysis/results/mqr_routed_memory_qualification_formal_10seed.json
```
