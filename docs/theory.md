# Theory notes (for the paper; Overleaf is written only when the authors say so)

Notation: contact network over windows t = t_lo .. t_obs - 1 with M contacts in all, n people,
K candidate sources left after pruning, S start times, P transmission rates (and R, Σ for the
recovery and susceptible-fraction grids when used). A hypothesis is h = (s, τ, p).

## 1. Mean-field likelihood (Lemmas 1, 2: proofs already in the manuscript)

Lemma 1. For every source, start and rate, the mean-field infection probability q_v(t)
upper-bounds the exact probability that v is infected by t (SI dynamics, independent
transmissions), by induction over windows using 1 - Π(1 - p q_u) ≥ P(some neighbour infects v).

Lemma 2. A candidate that cannot reach an infected observer before its report by a
time-respecting path has exact likelihood 0, and pruning it changes no score. The earliest
arrival test is exact, so pruning drops no possible source.

## 2. Test selection: what is guaranteed

Setting of Chen, Hassani, Karbasi and Krause (COLT 2015): a hidden hypothesis Y with a known
prior over n values, and tests X_e that are conditionally independent given Y. Under mean field
the reports of different people are conditionally independent given the full hypothesis
h = (s, τ, p), so the joint-hypothesis problem is exactly of this form, with Y = h and
X_e = person e's report (one of the windows, or "healthy"). Writing each test as a channel from
D_e(h) = p_e(· | h) (hypotheses that give person e the same report distribution are merged),
separability S(W_e) = min over distinct inputs of TV², S_min = min_e S(W_e).

Proposition 1 (from Chen et al., Theorem 2). The greedy policy that picks the test with the
largest mutual information about the full hypothesis h (the joint-hypothesis criterion,
"AKLD" in Sterchi et al. 2023, computed under mean field), run for k' tests, satisfies for
every k and δ > 0
    I(π_greedy[k']; h) ≥ I(π_OPT[k]; h) - δ (1 - exp(-k' / (k γ max{log n, log 1/δ}))),
with γ = 7 / S_min and n the number of hypotheses (K S P).

Scope, stated plainly: (i) the guarantee is relative to the mean-field model of the reports;
(ii) it covers the joint criterion; the source-only information gain ties with it in every
experiment (outputs/observer_placement, 7 networks × 3 seeds) but is not covered by the theorem;
(iii) S_min, computed in outputs/separability.json, can be small because some hypotheses give
nearly identical report distributions, so the constant is loose even though the bound holds.

## 3. Source sets with guaranteed coverage

Proposition 2 (split conformal prediction, Vovk et al. 2005; Lei et al. 2018). Let the
nonconformity score of an outbreak be the rank of its true source under any fixed localizer,
and k the ⌈(m + 1)(1 - α)⌉-th smallest rank over m calibration outbreaks. If calibration and
test outbreaks are exchangeable, the k top-ranked people contain the source with probability at
least 1 - α (ties broken against the method). Nothing is assumed about the network or the
model; a better localizer gives a smaller k. Checked empirically on the held-out halves
(outputs/conformal_sets.json: coverage 0.85-0.94 at nominal 0.90 over 100-140 test outbreaks).

## 4. Cost

| Method | Time per outbreak | Memory |
|---|---|---|
| Mean field (this work) | O(S P K (M + T n)) | O(K n) |
| Dynamic message passing | O(S P K (M + T E)), E directed edges | O(K E) |
| Belief propagation (SIB) | O(S P I Σ_v deg_v T²), I iterations, per candidate-free run | O(Σ_v deg_v T²) |
| Simulation (Sterchi et al.) | O(S P K N (M + n)), N = 10⁴ runs | O(N n) |
| Greedy test selection, per pick | mean field once + O(H n C) for H kept hypotheses, C report classes | O(H n C) |

Measured (outputs/bp_comparison, 200 outbreaks per network, one core): mean field 0.4-26 s, message
passing 2-295 s, belief propagation 149-1144 s; simulation-based selection 729-1184 s per pick vs
0.3 s for mean field (Hospital).

## 5. When mean field beats belief propagation

Belief propagation is exact on trees and approximate on loopy graphs, where messages are
correlated; mean field ignores correlations everywhere but its error does not grow with loops
in the same way, and it sums the start time and rate out over the whole grid at no extra cost.
Across the seven networks (BP comparison, true rate, 162-200 outbreaks each):

| Network | Clustering | Independent cycles per edge | MF - BP (MRR) |
|---|---|---|---|
| Growing PA (synthetic) | 0.018 | 0.48 | -0.034 (n.s.) |
| CollegeMsg | 0.109 | 0.86 | +0.056 |
| Office | 0.426 | 0.88 | +0.043 |
| High school | 0.475 | 0.92 | +0.119 |
| Hospital | 0.640 | 0.94 | +0.057 |
| Email-Eu | 0.407 | 0.94 | +0.123 |
| Primary school | 0.526 | 0.97 | +0.235 |

The only network where belief propagation is ahead is the only one that is close to a tree;
every real contact network is loopy, and there mean field is ahead on all six.
