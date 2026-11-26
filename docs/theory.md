# Theory notes (for the paper; Overleaf is written only when the authors say so)

Notation: contact network over windows t = t_lo .. t_obs - 1 with M contacts in all, n people,
K candidate sources left after pruning, J infected observers, S start times, P transmission
rates. A grid point is g = (τ, p) in the finite grid G (|G| = S P); a hypothesis is (s, g).
Reports O: each observer's infection window, or "healthy" at t_obs. Contact windows hold each
unordered pair at most once (`_bin_contacts` deduplicates), and each listed contact transmits
independently with probability p, from a node infected by t to its partner at t + 1.

Every statement below is checked numerically by `python -m experiments.verify_theory`
(section 6 lists what each check covers and its result).

## 1. Mean-field likelihood

Mean-field recursion, for a source s starting at τ:
    q_v(t + 1) = 1 - (1 - q_v(t)) Π_{u ~_t v} (1 - p q_u(t)),
where u ~_t v means u and v are in contact in window t. An observer infected at y contributes
q(y) - q(y - 1); an observer healthy at t_obs contributes 1 - q(t_obs - 1). The per-grid-point
likelihood L̃_g(s) is the product over observers (with a floor of 1e-4 per factor).

**Lemma 1 (mean field upper-bounds the exact marginals).** Let P_v(t) be the exact probability
that v is infected by t under the SI model above (same source, start and p). Then
P_v(t) <= q_v(t) for every v and t.

Proof. The infection indicators X_v(t) are increasing functions of the independent per-contact
transmission coins (more successful coins never remove an infection), so by Harris's inequality
E[f g] >= E[f] E[g] for any two non-negative decreasing functions f, g of them; since a product
of non-negative decreasing functions is again one, this extends by induction to any number of
factors. Given X(t), v escapes window t iff it is healthy and every infected partner's coin
fails, and the coins of window t are independent of X(t), so
    P(v healthy at t + 1) = E[(1 - X_v(t)) Π_{u ~_t v} (1 - p X_u(t))]
                          >= (1 - P_v(t)) Π_{u ~_t v} (1 - p P_u(t)).
The right side is decreasing in every P_u(t) and P_v(t), so if P(t) <= q(t) entrywise it is
>= (1 - q_v(t)) Π (1 - p q_u(t)) = 1 - q_v(t + 1). Hence P_v(t + 1) <= q_v(t + 1). At the
start P = q (the source is 1, everyone else 0), and induction over windows completes the proof. ∎

Remark. The bound needs the mean-field factor to match how many chances a pair has per window.
Here each pair appears once per window, so it does. If a pair listed k times in one window gave
k independent chances (option `weighted=True`), the factor is (1 - p q_u)^k, and the proof still
holds: after Harris, E[(1 - p X_u)^k] = 1 - (1 - (1 - p)^k) P_u (X_u is 0 or 1), which is
>= (1 - p P_u)^k because x -> (1 - p x)^k is convex and so lies below its chord on [0, 1].

**Lemma 2 (pruning).** Candidate s, which cannot start before b_s = max(t_lo, join_s), has exact
likelihood 0 for every grid point unless, for every infected observer j, some time-respecting
path leaves s at or after b_s and reaches j by its report y_j. The backward pass
`latest_departure` computes, for every j and every person u, the latest window ℓ_j(u) at which
u can be infected and still reach j by y_j; s is kept iff b_s <= ℓ_j(s) for all j.

Consequences, stated precisely: (i) no candidate with positive exact likelihood is dropped, and
in particular the true source is never dropped when the true start is >= t_lo (its infection
reached every infected observer along a time-respecting path); (ii) the exact posterior is
unchanged; (iii) under mean field with the floor, a dropped candidate would otherwise get a small
positive score, so pruning moves only impossible candidates to the bottom of the ranking and can
only improve the rank of the true source. (Checked: identical to the earlier n x n
earliest-arrival test on Hospital, High school and Email-Eu.)

## 2. Test selection (not part of the current claim)

Observer placement is not in the claim of the rebuilt paper (docs/ieee_trans_review.md §9). The
earlier Proposition 1 (from Chen, Hassani, Karbasi and Krause, COLT 2015) is kept out of the
paper unless that section returns; if it does, its constant γ = 7 / S_min and the form of the
δ term must first be checked line by line against the theorem as printed in that paper.

## 3. Source sets (dropped)

The split-conformal source sets are dropped from the contribution (X3: Dawkins et al. 2021
sets are comparable and need no calibration outbreaks). Dawkins et al. is cited; mean field can
serve as its test statistic. No proposition on sets appears in the paper.

## 4. Cost

| Method | Time per outbreak | Memory |
|---|---|---|
| Pruning (backward pass) | O(J (T n + M)) | O(J n) |
| Mean field (this work) | O(S P K (M + T n)) | O(B n), B = candidates per batch (<= 256) |
| Dynamic message passing | O(S P K (M + T E)), E directed edges | O(K E) |
| Belief propagation (SIB) | O(S P I Σ_v deg_v T²), I iterations | O(Σ_v deg_v T²) |
| Simulation (Sterchi et al.) | O(S P K N (M + n)), N = 10⁴ runs | O(N n) |

Measured (one core): mean field 0.4-26 s per outbreak on the six contact networks (200 outbreaks
each), belief propagation 149-1144 s; on sx-superuser (192,409 users) 571 s and 856 MB
(docs/ieee_trans_review.md §10).

## 5. What decides end-to-end accuracy when the start and rate are unknown

Setting. A localizer that does not know g sums it out. Under a prior ρ(g) u_g(s) (in the code ρ
is uniform over G, and u_g is uniform over the people who have joined by the first report; one
who joins after τ starts when they join), the exact posterior is
    P(s | O) = Σ_g W_g π_g(s),     W_g = ρ(g) Z_g / Σ_h ρ(h) Z_h,     π_g(s) = u_g(s) L_g(s) / Z_g,
with L_g(s) = P(O | s, g) and evidence Z_g = Σ_s u_g(s) L_g(s). So the posterior is a mixture:
the **weights** W_g say which (start, rate) explains the reports, and the **per-grid-point
posteriors** π_g say who the source is under that (start, rate).

Every localizer compared here has exactly this form with approximations W̃ and π̃:
- mean field: Z̃_g = Σ_s u_g(s) L̃_g(s), π̃_g ∝ u_g L̃_g (`mean_field_scores` is
  log Σ_g exp(log L̃_g(s)), which is this mixture up to a constant);
- belief propagation (SIB): score(s) = log Σ_g exp(Bethe log-likelihood_g) b_g(s), with b_g(s)
  its seed marginal. Writing c_g = Σ_s b_g(s), this is Σ_g Z̃_g π̃_g(s) with
  Z̃_g = exp(Bethe_g) c_g and π̃_g = b_g / c_g, an exact rewriting (so the unnormalized seed
  marginals are covered: their totals c_g become part of the evidence error);
- hybrid: mean-field W̃, belief-propagation π̃.
Rankings depend on P̃ only up to a constant factor, so W̃ may be normalized to sum to 1.

**Theorem 1.** Let P̃(s) = Σ_g W̃_g π̃_g(s), with W̃ a distribution on G and each π̃_g a
distribution on people, and let TV denote total variation distance.

(a) TV(P, P̃) <= TV(W, W̃) + Σ_g W_g TV(π_g, π̃_g).

(b) Suppose W̃_g ∝ ρ(g) Z̃_g with log Z̃_g = log Z_g + ε_g on the grid points where Z_g > 0,
and Z̃_g = 0 exactly where Z_g = 0. Let Δ = max_g ε_g - min_g ε_g (the spread of the
log-evidence errors). Then TV(W, W̃) <= tanh(Δ / 4), and this bound is attained.

(c) Draw (s*, g) from the prior and O from the model, and rank people by P̃(· | O). For any
reward that depends only on the rank of s*, lies in [0, 1] and does not increase with rank
(reciprocal rank, top-k hit),
    E[reward of ranking by P̃] >= E[reward of ranking by P] - 2 E[TV(P(· | O), P̃(· | O))],
and ranking by the exact posterior P maximizes the expected reward.

Proof. (a) P - P̃ = Σ_g (W_g - W̃_g) π̃_g + Σ_g W_g (π_g - π̃_g). Taking half the ℓ1 norm and
the triangle inequality, the first term is at most (1/2) Σ_g |W_g - W̃_g| ||π̃_g||_1 =
TV(W, W̃), because each π̃_g sums to 1; the second is at most Σ_g W_g TV(π_g, π̃_g).

(b) Grid points with Z_g = 0 have W_g = W̃_g = 0 and do not affect TV; drop them. Let
C = Σ_h W_h e^{ε_h}; then W̃_g = W_g r_g with r_g = e^{ε_g} / C, so Σ_g W_g r_g = 1, and
r_g ∈ [m, M] with m = e^{min ε} / C, M = e^{max ε} / C, M / m = e^Δ, and m <= 1 <= M (a
W-average of the r_g equals 1). Now TV(W, W̃) = Σ_g W_g (r_g - 1)_+. The function
x -> (x - 1)_+ is convex, so on [m, M] it lies below its chord: (r - 1)_+ <= (r - m)(M - 1) / (M - m).
Averaging under W and using Σ W_g r_g = 1 gives TV <= (1 - m)(M - 1) / (M - m). With k = e^Δ and
M = k m this is f(m) = (1 - m)(k m - 1) / (m (k - 1)) = (k + 1 - k m - 1/m) / (k - 1), whose
maximum over m > 0 is at m = k^{-1/2}, with value (√k - 1)² / (k - 1) = (√k - 1) / (√k + 1)
= tanh(Δ / 4). The bound is attained by two grid points with W = (√k, 1) / (√k + 1) and
ε = (0, Δ), which give W̃ = (1, √k) / (1 + √k) and TV = (√k - 1) / (√k + 1).

(c) Fix O and write p = P(· | O), q = P̃(· | O). Let R_q(s) be the reward when the source is s
and people are ranked by q (ties broken in any fixed way), so the expected reward under p is
E_p[R_q] = Σ_s p(s) R_q(s). For any c with values in [0, 1], |Σ_s (p - q)(s) c(s)| <= TV(p, q),
because Σ_s (p - q)(s) = 0. Hence E_p[R_q] >= E_q[R_q] - TV >= E_q[R_p] - TV >= E_p[R_p] - 2 TV.
The middle step is the rearrangement inequality: for a reward that does not increase with rank,
giving the best ranks to the largest probabilities maximizes Σ_s q(s) R(s). Taking the
expectation over O gives the claim; the same rearrangement argument under p shows that ranking by
p is optimal. ∎

**Corollary 1 (what the theorem says about the methods).** The loss of any of these localizers
in expected reciprocal rank, against the exact posterior on the same grid, is at most
    2 E[ tanh(Δ_O / 4) + Σ_g W_g TV(π_g, π̃_g) ].
The evidence enters only through Δ, the spread of its errors across grid points: an error
common to every (start, rate) pair costs nothing, while an approximation with accurate
per-grid-point posteriors can still be far off if its evidence is badly wrong at even one grid
point (the bound in (b) tends to 1 as Δ grows, and the two-point example attains it).

Scope, stated plainly.
1. The theorem is about agreement with the exact posterior on the same grid under the SI model.
   In the experiments the true start need not lie on the start grid, and outbreaks start "at a
   contact", not uniformly; the theorem then bounds the distance to the grid posterior, not to
   the truth.
2. It does not say that the mean-field evidence is accurate, nor that belief propagation's is
   not. Both facts are measured, not proved: told the true (start, rate), the two per-grid-point
   posteriors give the same accuracy (X1: BP - MF from -0.009 to +0.002 in MRR, all intervals
   across 0, 200 outbreaks x 4 networks); belief propagation fails to converge at some grid point
   in 78-98% of outbreaks; and replacing its weights by mean-field weights (the hybrid) removes
   its whole loss (X2). In the language of the theorem, the second term is about the same for
   both and the difference lies in Δ. The theorem shows why the weights are the place to look;
   the experiments show that the difference is there.
3. Lemma 1 bounds the mean-field marginals, not the likelihood or the evidence (a likelihood
   mixes q(y) - q(y - 1) and 1 - q terms, which move in opposite directions), so no claim of the
   form "the mean-field evidence is conservative" is made.

## 6. Verification (experiments/verify_theory.py)

| Statement | Check | Result |
|---|---|---|
| Lemma 1 | exact SI marginals by enumerating every coin outcome on random temporal graphs (3-6 people, 2-5 windows, <= 14 contacts, p ∈ {0.1, 0.3, 0.5, 0.9}) vs `mean_field_marginals` | passed: max(exact - mean field) = 6.7e-14 (rounding) |
| Theorem 1 (a) | 100,000 random mixtures (1-11 grid points, 2-14 people), 30% with one grid point off by e^20 | passed: bound never exceeded |
| Theorem 1 (b) | same instances; plus the two-point tightness example for Δ = 0.5, 2, 8 | passed; tight: TV = tanh(Δ/4) = 0.1244, 0.4621, 0.9640 |
| Theorem 1 (c) | same instances; expected reciprocal rank and top-1 | passed: loss never above 2 TV |
| Lemma 2 | pruning identical to the n x n earliest-arrival test (Hospital, High school, Email-Eu); mean-field scores bit-identical | passed (commit "Mean field memory fix") |

## 7. Removed from the earlier draft, and why

- Old §5 ("mean field beats belief propagation on loopy networks", with a clustering /
  cycles-per-edge table at damping 0.5). With damping 0.8 belief propagation is not beaten on
  every loopy network (review §8.1), and X1/X2 show the difference comes from the evidence
  weights, not from loops degrading the marginals. The loop explanation is withdrawn; Theorem 1
  and the X1/X2 measurements replace it.
- Old §3 (conformal source sets): dropped with the contribution (X3).
- Old §2 (Proposition 1, test selection): not in the current claim; see §2.
