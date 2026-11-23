# Strict internal review (2026-11-22)

Bar: an IEEE Transactions paper (e.g. IEEE TNSE, TSP). Read for calibration: Yu & Tan, "Unraveling
the viral spread of misinformation: maximum-likelihood estimation and starlike tree
approximation in Markovian spreading models" (IEEE TSP 2025): one estimator, a formal model,
exactness/approximation results as lemmas and theorems, then experiments on synthetic and real
networks against standard baselines. A Transactions paper is accepted for one clear,
well-supported contribution, not for breadth. Everything below is judged against that.

Every number is taken from `outputs/tables.json` (rebuilt by `python -m experiments.make_tables`).

## 1. The one claim

> A mean-field likelihood with the start time and spreading rate summed out makes Bayesian
> source inference on temporal contact networks near-optimal and fast enough to act on:
> on real contact networks it localizes better than belief propagation at 40-1400x lower cost,
> its speed makes active test selection and calibrated source sets practical, and on real
> outbreaks whose origin is hidden from the reports it finds the origin where heuristics fail.

Everything kept in the paper must support one clause of this sentence.

## 2. Claim-by-claim verdict

| # | Clause | Evidence | Verdict |
|---|---|---|---|
| C1 | near-optimal | MF vs exact Monte Carlo likelihood, Hospital 200 outbreaks (told: 0.470 vs 0.475; grid: MF 0.352 vs exact 0.324). Office and High school running. | Holds for Hospital; must hold on the two new networks before the word "near-optimal" is used without qualification |
| C2 | better than BP on real contact networks | 200 outbreaks per network (202 Primary school). MF - BP: Hospital +0.057 (p 2e-6), Primary +0.230 (p 1e-18), Office +0.043 (p 2e-4), High school +0.119 (p 6e-9), Email-Eu +0.123 (p 1e-9), CollegeMsg +0.066 [+0.016, +0.116] (p 0.048) | Holds on 6/6 real networks, all significant |
| C3 | 40-1400x lower cost | Measured seconds per outbreak, same machine class, single core | Holds |
| C3b | when BP wins | Growing PA (synthetic, clustering 0.018, the only near-tree graph): MF - BP -0.034 (n.s.) | Explained by theory (BP exact on trees); kept as the boundary of the claim |
| C4 | active test selection practical | 120 outbreaks per network (3 seeds): our criterion ties the joint (Sterchi-type) criterion on all 7 networks and beats the other 7 strategies on 6/7; Sterchi's own simulation pipeline, 16 Hospital outbreaks: 0.566 MRR at 611 s per pick vs ours 0.579 at 0.3 s | Holds; Primary school: "test the top suspect" is better (-0.043 [-0.073, -0.014]); on CollegeMsg it is not significantly worse (+0.039 [-0.010, +0.089]) |
| C5 | calibrated source sets | Split conformal, coverage 0.85-0.94 at nominal 0.90; MF sets 18-32 people vs BP 41-704 | Holds; recompute after top-up |
| C6 | real outbreaks, origin hidden from the reports | Pre-registered, run once: SARS 2003 MF ranks China 1st (earliest detection 16th, distance center 8.5, LPSI 14); H1N1 2009 MF 3rd (earliest detection 1.5, LPSI 2). Robustness over 5 window/look-back settings: SARS MF 1-7 (best in every setting), H1N1 MF 1-3; mean reciprocal rank MF 0.50, earliest detection 0.37, LPSI 0.23 | Strong case study, but n = 2: worded as a case study, never as a statistical claim |
| C7 | real cascades (Twitter, 381 untouched) | MRR tie with undirected distance center (+0.016 [-0.015, +0.046]); better than directed distance center (+0.038 [+0.006, +0.070]); top-1 45% vs 39% (McNemar p = 0.003); top-10 61% vs 65% (p = 0.04); 90% sets 572 vs 464 | Partial support: MF is the most precise, not the most robust; report as such |
| C8 | SARS-CoV-2 variants | Pre-registered, single-country origins (6): MF 0.308 vs earliest detection 0.464. Pre-registered, origin named as a region (7): MF 0.694 vs effective distance 0.476, earliest detection 0.424 (MF names the right region first in 4/7) | Mixed: MF loses where the label is the first sequencing country and wins where the label is a region; both reported, the first with the label explanation |

## 3. Fix 1: message passing vs mean field

Finding: dynamic message passing (DMP, Lokhov et al. 2014), plugged into the same likelihood
(start and rate summed out), ties MF on 5/7 networks and is better on CollegeMsg (-0.058,
p = 0.01) and Office (-0.018, p = 0.02), at 6-11x the cost (2-295 s vs 0.4-26 s).

Resolution: the contribution is the likelihood (start and rate summed out, the observation
model, pruning, and what it enables), not the marginal engine. DMP is presented as the
alternative engine in an ablation: same framework, same accuracy on most networks, a small gain
on the sparsest networks, an order of magnitude slower. Checked against Lokhov et al. (PRE 2014, arXiv 1303.5315): static networks, a
snapshot of states at one time, the snapshot time t0 inferred, the infection and recovery rates
given. Ours: evolving contact networks, timed reports from sparse observers, start time and rate
summed out. The DMP engine inside our likelihood is therefore our construction, not their
method, and is labelled "our likelihood, DMP engine". The paper must not claim MF beats DMP.

## 4. Fix 2: the test-selection guarantee

Finding: Chen et al. (COLT 2015, Theorem 2) applies exactly to the joint-hypothesis criterion
under the mean-field model, but its constant γ = 7 / S_min is vacuous here: S_min ≈ 1e-12 on all 7
networks (some hypotheses give nearly identical report distributions). Even after
merging hypotheses closer than the 5% quantile of pairwise squared TV (1e-4 to 5e-3), γ ≥ 1400,
so the bound says nothing for 10 tests.

Resolution: the guarantee is removed as a contribution. The paper states the empirical result
(ties the joint criterion, beats seven strategies, ~2000-4000x faster than simulation) and cites
Chen et al. only to say the criterion is the one with known guarantees in the noiseless/separable
limit. A vacuous theorem would weaken, not strengthen, the paper.

## 5. Issues found and what was done

| Severity | Issue | Action |
|---|---|---|
| High | "Near-optimal" rested on one network (Hospital) | Exact likelihood now running on Office and High school |
| High | Twitter: MF used follow directions, distance and Jordan centers did not | Directed versions added as extra baselines (running); undirected kept |
| High | Twitter fitted model has zero spread between non-sources (rate 0): there MF is a "source broadcasts to followers, plus background" model, not multi-hop propagation | Must be stated plainly; it is what the training data chose |
| High | Variants: single-country origin labels are essentially "first to sequence it", which makes earliest detection correct by construction | Reported in the main text next to the regional set, where the label is an inferred region and MF wins; the contrast is the finding |
| Medium | CollegeMsg and Primary school had 168 and 162 outbreaks in the BP table | Top-up to 200 running |
| Medium | Test selection: CollegeMsg had 21 complete outbreaks, Email-Eu 101 | Top-up running |
| Medium | H1N1/SARS: one fixed window and look-back | Robustness runs (window 14/42 days, look-back 90/210 days) running; all reported |
| Medium | H1N1/SARS use the 2014 OpenFlights network for 2003 and 2009 | Stated; identical for every method |
| Low | Variants regional mode loads more cluster files, so sequencing effort per country-week is a max over more files | Totals are the same across files; noted |
| Low | Twitter popularity prior added nothing (beta 0.5 chosen on training; test -0.005) | Removed from the method in the paper; pre-registered column still reported in the supplement |

## 6. Keep / move / remove

Keep (main text):
- Model and estimator; Lemma 1 (upper bound), Lemma 2 (safe pruning); cost analysis.
- Exactness check against the exact likelihood (3 networks).
- MF vs BP on 6 real contact networks + the near-tree boundary (Growing PA), with the loopiness table.
- DMP as engine ablation.
- Active test selection (vs Sterchi 2023 and 7 strategies), with cost.
- Conformal source sets (coverage theorem is standard; the result is the set sizes).
- Real outbreaks, all pre-registered and scored once: SARS 2003 / H1N1 2009 (case study with robustness), SARS-CoV-2 clusters (13: MF wins where the origin is an inferred region, loses where it is the first sequencing country), Twitter (381 untouched cascades).
  Story: MF wins when the origin is genuinely hidden from the reports, and loses when the label is defined by who reported first.

Move to supplement:
- SARS-CoV-2 development variants (10) and per-variant tables.
- Extensions not used in the main results (recovery, immunity, weighted/directed links, detection model details).
- MF variants and robustness tables (floors, grids).

Remove:
- Test-selection theorem (vacuous constant).
- Twitter popularity prior, "Most followers" (kept only in the supplement table).
- Chain-rule clamping, exact rescoring of a shortlist, hybrid MF+BP (failed; not mentioned).
- Old Boltzmann/curvature material and learned (GNN) models.

## 7. Still open before writing

1. Exact likelihood on Office and High school (C1): the run hit the 6-hour limit; results are being downloaded.
2. Done: BP at 200 (C2); test selection at 120 and Sterchi at 16 (C4); directed Twitter baselines (C7); H1N1/SARS robustness (C6); regional variants (C8).
3. Every number in this file was re-checked against `outputs/tables.json` on 2026-11-23, except C1.
