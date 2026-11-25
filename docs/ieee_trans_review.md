# Referee report, as for IEEE Transactions on Network Science and Engineering (2026-11-25)

Scope: everything finalised so far (results in `outputs/tables.json`, `docs/internal_review.md`),
excluding the four checks still running (BP damping, trivial Twitter baselines, selection
temperature, observer fraction). Written as a strict Transactions referee would, after reading
the closest prior work.

## 1. Summary of the submission (as it would be submitted)

Mean-field (MF) likelihood for single-source localization on temporal contact networks from
timed reports of a sparse set of observers, with the start time and spreading rate summed out
over a grid. Claims: near-exact; better than belief propagation (BP) on six real contact
networks at 30-1500x lower cost; makes active test selection and calibrated source sets
practical; finds the origin of real outbreaks (SARS 2003, H1N1 2009, Omicron clusters, Twitter).

## 2. Novelty: moderate, and at risk on three fronts

| Component | Closest prior work | Verdict |
|---|---|---|
| MF forward recursion on temporal contacts with test observations | Baker et al., PNAS 2021 (simple mean field and BP for risk inference from contact-tracing data); Lokhov et al., PRE 2014 (DMP for the origin, static network) | Not new. Must be cited as the engine; our use for source likelihood from timed reports is an adaptation |
| Summing out the unknown start time | Antulov-Fantulin et al., PRL 2015 (marginalise over start times, temporal networks); Sterchi et al., Sci Rep 2023 (source and start jointly unknown) | Not new as an idea; new only in being exact under MF and cheap |
| Active test selection | Sterchi et al. 2023 (same joint criterion, by simulation) | Our criterion ties theirs; the contribution is cost (0.9 s vs 611 s per pick), not a new criterion |
| Source sets with guaranteed coverage | Dawkins, Li and Xu, ICML 2021 (confidence sets for the source, finite-sample, any network, no calibration outbreaks needed) | **Serious.** Our split-conformal sets need exchangeable calibration outbreaks; theirs do not. Not compared, not cited. As it stands a referee calls C5 non-novel |
| Lemma 1 (MF upper-bounds SI marginals), Lemma 2 (safe pruning) | Classical MF/SI results; pruning by time-respecting reachability is standard | Correct but shallow; cannot carry the theory |

What is genuinely new: (i) the empirical finding that a cheap MF likelihood, used end to end with
unknown start and rate, beats BP on every real contact network tested; (ii) the explanation of
why (Section 3); (iii) the pre-registered real-outbreak evaluation with independently known
origins. A TNSE referee would rate novelty "incremental unless (ii) is made rigorous".

## 3. Technical depth: the mechanism claim is wrong as written

`docs/theory.md` §5 says MF beats BP because real networks are loopy and BP is exact only on
trees (7 networks, correlation with clustering). The project's own controlled data contradict
that:

- Told the true start and rate, BP is **better** than MF on High school (0.485 vs 0.378),
  Primary school (0.714 vs 0.281) and dense synthetic graphs (1.0 vs 0.83-0.90); MF is better
  only on Hospital (0.469 vs 0.407). (`outputs/mechanism/diagnostics*.jsonl`, 3-30 outbreaks each.)
- BP fails to converge on 83-100% of the (start, rate) grid points on the real networks, but
  on only 0-33% when told the truth.
- In the controlled synthetic grid, BP wins at degree 4 (more cycles) and loses at degree 16-32
  (fewer cycles): loop density does not order the gap.

So the supported mechanism is: **BP's Bethe evidence is unreliable at the wrong (start, rate),
where it does not converge, and that corrupts the sum over start and rate; MF's evidence is a
finite forward recursion and stays usable.** MF wins as an end-to-end estimator under parameter
uncertainty, not because its marginals are better. This is a sharper and more defensible
contribution, but it is currently based on 3-30 outbreaks per network and is not in the text.

Consequences:
- Claim C2 must be rephrased ("with unknown start and rate").
- A referee will ask for the obvious stronger BP baseline: BP whose (start, rate) weights come
  from a convergent evidence (for example MF weights, or dropping non-converged grid points).
  The repository's "hybrid" (MF weights, BP marginals) is exactly this and must be reported, not
  dropped; if it beats MF, the paper's hero changes.
- §5 of theory.md must be replaced.

## 4. Experimental evaluation

Strengths: identical outbreaks across methods; 200 outbreaks per network; paired bootstrap
intervals and Wilcoxon tests; pre-registered real test sets scored once; losses reported.

Weaknesses a referee will raise:

E1. Scale. Largest contact network 1,899 nodes; Twitter subgraphs at most 4,000 people. Memory
is O(K n) per grid point. TNSE expects 10^4-10^5 nodes or an honest bound. No scalability
experiment exists.

E2. Baseline coverage. Learned models (GCNSI, temporal GCN, Arrival GNN) were beaten on 6
networks (`outputs/mean_field_vs_learned`, 2 seeds, MF 0.46-0.53 vs best GNN 0.31-0.40), but
they are 2019-2022 models, are not on the 200-outbreak protocol, and the soft-margin estimator
of Antulov-Fantulin et al. (the standard temporal-network baseline) is run only on Growing PA.

E3. Confidence sets: no comparison to Dawkins et al. 2021 (see §2).

E4. Small effects. Office +0.043, Hospital +0.057, CollegeMsg +0.066 (Holm p = 0.048) in MRR;
top-1 differences are a few points. A referee asks for effect sizes in people (rank, set size),
not only MRR.

E5. Real outbreaks. Strongest evidence is n = 2 (SARS, H1N1) with a 2014 flight network for 2003
and 2009. Twitter is a tie in MRR. Single-country variants are a loss. Region-level variants win
(0.694 vs 0.424, chance 0.225) on 7 clusters. Together: suggestive, not conclusive; must be
presented as such.

E6. Single synthetic family. Growing PA is the only synthetic network; no SIR, no
heterogeneous rates, no reporting delays in the main simulations (the code has them).

## 5. Presentation risks (from the OJCS round)

- One naming table for methods (OJCS R2.7 was about this).
- All settings listed: grid sizes, floor 1e-4, temperature 2.0, 100-hypothesis cut, BP damping
  and priors (OJCS R2.2).
- Counts and seeds in one table (OJCS R2.4).
- Derived differences from unrounded values (OJCS R2.6): `make_tables.py` does this.

## 6. Recommendation

As the results stand: **major revision**. Publishable at Transactions level if (a) the mechanism
in §3 is established on enough outbreaks and becomes the central technical contribution,
(b) the stronger BP variant is reported, (c) the confidence-set comparison to Dawkins et al. is
added or the claim dropped, and (d) a scalability experiment is added. Without (a), a referee
will judge the paper an incremental, well-executed benchmark.

## 7. Required new executions

| # | Run | Why |
|---|---|---|
| X1 | MF vs BP told the true start and rate, with BP convergence rates, 200 outbreaks on Hospital, High school, Primary school, Office | Establish the mechanism (§3) at the same scale as the main table |
| X2 | Hybrid (MF weights, BP marginals) on the same 200 outbreaks | The stronger BP baseline a referee will demand |
| X3 | Dawkins et al. 2021 confidence sets vs our conformal sets (size, coverage), Hospital and Office | Novelty of C5 |
| X4 | Scalability: MF time and memory on a 10^4-10^5-node temporal network (e.g. SNAP sx-mathoverflow / wiki-talk temporal), with and without pruning | E1 |
| X5 | Soft-margin estimator (Antulov-Fantulin) and the learned models on the 200-outbreak protocol, at least 3 networks | E2 |
| X6 | SIR and heterogeneous-rate outbreaks on 2 networks (code exists) | E6 |

## 8. Outcomes of the four checks (2026-11-25), decided by rules fixed beforehand

| Check | Result | Rule fixed in advance | Decision |
|---|---|---|---|
| M1 BP damping (20 outbreaks each) | Damping 0.8 helps BP: Hospital 0.398 (0.5: 0.316), Office 0.574 (0.5: 0.534); Primary school stays poor at every damping (<= 0.155); BP fails to converge on 20-73% of runs at every damping | "If BP improves materially at another damping, rerun the BP table at that damping" | BP table rerun at damping 0.8 on all 6 real networks, same outbreaks (running). The main BP comparison uses whichever damping is better for BP |
| M3 Twitter trivial baselines (681 cascades) | "Infected followers" (count of a candidate's followers who retweeted) 0.531 / 0.526 vs mean field 0.518 / 0.506; difference -0.013 [-0.039, +0.012] and -0.020 [-0.044, +0.004] | "If the trivial follower baselines match mean field, the Twitter section is cut" | **Twitter removed as evidence for the claim.** The fitted model has no spread between ordinary users, so it reduces to follower counting. Kept, if at all, as one sentence in limitations |
| M4 Selection temperature 1 / 2 / 4 | Our criterion 0.507-0.551 (Hospital), 0.629-0.660 (Email-Eu); joint criterion 0.537-0.674 | none | Robust within +/-0.04; report T = 2 with this range |
| M5 Observer fraction 5% / 20% (40 each) | MF >= BP in 5/6 settings (Office 5%: -0.008, tie); significant on High school only | none | Supports C2 directionally; reported as robustness |

### 8.1 BP table rerun at damping 0.8 (200 outbreaks per network, same outbreaks, all methods recomputed; GitHub run 45626125350)

Damping 0.8 is better for BP on every network (e.g. Email-Eu 0.320 -> 0.392, High school 0.315 -> 0.401),
so by the rule above it is the main BP baseline. Mean field minus BP (MRR, bootstrap 95%):

| Network | MF | BP (0.8) | MP | MF - BP (0.8) |
|---|---|---|---|---|
| Hospital | 0.298 | 0.266 | 0.297 | +0.032 [-0.004, +0.068] |
| Primary school | 0.366 | 0.196 | 0.371 | **+0.170 [+0.117, +0.224]** |
| Office | 0.459 | 0.448 | 0.477 | +0.011 [-0.024, +0.045] |
| High school | 0.434 | 0.401 | 0.442 | +0.033 [-0.010, +0.077] |
| Email-Eu | 0.443 | 0.392 | 0.440 | **+0.051 [+0.004, +0.098]** |
| CollegeMsg | 0.395 | 0.405 | 0.448 | -0.010 [-0.057, +0.039] |

**Verdict change, caused by this new evidence:** claim C2 "MF beats BP on all six real networks" no
longer holds against BP at its best damping. What holds: MF is never significantly worse than
BP, is clearly better on 2/6 (Primary school large, Email-Eu), is directionally better on 3 more,
needs no damping tuning, and is 30-1500x cheaper. BP's accuracy moves by up to 0.09 MRR with
damping; MF has no such setting. Message passing (our likelihood with a DMP engine) is the best
of the three on CollegeMsg and Office. The headline must become "matches or beats tuned BP at a
fraction of the cost, with no tuning", and the mechanism section must say that damping that makes
BP converge more often closes most of the gap, which is consistent with the non-convergence mechanism.
