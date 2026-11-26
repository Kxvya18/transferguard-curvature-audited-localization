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

## 9. Outcomes of X1-X6 (GitHub run 45619114205; `outputs/trans_runs_summary.txt`, `experiments/summarize_trans_runs.py`)

X1/X2 and X6 ran BP at damping 0.5 (launched before 8.1). The told-the-truth comparison is not
affected by damping much (BP converges when told: 76-99%), but end-to-end BP numbers in X1/X6
are for damping 0.5 and are quoted as such.

| Item | Result | Decision |
|---|---|---|
| X1 mechanism, 200 outbreaks x 4 networks | Told the true start and rate, BP and MF **tie** on all four (BP - MF: -0.009, +0.002, -0.004, +0.002; all intervals across 0). BP fails to converge at some grid point in 78-98% of outbreaks, but only 1-24% when told. MF's end-to-end advantage sits in the outbreaks where BP failed somewhere on the grid (High school +0.148, Office +0.068); where BP converged everywhere the gap is gone (n small) | **Mechanism established, with a correction:** the earlier small-sample finding "BP is better than MF when told" (3-30 outbreaks) does not hold at 200; the marginals are equally good. The whole end-to-end difference comes from BP's evidence at the wrong (start, rate). Verdict change caused by the larger sample |
| X2 hybrid (MF weights, BP marginals) | Ties MF on all four (+0.006, +0.018, -0.001, -0.004) and beats plain BP | Confirms the mechanism: give BP reliable weights and its loss disappears. MF stays the method (same accuracy, 30-1500x cheaper); hybrid reported as the strongest BP variant |
| X3 Dawkins et al. sets (true rate given to Dawkins, 90%) | Hospital: Dawkins 20.7 people, coverage 0.92, 133 s/outbreak vs ours 25, 0.92. Office (187 of 200, run hit the time limit): Dawkins 33, coverage 0.99 (over-covers), 993 s vs ours 18, 0.92 | Comparable, and Dawkins needs no calibration outbreaks. **By the rule, the conformal-set contribution is dropped**; Dawkins sets are cited, and MF is reported as a usable test statistic for them |
| X4 scalability | sx-mathoverflow, all 24,759 users: 29 s, 9.4 GB; 10,000 users 4.5 s, 1.5 GB; sx-askubuntu 20,000 users 17 s, 6.1 GB; 50,000 and 157,222 users fail (out of memory on a 16 GB runner). MF beats distance center at every size | Time scales well, memory does not: all candidates are propagated at once (candidates x n per step). Fix: propagate candidates in batches (memory ~ batch x n). Until then, the stated limit is ~25k nodes on 16 GB |
| X5 baselines, 180 outbreaks x 4 networks | MF beats SME (Antulov-Fantulin 2015) everywhere by +0.14 to +0.28. Against learned models: MF better on High school (+0.15/+0.16) and Office (+0.08/+0.10); tie with Temporal GCN on Hospital (-0.050 [-0.107, +0.008]) and with Arrival GNN on Email-Eu (-0.014 [-0.068, +0.040]) | Replaces the 2-seed claim "MF beats the learned models on 6 networks" with: better on 2/4, tie on 2/4, with no training (the GNNs train on 600 outbreaks of the same network and rate) |
| X6 SIR / uneven rates, 100 each x 4 networks | MF >= BP (damping 0.5) in 8/8, significant in 5; MF >> time-respecting TOA in 8/8 (+0.12 to +0.36); MF = message passing | Robustness to model error supported; BP column to be quoted as damping 0.5 |

### Claim after X1-X6 and 8.1 (stable from here unless new evidence)

Under unknown start time and rate, a mean-field likelihood localizes the source as well as or
better than belief propagation at its best damping, at 30-1500x lower cost and with nothing to
tune, because what decides end-to-end accuracy is the reliability of the evidence used to sum
out (start, rate), not the accuracy of the marginals: told the truth both are equally accurate,
BP's evidence breaks where BP does not converge (78-98% of outbreaks), and giving BP mean-field
weights removes its entire loss. Mean field is within noise of the exact likelihood on 3 of 4
networks (particle-filter reference).

Remaining before writing: (1) batched candidates for X4 (memory), (2) the theorem in theory.md
replacing section 5, (3) Biazzo et al. 2022 cited and, if feasible, compared, (4) X1/X6 BP at
damping 0.8 for consistency (optional; the told comparison does not depend on it).

## 10. X4 rerun after the memory fix (GitHub run 45744233371)

Fix: pruning by a backward latest-departure pass (observers x n instead of n x n) and candidates
propagated in batches (memory ~ batch x n). Results are bit-identical to the old code
(checked on Hospital, High school, Email-Eu, Office).

| Network | Users | Contacts | Outbreaks | Median time | Peak memory | MF - distance center (RR, 95% CI) |
|---|---|---|---|---|---|---|
| sx-mathoverflow top 1,000 | 1,000 | 128,820 | 20 | 0.4 s | 5 MB | +0.236 [+0.066, +0.422] |
| sx-mathoverflow top 5,000 | 5,000 | 228,753 | 20 | 4.1 s | 37 MB | +0.135 [+0.002, +0.295] |
| sx-mathoverflow, all | 24,759 | 294,293 | 20 | 30 s | 183 MB (was 9.4 GB) | +0.111 [-0.064, +0.294] |
| sx-askubuntu top 50,000 | 50,000 | 386,134 | 10 | 55 s | 369 MB (was out of memory) | -0.110 [-0.394, +0.151] |
| sx-askubuntu, all | 157,222 | 549,914 | 10 | 286 s | 695 MB (was out of memory) | -0.078 [-0.238, +0.071] |
| sx-superuser, all | 192,409 | 857,643 | 10 | 571 s | 856 MB | **-0.322 [-0.592, -0.070]** |
| wiki-talk-temporal, all | ~1.1M | - | 0 | - | - | runner shut down after 41 min (most likely out of memory loading or propagating; not diagnosed) |

Decision:
- **Cost: closed.** Memory now grows linearly in n; mean field runs on 1.9 x 10^5 nodes in
  under 1 GB and about 10 minutes per outbreak on one core. The stated limit becomes
  ~2 x 10^5 nodes; 10^6 (wiki-talk) is not shown and is not claimed.
- **Accuracy at this scale: a new negative result, recorded as is.** With 1% observers on
  10^5-node Q&A networks, outbreaks have only 3-4 infected observers. Mean field beats distance
  center on sx-mathoverflow (significant at 1k and 5k), ties on sx-askubuntu, and is
  significantly worse on sx-superuser (n = 10, so the size of the gap is uncertain but its
  sign is not). The earlier sentence "MF beats distance center at every size" (section 9,
  20k users) does not hold beyond ~25k nodes and is withdrawn.
- For the paper: the scalability section reports cost (time, memory) at every size and accuracy
  only with this caveat. Before writing, one check is worth doing: 50 outbreaks on
  sx-superuser and sx-askubuntu, to tell whether the loss is a sparse-observation regime
  (3 infected observers) or something specific to these networks.

### 10.1 Check of the large-network accuracy drop (rules fixed before running)

Design: sx-superuser (all) and sx-askubuntu (all), observers 1% and 10%, 50 outbreaks per cell
(5 shards x 10, seeds 1300-1304 for 1%, 1400-1404 for 10%). Each row also holds mean field told
the true start and rate ("MF, told"). Paired bootstrap 95% intervals on reciprocal rank.

- R1 (observation regime): if at 10% observers MF - distance center has an interval that is not
  below 0 on both networks, the 1% loss is a sparse-observation regime; the paper states the
  regime (few infected observers) as a limit. If MF is still significantly worse at 10%, the
  loss is network-specific and must be diagnosed before the scalability section is written.
- R2 (where the loss comes from): if MF, told - MF has an interval above 0 and MF, told is not
  below distance center, the loss comes from summing out (start, rate) on these daily-binned
  networks; otherwise it is the likelihood itself (mean-field independence on these graphs).
- Both rules are also reported stratified by number of infected observers (3, 4-5, 6+).

### 10.2 Result of the check (GitHub run 45757095913, 50 outbreaks per cell)

| Network, observers | Infected observers (median, range) | MRR MF / MF told / distance | MF - distance | MF told - MF |
|---|---|---|---|---|
| sx-superuser, 1% | 3 (3-5) | 0.203 / 0.311 / 0.137 | +0.067 [-0.025, +0.160] | +0.108 [+0.032, +0.193] |
| sx-superuser, 10% | 4 (3-13) | 0.367 / 0.524 / 0.238 | +0.128 [-0.007, +0.261] | +0.157 [+0.063, +0.253] |
| sx-askubuntu, 1% | 3 (3-5) | 0.287 / 0.443 / 0.157 | **+0.130 [+0.004, +0.254]** | +0.155 [+0.063, +0.254] |
| sx-askubuntu, 10% | 4 (3-11) | 0.392 / 0.539 / 0.187 | **+0.205 [+0.111, +0.302]** | +0.147 [+0.065, +0.231] |

Decision (rules of 10.1):
- **The loss in 10 does not replicate.** On 50 fresh outbreaks at the same setting (1%, all
  users), MF is ahead of distance center on both networks (significant on sx-askubuntu,
  +0.067 with an interval across 0 on sx-superuser). The 10-outbreak result (-0.322 on
  sx-superuser) was a small-sample result: reciprocal rank is heavy-tailed, and 10 outbreaks
  were too few. **Correction to 10:** the sentence "the size of the gap is uncertain but its
  sign is not" was wrong and is withdrawn; so is "a new negative result". Verdict change caused
  by the larger sample, as in X1.
- R1: at 10% observers MF - distance is not below 0 on either network (significant on
  sx-askubuntu). There is no network-specific loss to diagnose.
- R2: MF told beats MF by +0.11 to +0.16 in all four cells (all intervals above 0), and MF told
  beats distance center everywhere. So on these daily-binned networks, summing out the start and
  rate costs 0.11-0.16 MRR. This is the first term of Theorem 1 (theory.md §5): the cost of not
  knowing (start, rate) is in the weights. It is reported as a limit, not hidden.
- For the paper: X4 reports cost at every size (§10 table) and accuracy from these 50-outbreak
  cells (pooled with nothing; the 10-outbreak runs are kept in outputs/scalability for the
  record and described as too small). Statement: up to 1.9 x 10^5 nodes, MF ranks the source
  ahead of or level with distance center, with significant gains on sx-askubuntu; knowing the
  start and rate would add 0.11-0.16.

## 11. Biazzo et al. 2022 (autoregressive neural network), rules fixed before running

Paper: Biazzo, Braunstein, Dall'Asta, Mazza, "A Bayesian generative neural network framework for
epidemic inference problems", Sci. Rep. 12:19673 (2022). Code: github.com/ocadni/annfore
(commit ee39c69) and github.com/ocadni/annfore-results. Their patient-zero setting: start time
and rate known, full final snapshot, N ~ 100, T = 15, 10,000 annealing steps x 10,000 samples,
on a GPU (their batch scripts allow 25 h per 5 instances). Reported on their hospital network:
patient zero found in 74% of instances vs 54% for SIB.

Our runs (localization/ann_biazzo.py, experiments/compare_ann.py): their library and default
settings, told the true start and rate (it needs them), our sparse timed reports (10% observers),
on GitHub Actions CPUs.

- Stage A, conversion check: 20 tiny temporal graphs (5-7 people, 4-5 windows, 5-7 contacts),
  exact source posterior by enumerating every transmission coin. **Pass if mean TV(exact, ANN)
  <= 0.05 at 1000 x 1000** (it is meant to be near-exact on tiny graphs). If it fails, the
  conversion or the budget is wrong, and no comparison is run until that is fixed. Mean field
  (told) TV is reported alongside for reference.
- Stage B, budget: Hospital, 8 outbreaks (seed 101) at 100, 300 and 1000 annealing steps x 1000
  samples, for time and accuracy. Main budget = the largest at which 40 outbreaks per network fit
  in the Actions time limit; the MRR at a third of that budget is reported as a saturation check.
- Stage C, comparison: Hospital and Office, 40 outbreaks each, seed 101 (the same outbreaks as the
  belief-propagation tables). Paired bootstrap 95% intervals on reciprocal rank:
  ANN told vs MF told (same information) and ANN told vs MF (not told).
- Decision: if ANN told beats MF told (interval above 0), the paper says so: the network is more
  accurate given the start and rate, at its measured cost, and needs both (or its own EM for the
  rate, start still given). If they tie, MF matches it with no training at a fraction of the
  cost. Biazzo et al. is cited in either case.

### 11.1 Stages A and B (GitHub runs 45777152246, 45778422738)

- **Stage A passes.** 20 tiny graphs: mean TV(exact, ANN) = 0.006 (max 0.056), same top person as
  the exact posterior in 20/20; mean field told 0.015 (max 0.197). The conversion to annfore
  (time convention, observations, no recovery) is correct.
- **Stage B, Hospital, 8 outbreaks (1000 samples per step):**

| Annealing steps | MRR ANN told | MRR MF told | MRR MF | Median s per outbreak (ANN) |
|---|---|---|---|---|
| 100 | 0.266 | 0.461 | 0.323 | 152 |
| 300 | 0.266 | 0.461 | 0.323 | 406 |
| 1000 | 0.385 | 0.461 | 0.323 | 1210 |

  Not saturated (0.266 -> 0.385 from 300 to 1000 steps). By the rule, no comparison until the
  budget saturates: next, 3000 x 1000, 1000 x 3000 and 1000 x 10000 (their sample count) on the
  same 8 outbreaks. In most outbreaks where ANN misses, it ranks the source 40-70 of 75 (no mass
  on it), i.e. the trained network concentrates on other explanations; whether more training
  fixes this is what the next budgets show.
