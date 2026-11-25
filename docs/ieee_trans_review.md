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
