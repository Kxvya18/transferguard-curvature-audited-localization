# Reviewer-style review (2026-11-25)

Written the way the OJCS reviewers wrote theirs: every equation, number and comparison is
checked for consistency, reproducibility, degeneracy and fairness. Reviewer 2 of the OJCS
round found (1) an inconsistent model definition, (2) components left as placeholders,
(3) training/evaluation mismatch, (4) seed and figure inconsistencies, (5) an evaluation that
was mathematically degenerate, (6) rounding errors in derived differences, (7) undefined
methods, (8) trivial transfer graphs. Each point below is tested against those failure types.

Status: DONE (verified), RUNNING (GitHub Actions), WRITE (must be handled in the manuscript).

## Major comments

M1. Is belief propagation set up fairly? (type 3, fairness)
BP (SIB) runs with damping 0.5, at most 1000 iterations, tolerance 1e-6, seed prior 1/n and
susceptible prior 0.5. A reviewer will ask whether BP loses because of these settings and how
often it fails to converge. RUNNING: damping 0.2 / 0.5 / 0.8 with convergence rates on the same
outbreaks (Hospital, Office, Primary school, where BP is weakest). WRITE: list every BP
setting, and that BP gets the same (start, rate) grid as mean field.

M2. Is the near-optimality claim supported? (type 5, degeneracy)
Told the true start and rate, MF equals the Monte Carlo exact likelihood on Hospital
(+0.005 [-0.020, +0.030]) and Office (-0.001 [-0.038, +0.035]). On High school the estimate is
zero for every candidate in 28% of outbreaks, so it is not a valid reference there. With start
and rate unknown, MF scores above the "exact" estimate on all three networks, which shows that
estimate is too noisy to be a ceiling. DONE. WRITE: claim only "matches the exact likelihood
where it can be estimated (2 networks)".

M3. Twitter: is the method degenerate? (type 5)
The model fitted on the training cascades has zero spreading rate between non-sources: the
source broadcasts to its followers and everyone else retweets only through the background
rate. Mean field may then reduce to "who has the most retweeting followers". RUNNING: two
trivial baselines on the same 681 cascades ("infected followers", "follower retweet share").
If mean field does not beat them, the Twitter section is removed or reframed.

M4. Are the hyperparameters of test selection justified? (type 2)
Posterior temperature 2.0 and the 100-hypothesis cut were fixed early and never varied.
RUNNING: temperature 1 and 4 for both information criteria (Hospital, Email-Eu). WRITE: the
cut is reported as a cost/accuracy setting.

M5. Do results depend on how many people are observed? (type 4)
Every main comparison uses 10% observers. RUNNING: 5% and 20% for MF vs BP vs message passing
(Hospital, Office, High school, 40 outbreaks each).

M6. Region-level variants: is the score inflated by region size? (type 5)
A region's score is its best-ranked country, so a large region scores well by chance. Chance
level, computed exactly: MRR 0.225 (per variant 0.085 to 0.427; "Asia" has 52 countries and
chance 0.427). Mean field 0.694, earliest detection 0.424. DONE. WRITE: report chance next to
every method; the convincing cases are the small regions (Southern Africa 1st, South America
1st, Southern Asia 3rd), not "Asia".

M7. Multiple comparisons. (type 4)
Six networks are tested for MF vs BP. Holm-adjusted p: 6.6e-18 to 4.8e-2; all six stay below
0.05, but CollegeMsg (0.048) is borderline and was not significant at 168 outbreaks (p = 0.25).
DONE. WRITE: call CollegeMsg "weak", not "significant".

M8. Single-country variants: the method loses. (honesty)
Pre-registered, MF 0.308 vs earliest detection 0.464. Explained by labels that equal the first
sequencing country. DONE. WRITE: in the main text, beside the region-level set, with the
reason stated before the result, not after.

M9. SARS 2003 / H1N1 2009. (type 8, data construction)
Two outbreaks; untrained model; the flight network is from about 2014. Robustness over five
window/look-back settings is done (SARS: MF 1st to 7th, best in every setting; H1N1: 1st to
3rd). WRITE: data construction in full (WHO table to first-report day, name mapping, every
country an observer), and "case study", never a statistical claim.

## Minor comments

m1. Every derived difference is computed from unrounded values by `experiments/make_tables.py`
(paired bootstrap, seed 0). WRITE: quote numbers only from `outputs/tables.json`.

m2. Seed and count consistency: BP comparison 200 outbreaks per network (Primary school 202,
because the top-up seeds overlapped by 2); test selection 120 per network (3 seeds x 40); Sterchi
16; Twitter 300 development + 381 test. WRITE: one table of counts and seeds; use 200 for
Primary school by dropping the last 2 (decided now, before writing).

m3. Method names: "Information gain (ours)" is the source-only criterion; "Joint disagreement
(AKLD, Sterchi 2023)" is Sterchi's criterion computed with mean field; "AKLD by simulation" is
Sterchi's own pipeline. WRITE: one naming table used in text, tables and figures alike.

m4. Message passing is our likelihood with a DMP engine (Lokhov et al. used static networks,
one snapshot, known rates). WRITE: label it so; never call it "Lokhov et al." as a baseline.

m5. Cost: BP/MF time ratio is 29x (CollegeMsg) to 1516x (Primary school), not "40-1400x".
WRITE: "30-1500x", single core, BP in C++ (SIB), MF in NumPy.

m6. Sterchi comparison: 16 outbreaks; ours 0.579 vs simulation 0.566 MRR is a tie, not a
win; the result is the cost (0.9 s vs 611 s per pick). WRITE accordingly.

m7. Test selection on Primary school: "test the top suspect" is better (-0.043
[-0.073, -0.014]). WRITE: reported, not hidden.

m8. Variants development run (10) informed the model; only the held-out sets count. WRITE.

## Is all required execution done?

Not yet. Four checks are running (M1, M3, M4, M5). After they finish:
- If BP improves materially at another damping, the BP table is rerun at that damping.
- If the trivial follower baselines match mean field on Twitter, the Twitter section is cut.
Everything else is complete; the remaining items are writing items.
