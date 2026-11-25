# New method components: prior art and pilot rules (fixed 2026-11-25, before any pilot result)

Two candidate additions to the method. Each is piloted on Hospital and High school; the rules
below decide whether it enters the paper. They were written before the pilots were run.

## Prior art found

| Work | What it does | Overlap with us |
|---|---|---|
| Biazzo, Braunstein, Dall'Asta, Mazza, Sci Rep 12:19673 (2022), arXiv 2111.03383 | Autoregressive neural network fitted by gradient to the posterior of SIR states on temporal contact networks (incl. a hospital network, ~330 people); patient zero from snapshot states; global rates (lambda, mu) learned by an EM-like gradient loop; 1 h to 2 days per instance on a GPU; reports SIB (BP) failing on dense/structured networks; patient zero 74% vs SIB 54% (hospital) | **Must be cited, and it is a missing baseline** (newer than our learned models, beats SIB). Their setting is snapshots with known rates in the patient-zero experiments; ours is timed sparse observers with start and rate unknown. Their gradient parameter learning overlaps #2 for global rates |
| Braunstein and Ingrosso, Sci Rep 2016 ("Inference of causality in epidemics on temporal contact networks") | BP on temporal contacts for source, infection times and epidemic parameters (parameters from the BP free energy) | Parameter learning from BP evidence: exactly what our mechanism says is unreliable when BP does not converge. #2 is the reliable-evidence counterpart |
| Kazemitabar and Amini (arXiv 1906.03052; UCLA PhD thesis 2020, ch. 2) | Importance sampling of the likelihood of a full infected snapshot on a static network (continuous heterogeneous SI), proposal = dynamics restricted to the infected subgraph; sub-Gaussian error bound; their "mean-field" is a product-form fit by a linear system, unrelated to ours | IS for the source likelihood is not new. Different data (full snapshot vs sparse timed observers), static network, and a proposal not guided by a forward approximation |
| Ju, Heng and Jacob, arXiv 2101.12156 (2021) | SMC for agent-based transmission models with proposals that look ahead to future observations | Guided proposals for network epidemics exist (for parameter inference, not the source) |
| Chopra et al., GradABM, AAMAS 2023 | Differentiable agent-based epidemic simulator for calibration | Differentiable epidemic models exist; not for source localization or per-person rates on contact networks |
| Antulov-Fantulin et al., PRL 2015; Dawkins et al., ICML 2021 | Monte Carlo likelihoods (soft margin; importance sampling for test statistics) | Already cited |

What would still be new:
- **#2:** learning per-person (or per-group) infectiousness by differentiating the mean-field likelihood over past outbreaks whose sources are unknown, in seconds on a CPU; then localizing with the learned rates. The global-rate part is not new (Biazzo 2022); the per-person part and the reliable-evidence argument against BP-based parameter learning are.
- **#1:** importance sampling of the exact likelihood of sparse timed reports on a temporal network, with a proposal conditioned on the reports and twisted by a mean-field look-ahead. Incremental over Kazemitabar-Amini and Ju-Heng-Jacob. Its main value is as an exact reference that works where plain Monte Carlo fails (High school), and as a per-outbreak check of mean field.

## Pilot #2: learned per-person infectiousness

Data: outbreaks with uneven infectiousness, each person's factor drawn once per network
(log-normal, sigma 1, mean 1) and kept across outbreaks (a person trait). 10% observers,
network rate and horizon from the main protocol. 150 training outbreaks (sources never
used), 60 test outbreaks, seed 1301.

Methods on the same test outbreaks:
- MF grid (the current method, rate grid, start summed out);
- MF learned: per-person log-factors (L2 penalty) and a global rate, fitted by gradient ascent of
  the marginal log-likelihood of the training outbreaks (source and start summed out);
- MF oracle: told the true factors (ceiling on what learning can give).

Rule: #2 enters the paper only if MF learned beats MF grid by >= 0.03 MRR with a paired
bootstrap 95% interval excluding 0 on BOTH networks. If the oracle itself does not beat
MF grid by 0.03, heterogeneity does not matter here and #2 is dropped regardless.

## Pilot #1: importance-sampled exact likelihood

Data: the outbreaks of `limits_exact_told.py` (seed 901), told the true start and rate, 40 per network.

Estimator: forward SI simulation in which each observer is forced to match its report (not
infected before its report time, infected at it, healthy observers never infected); the
weight is the product of the probabilities of the forced events, so the estimate of
P(reports | s) is unbiased. Variant "twisted": transmissions to non-observers are tilted
toward a mean-field look-ahead and the weight corrected accordingly.

Rules:
- Feasibility: estimate nonzero for the true source in >= 99% of outbreaks with N <= 1000
  particles (plain Monte Carlo failed in 28% on High school).
- #1 is claimed as a method only if its rank (told the truth) beats MF told the truth by
  >= 0.03 MRR with the interval excluding 0 on at least one network AND it can run with start and rate unknown at
  <= 100x MF cost. Otherwise it is used only as the exact reference (supplement), which fixes
  M2 on High school.
