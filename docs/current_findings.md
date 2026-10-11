# Current Evidence and Claim Boundaries

Results below distinguish two observation contracts. The v3 controlled protocol
uses a sparse simulated alert channel. The v4 protocol replaces that channel
with raw activity change so synthetic and IoT-23 features have the same meaning.
The two protocols answer different questions and must not be pooled.

## Supported findings

1. The v3 blind-origin synthetic protocol is nontrivial. The source has a direct alert in only 14.6% of the 96 baseline-audit incidents. Uniform expected MRR is 0.027, degree MRR is 0.029, Boltzmann-only MRR is 0.026, and outbound-activity MRR is 0.080.
2. Under v3, mixed-domain temporal learning is strong. On 480 held-out stress scenarios spanning n in {100, 220, 400}, m in {1, 2, 3, 4, 5}, and beta in {0.5, 1.0, 1.5, 2.0}, the safe hybrid reaches MRR 0.946 and hit@1 0.919.
3. Under v3, safe residual prior fusion provides a small statistically supported ranking gain over the architecture-matched temporal-only model: paired mean delta MRR +0.02345, bootstrap 95% CI [0.01038, 0.03703], paired sign-flip p=0.00051.
4. The gain is a robustness result, not a universal per-incident improvement. The hybrid wins 19 of 60 regime cells, ties 34, and loses 7. At the incident level it wins 8.1%, ties 88.3%, and loses 3.5%.
5. The safety gate learns to reduce prior strength as structural concentration rises: mean strength decreases from 0.0338 at beta=0.5 to 0.0114 at beta=2.0. The earlier opaque fusion lost 0.059 MRR at beta=2.0; v3 instead gains 0.067 MRR there.
6. Under the cross-dataset-aligned v4 activity protocol, temporal learning is stronger still: temporal-only MRR is 0.9854 and hit@1 is 0.9708 over the same 480-cell stress design.
7. IoT-23 ingestion is now correct and leakage-controlled for the implemented task. Five scenarios yield 1,194,491 parsed events in total, and attack labels are used only to define the evaluation origin and onset.
8. The v4 model ranks the labeled malicious-flow source first in all five IoT-23 scenarios. This is a successful feature-contract transfer sanity check, not evidence of superiority, because simple degree and outbound-activity rankings also achieve rank one.
9. Uncertainty scores detect the deliberately severe synthetic deployment shift. For safe hybrid, AUROC is 0.964 using normalized predictive entropy and 0.947 using MC-dropout mutual information; temporal-only reaches 0.924 and 0.936, respectively.
10. Selective prediction is effective in distribution. The raw temporal-only model has 97.2% full-coverage accuracy and 100% accuracy at 90% coverage on 144 independent in-distribution graphs.
11. At a strict two-window decision horizon with future aggregate activity removed, temporal-only reaches origin MRR 0.794 and hit@1 0.708 over 120 fresh scenarios spanning all 60 n/m/beta cells.
12. Posterior-weighted expected graph distance recovers useful early order: temporal-only pairwise order accuracy is 0.702 versus 0.667 for activity change and 0.643 for degree; the oracle-origin distance ceiling is 0.752.
13. Source localization and containment are empirically different objectives. At k=3, source-only temporal quarantine prevents 22.4% of future compromises, whereas the label-free probability-times-impact reranker prevents 57.3%.
14. The decision-aware reranker beats source-only and random quarantine at k in {1,3,5} with paired sign-flip p<1e-5. At k=1 it also exceeds degree by +0.0759 prevented fraction, bootstrap 95% CI [0.0146, 0.1392], p=0.0173.
15. At a retrospective 5% ID false-rejection operating point, normalized entropy rejects 75.0% of OOD hybrid scenarios while retaining 25.0%; accepted OOD accuracy rises from 33.8% to 67.5%. Temporal-only MC mutual information rejects 68.1% and raises accepted accuracy from 45.6% to 64.7%.
16. Independent training-seed evaluation is stable. On one fixed 120-graph test, hybrid mean MRR is 0.9863 (three-seed normal-approximation 95% CI [0.9793, 0.9934]) and temporal-only is 0.9870 ([0.9774, 0.9966]); both have mean hit@1 0.9778.
17. Temporal modeling is necessary on the aligned held-out comparison: temporal-only MRR is 0.9861 versus 0.7942 for static GCN and 0.9333 for the windowed TGAT-style attention baseline on the identical fixed 120-graph test with the same optimizer seed.
18. Frozen synthetic-to-real source ranking reaches MRR 1.0 on five IoT-23 scenarios for both models. On two TON_IoT network captures, hybrid reaches MRR 1.0 and temporal-only 0.583, but activity-change and outbound-activity baselines also reach 1.0. Using the pre-specified synthetic-ID 5% false-rejection entropy threshold, both models flag 40% of IoT-23 captures and 100% of TON_IoT captures as shifted.
19. The CPU model contains 147,104 parameters. Median single-graph inference latency rises from roughly 6 ms at n=100 to 30-31 ms at n=400; measured process peak RSS is 218-258 MB.
20. In the anytime cold-start experiment, the learned Fisher-bounded gate reduces mean prior strength from 0.144 under low evidence to 0.055 under full evidence. The broad low-evidence hybrid gain is not significant: delta MRR +0.00668, graph-clustered 95% CI [-0.00537, +0.01885].
21. A validation-fixed selective rule is independently confirmed on a second 120-graph grid. Applying the hybrid only above the validation Q75 Fisher-reliability threshold selects 31.7% of graphs and improves low-evidence MRR from 0.43218 to 0.44067 (delta +0.00849, 95% CI [+0.00242, +0.01548]) and Hit@1 from 0.35903 to 0.37431 (delta +0.01528, 95% CI [+0.00556, +0.02569]).
22. In the confirmatory high-reliability subgroup, the inherited prior adds +0.02682 low-evidence MRR, 95% CI [+0.00780, +0.04671]. The high-minus-low reliability interaction is +0.03307, 95% CI [+0.00629, +0.06090].

## Falsified or unsupported claims

1. A raw Boltzmann prior does not outperform simple topology or temporal learning under severe attack-entry misspecification.
2. Fisher information alone is not an applicability test. High Fisher information can coexist with a precise but operationally irrelevant structural prior.
3. IoT-23 does not establish multi-hop provenance reconstruction. Its labels identify malicious flows, not true compromise chains.
4. IoT-23 is not discriminating for source localization in the selected scenarios: degree, outbound activity, earliest seen, and Boltzmann rank all place the infected scanning source first.
5. The present results do not establish nontrivial cross-dataset provenance transfer, deployment-calibrated abstention, or real compromise-order recovery. Adaptive intervention has been tested synthetically, but its conclusions reverse between fixed and rerouting attackers.
6. Under v4, the Boltzmann fusion does not improve the architecture-matched temporal model. Hybrid MRR is 0.9768 versus 0.9854; paired delta MRR is -0.00861 with bootstrap 95% CI [-0.01729, -0.00056]. The two-sided sign-flip p-value is 0.0559, so the directional bootstrap result must not be described as a universal significant degradation.
7. Prior reliability derived from beta estimation does not predict operational utility in v4. The highest reliability quartile has mean delta MRR -0.015, while the lowest quartile has +0.0083. Statistical identifiability and task relevance are separate properties.
8. Standard temperature scaling is not shift-safe here. It fits T=0.37-0.40 on an easy calibration split, sharpens probabilities, and worsens held-out ID NLL as well as OOD NLL/ECE. Raw probabilities remain the primary reported result.
9. Split-conformal singleton sets achieve at least nominal coverage in distribution (95.8% hybrid, 97.2% temporal-only at nominal 90%) but collapse to model accuracy under OOD shift (33.8% and 45.6%). No distribution-free OOD coverage claim is valid.
10. OOD abstention identifies shift but does not make accepted OOD predictions reliably safe: at 50% OOD coverage, temporal-only accuracy is only 61.3%. Deployment should abstain at the scenario level when shift is detected, not merely discard a small tail of node-level predictions.
11. The decision-aware reranker does not significantly beat the strongest centrality/activity heuristics at every budget. At k=3 its mean advantages over degree (+0.0353) and outbound activity (+0.0401) have confidence intervals crossing zero; at k=5 it ties outbound activity.
12. The earlier fixed-cascade intervention result is retrospective and cannot be generalized to adaptive attackers. The new adaptive simulator explicitly models rerouting and reverses the reranker's advantage.
13. OOD thresholds above are test-set operating characteristics, not deployment-calibrated cutoffs. A deployment needs a separate clean reference stream or online recalibration policy.
14. The fixed-cascade intervention reranker does not survive adaptive rerouting. At k=3, decision-aware net prevention is 0.341 versus 0.480 for source-only ranking; paired delta -0.139 with bootstrap 95% CI [-0.225, -0.054].
15. Across three training seeds, the Boltzmann residual prior has no mean benefit on the aligned test (delta MRR -0.00063). It cannot be presented as the main empirical improvement.
16. The self-supervised auxiliary objective is not supported by the current one-seed ablation: best validation MRR is 0.9896 with SSL versus 0.9965 without it. Multi-seed SSL ablation would be needed to determine whether this difference is systematic.
17. Real transfer is not superiority evidence: IoT-23 and TON_IoT source targets are also solved by simple activity baselines, TON has only two usable captures, and one packaged TON ground-truth file does not overlap its processed capture timestamps.
18. The universal cold-start claim is false. Boltzmann fusion does not significantly improve all sparse-telemetry graphs, and the observed gain does not consistently shrink to zero at full temporal evidence.

## Defensible current claim

Under controlled hub-growth, density, graph-size, sparse-observation, and telemetry-noise shifts, temporal graph modeling robustly improves attack-origin ranking over static graph inference and detects severe deployment shift. Fisher information makes the inherited Boltzmann prior operationally selective: a validation-fixed policy invokes it on approximately one-third of networks and produces a small independently confirmed cold-start gain. Intervention quality remains regime-dependent, and the current SSL auxiliary loss is unsupported.

## Submission blockers still open

1. Add a real dataset with defensible multi-hop compromise chronology; IoT-23 and TON_IoT now provide source-transfer checks only.
2. Run the SSL and TGAT-style comparisons across multiple independent training seeds; their current comparison uses one optimizer seed.
3. Replace unsafe confidence sharpening with an OOD-conditioned calibrator or domain-conditional calibration protocol, validated on independent deployment shifts.
4. Develop a policy that selects source-focused versus impact-focused quarantine from observable attacker-dynamics evidence, then validate it on real chronology.
5. Cache generated samples to reduce repeated beta-estimation cost. Parameter count, construction time, CPU latency, and process RSS are now measured.
6. Evaluate graph-construction alternatives and validate the implemented partial-order/early-compromise output on real chronology. IoT-23 alone cannot validate that output.
