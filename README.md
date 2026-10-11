# TRANSFERGUARD (`ba2`)

`ba2` is the new active research project derived from the earlier Boltzmann-Barabasi work in [`/Users/boolu/Desktop/simulations/ALL_PROJECTS/ba`](/Users/boolu/Desktop/simulations/ALL_PROJECTS/ba).

Working title under evidence-driven revision:

**TRANSFERGUARD: Shift-Aware Temporal Attack-Origin Localization and Adaptive Containment in Evolving IoT Networks**

## Project Intent

This repository does **not** continue the old BA-only paper. It reuses the strongest ideas from `ba` as a structural-prior layer and rebuilds the system around a more modern security problem:

- localizing likely attack origins under label scarcity
- ranking early compromise nodes
- operating under cross-scenario and cross-dataset shift
- providing calibrated uncertainty and abstention
- measuring intervention utility, not just detection accuracy

## Current Status

The project now includes:

- audited legacy-to-new theory provenance and a 15+ paper gap synthesis
- heterogeneous preferential-growth IoT simulation with controlled domain shifts
- sequential beta estimation and reliability-shrunk Boltzmann origin priors
- temporal graph learning with architecture-matched no-prior ablations
- corrected IoT-23 ingestion for 1,194,491 events across five scenarios
- 480-scenario stress grids covering n={100,220,400}, m={1,...,5}, and beta={0.5,1,1.5,2}
- temperature, ECE, Brier, NLL, conformal, risk-coverage, and MC-dropout audits
- OOD tests combining unseen structure and severe telemetry degradation
- two-window early-order inference and time-respecting quarantine evaluation
- decision-aware probability-times-impact reranking with paired statistical tests
- three independent optimizer seeds for hybrid and temporal-only models
- static GCN and windowed TGAT-style reference baselines on an identical fixed test
- an adaptive attacker that reroutes after quarantine
- frozen source-ranking and shift-detection checks on IoT-23 and TON_IoT
- component ablations for temporal encoding, Boltzmann fusion, SSL, uncertainty, and intervention reranking
- anytime cold-start training with a Fisher-bounded Boltzmann residual gate
- discovery and independent confirmation of a validation-fixed selective-prior policy
- CPU runtime, peak process memory, parameter count, and inference-latency profiling
- conference-ready figures generated only from persisted result artifacts

The current evidence is intentionally bounded. IoT-23 is a feature-transfer sanity
check rather than a multi-hop provenance benchmark, and standard temperature
scaling is not OOD-safe in the observed experiments. The Boltzmann residual and
current SSL loss does not improve aligned mean ranking, while intervention-reranker
utility reverses under adaptive rerouting. The Boltzmann prior is useful only
selectively: a validation-fixed Fisher-reliability policy confirms a small
cold-start gain on a second held-out grid. See `docs/current_findings.md` and
`docs/cold_start_fisher_selective.md` before drafting claims.

See:

- [`docs/legacy_audit.md`](/Users/boolu/Desktop/simulations/ALL_PROJECTS/ba2/docs/legacy_audit.md)
- [`docs/inherited_components.md`](/Users/boolu/Desktop/simulations/ALL_PROJECTS/ba2/docs/inherited_components.md)
- [`docs/migration_decisions.md`](/Users/boolu/Desktop/simulations/ALL_PROJECTS/ba2/docs/migration_decisions.md)
- [`docs/literature_matrix.md`](/Users/boolu/Desktop/simulations/ALL_PROJECTS/ba2/docs/literature_matrix.md)
- [`docs/gap_synthesis.md`](/Users/boolu/Desktop/simulations/ALL_PROJECTS/ba2/docs/gap_synthesis.md)
- [`docs/design_decisions.md`](/Users/boolu/Desktop/simulations/ALL_PROJECTS/ba2/docs/design_decisions.md)

## Repository Skeleton

The codebase is being organized around the following modules:

- `loaders/`
- `graph_builders/`
- `synthetic/`
- `priors/`
- `self_supervised/`
- `fusion/`
- `transfer/`
- `uncertainty/`
- `interventions/`
- `evaluation/`
- `experiments/`
- `paper_assets/`

## Boundary Rules

- `ba` remains reference-only.
- All new implementation, experiments, outputs, and manuscript assets must live under `ba2`.
- Ideas inherited from `ba` must be explicitly documented as inherited rather than claimed as newly invented here.
