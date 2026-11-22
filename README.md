# Mean-field source localization on evolving networks

Given a network whose contacts happen at specific times, and a small set of
observer nodes that each report when they were infected (or that they are still
healthy), rank every node by how likely it is to be where the spread started.

## Method

`localization/mean_field.py` computes, for every candidate source at once, the
mean-field probability that each node is infected by each time step, and scores
the candidate by how well those probabilities explain the observers' reports.
The unknown start time and transmission probability are summed out over a grid.
It needs no training and no random simulations.

## Layout

- `localization/temporal_graph.py`: evolving-network container, SNAP loader, synthetic growing network.
- `localization/spread.py`: SI spreading over time windows, earliest arrival, outbreak simulation.
- `localization/mean_field.py`: the mean-field likelihood localizer.
- `localization/estimators.py`: training-free baselines (Jordan centre, distance centre, time-of-arrival fits, LPSI, SME, simulation posterior).
- `localization/learned.py`, `localization/arrival_features.py`: learned baselines (GCNSI, temporal GCN, Arrival GNN).
- `experiments/compare_mean_field.py`: mean-field against training-free baselines on identical outbreaks.
- `experiments/compare_learned.py`: mean-field against learned baselines trained on the same network.
- `experiments/benchmark_localization.py`: full benchmark (seeds, observer fractions, rate-shift tests).
- `tests/test_localization.py`: unit tests.

## Data

All data are open; nothing was requested from anyone. Not tracked; the GitHub Actions
workflows in `.github/workflows/` download each file at a fixed version, and so can you:

| Folder in `data/raw/` | Files | Source |
|---|---|---|
| `temporal/` | `CollegeMsg.txt.gz`, `email-Eu-core-temporal.txt.gz` | SNAP (snap.stanford.edu/data) |
| `sociopatterns/` | `sp_hospital`, `sp_primary_school`, `sp_office` `.csv.zip`; `sp_high_school_2012.csv.zip` | Netzschleuder (`networks.skewed.de/net/<name>/files/<name>.csv.zip`; high school: `sp_high_school_new/files/2012.csv.zip`) |
| `higgs/` | `higgs-social_network.edgelist.gz`, `higgs-activity_time.txt.gz` | SNAP (De Domenico et al. 2013) |
| `global/` | `airports.dat`, `routes.dat` | OpenFlights, commit 3b0e4d8 |
| `global/covariants/` | `<cluster>.json` | CoVariants `cluster_tables/<cluster>_data.json`, commit f42c1e2 |
| `global/un_regions.csv` | UN geoscheme | lukes/ISO-3166-Countries-with-Regional-Codes `all/all.csv`, commit 145f1ad |
| `heldout/` | H1N1 2009 and SARS 2003 WHO country tables | Kaggle `imdevskp/h1n1-swine-flu-2009-pandemic-dataset`, `imdevskp/sars-outbreak-2003-complete-dataset` |

Belief propagation needs `sib` (`pip install git+https://github.com/sibyl-team/sib.git@5b762da`).

## Reproducing the paper

Every table: `python -m experiments.make_tables` (reads `outputs/`, writes `outputs/tables.json`).
Seeds are fixed in each script (BP comparison 101-113, test selection 301-303, Twitter 1001).

| Result | Command | Output |
|---|---|---|
| Mean field vs BP vs message passing | `python -m experiments.compare_bp <network> 0.1 40 <seed> [true]` | `outputs/bp_comparison/` |
| Test selection (9 strategies) | `python -m experiments.observer_placement <network> 40 10 <seed>` | `outputs/observer_placement/` |
| Sterchi et al. 2023 by simulation | `SHARD=i/16 python -m experiments.observer_placement Hospital 16 10 301 simulated` | `outputs/observer_placement/Hospital_simulated_*` |
| Exact likelihood (limits) | `experiments/limits_*.py` | `outputs/limits/` |
| Source sets with coverage | `python -m experiments.conformal_sets` | `outputs/conformal_sets.json` |
| Separability (selection guarantee) | `python -m experiments.separability` | `outputs/separability.json` |
| Twitter (Higgs) | `real_cascades fit 200`, then `real_cascades score train 0 200`, `score test 0 300`, `score test 300 100000`; `twitter_v3_summary` | `outputs/real_cascades/` |
| SARS-CoV-2 variants | `python -m experiments.global_variants` (`HELDOUT=1`, `HELDOUT=2` for the pre-registered sets) | `outputs/real_epidemics/` |
| H1N1 2009, SARS 2003 | `python -m experiments.independent_origins` | `outputs/real_epidemics/independent_origins.jsonl` |

The real-outbreak test sets were fixed before any method was run on them:
`docs/preregistration_real_outbreaks.md`. Theory notes: `docs/theory.md`.
