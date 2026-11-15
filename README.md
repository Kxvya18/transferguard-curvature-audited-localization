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

Download `CollegeMsg.txt.gz` and `email-Eu-core-temporal.txt.gz` from SNAP into
`data/raw/temporal/` (not tracked).

## Run

```bash
python -m unittest tests.test_localization
python -m experiments.compare_mean_field "Email-Eu" 0.1 40
python -m experiments.compare_learned "Email-Eu" 0.1 201 60
```

## Status

Pilot results are in `outputs/`. They are preliminary (one or two seeds) and
the comparison on CollegeMsg is incomplete.
