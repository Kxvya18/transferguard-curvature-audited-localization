# Fisher-Selective Cold-Start Experiment

## Research question

Does the inherited Boltzmann reverse-time prior improve attack-origin ranking
when temporal telemetry is scarce, and can Fisher information identify when the
prior should be trusted?

## Controlled protocol

- One anytime model is trained on randomly sampled prefixes of 1, 2, 3, or 6
  windows and telemetry retention of 25%, 50%, 75%, or 100%.
- Hybrid and temporal-only models use identical architectures, graph samples,
  masks, optimizer seeds, and checkpoint criteria.
- The temporal encoder is prior-blind. The Boltzmann prior enters only through
  an auditable residual log-prior term.
- Fisher reliability multiplicatively bounds prior strength.
- Future six-window aggregate activity is removed from every prefix.
- Three optimizer seeds are trained for each method for 20 epochs.
- The evidence-aware cold-start model has 165,728 trainable parameters.
- Training spans n in {120, 220, 320}, m in {1, 2, 3, 4, 5}, and beta in
  {0.5, 1.0, 1.5, 2.0}.
- Each test grid contains 120 fresh graphs spanning n in {100, 220, 400}, all
  five m values, and all four beta values.

## Discovery result

The broad pre-specified hypothesis failed. Across all low-evidence graphs,
hybrid-minus-temporal MRR was +0.00668 with graph-clustered 95% CI
[-0.00537, +0.01885]. The learned gate nevertheless reduced mean residual
strength from 0.144 under low evidence to 0.055 under full evidence.

Using reliability cut points estimated only from the independent validation
cache, the highest-reliability group gained +0.02907 MRR under low evidence,
95% CI [+0.01088, +0.04824]. This motivated a fixed selective rule: use the
hybrid only above the validation Q75 Fisher-reliability threshold, otherwise use
the temporal-only model.

## Independent confirmation

The fixed rule was evaluated on a second 120-graph test grid with a new graph
seed base and no threshold refitting.

- The rule selected 38/120 graphs (31.7%).
- Low-evidence MRR increased from 0.43218 to 0.44067: delta +0.00849,
  graph-clustered 95% CI [+0.00242, +0.01548].
- Low-evidence Hit@1 increased from 0.35903 to 0.37431: delta +0.01528,
  95% CI [+0.00556, +0.02569].
- Within the validation-defined highest-reliability group, low-evidence delta
  MRR was +0.02682, 95% CI [+0.00780, +0.04671].
- The high-minus-low reliability interaction was +0.03307, 95% CI
  [+0.00629, +0.06090].

## Claim boundary

Supported:

> Fisher information is an operational applicability criterion for the
> inherited Boltzmann prior. A validation-fixed selective policy uses the prior
> on approximately one-third of networks and yields a small but reproducible
> improvement in cold-start source ranking.

Not supported:

> The Boltzmann prior improves every low-observability network, or its advantage
> necessarily disappears when all temporal windows are available.

The inherited BA framework is therefore conditionally necessary rather than a
universal replacement for temporal learning.

## Artifacts

- Discovery summary: `outputs/cold_start_v1/evaluation/summary.json`
- Confirmatory summary: `outputs/cold_start_v1/confirmatory_evaluation/summary.json`
- Per-graph metrics: `outputs/cold_start_v1/confirmatory_evaluation/model_sample_metrics.csv`
- Figure: `paper_assets/figures/fig_cold_start_fisher_selective_v6.png`
