# Pre-registered test sets for the real outbreaks

Fixed on 2026-11-22, committed before any method was run on the outbreaks below. Every
outbreak already scored (the 10 variants in `outputs/real_epidemics/variants.jsonl`, the
283 Twitter cascades in `outputs/real_cascades/twitter_test_part*.jsonl`) is from now on
development data: models may be changed after looking at them. The sets below are scored
once, with the final code, and every result is reported.

## SARS-CoV-2 variants (CoVariants, OpenFlights)

Rule: every CoVariants cluster (web/src/content/en/clusters/*.mdx, fetched 2026-11-22, kept in
`data/raw/global/covariants_text/`) whose description names a single country as where it
arose, originated, emerged or was first detected, and that is not a development variant.

| Cluster | Origin (CoVariants wording) |
|---|---|
| 21L.Omicron (BA.2) | "South Africa, where this variant is suspected to have originated" |
| 22A.Omicron (BA.4) | "thought to have arisen in South Africa" |
| 22B.Omicron (BA.5) | "likely arose in early 2022, possibly in South Africa" |
| 23H.Omicron | "likely arose in May or June 2023, likely in China" |
| 24B.Omicron | "likely emerged in late-2023 or early 2024, possibly in India" |
| 24C.Omicron | "likely emerged early 2024, possibly in India" |

Secondary set (origin named only as a region; score = rank of the best-ranked country of
that region, regions as UN geoscheme subregions, or continents where a continent is named):
22E (Middle and Western Africa), 22F (Southern Asia), 23A (Northern America), 23B (Asia),
23G (South America), 23I (Southern Africa), 24A (Europe).

Protocol: as for the development variants (`experiments/global_variants.py`): same network,
reports, observation window (6 periods after the first detection, start up to 10 before),
same baselines. Mean field's parameters are fitted on the 10 development variants only.

## Twitter (Higgs)

Rule: the usable cascades of the later (test) half, in order of first retweet, after the 300
already used for development (`real_cascades.test`, count 300). All of them are scored.
Mean field's parameters are fitted on the earlier (training) half only, as before.

## Reported for every method, on every set

Mean reciprocal rank, top-1 and top-10 rate, the size of the 90% conformal source set
(calibrated on the development outbreaks of the same dataset), paired differences to the best
baseline with bootstrap 95% intervals and a paired Wilcoxon test, and every individual rank.

## Addendum, 2026-11-22: two outbreaks with independently known origins

Fixed before any method was run on them. These test what the variants cannot: their origins
are established by epidemiological investigation, not by who reported first.

| Outbreak | Origin | Data (open) |
|---|---|---|
| H1N1 pandemic, 2009 | Mexico | WHO situation updates by country, 24 Apr - 6 Jul 2009 (Kaggle imdevskp/h1n1-swine-flu-2009-pandemic-dataset, data.csv) |
| SARS, 2003 | China (Guangdong) | WHO cumulative case tables by country, 17 Mar - 11 Jul 2003 (Kaggle imdevskp/sars-outbreak-2003-complete-dataset) |

Protocol (`experiments/independent_origins.py`):
- Network: countries linked by airline routes (OpenFlights, as for the variants), one contact
  per route per day. Country names are matched to OpenFlights by a fixed table in the script;
  territories go to their own OpenFlights entry where one exists, otherwise they are dropped.
- Report: a country's first day with at least one WHO-reported case. Every country in the
  network is an observer (WHO collected reports from all member states); a country with no case
  by the observation day reports "not infected".
- Observation day: 28 days after the first report anywhere. The start may be up to 150 days
  before the first report (SARS emerged months before the WHO tables begin).
- Mean field: no training data exist for these diseases, so it is untrained: the average over
  a fixed grid of rate per route per day (1e-5, 1e-4, 1e-3), outside introductions per day
  (0, 1e-5, 1e-4), and chance per day that an infected country reports (0.02, 0.1, 0.5).
- Baselines on the same reports: earliest detection, distance center, Jordan center, LPSI
  (alpha 0.5, no tuning data), Pinto-style, effective distance (Brockmann and Helbing 2013).
- Reported: the rank of the origin under every method (ties share the average rank).
