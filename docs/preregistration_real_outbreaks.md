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
