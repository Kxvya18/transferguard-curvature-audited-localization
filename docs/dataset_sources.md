# Dataset Sources for `ba2`

This document records the real-data sources being used for TRANSFERGUARD and how they map to the project goals.

## Primary Real Dataset

### IoT-23

Official source:

- [Stratosphere IPS IoT-23 dataset page](https://www.stratosphereips.org/datasets-iot23)
- [IoT-23 release post with direct archive links](https://www.stratosphereips.org/blog/2020/1/22/aposemat-iot-23-a-labeled-dataset-with-malicious-and-benign-iot-network-traffic)

Current acquisition choice:

- official small archive containing `conn.log.labeled` files

Direct archive used:

- `https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/iot_23_datasets_small.tar.gz`

Why it is central:

- real IoT malware traffic
- labeled Zeek connection logs
- scenario-based structure that is suitable for temporal graph construction

## Secondary / Transfer Dataset

### TON_IoT

Official source:

- [UNSW TON_IoT project page](https://research.unsw.edu.au/projects/toniot-datasets)

Important note:

- the official UNSW page exposes the dataset through a public SharePoint link rather than a stable plain-file URL
- because of that, acquisition may require resolving the current redirect chain or using a public mirror that preserves the published dataset contents

Fallback mirror used only if the official file endpoint is unstable:

- [Zenodo record referencing TON_IoT and IoT-23](https://zenodo.org/records/19367312)

Why TON_IoT matters:

- more heterogeneous than a single network-flow dataset
- useful for cross-dataset transfer and few-shot adaptation experiments
- includes security-event ground-truth metadata in the published structure

## Optional Tertiary Datasets

Potential next additions:

- Edge-IIoTset
- N-BaIoT
- UNSW-NB15 or Bot-IoT

These are not being prioritized ahead of IoT-23 and TON_IoT because the current research question is about:

- source localization
- early-compromise ranking
- uncertainty under shift

rather than maximizing the number of benchmark names.

## Data-Engineering Constraint

The project requirement is not merely to classify rows. For every real dataset we acquire, we will convert raw or processed records into:

- communication events
- time-windowed graphs
- scenario summaries
- candidate provenance-style subgraphs

so that the evaluation reflects the project’s actual scientific target rather than a tabular IDS shortcut.
