# First-Pass Design Decisions for `ba2`

This file converts the audit and literature synthesis into concrete project decisions.

## Single-Sentence Thesis

`ba2` will test whether a **Boltzmann structural prior**, fused with **self-supervised temporal graph representations**, can improve **attack-origin localization** and **early-compromise ranking** in **unseen IoT environments** under **label scarcity and domain shift**, while also producing **calibrated uncertainty** for abstention and containment decisions.

## Main Task Definition

Primary tasks:

- source-node localization
- early-compromise node ranking
- optional pairwise precedence / partial-order inference

Secondary tasks:

- attack / benign detection
- attack-family prediction

Reason:

- primary tasks match the gap we found
- secondary tasks are useful supporting outputs, but not the center of novelty

## Data Strategy

### Real datasets

Priority order:

1. IoT-23
2. TON_IoT
3. Edge-IIoTset or N-BaIoT

Selection logic:

- IoT-23 is the most natural primary anchor because it contains realistic IoT malware scenarios and device-level traffic context.
- TON_IoT is a good transfer target because it offers richer scenario diversity.
- A third dataset is useful only if graph construction remains clean and reproducible.

### Synthetic datasets

We will build a heterogeneous generator for controlled experiments with:

- role types
- role-conditioned attachment
- tunable `beta`
- aging and fitness
- communities
- rewiring noise
- spread cascades
- observation dropout
- source-target shift knobs

Reason:

- real datasets rarely give exact ground-truth chronology or counterfactual control
- synthetic data lets us measure source localization and early-compromise ranking precisely

## Graph Construction Decisions

We should not commit to one graph view without comparison.

Planned graph views:

1. device-to-device communication graph
2. heterogeneous device-IP-domain-service graph
3. time-windowed event graph

Evaluation question:

- which graph view best supports origin localization and transfer, not only classification?

Initial default:

- start with time-windowed heterogeneous communication graphs

Reason:

- this retains actor identity and temporal context without requiring full host provenance

## Prior Module Decisions

### Prior outputs to compute

- `beta_hat`
- Fisher-information-based reliability of `beta_hat`
- hub concentration features
- reverse-time origin score
- ambiguity / recoverability score
- role-aware prior score

### Prior variants to compare

1. vanilla BA prior
2. heterogeneous role-aware BA prior
3. empirical-statistics-regularized prior

### Prior usage

The prior will not directly make final decisions.

It will instead feed:

- fusion features
- uncertainty features
- ablation comparisons

## Self-Supervised Encoder Decisions

### Default model family

Start with a temporal heterogeneous GraphSAGE or temporal GAT backbone rather than a very large transformer.

Reason:

- better engineering simplicity
- more likely to train robustly on moderate hardware
- easier ablations

### SSL objectives

Initial objective set:

- masked node / edge attribute prediction
- temporal next-edge or next-window prediction
- contrastive consistency between adjacent windows

Optional later objective:

- role reconstruction or subgraph discrimination

Reason:

- these align well with unlabeled network data and avoid strong label leakage

## Fusion Decisions

We will compare at least three fusion styles.

### Fusion A: early baseline

- concatenation of prior features and learned embeddings into an MLP

Purpose:

- easy baseline that clarifies whether the prior carries additive value

### Fusion B: gated fusion

- learned gate over prior features versus SSL features

Purpose:

- shows whether the model learns when structure matters

### Fusion C: energy-style fusion

- treat prior score as energy / log-prior term and learned evidence as log-likelihood-style term

Purpose:

- most faithful to the project’s scientific identity

## Transfer Strategy Decisions

### Evaluation regimes

- in-domain supervised
- cross-scenario transfer
- cross-dataset transfer
- unseen attack family
- unseen role composition
- few-shot adaptation at 1%, 5%, 10% labels

### Adaptation strategies

Start with:

1. frozen pretrained encoder + lightweight target head
2. feature alignment with a simple domain-adaptation loss

Add later if needed:

- source-free adaptation
- prototype-based few-shot adaptation

Reason:

- gives a strong and manageable first experiment grid

## Uncertainty Decisions

### Initial uncertainty stack

- deep ensemble or MC dropout for epistemic uncertainty
- temperature scaling on validation splits
- conformal wrapper for set-valued or abstention-aware decisions
- prior ambiguity score as an additional uncertainty feature

### Core uncertainty outputs

- node posterior entropy
- calibration error
- risk-coverage curve
- abstention-quality curve

Reason:

- these are the minimum needed to make “trustworthy provenance inference” a real claim

## Intervention Evaluation Decisions

We will add a simulated containment task:

- quarantine top-k ranked candidate origins or early-compromise nodes
- measure prevented downstream spread or residual compromised mass

Reason:

- this converts attribution quality into operational value
- it fits ANTIC better than another classification table

## Baseline Decisions

Baselines should include both classical and modern families.

### Structural / heuristic baselines

- degree centrality
- betweenness
- PageRank
- random
- shortest-path or diffusion heuristics where applicable

### Learned baselines

- temporal GNN without prior
- prior-only scoring
- simple MLP over flow statistics
- few-shot/meta-learning comparator where feasible

### Required ablations

- no prior
- no SSL
- no transfer adaptation
- no uncertainty calibration
- no intervention-aware thresholding

## Claim Discipline

The main claim should stay close to this form:

> A physics-informed structural prior derived from hub-dominated graph growth, when fused with self-supervised temporal graph representations, improves attack-origin localization and early-compromise ranking in unseen IoT deployments under label scarcity and domain shift, while providing calibrated uncertainty for safer intervention decisions.

We should not promise:

- universal zero-shot generalization
- exact provenance recovery
- full causality from sparse data

## Immediate Next Implementation Stages

1. Create a small reusable Python package skeleton and config layout.
2. Port the minimal structural-prior code from `ba` into `ba2/priors/`.
3. Build dataset-ingestion stubs for IoT-23 and TON_IoT.
4. Implement graph builders for two candidate graph views.
5. Build the synthetic heterogeneous IoT generator.
6. Add a minimal SSL encoder and prior-only baseline.
7. Run the first end-to-end synthetic experiments before touching real transfer experiments.
