# Gap Synthesis for `ba2`

This document distills the literature review into concrete design pressure.

## Core Observation

The literature is fragmented across four partially disconnected camps:

1. provenance-based intrusion analysis
2. graph-based IoT intrusion detection
3. graph transfer / few-shot adaptation
4. graph uncertainty / calibration

Each camp solves an important piece, but none combine them into a single system that is well suited for:

- unseen IoT deployments
- weak labels
- origin localization rather than only attack classification
- abstention when evidence is structurally ambiguous

## Recurrent Assumptions We Keep Seeing

### Assumption A: rich logging is available

Common in provenance papers:

- whole-system provenance
- dense host logging
- strong causality traces

Why this is a problem:

- IoT and edge environments often cannot afford full temporal logging
- even if logging exists, it may be partial, delayed, or lossy

Design implication for `ba2`:

- the method should operate on lighter graph abstractions from flows, windows, and interaction summaries
- synthetic experiments must include incomplete observations

### Assumption B: the main task is attack classification

Common in IoT IDS papers:

- benign vs malicious
- attack-family classification
- known/unknown attack type discrimination

Why this is a problem:

- analysts often need “where did this start?” and “what should I quarantine first?”
- classification alone does not give provenance or actionability

Design implication for `ba2`:

- primary outputs should be source probability, early-compromise ranking, and top intervention candidates

### Assumption C: train and test distributions are close

Common in both security and graph-learning papers:

- random splits within the same dataset
- scenario splits with weak distribution shift

Why this is a problem:

- new deployments change device mix, topology, traffic intensity, and attack families

Design implication for `ba2`:

- cross-scenario, cross-dataset, and role-composition transfer must be first-class experiments

### Assumption D: confidence scores are enough

Common across many ML security systems:

- report probabilities
- skip calibration
- no risk-coverage analysis

Why this is a problem:

- in security, confident errors are operationally dangerous

Design implication for `ba2`:

- uncertainty must be evaluated, calibrated, and tied to abstention

### Assumption E: structure is either ignored or purely learned

Common split:

- classical IDS papers underuse graph structure
- GNN papers learn structure implicitly but rarely use explicit structural priors

Why this is a problem:

- hub-mediated spread and central-broker dependence are real in many IoT topologies
- purely learned models may overfit domain-specific patterns

Design implication for `ba2`:

- keep the Boltzmann/BA prior as an explicit, interpretable bias
- test when the prior helps and when it hurts

## What the Literature Still Lacks

### 1. Structural-prior plus temporal-SSL fusion

We found good structural priors in the old `ba` lineage and good SSL in recent provenance / graph-IDS work, but not a convincing fusion of:

- explicit graph-growth prior
- temporal learned evidence
- target-domain adaptation

### 2. Origin localization under label scarcity

Many papers detect attacks. Fewer papers localize causes. Very few do so under:

- few-shot labels
- cross-dataset shift
- partial observability

### 3. Calibration-aware provenance inference

The provenance literature talks about attribution quality but rarely about:

- calibrated node-level posterior uncertainty
- valid abstention
- risk-coverage tradeoffs

### 4. Intervention-value evaluation

Most papers stop at:

- accuracy
- F1
- PR-AUC

They do not show whether the inferred rankings improve:

- quarantine selection
- containment speed
- compromise spread reduction

## Why the Old `ba` Prior Still Matters

The audit does not say “forget BA.” It says:

- use BA carefully
- narrow its role
- let it do the job it is actually good at

Specifically, the old prior is still useful because it gives `ba2`:

- a mechanistic concentration score
- a principled way to express structural preference for hubs / brokers
- an ambiguity notion that can feed uncertainty

This is a differentiator because many modern graph-security papers rely entirely on learned features with weak interpretability.

## The Precise Niche for `ba2`

The niche is not:

- another IDS benchmark
- another pure provenance system
- another graph domain adaptation paper

The niche is:

> attack-origin localization and early-compromise ranking in unseen IoT networks using a hybrid of explicit structural priors and self-supervised temporal graph learning, with calibrated uncertainty and intervention-aware evaluation.

## Implications for Scope Discipline

To stay strong, `ba2` should avoid three traps.

### Trap 1: promising exact chronology

We should not promise exact full attack reconstruction from sparse logs.

Safer target:

- ranked origin and early-compromise candidates
- optional partial-order predictions

### Trap 2: overclaiming zero-shot generalization

We should not claim universal zero-shot robustness.

Safer target:

- improved robustness under tested cross-domain settings
- explicit failure analysis under severe shift

### Trap 3: letting thermodynamics dominate again

The structural prior should be visible and important, but the conference paper should still read like:

- a network-security system paper

not like:

- a physics paper with IoT nouns added afterward

## Decision-Level Synthesis

Because of the gaps above, `ba2` should be built around the following commitments:

1. Main task: source localization and early-compromise ranking
2. Main learning engine: self-supervised temporal graph encoder
3. Main differentiator: explicit Boltzmann structural prior
4. Main robustness story: transfer and few-shot adaptation
5. Main trustworthiness story: calibration, abstention, and ambiguity awareness
6. Main operational story: intervention utility
