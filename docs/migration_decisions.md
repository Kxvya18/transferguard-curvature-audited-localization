# Migration Decisions from `ba` to `ba2`

This document records what we are intentionally keeping, changing, or abandoning.

## Decision Summary

| Item | Decision | Rationale |
|---|---|---|
| Boltzmann attachment prior | Keep, but demote from full story to prior module | Strong differentiator, but insufficient alone |
| `beta` identifiability machinery | Keep and refactor | Still valuable as a structural-confidence signal |
| Hamiltonian terminology | Keep cautiously | Useful interpretability hook, but should not dominate claims |
| Exact BA paper framing | Abandon | Too narrow and not application-facing |
| Reverse-time reconstruction | Reframe | Convert from exact order recovery to probabilistic provenance ranking |
| BA-only synthetic generator | Replace with heterogeneous generator | Need roles, drift, attack propagation, partial observation |
| Pure theorem-first narrative | Abandon for conference target | ANTIC fit requires system value and empirical strength |
| Figure-generation habits | Keep | Good reproducibility pattern |
| Old manuscript assets | Do not port directly | They reflect the wrong scientific center of gravity |

## Decisions in Detail

### Keep: Boltzmann prior as a structural lens

We will keep the core idea that hub-dominated graph structure contains signal about likely early nodes and concentration of interaction.

How it changes:

- from “the network is fully explained by a BA Hamiltonian”
- to “a Boltzmann-style structural prior supplies useful inductive bias”

### Keep: `beta` estimation, but reinterpret it

Legacy meaning:

- inverse-temperature / preferential attachment strength

`ba2` meaning:

- degree-concentration parameter
- structural hub-dominance score
- prior reliability indicator

This parameter becomes useful when:

- topological spread is concentrated through gateways, brokers, or hubs
- the environment is structurally skewed

It becomes less useful when:

- communication is flat
- role constraints dominate degree
- graph formation is far from preferential concentration

### Reframe: reconstruction

We will not claim:

- exact attack chronology recovery from structure alone

We will instead target:

- source-node localization
- early-compromise ranking
- partial-order precedence estimates
- provenance-path candidates with uncertainty

This keeps the spirit of the old work while making the claim defendable.

### Replace: homogeneous BA generator

The old generator is not enough.

`ba2` synthetic generation must support:

- device roles
- role-conditioned attachment
- aging
- fitness
- modular structure
- rewiring / noise
- attack spread processes
- observation loss
- source-target domain shifts

### Add from scratch: learned temporal evidence

This is the biggest scientific redesign.

Reason:

- structure alone will not survive domain shift
- temporal attack patterns cannot be read from degree concentration alone
- few-shot and zero-shot claims require representation learning

### Add from scratch: calibration and abstention

Reason:

- security systems fail badly when confidently wrong
- transfer settings need a mechanism for “do not trust this prediction”

### Add from scratch: intervention utility

Reason:

- conference reviewers will ask “what does better localization buy us?”
- containment ranking creates a concrete downstream value proposition

## Claims Explicitly Abandoned

We are abandoning the following as main-paper claims:

- structural thermodynamics alone is the main contribution
- exact historical reconstruction is the primary security deliverable
- BA synthetic evidence alone is enough for deployment relevance
- forward prediction should be judged mainly by flat top-k node accuracy

## Claims Explicitly Retained

We retain these ideas in adapted form:

- structural concentration is informative
- reverse-time reasoning can produce useful origin evidence
- ambiguous structure should be quantified rather than hidden
- sequential likelihood and uncertainty around `beta` are publishable support tools

## Migration Principle

Every inherited component must pass this test:

> Does this help localize attack origins or early compromise in a way that remains useful under heterogeneity, label scarcity, and domain shift?

If yes:

- keep and refactor

If partly:

- keep as auxiliary prior or diagnostic

If no:

- leave it in `ba`
