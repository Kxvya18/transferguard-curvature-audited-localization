# Inherited Components Map

This file separates inherited concepts from genuinely new `ba2` design components.

## Inherited Theory and Algorithms from `ba`

### Structural prior over graph concentration

Inherited idea:

- preferential-attachment style growth can be represented by a Boltzmann-weighted attachment rule parameterized by `beta`

`ba2` reuse:

- convert this from the full scientific story into a structural prior module
- estimate a concentration parameter on observed communication graphs or synthetic IoT-like graphs
- use it as prior evidence for likely origin nodes and early-compromise candidates

### Sequential log-likelihood and `beta` estimation

Inherited implementation targets:

- sequential without-replacement attachment likelihood
- `beta` MLE
- Fisher information / confidence interval logic

`ba2` reuse:

- keep the mathematical identifiability logic where it remains valid
- repurpose confidence in `beta` as one part of the system’s structural-confidence profile

### Reverse-time removal / least-action intuition

Inherited idea:

- nodes that are easier to remove under the reverse model are more likely to be late additions

`ba2` reuse:

- reinterpret as early-compromise versus late-stage infection ranking
- use as a prior signal, not a stand-alone exact reconstruction claim

### Recoverability / ambiguity

Inherited idea:

- some graph states are structurally ambiguous and resist exact historical recovery

`ba2` reuse:

- treat ambiguity as a strength, not a weakness
- turn it into an explicit uncertainty feature
- expose “insufficient structural evidence” as a reason to abstain

### Controlled synthetic experimentation

Inherited practice:

- synthetic graph generation with explicit parameters
- reproducible seeds
- figure-and-CSV outputs

`ba2` reuse:

- preserve this experimental discipline
- extend it to heterogeneous IoT-like generators and transfer-shift stress tests

## Reused Code Candidates

These should be ported selectively into `ba2`, likely after refactoring.

### Good port candidates

- attachment probability utilities
- sequential likelihood computation
- `mle_beta`
- `fisher_information`
- reconstruction score helpers
- plotting / CSV helper patterns

### Port-with-caution candidates

- thermodynamic trace routines
- Hamiltonian-based diagnostics
- reverse-time Viterbi-style routines

Reason:

- these are conceptually useful but currently too tied to homogeneous BA assumptions

### Do-not-port-as-is candidates

- BA-paper-specific figure orchestration
- manuscript-specific theorem plotting logic
- claims hardcoded around equilibrium / non-equilibrium physics

## New `ba2` Components That Are Not Inherited

These are genuinely new and must be designed from scratch.

### 1. Real-data IoT graph builders

Needed because `ba` has no notion of:

- devices
- flows
- roles
- time windows
- protocol-specific graph views

### 2. Temporal self-supervised learning

Needed because `ba` has no learned representation module.

Planned role:

- encode unlabeled temporal graph structure
- reduce dependence on attack labels
- support transfer to new environments

### 3. Fusion model

Needed because `ba` only has structural scoring.

Planned role:

- fuse structural prior, temporal evidence, and metadata
- output source probability, early-compromise ranking, and calibrated confidence

### 4. Domain adaptation / few-shot adaptation

Needed because `ba` is single-domain synthetic only.

Planned role:

- support cross-scenario and cross-dataset adaptation
- operate under 0%, 1%, 5%, and 10% label regimes

### 5. Predictive uncertainty and abstention

Needed because `ba` only estimates uncertainty on `beta`, not on downstream decisions.

Planned role:

- identify when provenance localization is too uncertain
- produce risk-coverage and abstention analyses

### 6. Intervention utility layer

Needed because `ba` stops at inference.

Planned role:

- ask whether top-ranked suspected origins actually improve containment decisions
- make the system useful for ANTIC-style application framing

## Provenance Statement

To keep scientific provenance clean:

- `ba` contributes the structural-prior lineage
- `ba2` contributes the transfer-learning, uncertainty, intervention, and IoT-security framing
- any manuscript should explicitly say that the Boltzmann prior layer is adapted from earlier preferential-attachment inference work rather than presented as first appearing in `ba2`
