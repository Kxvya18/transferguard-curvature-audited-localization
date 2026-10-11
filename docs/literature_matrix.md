# Literature Matrix for `ba2`

This matrix focuses on primary papers most relevant to provenance-aware intrusion analysis, IoT graph security, transfer under label scarcity, and uncertainty-aware graph inference.

Notes:

- “Transfer” means explicit cross-domain, zero-shot, few-shot, or open-set adaptation.
- “Uncertainty” means calibration, abstention, Bayesian uncertainty, conformal prediction, or a directly related uncertainty mechanism.
- The final “Gap for `ba2`” column is interpretive and reflects our synthesis rather than the original authors’ claim.

## Matrix

| # | Paper | Problem solved | Setting | Data / logging assumptions | Supervision assumptions | Graph assumptions | Transfer? | Uncertainty? | Outputs | Strengths | Main limitations | Gap for `ba2` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [Han et al., “Provenance-based Intrusion Detection: Opportunities and Challenges” (USENIX TaPP 2018)](https://www.usenix.org/conference/tapp2018/presentation/han) | Frames provenance-based IDS opportunities and limitations | Host / provenance | Requires detailed provenance capture | Conceptual, not a transfer design | Rich provenance graphs | No | No | Detection / framing | Strong problem framing; attack-agnostic motivation | No concrete transfer, calibration, or IoT solution | Use as motivation for provenance value, not as a system recipe |
| 2 | [ProvNet-IoT (DFRWS APAC 2022)](https://doi.org/10.1016/j.fsidi.2022.301441) | Provenance-based IoT network forensics | IoT network forensics | Relies on network-layer evidence and modeled artifacts | Uses labeled IoT datasets for validation | Provenance-like IoT interaction graph | No | No | Attack-scenario reconstruction / evidence | IoT-native forensic framing | Not a modern transfer-learning system; limited uncertainty | `ba2` can modernize provenance reasoning with learned and calibrated inference |
| 3 | [KAIROS (IEEE S&P 2024)](https://spg.cs.ubc.ca/publication/2024-sp/) | Practical provenance-based intrusion detection and investigation | Host provenance | Requires whole-system provenance streams | Mainly benign-learning anomaly detection | Temporal provenance graph | No | Limited implicit anomaly scores only | Node anomalies and summary attack graph | Strong temporal graph learning + reconstruction | Heavy provenance assumptions; not transfer-focused; not IoT-focused | `ba2` should borrow the “quality of attribution” mindset without needing full host provenance |
| 4 | [MAGIC (USENIX Security 2024)](https://www.usenix.org/conference/usenixsecurity24/presentation/jia-zian) | APT detection via masked graph representation learning | Provenance / enterprise security | Requires provenance graphs | Self-supervised pretraining, low attack dependence | Masked graph representation on provenance graphs | Limited concept-drift adaptation | No explicit calibration | Entity-level and batch-level anomaly detection | Very relevant SSL precedent | Focused on APT detection, not source localization under transfer | Supports `ba2` decision to include masked graph SSL objectives |
| 5 | [Yu et al., “Cost-effective Attack Forensics by Recording and Correlating File System Changes” (USENIX Security 2024)](https://www.usenix.org/conference/usenixsecurity24/presentation/yu-le) | Low-cost attack forensics without dense temporal logging | IoT / resource-constrained systems | Assumes full temporal logging is expensive; uses file-system deltas | Not a transfer-learning paper | Spatially correlated provenance-like graph | No | No | Causality / attack forensics | Important practical bottleneck: logging cost | Not graph-transfer or uncertainty-aware | Validates `ba2` focus on partial observability and approximate provenance |
| 6 | [Few-Shot Learning-Based Malicious IoT Traffic Detection with Prototypical GNNs (IEICE 2023)](https://doi.org/10.1587/transinf.2022OFP0004) | Few-shot IoT attack detection | IoT traffic | Uses flow-derived representations | Few-shot labeled episodes | GNN over traffic representations | Yes, few-shot | No | Attack classification | Strong label-scarcity relevance | Not provenance-aware, not transfer across graph domains, not calibrated | `ba2` should move from few-shot classification to source localization + provenance ranking |
| 7 | [Source Free Unsupervised Graph Domain Adaptation (arXiv 2021/2023)](https://doi.org/10.48550/arXiv.2112.00955) | Domain adaptation when source graph is unavailable | General graph ML | Assumes pretrained source model only | Unsupervised target adaptation | Standard graph DA assumptions | Yes | No | Node classification | Strong for privacy-aware adaptation | Not a security paper; not temporal; not provenance-aware | Useful adaptation blueprint for cross-dataset transfer in `ba2` |
| 8 | [Diffusion Network Inference from Partial Observations (AAAI 2021)](https://ojs.aaai.org/index.php/AAAI/article/view/16918) | Infer diffusion structure from partial observations | General diffusion networks | Missing timestamps and missing infection data | Weakly observed cascades | Diffusion network structure | No | No | Inferred structure / transmission probabilities | Directly relevant to partial observation reasoning | Focuses on diffusion-network recovery, not security graphs or transfer | Supports `ba2` handling of incomplete compromise observations |
| 9 | [FedSTGCN (FITEE 2025)](https://doi.org/10.1631/FITEE.2400932) | Federated spatiotemporal GNN for IoT intrusion detection | IoT / edge privacy | Multi-site distributed traffic datasets | Supervised with FL | Spatiotemporal graph | Indirect, via federated setting | No | Intrusion detection | Shows temporal GNNs matter in IoT | Still mainly classification; no provenance; no calibrated abstention | Good comparator for temporal modeling, but `ba2` needs attribution rather than only classification |
| 10 | [GraphIDS: Self-Supervised Learning of Graph Representations for Network Intrusion Detection (NeurIPS 2025)](https://mlanthology.org/neurips/2025/guerra2025neurips-selfsupervised/) | Self-supervised graph intrusion detection | Network traffic | Learns from largely unlabeled traffic | Self-supervised masked autoencoding | Local communication graph + transformer decoder | Not explicit transfer | No explicit calibration | Intrusion scores on flows | Very relevant modern SSL baseline | Focus is anomaly detection, not attack-origin localization or provenance ranking | Strong evidence that SSL should be central in `ba2` |
| 11 | [Sometimes Simpler is Better: A Comprehensive Analysis of State-of-the-Art Provenance-Based Intrusion Detection Systems (USENIX Security 2025)](https://www.usenix.org/conference/usenixsecurity25/presentation/bilot) | Audits PIDS practicality and benchmarking flaws | Host provenance | Requires standardized provenance datasets | Mainly anomaly-detection evaluation | Provenance graphs | No | No | Detection quality / practicality analysis | Excellent critique of current PIDS evaluation | Not an IoT transfer solution itself | Important warning: `ba2` must optimize analyst-usable attribution, not inflated graph-level metrics |
| 12 | [ORTHRUS (USENIX Security 2025)](https://www.usenix.org/conference/usenixsecurity25/presentation/jiang-baoxiang) | High-quality attribution in provenance IDS | Host provenance | Requires detailed provenance logs | Mostly weakly supervised anomaly framing | Spatiotemporal provenance graph | No | No explicit calibration | Node-level attribution and attack-path reconstruction | Strongest recent “quality of attribution” precedent | Heavy logging assumptions; not transfer; not IoT | `ba2` should match this attribution mindset on lighter IoT graphs |
| 13 | [MAML-SAMIoT (Computer Networks 2026)](https://www.sciencedirect.com/science/article/pii/S1389128625008503) | Few-shot intrusion detection via meta-learning | IoT / cloud-fog-edge | Uses curated few-shot IoT datasets | Few-shot meta-learning | Not primarily graph-provenance | Yes, few-shot | No | Attack classification | Strong label-scarcity design | Not provenance-aware; not structural-prior aware; not calibrated | Useful few-shot baseline family for `ba2` |
| 14 | [Systematic Evaluation of Few-Shot Learning for Unseen IoT Network Attack Detection (CMES 2026)](https://www.techscience.com/CMES/v147n1/67129/html) | Benchmarks few-shot detection for unseen IoT attacks | IoT IDS | Uses multiple datasets for unseen attacks | Few-shot supervised | Mostly non-provenance / traffic feature setting | Yes | No | Unseen-attack detection | Good benchmarking perspective | Still attack-type classification, not source localization or provenance inference | Helps justify few-shot regime design in `ba2` |
| 15 | [MGIDS (ACM TOIT 2026)](https://doi.org/10.1145/3811424) | Graph-based IDS for MQTT IoT traffic | MQTT IoT | Mixed encrypted / unencrypted traffic, flow-derived graphs | Supervised two-stage classification | Dynamic grid graph over quantized flow signatures | Some unknown-attack evaluation | No | Attack-category classification | Strong protocol-specific graph construction idea | No attack-origin localization, no uncertainty calibration, limited transfer theory | Could inspire MQTT-specific graph-builder variant in `ba2` |
| 16 | [UniProv (Neurocomputing 2026)](https://www.sciencedirect.com/science/article/pii/S0925231226007423) | Unified pretraining for provenance graph representation learning | Provenance / security graphs | Requires provenance datasets at scale | Self-supervised multi-task pretraining | Provenance graphs with causality-preserving augmentation | Transfer across downstream provenance tasks | Bayesian uncertainty used for task balancing | Reusable graph representations | Very close to `ba2` philosophy on SSL | Still not about IoT source localization under domain shift | Strong design precedent for multi-task SSL plus uncertainty-aware training |
| 17 | [Open-Set Graph Domain Adaptation via Separate Domain Alignment (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/28765) | Open-set graph DA with unknown target classes | General graph DA | Source labeled, target unlabeled | Open-set unsupervised DA | Node-classification graph setting | Yes | Partial, via unknown-class handling | Known/unknown class predictions | Highly relevant to unseen-target conditions | Not temporal security; not provenance ranking | Supports `ba2` open-set or abstention-aware transfer branch |
| 18 | [Graph Domain Adaptation: A Generative View (TKDD 2024)](https://doi.org/10.1145/3631712) | Disentangled graph DA | General graph learning | Source/target distribution shift | Unsupervised DA | Graph classification DA | Yes | Some uncertainty modeling in random latent factors | Domain-invariant representations | Strong conceptual view of semantic/domain/random factors | Not security-specific or temporal | Useful when designing graph-invariant versus domain-specific separation |
| 19 | [Accurate and Scalable Estimation of Epistemic Uncertainty for Graph Neural Networks (ICLR 2024)](https://openreview.net/forum?id=ZL6yd6N1S2) | Better GNN epistemic uncertainty under shift | General graph ML | Standard graph benchmark assumptions | Supervised / pretrained GNNs | General GNN setting | Indirectly, evaluated under shifts | Yes | Calibrated uncertainty, OOD cues | Excellent uncertainty reference | Not a security/provenance system | Strong candidate method family for `ba2` uncertainty module |
| 20 | [Valid Conformal Prediction for Dynamic GNNs (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c0ea9a95e243818f62004e73d57831ca-Abstract-Conference.html) | Conformal uncertainty for dynamic GNNs | Dynamic graph ML | Dynamic graph inference scenarios | Prediction-set calibration assumptions | Dynamic graph setting | Indirect | Yes, explicit conformal validity | Valid prediction sets | Very relevant to abstention / validity under dynamics | Not security-specific; not provenance-aware | Best calibration reference for dynamic-graph abstention in `ba2` |

## High-Level Patterns Emerging from the Matrix

### Provenance papers

They are good at:

- causality
- attribution
- analyst relevance

They are weak at:

- transfer across deployments
- lightweight IoT graph construction
- calibrated abstention

### IoT IDS papers

They are good at:

- dataset benchmarking
- protocol-aware features
- classification accuracy

They are weak at:

- provenance / chronology / origin localization
- transfer rigor
- uncertainty discipline

### Transfer-learning papers

They are good at:

- label scarcity
- domain shift handling
- source-free or open-set reasoning

They are weak at:

- security provenance structure
- actionable early-compromise inference

### Uncertainty papers

They are good at:

- calibration
- prediction validity
- OOD sensitivity

They are weak at:

- integrating structural priors from generative graph assumptions
- security-specific intervention decisions

## Working Conclusion

No single paper we reviewed combines all four pieces that `ba2` needs:

- provenance-style origin reasoning
- IoT graph realism
- transfer/few-shot generalization
- calibrated uncertainty with abstention

That is the main opening for TRANSFERGUARD.
