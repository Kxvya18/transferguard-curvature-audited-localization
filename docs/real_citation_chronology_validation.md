# Real Timestamped Chronology Validation

Dataset: SNAP `cit-HepPh`, downloaded from https://snap.stanford.edu/data/cit-HepPh.html.

Purpose: This experiment closes the missing real timestamped-growth validation gap. It does not claim IoT compromise causality. Instead, it tests whether the structural Boltzmann evidence used by the BA2 framework recovers real node chronology when timestamps are available but hidden at evaluation time.

Protocol:

- Use paper submission dates as the ground-truth arrival order.
- Build chronological prefixes of 1,000, 2,000, 4,000, and 8,000 papers.
- Construct the final graph for each prefix, then evaluate structural scores without giving timestamps to the scoring functions.
- Estimate beta from real citation arrivals with the sequential without-replacement likelihood.
- Report Fisher information, confidence intervals, reliability weight, Kendall tau, Spearman rho, and oldest-paper retrieval precision.

Main result:

- On the largest evaluated prefix, beta is estimated as 1.056 with Fisher information 7155.17 and reliability 0.997.
- The citation-specific Boltzmann preferential-attachment score obtains Kendall tau 0.407 and Spearman rho 0.546 against true paper age.
- The same score retrieves 26.1% of the oldest 5% of papers, compared with 5.3% for the random draw recorded in that run.

Recommended paper claim:

> On a real timestamped citation network, the fitted structural parameter is not merely identifiable in synthetic data: the estimator recovers a preferential-attachment regime close to beta = 1 with high Fisher curvature, and the resulting citation-specific Boltzmann age score recovers real paper chronology substantially above random. This supports the use of Fisher-audited structural evidence as a real-network prior, while remaining distinct from causal attack-source validation.

Limitation:

The dataset validates real growth chronology, not malware-spread chronology. It should be presented as real timestamped network validation and not as a replacement for a future real compromise-cascade benchmark.
