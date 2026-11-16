"""Print mean field vs message passing vs belief propagation from outputs/bp_comparison."""

import glob

import numpy as np
import pandas as pd

from experiments.benchmark_localization import PROJECT_ROOT

d = pd.concat([pd.read_json(f, lines=True) for f in glob.glob(str(PROJECT_ROOT / "outputs" / "bp_comparison" / "*.jsonl"))])
methods = ["Mean-field", "Message passing", "Belief propagation"]
for (ds, rate), g in d.groupby(["dataset", "test_rate"]):
    diff = 1 / g["Mean-field"] - 1 / g["Belief propagation"]
    half = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("nan")
    print(f"{ds:15s} {rate:5s} n={len(g):3d} " + " | ".join(f"{m}={np.mean(1/g[m]):.3f}" for m in methods)
          + f" | MF-BP {diff.mean():+.3f}±{half:.3f} | " + " ".join(f"{g[m + ' seconds'].mean():.1f}s" for m in methods))
