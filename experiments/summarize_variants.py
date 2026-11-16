"""Print paired comparison of variants per dataset and rate.  python -m experiments.summarize_variants filtered"""

import glob
import sys

import numpy as np
import pandas as pd

from experiments.benchmark_localization import PROJECT_ROOT

d = pd.concat([pd.read_json(f, lines=True) for f in glob.glob(str(PROJECT_ROOT / "outputs" / f"variants_{sys.argv[1]}" / "*.jsonl"))])
methods = [c for c in d.columns if c not in ("dataset", "test_rate") and not c.endswith("seconds")]
base, new = methods[0], methods[-1]
for (ds, rate), g in d.groupby(["dataset", "test_rate"]):
    diff = 1 / g[new] - 1 / g[base]
    half = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("nan")
    cells = " | ".join(f"{m}={np.mean(1/g[m]):.3f} top1={np.mean(g[m]<=1):.2f}" for m in methods)
    print(f"{ds:15s} {rate:5s} n={len(g):3d} {cells} | diff {diff.mean():+.3f}±{half:.3f} | "
          + " ".join(f"{g[m + ' seconds'].mean():.1f}s" for m in methods))
