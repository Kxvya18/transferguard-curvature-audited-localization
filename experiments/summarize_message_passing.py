"""Print mean field vs message passing per dataset and test rate from outputs/message_passing_pilot."""

import glob

import numpy as np
import pandas as pd

from experiments.benchmark_localization import PROJECT_ROOT

d = pd.concat([pd.read_json(f, lines=True) for f in glob.glob(str(PROJECT_ROOT / "outputs" / "message_passing_pilot" / "*.jsonl"))])
for (ds, rate), g in d.groupby(["dataset", "test_rate"]):
    a, b = g["Mean-field"], g["Message passing"]
    diff = 1 / b - 1 / a
    half = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("nan")
    print(f"{ds:15s} {rate:5s} n={len(g):3d} MF={np.mean(1/a):.3f} top1={np.mean(a<=1):.2f} | "
          f"MP={np.mean(1/b):.3f} top1={np.mean(b<=1):.2f} | MP-MF {diff.mean():+.3f}±{half:.3f} | "
          f"{g['Mean-field seconds'].mean():.1f}s vs {g['Message passing seconds'].mean():.1f}s")
