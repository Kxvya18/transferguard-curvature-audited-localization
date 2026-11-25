"""MRR and paired bootstrap differences for the two method pilots (docs/pilot_new_components.md)."""
import json
import sys

import numpy as np

from experiments.benchmark_localization import PROJECT_ROOT

PILOTS = {"pilot_learned_rates": ("MF learned", "MF grid", ["MF oracle"]),
          "pilot_exact_is": ("Exact (particle filter), told", "MF, told", []),
          "pilot_exact_is_e2e": ("Exact (particle filter)", "Mean-field", [])}


def paired(a, b, reps=10000):
    rng = np.random.default_rng(0)
    d = 1 / a - 1 / b
    boot = rng.choice(d, size=(reps, len(d))).mean(axis=1)
    return d.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


for pilot, (new, old, extra) in PILOTS.items():
    for path in sorted((PROJECT_ROOT / "outputs" / pilot).glob("*.jsonl")):
        rows = [json.loads(line) for line in path.open()]
        if not rows:
            continue
        get = lambda k: np.array([r[k] for r in rows], dtype=float)
        line = f"{pilot} {rows[0]['network']} n={len(rows)} " + " ".join(
            f"{k} {np.mean(1 / get(k)):.3f}" for k in [old, new] + extra)
        m, lo, hi = paired(get(new), get(old))
        line += f" | {new} - {old}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        for k in extra:
            m, lo, hi = paired(get(k), get(old))
            line += f" | {k} - {old}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        if "source finite" in rows[0]:
            line += f" | source finite {np.mean([r['source finite'] for r in rows]):.3f}"
        if "Exact seconds" in rows[0]:
            line += f" | exact/MF time {np.median(get('Exact seconds') / get('MF seconds')):.0f}x"
        print(line)
