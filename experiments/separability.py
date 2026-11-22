"""Separability S_min (Chen et al., COLT 2015) of the test-selection problem under mean field.

Hypotheses are full (source, start, rate) triples; a test is one person's report, whose
distribution under each hypothesis comes from mean field. For each person, S is the
smallest squared total-variation distance between two hypotheses that give different report
distributions; S_min is the smallest S over people. Theorem 2 of Chen et al. then holds with
gamma = 7 / S_min. Also reported: the 5% quantile of the pairwise squared distances, which shows
whether S_min comes from a few near-identical hypotheses.

Run:  python -m experiments.separability
"""
import json
import numpy as np
from experiments.benchmark_localization import DATASETS, PROJECT_ROOT
from localization import estimators as est
from localization.mean_field import mean_field_loglik, _start_times, _contact_matrices
from localization.spread import simulate_outbreak

PROBS = (0.1, 0.3, 0.5, 0.7, 0.9)


def report_distributions(ctx, top):
    """dist[h, v, c] for hypotheses h = (start, rate, source in top)."""
    ob, graph = ctx.outbreak, ctx.graph
    n, t_obs, n_times = graph.n_nodes, ob.t_observe, ob.t_observe - ctx.t_lo
    contacts, out = _contact_matrices(ctx), []
    rows = np.arange(len(top))
    for tau in _start_times(ctx, 8):
        begin = np.maximum(tau, graph.join[top])
        for p in PROBS:
            q = np.zeros((len(top), n), dtype=np.float64)
            prev = np.zeros_like(q)
            d = np.zeros((len(top), n, n_times + 1))
            for t in range(int(tau), t_obs):
                q[rows[begin == t], top[begin == t]] = 1.0
                d[:, :, t - ctx.t_lo] = q - prev
                prev = q.copy()
                if contacts[t - ctx.t_lo] is not None:
                    q = 1.0 - (1.0 - q) * np.exp(np.log1p(-p * q) @ contacts[t - ctx.t_lo])
            d[:, :, -1] = 1.0 - prev
            out.append(d[begin <= ob.observed_times[ob.observed_times < t_obs].min()])
    return np.concatenate(out)


def main(count=5, top_k=20):
    result = {}
    for name in DATASETS:
        loader, p, horizon, lookback = DATASETS[name]
        graph = loader(301)
        rng = np.random.default_rng(301)
        values, q05 = [], []
        while len(values) < count:
            ob = simulate_outbreak(graph, rng, p, horizon, 0.1)
            if ob is None:
                continue
            ctx = est.build_context(graph, ob, lookback)
            score = np.logaddexp.reduce(mean_field_loglik(ctx), axis=0)
            top = np.argsort(-np.where(np.isfinite(score), score, -np.inf))[:top_k]
            dist = report_distributions(ctx, top)  # hypotheses x people x classes
            untested = np.setdiff1d(np.arange(graph.n_nodes), ob.observers)
            smallest, quant = np.inf, []
            for v in untested:
                x = dist[:, v, :]
                tv2 = (0.5 * np.abs(x[:, None, :] - x[None, :, :]).sum(-1)) ** 2
                tv2 = tv2[np.triu_indices(len(x), 1)]
                tv2 = tv2[tv2 > 1e-12]
                if len(tv2):
                    smallest = min(smallest, tv2.min())
                    quant.append(np.quantile(tv2, 0.05))
            values.append(smallest)
            q05.append(float(np.median(quant)))
        result[name] = {"S_min (median over outbreaks)": float(np.median(values)), "gamma = 7 / S_min": float(7 / np.median(values)),
                        "median 5% quantile of squared TV": float(np.median(q05))}
        print(name, result[name], flush=True)
        (PROJECT_ROOT / "outputs" / "separability.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
