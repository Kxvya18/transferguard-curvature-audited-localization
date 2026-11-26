"""Numerical checks of every statement in docs/theory.md (Lemma 1, Theorem 1 (a)-(c), Corollary 1).

Each check either passes on every random instance or prints the first counterexample.
Run:  python -m experiments.verify_theory
"""
import itertools

import numpy as np

from localization.mean_field import mean_field_marginals
from localization.temporal_graph import TemporalGraph, _join_times

rng = np.random.default_rng(0)
TOL = 1e-12


def tv(a, b):
    return 0.5 * np.abs(a - b).sum()


def random_dist(k, sparse=False):
    x = rng.exponential(size=k) ** (3 if sparse else 1)
    return x / x.sum()


def reciprocal_rank_reward(p, q):
    """Expected reciprocal rank of the true source s ~ p when people are ranked by q (ties broken by index)."""
    order = np.argsort(-q, kind="stable")
    rank = np.empty(len(q))
    rank[order] = np.arange(1, len(q) + 1)
    return np.sum(p / rank)


def check_theorem(trials=100_000):
    worst_a = worst_b = worst_c = -np.inf
    for _ in range(trials):
        G, n = rng.integers(1, 12), rng.integers(2, 15)
        W = random_dist(G, sparse=True)  # exact weights P(g | O)
        pi = np.array([random_dist(n, sparse=True) for _ in range(G)])  # exact P(s | g, O)
        eps = rng.normal(scale=rng.choice([0.1, 1, 5]), size=G)  # log-evidence errors
        if rng.random() < 0.3:
            eps[rng.integers(G)] += 20  # one broken grid point, as with non-converged BP
        W_t = W * np.exp(eps - eps.max())
        W_t /= W_t.sum()
        pi_t = np.array([random_dist(n, sparse=True) if rng.random() < 0.5 else pi[g] for g in range(G)])
        P, P_t = W @ pi, W_t @ pi_t
        # (a) decomposition bound
        bound_a = tv(W, W_t) + np.sum(W * np.array([tv(pi[g], pi_t[g]) for g in range(G)]))
        worst_a = max(worst_a, tv(P, P_t) - bound_a)
        # (b) evidence error -> weight error
        delta = eps.max() - eps.min()
        worst_b = max(worst_b, tv(W, W_t) - np.tanh(delta / 4))
        # (c) any rank-based reward in [0, 1] loses at most 2 TV against ranking by the exact posterior
        worst_c = max(worst_c, (reciprocal_rank_reward(P, P) - reciprocal_rank_reward(P, P_t)) - 2 * tv(P, P_t))
        worst_c = max(worst_c, (P.max() - P[np.argmax(P_t)]) - 2 * tv(P, P_t))  # top-1
    print(f"Theorem 1 (a): max(TV - bound) = {worst_a:.2e}  {'PASS' if worst_a <= TOL else 'FAIL'}")
    print(f"Theorem 1 (b): max(TV(W, W~) - tanh(D/4)) = {worst_b:.2e}  {'PASS' if worst_b <= TOL else 'FAIL'}")
    print(f"Theorem 1 (c): max(loss - 2 TV) = {worst_c:.2e}  {'PASS' if worst_c <= TOL else 'FAIL'}")
    # (b) is tight: two grid points, W = (1/(1+k^(1/2)), ...) with errors (0, D)
    for D in [0.5, 2.0, 8.0]:
        k = np.exp(D)
        W = np.array([np.sqrt(k), 1.0]) / (np.sqrt(k) + 1)  # chosen so the bound is attained
        W_t = W * np.exp([0.0, D])
        W_t /= W_t.sum()
        print(f"  tightness D={D}: TV {tv(W, W_t):.6f} vs tanh(D/4) {np.tanh(D / 4):.6f}")


def exact_marginals(graph, source, t_begin, t_end, p):
    """Exact P(v infected by t) for SI with one independent coin per contact, by enumerating all coins."""
    contacts = [(t, int(u), int(v)) for t in range(t_begin, t_end) for u, v in graph.windows[t]]
    out = np.zeros((t_end - t_begin + 1, graph.n_nodes))
    for coins in itertools.product([0, 1], repeat=len(contacts)):
        weight = np.prod([p if c else 1 - p for c in coins])
        infected = np.zeros(graph.n_nodes, dtype=bool)
        infected[source] = True
        out[0] += weight * infected
        k = 0
        for t in range(t_begin, t_end):
            now = infected.copy()  # one hop per window: only those infected by t transmit
            while k < len(contacts) and contacts[k][0] == t:
                _, u, v = contacts[k]
                if coins[k]:
                    if now[u]:
                        infected[v] = True
                    if now[v]:
                        infected[u] = True
                k += 1
            out[t - t_begin + 1] += weight * infected
    return out


def check_lemma1(graphs=300):
    worst = -np.inf
    for _ in range(graphs):
        n, T = rng.integers(3, 7), rng.integers(2, 6)
        windows = []
        for _ in range(T):
            pairs = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < 0.35]
            windows.append(np.array(pairs, dtype=np.int64).reshape(-1, 2))
        if sum(len(w) for w in windows) > 14:
            continue
        graph = TemporalGraph("tiny", n, windows, _join_times(n, windows))
        p = rng.choice([0.1, 0.3, 0.5, 0.9])
        source = int(rng.integers(n))
        exact = exact_marginals(graph, source, 0, T, p)
        q = mean_field_marginals(graph, source, 0, T, p)
        worst = max(worst, np.max(exact - q))
    print(f"Lemma 1: max(exact - mean field) = {worst:.2e}  {'PASS' if worst <= 1e-12 else 'FAIL'}")


if __name__ == "__main__":
    check_lemma1()
    check_theorem()
