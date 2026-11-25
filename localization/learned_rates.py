"""Mean-field likelihood with one infectiousness per person, written in torch so that the
infectiousness can be learned by gradient ascent from past outbreaks whose sources are unknown.

Same recursion as `mean_field.mean_field_loglik` (SI, unweighted contacts, start time summed
over the same grid), except that person u passes the infection along a contact with
probability rate[u] instead of one shared p:

    q[v](t + 1) = 1 - (1 - q[v](t)) * prod over contacts u~v in window t of (1 - rate[u] q[u](t)).
"""

from __future__ import annotations

import numpy as np
import torch

from .estimators import INFEASIBLE
from .mean_field import _start_times
from .spread import earliest_arrival


class Prepared:
    """Everything about one outbreak that does not depend on the rates."""

    def __init__(self, ctx, n_start_times: int = 8):
        ob, graph = ctx.outbreak, ctx.graph
        self.n, self.t_lo, self.t_obs = graph.n_nodes, ctx.t_lo, ob.t_observe
        y = ob.observed_times.astype(int)
        infected = y < self.t_obs
        first = int(y[infected].min())
        arrival = earliest_arrival(graph, ctx.t_lo, self.t_obs)
        self.possible = np.flatnonzero((graph.join <= first) & np.all(arrival[:, ob.observers[infected]] <= y[infected], axis=1))
        starts = _start_times(ctx, n_start_times)
        # one row per (start, candidate); a candidate cannot start before it joins or after the first report
        begin = np.maximum(starts[:, None], graph.join[self.possible][None, :])
        self.valid = torch.tensor((begin <= first).ravel())
        self.begin = begin.ravel()
        self.cand = np.tile(self.possible, len(starts))
        self.n_starts = len(starts)
        self.observers = torch.tensor(ob.observers)
        self.y = torch.tensor(y)
        self.infected = torch.tensor(infected)
        self.t0 = int(starts.min())
        self.contacts = {}
        for t in range(self.t0, self.t_obs):
            c = graph.windows[t]
            if len(c):
                pairs = np.unique(np.sort(c, axis=1), axis=0)
                a = torch.zeros(self.n, self.n)
                a[pairs[:, 0], pairs[:, 1]] = 1.0
                a[pairs[:, 1], pairs[:, 0]] = 1.0
                self.contacts[t] = a


def loglik(prep: Prepared, rate: torch.Tensor, floor: float = 1e-4) -> torch.Tensor:
    """Log-likelihood of the reports, shape (n_starts, n_candidates), for per-person rates."""
    rows = prep.begin.shape[0]
    q = torch.zeros(rows, prep.n)
    at = torch.zeros(rows, len(prep.y))
    before = torch.zeros(rows, len(prep.y))
    healthy = None
    index = torch.arange(rows)
    for t in range(prep.t0, prep.t_obs + 1):
        live = torch.tensor(prep.begin == t)
        if live.any():
            q = q.index_put((index[live], torch.tensor(prep.cand)[live]), torch.ones(int(live.sum())))
        current = q[:, prep.observers]
        at = torch.where(prep.y[None, :] == t, current, at)
        before = torch.where(prep.y[None, :] - 1 == t, current, before)
        if t == prep.t_obs - 1:
            healthy = 1.0 - current
        if t == prep.t_obs:
            break
        a = prep.contacts.get(t)
        if a is not None:
            log_escape = torch.log1p(-(rate[None, :] * q).clamp(max=1 - 1e-6)) @ a
            q = 1.0 - (1.0 - q) * torch.exp(log_escape)
    like = torch.where(prep.infected[None, :], at - before, healthy)
    out = torch.log(like.clamp(min=floor)).sum(dim=1)
    out = torch.where(prep.valid, out, torch.full_like(out, -torch.inf))
    return out.reshape(prep.n_starts, -1)


def scores(prep: Prepared, rate: torch.Tensor) -> np.ndarray:
    """One score per node: start time summed out (same convention as `mean_field_scores`)."""
    with torch.no_grad():
        s = torch.logsumexp(loglik(prep, rate), dim=0).numpy()
    out = np.full(prep.n, INFEASIBLE)
    out[prep.possible] = np.where(np.isfinite(s), s, INFEASIBLE)
    return out


def evidence(prep: Prepared, rate: torch.Tensor) -> torch.Tensor:
    """log P(reports), source and start summed out under a uniform prior (up to a constant)."""
    return torch.logsumexp(loglik(prep, rate).ravel(), dim=0)


def fit(preps: list[Prepared], n: int, p0: float, steps: int = 150, penalty: float = 1.0, lr: float = 0.05,
        log=None) -> torch.Tensor:
    """Learn rate[u] = sigmoid(b + a[u]) from past outbreaks by maximising the summed evidence
    minus penalty * sum(a ** 2) / 2 (a Gaussian prior: most people near the shared rate)."""
    b = torch.tensor(float(np.log(p0 / (1 - p0))), requires_grad=True)
    a = torch.zeros(n, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=lr)
    for step in range(steps):
        opt.zero_grad()
        rate = torch.sigmoid(b + a)
        total = sum(evidence(p, rate) for p in preps)
        loss = -(total - 0.5 * penalty * (a ** 2).sum()) / len(preps)
        loss.backward()
        opt.step()
        if log is not None and step % 10 == 0:
            log(step, float(-loss), float(torch.sigmoid(b)))
    return torch.sigmoid(b + a).detach()
