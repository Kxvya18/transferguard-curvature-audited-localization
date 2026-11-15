"""Baseline source estimators that need no training.

Every estimator sees only what a defender would have: the contact log up to the
observation time, the observer set, and each observer's infection time (or the
fact that it is still healthy). None of them sees the true source or start time.

Each returns one score per node; higher means more likely to be the source.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.special import logsumexp

from .spread import Outbreak, earliest_arrival, spread
from .temporal_graph import TemporalGraph

INFEASIBLE = -1e12


@dataclass
class Context:
    """Quantities shared by several estimators for one outbreak."""

    graph: TemporalGraph
    outbreak: Outbreak
    t_lo: int  # earliest start time the estimators consider
    static: nx.Graph  # union of contacts in [t_lo, t_observe)
    hop: np.ndarray  # static hop distance, shape (n_nodes, n_infected_observers)

    @property
    def infected(self) -> np.ndarray:
        return self.outbreak.infected_observers

    @property
    def infected_times(self) -> np.ndarray:
        ob = self.outbreak
        return ob.observed_times[ob.observed_times < ob.t_observe]


def build_context(graph: TemporalGraph, outbreak: Outbreak, max_lookback: int) -> Context:
    t_lo = max(outbreak.t_observe - max_lookback, 0)
    static = graph.static_graph(t_lo, outbreak.t_observe)
    infected = outbreak.infected_observers
    hop = np.full((graph.n_nodes, len(infected)), np.inf)
    for j, obs in enumerate(infected):
        for node, dist in nx.single_source_shortest_path_length(static, int(obs)).items():
            hop[node, j] = dist
    return Context(graph, outbreak, t_lo, static, hop)


def _feasible(ctx: Context) -> np.ndarray:
    """A source must have joined no later than the first observed infection."""
    return ctx.graph.join <= ctx.infected_times.min()


# ---------------------------------------------------------------- static baselines


def jordan_center(ctx: Context) -> np.ndarray:
    """Minimise the largest hop distance to an infected observer."""
    score = -ctx.hop.max(axis=1)
    return np.where(np.isfinite(score), score, INFEASIBLE)


def distance_center(ctx: Context) -> np.ndarray:
    """Minimise the total hop distance to the infected observers."""
    score = -ctx.hop.sum(axis=1)
    return np.where(np.isfinite(score), score, INFEASIBLE)


def _time_of_arrival_energy(delays: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Least-squares fit of observed times as start + speed * delay, per candidate.

    Returns the residual sum of squares; np.inf where an infected observer is
    unreachable from the candidate.
    """
    energy = np.full(len(delays), np.inf)
    for s, d in enumerate(delays):
        if not np.all(np.isfinite(d)):
            continue
        design = np.column_stack([np.ones_like(d), d])
        coef, *_ = np.linalg.lstsq(design, times, rcond=None)
        if coef[1] < 0:  # infection cannot arrive earlier at farther observers
            coef = np.array([times.mean(), 0.0])
        energy[s] = float(np.sum((times - design @ coef) ** 2))
    return energy


def pinto_style(ctx: Context) -> np.ndarray:
    """Time-of-arrival fit on static hop distances (after Pinto et al., 2012)."""
    energy = _time_of_arrival_energy(ctx.hop, ctx.infected_times)
    return np.where(np.isfinite(energy), -energy, INFEASIBLE)


def lpsi(ctx: Context, alpha: float = 0.5, iterations: int = 60) -> np.ndarray:
    """Label propagation of observer states (LPSI, Wang et al., 2017)."""
    ob = ctx.outbreak
    labels = np.zeros(ctx.graph.n_nodes)
    labels[ob.observers] = np.where(ob.observed_times < ob.t_observe, 1.0, -1.0)
    adjacency = nx.to_scipy_sparse_array(ctx.static, nodelist=range(ctx.graph.n_nodes), format="csr")
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(np.maximum(degree, 1e-12)), 0.0)
    score = labels.copy()
    for _ in range(iterations):
        score = alpha * inv_sqrt * (adjacency @ (inv_sqrt * score)) + (1 - alpha) * labels
    return score


# ---------------------------------------------------------------- time-respecting and simulation baselines


def temporal_screen(ctx: Context, n_start_times: int = 8) -> np.ndarray:
    """Time-of-arrival fit on time-respecting arrival delays.

    Same fit as `pinto_style`, but the delay from a candidate to an observer is
    the earliest arrival along contacts that actually happened in time order.
    In a temporal network the paths that exist depend on when the outbreak
    starts, so the fit is repeated for several start times between t_lo and the
    first observed infection, and each candidate keeps its best fit.
    """
    ob = ctx.outbreak
    first_infection = int(ctx.infected_times.min())
    starts = np.unique(np.linspace(ctx.t_lo, max(first_infection, ctx.t_lo), n_start_times).round().astype(int))
    best = np.full(ctx.graph.n_nodes, np.inf)
    for tau in starts:
        arrival = earliest_arrival(ctx.graph, int(tau), ob.t_observe)
        begin = np.maximum(tau, ctx.graph.join)
        delays = arrival[:, ctx.infected] - begin[:, None]
        best = np.minimum(best, _time_of_arrival_energy(delays, ctx.infected_times))
    score = np.where(np.isfinite(best), -best, INFEASIBLE)
    return np.where(_feasible(ctx), score, INFEASIBLE)


@dataclass
class Posterior:
    score: np.ndarray  # ranking score for every node
    probability: np.ndarray  # posterior over the screened candidates (0 elsewhere)
    temperature: float
    candidates: np.ndarray


def _top_candidates(screen: np.ndarray, k: int) -> np.ndarray:
    order = np.argsort(-screen)[:k]
    return order[screen[order] > INFEASIBLE]


def _simulate_from(
    ctx: Context,
    rng: np.random.Generator,
    candidates: np.ndarray,
    prob_range: tuple[float, float],
    n_simulations: int,
) -> np.ndarray:
    """Observer infection times simulated from each candidate: shape (K, S, observers).

    Start time is uniform over the candidate's feasible range (from when it
    could have started up to the first observed infection, inclusive) and the
    transmission probability is uniform over `prob_range`. Observers still
    healthy at t_observe get the value t_observe.
    """
    ob = ctx.outbreak
    first_infection = int(ctx.infected_times.min())
    out = np.empty((len(candidates), n_simulations, len(ob.observers)))
    for row, cand in enumerate(candidates):
        earliest = max(ctx.t_lo, int(ctx.graph.join[cand]))
        starts = rng.integers(earliest, max(first_infection, earliest) + 1, size=n_simulations)
        init = np.full((n_simulations, ctx.graph.n_nodes), np.inf)
        init[np.arange(n_simulations), cand] = starts
        probs = rng.uniform(*prob_range, size=n_simulations)
        times = spread(ctx.graph, init, int(starts.min()), ob.t_observe, probs, rng)
        out[row] = np.minimum(times[:, ob.observers], ob.t_observe)
    return out


def _rank_with_posterior(screen: np.ndarray, candidates: np.ndarray, log_post: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Screened candidates are ranked by posterior, above everyone else; the
    rest keep their screening order (screening scores are <= 0)."""
    probability = np.zeros(n)
    probability[candidates] = np.exp(log_post)
    score = screen.copy()
    score[candidates] = 1e6 + log_post
    return score, probability


def sme_posterior(
    ctx: Context,
    rng: np.random.Generator,
    screen: np.ndarray,
    prob_range: tuple[float, float] = (0.1, 0.9),
    n_candidates: int = 50,
    n_simulations: int = 100,
    width: float = 0.25,
) -> Posterior:
    """Soft-margin estimator (Antulov-Fantulin et al., PRL 2015), adapted to observers.

    Uses only WHICH observers are infected, not when: each simulation is scored
    by exp(-(1 - J)^2 / width^2), where J is the Jaccard similarity between the
    simulated and observed sets of infected observers.
    """
    ob = ctx.outbreak
    candidates = _top_candidates(screen, n_candidates)
    simulated = _simulate_from(ctx, rng, candidates, prob_range, n_simulations)
    observed = ob.observed_times < ob.t_observe
    sim_inf = simulated < ob.t_observe
    inter = (sim_inf & observed).sum(axis=2)
    union = np.maximum((sim_inf | observed).sum(axis=2), 1)
    jaccard = inter / union
    likelihood = np.exp(-((1.0 - jaccard) ** 2) / width**2).mean(axis=1)
    log_post = np.log(np.maximum(likelihood, 1e-300))
    log_post -= logsumexp(log_post)
    score, probability = _rank_with_posterior(screen, candidates, log_post, ctx.graph.n_nodes)
    return Posterior(score, probability, width, candidates)


def boltzmann_posterior(
    ctx: Context,
    rng: np.random.Generator,
    prob_range: tuple[float, float] = (0.1, 0.9),
    n_candidates: int = 50,
    n_simulations: int = 100,
    screen: np.ndarray | None = None,
    min_ess: float = 10.0,
) -> Posterior:
    """Simulation posterior with a Boltzmann kernel over the screened candidates.

    For each candidate s, simulate outbreaks from s with a start time drawn
    uniformly from its feasible range and a transmission probability drawn
    uniformly from `prob_range` (the true value is not given to the method).
    Each simulation's energy is
    E = sum over ALL observers of |simulated time - observed time| (healthy
    observers count as infected at t_observe). The posterior is

        p(s | y) ∝ mean_i exp(-E_si / T),

    with the temperature T chosen by maximum marginal likelihood under a
    Laplace observation model with scale T per observer.

    `screen` defaults to `temporal_screen`; passing another screening score
    gives the ablation without time-respecting screening.
    """
    ob = ctx.outbreak
    if screen is None:
        screen = temporal_screen(ctx)
    candidates = _top_candidates(screen, n_candidates)
    n_obs = len(ob.observers)
    simulated = _simulate_from(ctx, rng, candidates, prob_range, n_simulations)
    energies = np.abs(simulated - ob.observed_times[None, None, :]).sum(axis=2)

    temperatures = np.geomspace(0.02, 50.0, 100)
    best_t, best_ll = temperatures[0], -np.inf
    for temp in temperatures:
        per_candidate = logsumexp(-energies / temp, axis=1) - np.log(n_simulations)
        loglik = logsumexp(per_candidate) - np.log(len(candidates)) - n_obs * np.log(2 * temp)
        if loglik > best_ll:
            best_t, best_ll = temp, loglik

    # Monte Carlo floor: with a finite number of simulations the likelihood
    # estimate collapses onto one lucky run when T is too small. Raise T until
    # the best candidate's kernel weights have an effective sample size of at
    # least `min_ess` simulations.
    best_row = int(np.argmin(energies.min(axis=1)))
    for temp in temperatures[temperatures >= best_t]:
        w = np.exp(-(energies[best_row] - energies[best_row].min()) / temp)
        if w.sum() ** 2 / np.sum(w**2) >= min_ess:
            best_t = temp
            break

    log_post = logsumexp(-energies / best_t, axis=1)
    log_post -= logsumexp(log_post)
    score, probability = _rank_with_posterior(screen, candidates, log_post, ctx.graph.n_nodes)
    return Posterior(score, probability, float(best_t), candidates)
