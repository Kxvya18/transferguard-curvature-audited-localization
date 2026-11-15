"""Discrete-time evolving graphs: loaders for real contact logs and a growing generator.

A `TemporalGraph` is a list of time windows. Each window holds the undirected
contacts that happened in it, as an array of node pairs. A node "joins" at the
first window in which it has a contact; before that it cannot spread or catch
anything.
"""

from __future__ import annotations

import gzip
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np


@dataclass
class TemporalGraph:
    name: str
    n_nodes: int
    windows: list[np.ndarray]  # windows[t] has shape (k_t, 2)
    join: np.ndarray  # first window with a contact, per node

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    def static_graph(self, t_start: int, t_end: int) -> nx.Graph:
        """Union of all contacts in windows t_start .. t_end - 1."""
        graph = nx.Graph()
        graph.add_nodes_from(range(self.n_nodes))
        for t in range(max(t_start, 0), min(t_end, self.n_windows)):
            graph.add_edges_from(map(tuple, self.windows[t]))
        return graph


def _join_times(n_nodes: int, windows: list[np.ndarray]) -> np.ndarray:
    join = np.full(n_nodes, len(windows), dtype=int)
    for t, contacts in enumerate(windows):
        if len(contacts):
            nodes = contacts.ravel()
            join[nodes] = np.minimum(join[nodes], t)
    return join


def _bin_contacts(data: np.ndarray, name: str, window_seconds: int, drop_empty: bool) -> TemporalGraph:
    """Turn rows (src, dst, time) into a TemporalGraph with fixed-length windows.

    With `drop_empty`, windows without any contact (nights, weekends) are removed,
    so time counts only periods in which people actually met.
    """
    data = data[data[:, 0] != data[:, 1]]
    ids, relabeled = np.unique(data[:, :2], return_inverse=True)
    pairs = relabeled.reshape(-1, 2)
    window = (data[:, 2] - data[:, 2].min()) // window_seconds
    if drop_empty:
        _, window = np.unique(window, return_inverse=True)
    windows = []
    for t in range(int(window.max()) + 1):
        contacts = np.unique(np.sort(pairs[window == t], axis=1), axis=0)
        windows.append(contacts.astype(np.int64))
    return TemporalGraph(name, len(ids), windows, _join_times(len(ids), windows))


def load_snap_temporal(path: Path, name: str, window_seconds: int) -> TemporalGraph:
    """Read a SNAP 'src dst unix_time' file and bin contacts into windows."""
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            src, dst, stamp = line.split()[:3]
            rows.append((int(src), int(dst), int(stamp)))
    return _bin_contacts(np.asarray(rows, dtype=np.int64), name, window_seconds, drop_empty=False)


def load_netzschleuder_contacts(path: Path, name: str, window_seconds: int) -> TemporalGraph:
    """Read a Netzschleuder CSV zip ('source, target, time' in edges.csv), e.g. the
    SocioPatterns face-to-face contact data, dropping windows without contacts."""
    raw = zipfile.ZipFile(path).read("edges.csv").decode("utf-8")
    data = np.loadtxt(io.StringIO(raw), delimiter=",", comments="#", dtype=np.int64).reshape(-1, 3)
    return _bin_contacts(data, name, window_seconds, drop_empty=True)


def growing_network(
    n_nodes: int,
    n_windows: int,
    rng: np.random.Generator,
    edges_per_node: int = 2,
    contact_prob: float = 0.35,
    seed_nodes: int = 20,
) -> TemporalGraph:
    """Preferential-attachment growth with intermittent contacts.

    Nodes arrive at an even rate and attach to `edges_per_node` existing nodes
    with probability proportional to degree + 1. Every existing tie is active as
    a contact in a window with probability `contact_prob`.
    """
    arrival = np.zeros(n_nodes, dtype=int)
    degree = np.zeros(n_nodes)
    ties = []
    for node in range(1, seed_nodes):
        other = int(rng.integers(node))
        ties.append((node, other))
        degree[[node, other]] += 1
    rate = (n_nodes - seed_nodes) / n_windows
    for node in range(seed_nodes, n_nodes):
        arrival[node] = int((node - seed_nodes) / rate)
        weights = degree[:node] + 1.0
        targets = rng.choice(node, size=edges_per_node, replace=False, p=weights / weights.sum())
        for other in targets:
            ties.append((node, int(other)))
            degree[[node, other]] += 1
    ties = np.asarray(ties, dtype=np.int64)
    born = np.maximum(arrival[ties[:, 0]], arrival[ties[:, 1]])
    windows = []
    for t in range(n_windows):
        active = (born <= t) & (rng.random(len(ties)) < contact_prob)
        windows.append(ties[active])
    return TemporalGraph("Growing PA", n_nodes, windows, _join_times(n_nodes, windows))
