"""Learned source localizers.

All models are trained on outbreaks simulated on the same network as the test
outbreaks (different sources and start times), which is the usual protocol for
learned localizers and the most favourable setting for them.

- GCNSI (Dong et al., CIKM 2019): a GCN over the static contact graph whose
  inputs are observer states and their label-propagation smoothings.
- Temporal GCN: one GCN layer per time bin (up to 8 bins over the lookback),
  chained by a GRU, so the model sees when contacts happened as well as where.
- Arrival GNN (the proposed model): a residual GCN whose inputs add, for every
  node, how well "this node started it" explains the observers along
  time-respecting paths (see `arrival_features`).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch
from torch import nn

from .estimators import Context, lpsi


def _normalized_adjacency(graph: nx.Graph, n_nodes: int) -> torch.Tensor:
    """D^-1/2 (A + I) D^-1/2 as a sparse tensor."""
    edges = np.asarray(list(graph.edges()), dtype=np.int64).reshape(-1, 2)
    return _normalized_from_pairs(edges, n_nodes)


def _normalized_from_pairs(pairs: np.ndarray, n_nodes: int) -> torch.Tensor:
    loops = np.arange(n_nodes)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1], loops])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0], loops])
    degree = np.bincount(rows, minlength=n_nodes).astype(np.float32)
    values = 1.0 / np.sqrt(degree[rows] * degree[cols])
    index = torch.as_tensor(np.vstack([rows, cols]))
    return torch.sparse_coo_tensor(index, torch.as_tensor(values, dtype=torch.float32), (n_nodes, n_nodes)).coalesce()


def node_features(ctx: Context) -> torch.Tensor:
    """Per-node inputs shared by both learned models.

    Columns: is observer, observed infected, observed healthy, how early the
    observer was infected (0 to 1), and LPSI scores at two smoothing levels.
    """
    ob = ctx.outbreak
    n = ctx.graph.n_nodes
    lookback = max(ob.t_observe - ctx.t_lo, 1)
    infected = ob.observed_times < ob.t_observe
    feats = np.zeros((n, 6), dtype=np.float32)
    feats[ob.observers, 0] = 1.0
    feats[ob.observers[infected], 1] = 1.0
    feats[ob.observers[~infected], 2] = 1.0
    feats[ob.observers[infected], 3] = (ob.t_observe - ob.observed_times[infected]) / lookback
    feats[:, 4] = lpsi(ctx, alpha=0.5)
    feats[:, 5] = lpsi(ctx, alpha=0.9)
    return torch.as_tensor(feats)


def time_bins(ctx: Context, max_bins: int = 8) -> list[tuple[int, int]]:
    """Split [t_lo, t_observe) into at most `max_bins` consecutive bins."""
    edges = np.unique(np.linspace(ctx.t_lo, ctx.outbreak.t_observe, max_bins + 1).round().astype(int))
    return list(zip(edges[:-1], edges[1:]))


def bin_states(ctx: Context, bins: list[tuple[int, int]]) -> torch.Tensor:
    """Observer states at the end of each bin: shape (bins, nodes, 3).

    Columns: is observer, observed infected by then, observed still healthy then.
    """
    ob = ctx.outbreak
    times = ob.observed_times
    states = np.zeros((len(bins), ctx.graph.n_nodes, 3), dtype=np.float32)
    for i, (_, end) in enumerate(bins):
        states[i, ob.observers, 0] = 1.0
        states[i, ob.observers, 1] = (times < end).astype(np.float32)
        states[i, ob.observers, 2] = (times >= end).astype(np.float32)
    return torch.as_tensor(states)


class GCNSI(nn.Module):
    def __init__(self, in_dim: int = 6, hidden: int = 64, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.out = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, adjacency: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.dropout(torch.relu(torch.sparse.mm(adjacency, layer(x))))
        return self.out(x).squeeze(-1)


class TemporalGCN(nn.Module):
    def __init__(self, in_dim: int = 3, static_dim: int = 6, hidden: int = 64):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.static = nn.Linear(static_dim, hidden)
        self.out = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, windows: list[torch.Tensor], states: torch.Tensor, static_x: torch.Tensor) -> torch.Tensor:
        h = torch.zeros(states.shape[1], self.gru.hidden_size)
        for adjacency, x in zip(windows, states):
            message = torch.relu(torch.sparse.mm(adjacency, self.embed(x)))
            h = self.gru(message, h)
        return self.out(torch.cat([h, torch.relu(self.static(static_x))], dim=-1)).squeeze(-1)


def _gcnsi_inputs(ctx: Context):
    return _normalized_adjacency(ctx.static, ctx.graph.n_nodes), node_features(ctx)


def _temporal_inputs(ctx: Context):
    n = ctx.graph.n_nodes
    bins = time_bins(ctx)
    adjacency = []
    for start, end in bins:
        pairs = [ctx.graph.windows[t].reshape(-1, 2) for t in range(start, end)]
        adjacency.append(_normalized_from_pairs(np.unique(np.concatenate(pairs), axis=0) if pairs else np.zeros((0, 2), dtype=np.int64), n))
    return adjacency, bin_states(ctx, bins), node_features(ctx)


def _arrival_inputs(ctx: Context):
    from .arrival_features import arrival_features

    adjacency, static_x = _gcnsi_inputs(ctx)
    x = torch.cat([static_x, torch.as_tensor(arrival_features(ctx))], dim=1)
    return adjacency, x


class ArrivalGNN(nn.Module):
    """GCN over the contact graph whose inputs include time-respecting arrival features."""

    def __init__(self, in_dim: int = 14, hidden: int = 64, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.out = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, adjacency: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        own = torch.relu(self.encode(x))
        h = own
        for layer in self.layers:
            h = h + self.dropout(torch.relu(torch.sparse.mm(adjacency, layer(h))))  # residual
        return self.out(torch.cat([own, h], dim=-1)).squeeze(-1)


MODELS = {
    "GCNSI": (GCNSI, _gcnsi_inputs),
    "Temporal GCN": (TemporalGCN, _temporal_inputs),
    "Arrival GNN": (ArrivalGNN, _arrival_inputs),
}


def train_model(
    name: str,
    train_contexts: list[Context],
    seed: int,
    epochs: int = 30,
    lr: float = 5e-3,
) -> nn.Module:
    """Train with cross-entropy over nodes, one outbreak per step."""
    torch.manual_seed(seed)
    model_cls, make_inputs = MODELS[name]
    model = model_cls()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    data = [(make_inputs(ctx), ctx.outbreak.source) for ctx in train_contexts]
    order = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        for idx in order.permutation(len(data)):
            inputs, source = data[idx]
            logits = model(*inputs)
            loss = nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([source]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def predict(name: str, model: nn.Module, ctx: Context) -> np.ndarray:
    _, make_inputs = MODELS[name]
    return model(*make_inputs(ctx)).numpy().astype(float)
