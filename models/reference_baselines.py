"""Architecture-matched static and temporal-attention reference baselines."""

from __future__ import annotations

import math

import torch
from torch import nn

from self_supervised.temporal_encoder import GraphMessageLayer, _normalize_adjacency

try:
    from tgm.nn.encoder.tgn import (
        GraphAttentionEmbedding,
        IdentityMessage,
        LastAggregator,
        TGNMemory,
    )
except ImportError:  # pragma: no cover - exercised only without journal extras
    GraphAttentionEmbedding = IdentityMessage = LastAggregator = TGNMemory = None


def _masked_output(logits: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    logits = logits.masked_fill(mask <= 0, float("-inf"))
    return {"logits": logits, "probs": torch.softmax(logits, dim=-1)}


class StaticGCNOriginBaseline(nn.Module):
    """Static GCN over time-aggregated observable features."""

    def __init__(self, node_feature_dim: int, structural_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.input = nn.Linear(node_feature_dim + structural_dim, hidden_dim)
        self.layers = nn.ModuleList([GraphMessageLayer(hidden_dim, hidden_dim) for _ in range(3)])
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, node_features, adjacency, structural_features, node_mask):
        dynamic = node_features.mean(dim=1)
        hidden = self.input(torch.cat([dynamic, structural_features], dim=-1))
        norm = _normalize_adjacency(adjacency)
        for layer in self.layers:
            hidden = layer(hidden, norm)
        return _masked_output(self.head(hidden).squeeze(-1), node_mask)


class WindowedTGATOriginBaseline(nn.Module):
    """TGAT-style windowed baseline with graph propagation and temporal attention.

    This is an adaptation to snapshot tensors, not a claim of exact reproduction
    of the original event-stream TGAT implementation.
    """

    def __init__(
        self,
        node_feature_dim: int,
        structural_dim: int,
        hidden_dim: int = 96,
        heads: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_input = nn.Linear(node_feature_dim, hidden_dim)
        self.struct_input = nn.Linear(structural_dim, hidden_dim)
        self.graph_layer = GraphMessageLayer(hidden_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_attention = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def _time_encoding(self, steps: int, device, dtype) -> torch.Tensor:
        position = torch.arange(steps, device=device, dtype=dtype).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, self.hidden_dim, 2, device=device, dtype=dtype)
            * (-math.log(10_000.0) / self.hidden_dim)
        )
        encoding = torch.zeros((steps, self.hidden_dim), device=device, dtype=dtype)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency[: encoding[:, 1::2].shape[1]])
        return encoding

    def forward(self, node_features, adjacency, structural_features, node_mask):
        batch, steps, nodes, _ = node_features.shape
        norm = _normalize_adjacency(adjacency)
        structural = self.struct_input(structural_features)
        sequence = []
        time_encoding = self._time_encoding(steps, node_features.device, node_features.dtype)
        for step in range(steps):
            hidden = self.node_input(node_features[:, step]) + structural
            hidden = self.graph_layer(hidden, norm) + time_encoding[step]
            sequence.append(hidden)
        stacked = torch.stack(sequence, dim=2).reshape(batch * nodes, steps, self.hidden_dim)
        attended = self.temporal_attention(stacked)[:, -1]
        attended = attended.reshape(batch, nodes, self.hidden_dim)
        return _masked_output(self.head(attended).squeeze(-1), node_mask)


class TGMTGNOriginBaseline(nn.Module):
    """TGN baseline built from the maintained TGM/PyG implementation.

    Independent graphs are represented as a disjoint union and the TGN memory
    is reset for every minibatch. Repeated communication edges at each snapshot
    form the temporal event stream; messages contain observable endpoint
    features only.
    """

    implementation_source = "tgm-lib 0.1.0b1 / PyG TGNMemory"

    def __init__(
        self,
        node_feature_dim: int,
        structural_dim: int,
        hidden_dim: int = 96,
        max_batch_nodes: int = 4800,
    ):
        super().__init__()
        if TGNMemory is None:
            raise ImportError("TGMTGNOriginBaseline requires tgm-lib and torch-geometric")
        del structural_dim
        raw_msg_dim = node_feature_dim * 2
        time_dim = max(hidden_dim // 4, 8)
        message = IdentityMessage(raw_msg_dim, hidden_dim, time_dim)
        self.memory = TGNMemory(
            num_nodes=max_batch_nodes,
            raw_msg_dim=raw_msg_dim,
            memory_dim=hidden_dim,
            time_dim=time_dim,
            message_module=message,
            aggregator_module=LastAggregator(),
        )
        self.embedding = GraphAttentionEmbedding(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            msg_dim=raw_msg_dim,
            time_enc=self.memory.time_enc,
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.max_batch_nodes = int(max_batch_nodes)

    def _events(self, features, adjacency, step):
        batch_index, src, dst = (adjacency > 0).nonzero(as_tuple=True)
        nodes_per_graph = adjacency.size(1)
        source = batch_index * nodes_per_graph + src
        destination = batch_index * nodes_per_graph + dst
        messages = torch.cat(
            [features[batch_index, step, src], features[batch_index, step, dst]], dim=-1
        )
        times = torch.full(
            (source.numel(),), step + 1, dtype=torch.long, device=features.device
        )
        return source, destination, times, messages

    def forward(self, node_features, adjacency, structural_features, node_mask):
        del structural_features
        batch, steps, nodes, _ = node_features.shape
        total_nodes = batch * nodes
        if total_nodes > self.max_batch_nodes:
            raise ValueError(
                f"TGN disjoint batch has {total_nodes} nodes; maximum is {self.max_batch_nodes}"
            )
        # Independent graph batches must not share TGN state or autograd history.
        self.memory.detach()
        self.memory.reset_state()
        last_events = None
        for step in range(steps):
            events = self._events(node_features, adjacency, step)
            self.memory.update_state(*events)
            last_events = events
        node_ids = torch.arange(total_nodes, device=node_features.device)
        memory, last_update = self.memory(node_ids)
        if last_events is not None and last_events[0].numel() > 0:
            source, destination, times, messages = last_events
            edge_index = torch.stack([source, destination], dim=0)
            memory = self.embedding(memory, last_update, edge_index, times, messages)
        embeddings = memory.reshape(batch, nodes, -1)
        return _masked_output(self.head(embeddings).squeeze(-1), node_mask)
