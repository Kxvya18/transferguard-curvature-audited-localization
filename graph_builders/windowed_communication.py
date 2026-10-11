"""Time-windowed communication graph builder.

This is the first real-data oriented utility in `ba2`. It is intentionally
lightweight and works from generic event objects so we can adapt it later to
IoT-23, TON_IoT, or other flow datasets without rewriting graph logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx


@dataclass(frozen=True)
class CommunicationEvent:
    timestamp: float
    src: str
    dst: str
    protocol: str
    bytes_out: float = 0.0
    bytes_in: float = 0.0
    label: str | None = None


def build_time_windowed_graphs(
    events: Iterable[CommunicationEvent],
    window_size: float,
) -> list[tuple[float, nx.Graph]]:
    buckets: dict[int, list[CommunicationEvent]] = {}
    for event in events:
        bucket = int(event.timestamp // max(window_size, 1e-12))
        buckets.setdefault(bucket, []).append(event)

    graphs: list[tuple[float, nx.Graph]] = []
    for bucket in sorted(buckets):
        G = nx.Graph()
        for event in buckets[bucket]:
            if not G.has_node(event.src):
                G.add_node(event.src, kind="endpoint")
            if not G.has_node(event.dst):
                G.add_node(event.dst, kind="endpoint")

            if G.has_edge(event.src, event.dst):
                edge = G[event.src][event.dst]
                edge["count"] += 1
                edge["bytes_out"] += float(event.bytes_out)
                edge["bytes_in"] += float(event.bytes_in)
                edge["protocols"].add(event.protocol)
            else:
                G.add_edge(
                    event.src,
                    event.dst,
                    count=1,
                    bytes_out=float(event.bytes_out),
                    bytes_in=float(event.bytes_in),
                    protocols={event.protocol},
                )

        graphs.append((bucket * window_size, G))
    return graphs

