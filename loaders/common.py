"""Common loader helpers for heterogeneous IoT event ingestion."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from graph_builders.windowed_communication import CommunicationEvent


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() in {"", "-", "(empty)"}:
        return True
    return bool(pd.isna(value))


def _first_present(row: Mapping, names: list[str], default=None):
    for name in names:
        if name in row and not _is_missing(row[name]):
            return row[name]
    return default


def _safe_float(value, default: float = 0.0) -> float:
    if value in {None, "", "-"}:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def rows_to_events(
    frame: pd.DataFrame,
    timestamp_cols: list[str],
    src_cols: list[str],
    dst_cols: list[str],
    protocol_cols: list[str],
    bytes_out_cols: list[str] | None = None,
    bytes_in_cols: list[str] | None = None,
    label_cols: list[str] | None = None,
) -> list[CommunicationEvent]:
    """Convert a dataframe with flexible column names into event objects."""
    events: list[CommunicationEvent] = []
    columns = list(frame.columns)
    # Named tuples rewrite Zeek columns such as ``id.orig_h`` to positional
    # names. Zipping unnamed tuples preserves the source schema exactly.
    for values in frame.itertuples(index=False, name=None):
        event = row_to_event(
            dict(zip(columns, values)),
            timestamp_cols=timestamp_cols,
            src_cols=src_cols,
            dst_cols=dst_cols,
            protocol_cols=protocol_cols,
            bytes_out_cols=bytes_out_cols,
            bytes_in_cols=bytes_in_cols,
            label_cols=label_cols,
        )
        if event is not None:
            events.append(event)
    return events


def row_to_event(
    row: Mapping,
    timestamp_cols: list[str],
    src_cols: list[str],
    dst_cols: list[str],
    protocol_cols: list[str],
    bytes_out_cols: list[str] | None = None,
    bytes_in_cols: list[str] | None = None,
    label_cols: list[str] | None = None,
) -> CommunicationEvent | None:
    """Convert one source row while preserving dotted Zeek column names."""
    timestamp = _first_present(row, timestamp_cols)
    src = _first_present(row, src_cols)
    dst = _first_present(row, dst_cols)
    protocol = _first_present(row, protocol_cols, default="unknown")
    if timestamp is None or src is None or dst is None:
        return None

    bytes_out = _safe_float(
        _first_present(row, bytes_out_cols or [], default=0.0),
        default=0.0,
    )
    bytes_in = _safe_float(
        _first_present(row, bytes_in_cols or [], default=0.0),
        default=0.0,
    )
    raw_label = _first_present(row, label_cols or [], default=None)
    label = None if raw_label is None else str(raw_label).strip()
    return CommunicationEvent(
        timestamp=_safe_float(timestamp, default=0.0),
        src=str(src),
        dst=str(dst),
        protocol=str(protocol),
        bytes_out=bytes_out,
        bytes_in=bytes_in,
        label=label,
    )
