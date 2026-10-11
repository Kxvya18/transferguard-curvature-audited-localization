"""IoT-23 ingestion.

IoT-23 commonly distributes `conn.log.labeled` as a Zeek log. Those files are
not standard CSVs because the header is encoded in `#fields` metadata lines.
This loader parses the Zeek layout directly and also supports simpler CSV/TSV
exports when present.
"""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from .common import row_to_event, rows_to_events


def _read_zeek_log(path: Path) -> pd.DataFrame:
    fields: list[str] | None = None
    rows: list[list[str]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#fields"):
                fields = re.split(r"\s+", line[len("#fields") :].strip())
                continue
            if line.startswith("#"):
                continue
            if fields is None:
                continue
            values = re.split(r"\s+", line.strip(), maxsplit=len(fields) - 1)
            if len(values) == len(fields):
                rows.append(values)

    if not rows:
        return pd.DataFrame(columns=fields or [])
    return pd.DataFrame(rows, columns=fields)


def _load_zeek_events(path: Path) -> list:
    """Stream a CTU Zeek connection log without materialising a dataframe."""
    fields: list[str] | None = None
    events = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#fields"):
                fields = re.split(r"\s+", line[len("#fields") :].strip())
                continue
            if line.startswith("#") or fields is None:
                continue
            values = re.split(r"\s+", line, maxsplit=len(fields) - 1)
            if len(values) != len(fields):
                continue
            event = row_to_event(
                dict(zip(fields, values)),
                timestamp_cols=["ts", "timestamp", "time"],
                src_cols=["id.orig_h", "src_ip", "src", "source_ip"],
                dst_cols=["id.resp_h", "dst_ip", "dst", "destination_ip"],
                protocol_cols=["proto", "protocol", "service"],
                bytes_out_cols=["orig_bytes", "src_bytes", "bytes_out"],
                bytes_in_cols=["resp_bytes", "dst_bytes", "bytes_in"],
                label_cols=["detailed-label", "label", "attack", "scenario"],
            )
            if event is not None:
                events.append(event)
    return events


def load_iot23_events(path: str | Path) -> list:
    file_path = Path(path)
    if "conn.log" in file_path.name:
        return _load_zeek_events(file_path)
    else:
        sep = "\t" if file_path.suffix.lower() in {".tsv", ".log"} else ","
        frame = pd.read_csv(file_path, sep=sep, comment="#")
    return rows_to_events(
        frame=frame,
        timestamp_cols=["ts", "timestamp", "time"],
        src_cols=["id.orig_h", "src_ip", "src", "source_ip"],
        dst_cols=["id.resp_h", "dst_ip", "dst", "destination_ip"],
        protocol_cols=["proto", "protocol", "service"],
        bytes_out_cols=["orig_bytes", "src_bytes", "bytes_out"],
        bytes_in_cols=["resp_bytes", "dst_bytes", "bytes_in"],
        label_cols=["detailed-label", "label", "attack", "scenario"],
    )
