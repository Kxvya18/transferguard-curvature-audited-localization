"""TON_IoT ingestion with flexible CSV-column mapping."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import rows_to_events


def load_ton_iot_events(path: str | Path) -> list:
    file_path = Path(path)
    frame = pd.read_csv(file_path, low_memory=False)
    return rows_to_events(
        frame=frame,
        timestamp_cols=["ts", "timestamp", "time", "frame.time_epoch"],
        src_cols=["src_ip", "src", "source_ip", "srcip"],
        dst_cols=["dst_ip", "dst", "destination_ip", "dstip"],
        protocol_cols=["proto", "protocol", "service"],
        bytes_out_cols=["src_bytes", "sbytes", "bytes_out"],
        bytes_in_cols=["dst_bytes", "dbytes", "bytes_in"],
        label_cols=["label", "attack", "type", "category"],
    )
