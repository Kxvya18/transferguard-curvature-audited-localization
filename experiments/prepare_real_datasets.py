"""Prepare downloaded IoT datasets into graph-ready metadata summaries."""

from __future__ import annotations

import argparse
from collections import Counter
import errno
import json
import sys
import tarfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph_builders.windowed_communication import build_time_windowed_graphs
from loaders.iot23 import load_iot23_events
from loaders.ton_iot import load_ton_iot_events
from loaders.ton_iot_scenarios import discover_ton_iot_scenarios


def _extract_archive(archive_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.suffixes[-2:] == [".tar", ".gz"] or archive_path.name.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=target_dir)
        return
    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zip_handle:
            zip_handle.extractall(path=target_dir)
        return
    raise ValueError(f"Unsupported archive type: {archive_path}")


def _extract_iot23_logs_only(archive_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith("conn.log.labeled"):
                continue
            tar.extract(member, path=target_dir)


def _find_iot23_logs(root_dir: Path) -> list[Path]:
    return sorted(root_dir.rglob("conn.log.labeled"), key=lambda path: (path.stat().st_size, str(path)))


def _summarize_graphs(graphs: list[tuple[float, object]]) -> dict:
    if not graphs:
        return {
            "n_windows": 0,
            "mean_nodes": 0.0,
            "mean_edges": 0.0,
        }
    nodes = [graph.number_of_nodes() for _, graph in graphs]
    edges = [graph.number_of_edges() for _, graph in graphs]
    return {
        "n_windows": len(graphs),
        "mean_nodes": float(sum(nodes) / len(nodes)),
        "mean_edges": float(sum(edges) / len(edges)),
    }


def _summarize_labels(events: list) -> dict:
    labels = Counter(event.label for event in events if event.label)
    benign = sum(count for label, count in labels.items() if label.lower() == "benign")
    labeled = sum(labels.values())
    return {
        "n_labeled_events": labeled,
        "n_attack_events": labeled - benign,
        "label_counts": dict(labels.most_common()),
    }


def prepare_iot23(
    raw_dir: Path,
    processed_dir: Path,
    window_size: float,
    max_log_bytes: int | None = 500_000_000,
) -> dict:
    archives = sorted(raw_dir.glob("*.tar.gz"))
    extract_dir = processed_dir / "iot23_extracted"
    existing_logs = _find_iot23_logs(extract_dir) if extract_dir.exists() else []
    if not existing_logs:
        for archive in archives:
            try:
                _extract_iot23_logs_only(archive, extract_dir)
            except (tarfile.ReadError, EOFError):
                continue
            except OSError as exc:
                if exc.errno != errno.ENOSPC:
                    raise
                print(
                    f"[iot23] disk full during extraction of {archive}; continuing with extracted logs only",
                    flush=True,
                )
                break

    scenario_summaries = []
    skipped_summaries = []
    for log_path in _find_iot23_logs(extract_dir):
        size_bytes = int(log_path.stat().st_size)
        if max_log_bytes is not None and size_bytes > max_log_bytes:
            skipped_summaries.append(
                {
                    "scenario_path": str(log_path),
                    "size_bytes": size_bytes,
                    "reason": f"exceeds max_log_bytes={max_log_bytes}",
                }
            )
            print(f"[iot23] skipped oversized {log_path}", flush=True)
            continue
        events = load_iot23_events(log_path)
        if not events:
            raise ValueError(f"IoT-23 parser produced zero events for non-empty log: {log_path}")
        graphs = build_time_windowed_graphs(events, window_size=window_size)
        scenario_summaries.append(
            {
                "scenario_path": str(log_path),
                "size_bytes": size_bytes,
                "n_events": len(events),
                **_summarize_labels(events),
                **_summarize_graphs(graphs),
            }
        )
        print(f"[iot23] prepared {log_path}", flush=True)
    return {
        "dataset": "iot23",
        "n_scenarios": len(scenario_summaries),
        "scenarios": scenario_summaries,
        "n_skipped": len(skipped_summaries),
        "skipped": skipped_summaries,
    }


def prepare_ton_iot(raw_dir: Path, processed_dir: Path, window_size: float) -> dict:
    archives = sorted(raw_dir.glob("*.zip"))
    extract_dir = processed_dir / "ton_iot_extracted"
    for archive in archives:
        try:
            _extract_archive(archive, extract_dir)
        except zipfile.BadZipFile:
            continue

    file_summaries = []
    ton_root = extract_dir / "Ton_Iot"
    for scenario in discover_ton_iot_scenarios(ton_root):
        csv_path = Path(scenario["data_csv"])
        try:
            events = load_ton_iot_events(csv_path)
        except Exception:
            continue
        if not events:
            continue
        graphs = build_time_windowed_graphs(events, window_size=window_size)
        gt_path = scenario["ground_truth_csv"]
        gt_rows = 0
        if gt_path is not None and Path(gt_path).exists():
            try:
                import pandas as pd

                gt_rows = int(len(pd.read_csv(gt_path, low_memory=False)))
            except Exception:
                gt_rows = 0
        file_summaries.append(
            {
                "modality": str(scenario["modality"]),
                "scenario_name": str(scenario["scenario_name"]),
                "csv_path": str(csv_path),
                "ground_truth_csv": None if gt_path is None else str(gt_path),
                "ground_truth_rows": gt_rows,
                "n_events": len(events),
                **_summarize_graphs(graphs),
            }
        )
        print(f"[ton_iot] prepared {csv_path}", flush=True)
    return {
        "dataset": "ton_iot",
        "n_files": len(file_summaries),
        "files": file_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-size", type=float, default=300.0)
    parser.add_argument(
        "--max-iot23-log-bytes",
        type=int,
        default=500_000_000,
        help="Skip IoT-23 conn.log.labeled files larger than this many bytes.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["iot23", "ton_iot"],
        choices=["iot23", "ton_iot"],
        help="Datasets to prepare from the downloaded archives.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Users/boolu/Desktop/simulations/ALL_PROJECTS/ba2/data"),
    )
    args = parser.parse_args()

    raw_root = args.data_root / "raw"
    processed_root = args.data_root / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)

    summaries = {}
    out_path = processed_root / "dataset_preparation_summary.json"
    if out_path.exists():
        try:
            with out_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if isinstance(existing, dict):
                summaries.update(existing)
        except json.JSONDecodeError:
            pass
    iot23_dir = raw_root / "iot23"
    ton_dir = raw_root / "ton_iot"
    if "iot23" in args.datasets and iot23_dir.exists():
        summaries["iot23"] = prepare_iot23(
            iot23_dir,
            processed_root,
            args.window_size,
            max_log_bytes=args.max_iot23_log_bytes,
        )
    if "ton_iot" in args.datasets and ton_dir.exists():
        summaries["ton_iot"] = prepare_ton_iot(ton_dir, processed_root, args.window_size)

    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    print(f"Saved preparation summary to {out_path}")


if __name__ == "__main__":
    main()
