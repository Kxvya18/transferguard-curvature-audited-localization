"""Evaluate static-GCN and windowed-TGAT checkpoints on the fixed test grid."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from evaluation.metrics import evaluate_origin_localization
from experiments.cold_start_common import prepare_prefix_batch
from models.reference_baselines import StaticGCNOriginBaseline, TGMTGNOriginBaseline, WindowedTGATOriginBaseline
from synthetic.dataset import SyntheticOriginDataset, collate_temporal_graph_samples
from synthetic.iot_growth import SyntheticScenarioConfig


def main() -> None:
    with (ROOT / "configs/train_transferguard_mixed_no_prior.json").open() as handle: payload=json.load(handle)
    keys=set(SyntheticScenarioConfig.__dataclass_fields__); base=SyntheticScenarioConfig(**{k:v for k,v in payload.items() if k in keys})
    samples=[]; cell=0
    for n,m,beta in product((100,220,400),(1,2,3,4,5),(0.5,1.0,1.5,2.0)):
        samples.extend(SyntheticOriginDataset(replace(base,n_nodes=n,m=m,beta=beta,seed=610_000+cell*1009),2,seed_offset=70_000).samples); cell+=1
    specifications=[
        ("static_gcn",StaticGCNOriginBaseline,ROOT/"outputs/reference_baselines/static_gcn/best.pt"),
        ("windowed_tgat",WindowedTGATOriginBaseline,ROOT/"outputs/reference_baselines/windowed_tgat/best.pt"),
        ("tgm_tgn",TGMTGNOriginBaseline,ROOT/"outputs/journal_v1/reference_baselines/tgm_tgn/best.pt"),
    ]
    rows=[]
    for name,cls,path in specifications:
        if not path.exists():
            print(f"[reference-test] skip missing {path}", flush=True)
            continue
        checkpoint=torch.load(path,map_location="cpu",weights_only=False); model=cls(15,7,hidden_dim=96); model.load_state_dict(checkpoint["model_state_dict"]); model.eval(); metrics=[]
        for sample in samples:
            batch=prepare_prefix_batch(
                collate_temporal_graph_samples([sample]), 6, 1.0, 880_000 + len(metrics), True
            )
            with torch.no_grad(): out=model(batch["node_features"],batch["adjacency"],batch["structural_features"],batch["node_mask"])
            probs=out["probs"][0,:len(sample.node_ids)]; metrics.append(evaluate_origin_localization({i:float(probs[i]) for i in range(len(probs))},sample.origin_index))
        rows.append({"method":name,"n_test_samples":len(samples),"mrr":float(np.mean([x["mrr"] for x in metrics])),"hit1":float(np.mean([x["hit@1"] for x in metrics]))})
        print(f"[reference-test] {name}",flush=True)
    with (ROOT/"outputs/reference_baselines/fixed_test_summary.json").open("w") as handle: json.dump({"test_seed_contract":"identical to multiseed_v4 evaluation","rows":rows},handle,indent=2)


if __name__=="__main__": main()
