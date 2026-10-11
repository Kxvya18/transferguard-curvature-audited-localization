"""Plot independent-seed and component-ablation evidence."""

from __future__ import annotations

import json, os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
with (ROOT/"outputs/submission_evidence_v5/summary.json").open() as h: data=json.load(h)
plt.rcParams.update({"font.family":"STIXGeneral","mathtext.fontset":"stix","font.size":11,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.22})
fig,axes=plt.subplots(1,2,figsize=(11.5,4.4),constrained_layout=True); fig.suptitle("Independent-Seed Stability and Component Necessity",fontsize=16,fontweight="bold")
ax=axes[0]
for index,(method,label,color) in enumerate((("hybrid","Hybrid","#2A9D8F"),("temporal_only","Temporal only","#264653"))):
    rows=[x for x in data["multiseed"]["seed_rows"] if x["method"]==method]; values=[x["mrr"] for x in rows]; ax.scatter(np.full(len(values),index)+np.linspace(-.07,.07,len(values)),values,s=60,color=color,zorder=3)
    summary=next(x for x in data["multiseed"]["summary"] if x["method"]==method); lo,hi=summary["mrr_ci95_across_training_seeds"]; ax.errorbar(index,summary["mean_mrr"],yerr=[[summary["mean_mrr"]-lo],[hi-summary["mean_mrr"]]],fmt="D",ms=7,color="#E76F51",capsize=5,zorder=4)
ax.set_xticks((0,1),("Hybrid","Temporal only")); ax.set_ylim(.965,1.002); ax.set_ylabel("MRR on fixed 120-graph test"); ax.set_title("(a) Three independent training seeds")
ax=axes[1]; rows=data["ablation_table"]; labels=["Boltzmann\nprior","Temporal\nencoder","vs TGAT-style","SSL\nauxiliary"]; deltas=[x["delta"] for x in rows]; colors=["#2A9D8F" if x>0 else "#E76F51" for x in deltas]; ax.bar(np.arange(4),deltas,color=colors); ax.axhline(0,color="#333",lw=1); ax.set_xticks(np.arange(4),labels); ax.set_ylabel(r"ablation effect $\Delta$MRR"); ax.set_title("(b) Positive means component improves MRR")
ax.set_ylim(-0.028, 0.215)
for i,v in enumerate(deltas): ax.text(i,v+(0.004 if v>=0 else -0.004),f"{v:+.3f}",ha="center",va="bottom" if v>=0 else "top")
path=ROOT/"paper_assets/figures/fig_multiseed_ablation_v5.png"; fig.savefig(path,dpi=300,bbox_inches="tight"); print(path)
