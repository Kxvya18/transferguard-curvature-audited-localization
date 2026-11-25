"""Every table of the paper, rebuilt from the stored results in outputs/ (no new runs).

Writes outputs/tables.json and prints the tables. Intervals are paired bootstrap 95%
(10,000 resamples, seed 0); tests are paired Wilcoxon (ranks) or exact McNemar (top-k).

Run:  python -m experiments.make_tables
"""
import glob
import json

import numpy as np
from scipy.stats import binomtest, wilcoxon

from experiments.benchmark_localization import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs"
NETWORKS = ["Hospital", "Primary_school", "Office", "High_school", "Email-Eu", "CollegeMsg", "Growing_PA"]


def load(pattern):
    return [json.loads(line) for f in sorted(glob.glob(str(OUT / pattern))) for line in open(f)]


def paired(a, b):
    """Mean of a - b, bootstrap 95% interval, Wilcoxon p."""
    d = np.asarray(a) - np.asarray(b)
    boot = np.random.default_rng(0).choice(d, (10000, len(d))).mean(axis=1)
    p = float(wilcoxon(a, b).pvalue) if np.any(d) else 1.0
    return {"difference": float(d.mean()), "interval": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "p": p}


def bp_table(folder="bp_comparison"):
    """Mean field vs belief propagation vs message passing, true rate, identical outbreaks.
    `bp_comparison_damping0.8`: the same outbreaks, every method recomputed, BP damping 0.8."""
    table = {}
    for net in NETWORKS:
        rows = [r for r in load(f"{folder}/{net}_0.1*.jsonl") if r["test_rate"] == "true"]
        if not rows:
            continue
        rr = {m: 1 / np.array([r[m] for r in rows]) for m in ("Mean-field", "Belief propagation", "Message passing")}
        sec = {m: float(np.mean([r[m + " seconds"] for r in rows])) for m in rr}
        table[net] = {"outbreaks": len(rows), **{f"MRR {m}": float(v.mean()) for m, v in rr.items()},
                      **{f"seconds {m}": s for m, s in sec.items()},
                      "MF - BP": paired(rr["Mean-field"], rr["Belief propagation"]),
                      "MF - MP": paired(rr["Mean-field"], rr["Message passing"])}
    return table


def placement_table():
    """Test selection: MRR over 10 picks and tests until the source is ranked first."""
    mrr = lambda ranks: np.mean([1 / x for x in ranks])
    tests = lambda ranks: next((i for i, x in enumerate(ranks) if x <= 1), len(ranks))
    table = {}
    for net in NETWORKS:
        by = {}
        files = glob.glob(str(OUT / f"observer_placement/{net}.jsonl")) + glob.glob(str(OUT / f"observer_placement/{net}_seed30?.jsonl")) \
            + glob.glob(str(OUT / f"observer_placement/{net}_seed30?_shard*.jsonl")) + glob.glob(str(OUT / f"observer_placement/{net}_shard*.jsonl"))
        for f in files:
            if "simulated" in f:
                continue
            seed = f.split("_seed")[1][:3] if "_seed" in f else "301"
            count = {}
            for line in open(f):
                r = json.loads(line)
                k = r.get("outbreak", count.get(r["strategy"], 0))
                count[r["strategy"]] = count.get(r["strategy"], 0) + 1
                by.setdefault(r["strategy"], {})[(seed, k)] = r["ranks"]
        if not by:
            continue
        common = sorted(set.intersection(*[set(d) for d in by.values()]))
        ours = by["Information gain (ours)"]
        block = {"outbreaks": len(common)}
        for s, d in by.items():
            block[s] = {"MRR": float(np.mean([mrr(d[k]) for k in common])), "tests to rank 1": float(np.mean([tests(d[k]) for k in common]))}
            if s != "Information gain (ours)":
                block[s]["ours - this"] = paired([mrr(ours[k]) for k in common], [mrr(d[k]) for k in common])
        table[net] = block
    return table


def sterchi_table():
    sim = {r["outbreak"]: r for r in load("observer_placement/Hospital_simulated_shard*.jsonl")}
    if not sim:
        return {}
    main = {}
    for r in load("observer_placement/Hospital.jsonl"):
        main.setdefault(r["strategy"], []).append(r["ranks"])
    ks = sorted(sim)
    mrr = lambda ranks: float(np.mean([1 / x for x in ranks]))
    out = {"outbreaks": len(ks), "AKLD by simulation (Sterchi 2023)": {"MRR": float(np.mean([mrr(sim[k]["ranks"]) for k in ks])),
                                                                        "seconds per pick": float(np.mean([sim[k]["seconds per pick"] for k in ks]))}}
    for s in ("Information gain (ours)", "Joint disagreement (AKLD, Sterchi 2023)"):
        out[s] = {"MRR": float(np.mean([mrr(main[s][k]) for k in ks]))}
    return out


def real_table():
    out = {}
    for name in ("variants", "variants_heldout", "variants_regional", "independent_origins"):
        rows = load(f"real_epidemics/{name}.jsonl")
        if not rows:
            continue
        methods = [k for k, v in rows[0].items() if isinstance(v, float)]
        out[name] = {"outbreaks": [r.get("variant", r.get("outbreak")) for r in rows],
                     **{m: {"MRR": float(np.mean([1 / r[m] for r in rows])), "ranks": [r[m] for r in rows]} for m in methods}}
    summary = OUT / "real_cascades" / "twitter_v3_summary.json"
    if summary.exists():
        out["twitter"] = json.loads(summary.read_text())
        test = load("real_cascades/twitter_v3_test_300_part*.jsonl")
        beta = out["twitter"]["beta"]
        for k in (1, 10):
            a = np.array([r[f"Mean-field, beta {beta}"] <= k for r in test])
            b = np.array([r["Distance center"] <= k for r in test])
            n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
            out["twitter"][f"top-{k} vs distance center (test)"] = {"mean field only": n01, "distance center only": n10,
                                                                     "McNemar p": float(binomtest(n01, n01 + n10).pvalue)}
    return out


def main():
    tables = {"belief propagation": bp_table(), "belief propagation, damping 0.8": bp_table("bp_comparison_damping0.8"), "test selection": placement_table(), "Sterchi simulation": sterchi_table(),
              "real outbreaks": real_table()}
    for extra in ("conformal_sets.json", "separability.json"):
        if (OUT / extra).exists():
            tables[extra[:-5]] = json.loads((OUT / extra).read_text())
    (OUT / "tables.json").write_text(json.dumps(tables, indent=2))
    for net, t in list(tables["belief propagation"].items()) + [(n + " (BP damping 0.8)", t) for n, t in tables["belief propagation, damping 0.8"].items()]:
        d = t["MF - BP"]
        print(f"BP {net:15s} n={t['outbreaks']:3d} MF {t['MRR Mean-field']:.3f} BP {t['MRR Belief propagation']:.3f} MP {t['MRR Message passing']:.3f}"
              f"  MF-BP {d['difference']:+.3f} [{d['interval'][0]:+.3f},{d['interval'][1]:+.3f}] p={d['p']:.1e}")
    for net, t in tables["test selection"].items():
        print(f"Selection {net:15s} n={t['outbreaks']:3d} ours {t['Information gain (ours)']['MRR']:.3f}/{t['Information gain (ours)']['tests to rank 1']:.1f}")
    print("Sterchi", json.dumps(tables["Sterchi simulation"]))
    for name, t in tables["real outbreaks"].items():
        if name != "twitter":
            print(name, {m: round(v["MRR"], 3) for m, v in t.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
