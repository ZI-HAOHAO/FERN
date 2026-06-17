"""Cross-regime check: FERN++ submodular coverage vs value-per-byte greedy on IoT-23
(dense-evidence regime). Expectation: NO gain (dense regime doesn't stress selection),
reported honestly for cross-regime validation."""
import json, os
import numpy as np
import lightgbm as lgb
from collections import defaultdict
from retention import train_fern_scorer, byte_cost, keep_under_budget
from evaluation_suite import strat_split
from fern_submod import submod_offline
import iot23_pipeline as I23

EDGE_BUDGETS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]


def run(seeds=(0, 1, 2)):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from hardening_suite import iot23_value_labels
    flows = []
    for fn in sorted(os.listdir("data/raw/iot23")):
        if not fn.endswith(".csv"): continue
        fl = I23.parse_csv(os.path.join("data/raw/iot23", fn), 300000)
        tag = fn.replace("dataset", "s").replace(".csv", "")
        for f in fl: f["src"] = f"{tag}:{f['src']}"; f["dst"] = f"{tag}:{f['dst']}"
        flows += fl
    X = I23.featmat(flows); ys = I23.stage_y(flows)
    agg = {m: {"sr": defaultdict(list), "chain": defaultdict(list)} for m in ["greedy_vpb", "submod_off"]}
    for sd in seeds:
        rng = np.random.default_rng(sd)
        scen = [f["src"].split(":")[0] for f in flows]
        tr, te = strat_split([f"{s}|{y}" for s, y in zip(scen, ys)], rng)
        te = sorted(te, key=lambda i: flows[i]["t_first"])
        Xtr, Xte = X[tr], X[te]; fte = [flows[i] for i in te]
        ce = np.array([max(1.0, f["bytes_tot"]) for f in fte]); full = float(ce.sum())
        gt = I23.host_ground_truth(fte)
        if not gt: continue
        _, anc_tr = iot23_value_labels([flows[i] for i in tr])
        s_te, _ = train_fern_scorer(Xtr, anc_tr, Xte, dev=dev)
        sm = lgb.train({"objective": "multiclass", "num_class": len(I23.CLASSES), "verbose": -1,
                        "num_leaves": 31, "learning_rate": 0.1}, lgb.Dataset(Xtr, label=ys[tr]), 80)
        pst = np.array([I23.CLASSES[i] for i in np.argmax(sm.predict(Xte), 1)], dtype=object)
        for b in EDGE_BUDGETS:
            bb = b * full
            mg = keep_under_budget(np.argsort(-(s_te / np.sqrt(ce))), ce, bb)
            ms = submod_offline(s_te, ce, fte, pst, bb, alpha=0.5, rare_weight=False)
            for name, mask in [("greedy_vpb", mg), ("submod_off", ms)]:
                mal = np.zeros(len(fte), int); kept = np.where(mask)[0]
                mal[kept] = (pst[kept] != "Normal").astype(int)
                fid = I23.fidelity(fte, mal, pst, gt)
                agg[name]["sr"][b].append(fid["stage_recall"])
                agg[name]["chain"][b].append(fid["chain_completeness"])
        print(f"seed {sd} done", flush=True)
    out = {"budgets": EDGE_BUDGETS, "methods": {}}
    for m in agg:
        out["methods"][m] = {
            "stage_recall": [round(float(np.mean(agg[m]["sr"][b])), 4) for b in EDGE_BUDGETS],
            "chain": [round(float(np.mean(agg[m]["chain"][b])), 4) for b in EDGE_BUDGETS]}
    return out


if __name__ == "__main__":
    out = run()
    json.dump(out, open("outputs/fern_submod_iot23.json", "w"), indent=2)
    print(json.dumps(out, indent=1)); print("IOT23 SUBMOD DONE", flush=True)



