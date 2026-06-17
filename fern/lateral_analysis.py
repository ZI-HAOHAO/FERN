"""Dataset diagnostic: why is Edge-IIoTset Lateral recall 0? Show it is an upstream
stage-classifier / GT-confidence issue, not a retention failure.

Per stage on the test set (3 seeds), report:
  n_campaigns_with_stage  : GT campaigns containing the stage
  n_flows                 : true-stage flows in the test set
  exact / derived         : confidence composition of those flows (map_attack)
  stagehead_recall        : per-class recall of the train-only stage classifier
                            BEFORE any retention (does the head ever fire this stage?)
  fern_recall_5pct        : FERN-Greedy per-stage recall at 5% budget (for reference)
"""
import json
import numpy as np
from collections import defaultdict
from pilot_k1 import load_flows, make_xy
from retention import forensic_value_labels, train_fern_scorer, byte_cost, keep_under_budget
from fidelity import build_ground_truth
from stage_clf import train_stage_clf, predict_stages
from stage_mapping import map_attack, UNKNOWN, STAGE_ORDER
from evaluation_suite import strat_split, per_stage_recall


def run(seeds=(0, 1, 2)):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    flows = load_flows("data/processed/edge_flows_full.jsonl")
    X, _ = make_xy(flows, "edge-iiot")
    acc = {s: defaultdict(list) for s in STAGE_ORDER}
    for sd in seeds:
        rng = np.random.default_rng(sd)
        tr, te = strat_split([f["attack"] for f in flows], rng)
        ftr = [flows[i] for i in tr]
        fte = [flows[i] for i in te]
        Xtr, Xte = X[tr], X[te]
        ce = np.array([float(byte_cost(f)) for f in fte])
        full = float(ce.sum())
        gt = build_ground_truth(fte, "edge-iiot")
        # GT campaigns per stage
        camp_with = defaultdict(int)
        for cid, d in gt.items():
            for s in d["stages_ordered"]:
                camp_with[s] += 1
        # true stage + confidence per test flow
        true_stage, conf = [], []
        for f in fte:
            s, c = map_attack(f["attack"], "edge-iiot")
            true_stage.append(s if s not in (None, UNKNOWN) else None)
            conf.append(c)
        true_stage = np.array(true_stage, dtype=object)
        conf = np.array(conf, dtype=object)
        # stage classifier (train-only), predict on test -> pre-retention recall
        sm = train_stage_clf(Xtr, ftr, "edge-iiot")
        pst = predict_stages(sm, Xte)
        # FERN-Greedy 5% per-stage recall
        v = forensic_value_labels(ftr, "edge-iiot", n_rare_stages=3, anchor_focused=True)
        s_te, _ = train_fern_scorer(Xtr, v, Xte, dev=dev)
        mask = keep_under_budget(np.argsort(-(s_te / np.sqrt(ce))), ce, 0.05 * full)
        ps5 = per_stage_recall(fte, mask, pst, gt, "edge-iiot")
        for s in STAGE_ORDER:
            idx = np.where(true_stage == s)[0]
            acc[s]["n_campaigns_with_stage"].append(camp_with.get(s, 0))
            acc[s]["n_flows"].append(int(len(idx)))
            if len(idx):
                acc[s]["exact_frac"].append(float(np.mean([conf[i] == "exact" for i in idx])))
                acc[s]["stagehead_recall"].append(float(np.mean([pst[i] == s for i in idx])))
            if ps5.get(s) is not None:
                acc[s]["fern_recall_5pct"].append(ps5[s])
    out = {}
    for s in STAGE_ORDER:
        a = acc[s]
        out[s] = {
            "n_campaigns_with_stage": int(round(np.mean(a["n_campaigns_with_stage"]))) if a["n_campaigns_with_stage"] else 0,
            "n_flows": int(round(np.mean(a["n_flows"]))) if a["n_flows"] else 0,
            "exact_frac": float(np.mean(a["exact_frac"])) if a["exact_frac"] else None,
            "stagehead_recall": float(np.mean(a["stagehead_recall"])) if a["stagehead_recall"] else None,
            "fern_recall_5pct": float(np.mean(a["fern_recall_5pct"])) if a["fern_recall_5pct"] else None,
        }
    return out


if __name__ == "__main__":
    out = run()
    json.dump(out, open("outputs/lateral_analysis.json", "w"), indent=2)
    print(json.dumps(out, indent=1))
    print("LATERAL DONE", flush=True)



