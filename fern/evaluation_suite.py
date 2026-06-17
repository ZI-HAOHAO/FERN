"""Unified evaluation suite -resolves the offline/online inconsistency + adds demanded analyses.

ONE unified harness so every retention number is apples-to-apples: same test split, same
anchor-trained scorer, same train-only investigator, same ground truth, same stage-recall
definition, same byte-budget definition. For each (selector, budget) we record:
  stage_recall, chain, kept_byte_frac, budget_violation, n_kept, malicious_frac, anchor_frac
plus per-stage recall for the headline selector. This makes the budget strictly 
puts FERN-Greedy (value-per-byte) and FERN-Threshold (raw-score) in the SAME table.

Selectors:
  greedy            : offline, keep by g/sqrt(c) descending until budget (the old FERN rule)
  threshold_off     : offline, keep by raw g descending until budget (score-threshold, offline)
  threshold_on_fix  : online, chronological, keep if g>=tau (tau train-calibrated) & budget left
  threshold_on_adapt: online, window-adaptive tau (causal)

Parts:
  A  Edge-IIoTset unified retention + budget audit + per-stage recall
  B  IoT-23 unified retention + per-stage recall
  C  evidence-sparsity indices per dataset
  D  sparse-vs-dense controlled stress test (Edge): inject benign+flood redundancy, vary the
     anchor-byte-share, measure FERN-Threshold vs random advantage as a function of sparsity
"""
import argparse, json, os
import numpy as np
from collections import defaultdict

from pilot_k1 import load_flows, FEATS, make_xy
from retention import forensic_value_labels, anchor_labels, train_fern_scorer, byte_cost, keep_under_budget
from fidelity import build_ground_truth, forensic_fidelity, reconstruct
from stage_clf import train_stage_clf, predict_stages, STAGE_CLASSES
from stage_mapping import map_attack, UNKNOWN, STAGE_ORDER
import iot23_pipeline as I23

EDGE_BUDGETS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]


def strat_split(keys, rng, frac=0.7):
    b = defaultdict(list)
    for i, k in enumerate(keys): b[k].append(i)
    tr, te = [], []
    for k, idxs in b.items():
        idxs = list(idxs); rng.shuffle(idxs); c = int(frac * len(idxs)); tr += idxs[:c]; te += idxs[c:]
    return tr, te


def keep_threshold_offline(scores, costs, budget):
    """Keep by RAW score descending until byte budget (offline threshold = top-by-g)."""
    order = np.argsort(-scores)
    mask = np.zeros(len(scores), dtype=bool); spent = 0.0
    for i in order:
        if spent + costs[i] <= budget:
            mask[i] = True; spent += costs[i]
    return mask


def keep_threshold_online(scores, costs, thr, budget):
    mask = np.zeros(len(scores), dtype=bool); spent = 0.0
    for i in range(len(scores)):
        if scores[i] >= thr and spent + costs[i] <= budget:
            mask[i] = True; spent += costs[i]
    return mask


def calib_thr(scores, costs, budget):
    order = np.argsort(-scores); spent = 0.0; thr = np.inf
    for i in order:
        if spent + costs[i] > budget: break
        spent += costs[i]; thr = scores[i]
    return thr


def per_stage_recall(flows, mask, pred_stages, gt, dataset, host=False):
    """Recall of each stage's evidence across campaigns, for the retained+staged set."""
    mal = np.zeros(len(flows), dtype=int); kept = np.where(mask)[0]
    mal[kept] = (pred_stages[kept] != "Normal").astype(int)
    if host:
        pred = I23.host_reconstruct(flows, mal, pred_stages)
    else:
        pred = reconstruct(flows, mal, dataset, pred_stages=pred_stages)
    num = defaultdict(int); den = defaultdict(int)
    for cid, stages in gt.items():
        gt_stages = stages["stages_ordered"] if isinstance(stages, dict) else stages
        p = pred.get(cid, [])
        ps = p["stages_ordered"] if isinstance(p, dict) else p
        for s in gt_stages:
            den[s] += 1; num[s] += int(s in ps)
    return {s: (num[s] / den[s] if den[s] else None) for s in STAGE_ORDER}


def audit_row(flows, mask, costs, anchors, ys_mal):
    kf = float(costs[mask].sum() / costs.sum())
    nk = int(mask.sum())
    return {"kept_byte_frac": kf, "n_kept": nk,
            "malicious_frac": float(ys_mal[mask].mean()) if nk else 0.0,
            "anchor_frac": float(anchors[mask].mean()) if nk else 0.0}


# ---------------- Part A: Edge ----------------
def part_edge(flows, X, seeds=(0, 1, 2)):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    runs = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        tr, te = strat_split([f["attack"] for f in flows], rng)
        te = sorted(te, key=lambda i: flows[i]["t_first"])
        flows_tr = [flows[i] for i in tr]; fte = [flows[i] for i in te]
        Xtr, Xte = X[tr], X[te]
        ce = np.array([float(byte_cost(f)) for f in fte]); full = float(ce.sum())
        ctr = np.array([float(byte_cost(flows[i])) for i in tr])
        gt = build_ground_truth(fte, "edge-iiot")
        v = forensic_value_labels(flows_tr, "edge-iiot", n_rare_stages=3, anchor_focused=True)
        s_tr, _ = train_fern_scorer(Xtr, v, Xtr, dev=dev)
        s_te, _ = train_fern_scorer(Xtr, v, Xte, dev=dev)
        sm = train_stage_clf(Xtr, flows_tr, "edge-iiot")
        pst = predict_stages(sm, Xte)
        anc_te = anchor_labels(fte, "edge-iiot")
        ysm = np.array([0 if map_attack(f["attack"], "edge-iiot")[0] in (None,) else 1 for f in fte])
        res = {}
        for sel in ["greedy", "threshold_off", "threshold_on_fix", "threshold_on_adapt"]:
            curve = []
            for b in EDGE_BUDGETS:
                bb = b * full
                if sel == "greedy":
                    mask = keep_under_budget(np.argsort(-(s_te / np.sqrt(ce))), ce, bb)
                elif sel == "threshold_off":
                    mask = keep_threshold_offline(s_te, ce, bb)
                elif sel == "threshold_on_fix":
                    thr = calib_thr(s_tr, ctr, b * float(ctr.sum()))
                    mask = keep_threshold_online(s_te, ce, thr, bb)
                else:
                    from streaming_eval import stream_keep_adaptive
                    thr = calib_thr(s_tr, ctr, b * float(ctr.sum()))
                    mask = stream_keep_adaptive(s_te, ce, bb, 20, thr)
                fid = forensic_fidelity(fte, (pst != "Normal").astype(int) * mask.astype(int),
                                        "edge-iiot", gt=gt, pred_stages=pst)
                row = {"budget": b, "stage_recall": fid["stage_recall"], "chain": fid["chain_completeness"],
                       "budget_violation": bool(ce[mask].sum() > bb + 1)}
                row.update(audit_row(fte, mask, ce, anc_te, ysm))
                if sel in ("greedy", "threshold_off", "threshold_on_fix"):
                    row["per_stage"] = per_stage_recall(fte, mask, pst, gt, "edge-iiot")
                curve.append(row)
            res[sel] = curve
        runs.append(res)
    return aggregate(runs, EDGE_BUDGETS)


# ---------------- Part B: IoT-23 ----------------
def part_iot23(flows, X, seeds=(0, 1, 2)):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ys = I23.stage_y(flows)
    from hardening_suite import iot23_value_labels
    runs = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        scen = [f["src"].split(":")[0] for f in flows]
        tr, te = strat_split([f"{s}|{y}" for s, y in zip(scen, ys)], rng)
        te = sorted(te, key=lambda i: flows[i]["t_first"])
        Xtr, Xte = X[tr], X[te]; fte = [flows[i] for i in te]
        ce = np.array([max(1.0, f["bytes_tot"]) for f in fte]); full = float(ce.sum())
        ctr = np.array([max(1.0, flows[i]["bytes_tot"]) for i in tr])
        gt = I23.host_ground_truth(fte)
        if not gt: continue
        val_tr, _ = iot23_value_labels([flows[i] for i in tr])
        # anchor-focused teacher for IoT-23 too (host anchors)
        _, anc_tr = iot23_value_labels([flows[i] for i in tr])
        s_tr, _ = train_fern_scorer(Xtr, anc_tr, Xtr, dev=dev)
        s_te, _ = train_fern_scorer(Xtr, anc_tr, Xte, dev=dev)
        import lightgbm as lgb
        sm = lgb.train({"objective": "multiclass", "num_class": len(I23.CLASSES), "verbose": -1,
                        "num_leaves": 31, "learning_rate": 0.1}, lgb.Dataset(Xtr, label=ys[tr]), 80)
        pst = np.array([I23.CLASSES[i] for i in np.argmax(sm.predict(Xte), 1)], dtype=object)
        _, anc_te = iot23_value_labels(fte)
        ysm = (ys[te] != 0).astype(int)
        res = {}
        for sel in ["greedy", "threshold_off", "threshold_on_fix"]:
            curve = []
            for b in EDGE_BUDGETS:
                bb = b * full
                if sel == "greedy":
                    mask = keep_under_budget(np.argsort(-(s_te / np.sqrt(ce))), ce, bb)
                elif sel == "threshold_off":
                    mask = keep_threshold_offline(s_te, ce, bb)
                else:
                    thr = calib_thr(s_tr, ctr, b * float(ctr.sum()))
                    mask = keep_threshold_online(s_te, ce, thr, bb)
                mal = np.zeros(len(fte), dtype=int); kept = np.where(mask)[0]
                mal[kept] = (pst[kept] != "Normal").astype(int)
                fid = I23.fidelity(fte, mal, pst, gt)
                row = {"budget": b, "stage_recall": fid["stage_recall"], "chain": fid["chain_completeness"],
                       "budget_violation": bool(ce[mask].sum() > bb + 1)}
                row.update(audit_row(fte, mask, ce, anc_te, ysm))
                row["per_stage"] = per_stage_recall(fte, mask, pst, gt, "iot23", host=True)
                curve.append(row)
            res[sel] = curve
        runs.append(res)
    return aggregate(runs, EDGE_BUDGETS)


def aggregate(runs, budgets):
    if not runs: return {}
    sels = list(runs[0].keys())
    out = {"budgets": budgets, "n_seeds": len(runs), "selectors": {}}
    for sel in sels:
        agg = []
        for bi in range(len(budgets)):
            keys = ["stage_recall", "chain", "kept_byte_frac", "n_kept", "malicious_frac", "anchor_frac"]
            d = {k: {"mean": float(np.mean([r[sel][bi][k] for r in runs])),
                     "std": float(np.std([r[sel][bi][k] for r in runs]))} for k in keys}
            d["budget"] = budgets[bi]
            d["budget_violation_any"] = bool(any(r[sel][bi]["budget_violation"] for r in runs))
            if "per_stage" in runs[0][sel][bi]:
                ps = {}
                for s in STAGE_ORDER:
                    vals = [r[sel][bi]["per_stage"].get(s) for r in runs if r[sel][bi]["per_stage"].get(s) is not None]
                    ps[s] = float(np.mean(vals)) if vals else None
                d["per_stage"] = ps
            agg.append(d)
        out["selectors"][sel] = agg
    return out


# ---------------- Part C: sparsity indices ----------------
def sparsity_edge(flows):
    gt = build_ground_truth(flows, "edge-iiot")
    mal_bytes = 0.0; anchor_bytes = 0.0; flow_bytes = []
    stage_flow_counts = defaultdict(lambda: defaultdict(int))  # campaign->stage->count
    anchors = anchor_labels(flows, "edge-iiot")
    for i, f in enumerate(flows):
        st, _ = map_attack(f["attack"], "edge-iiot")
        if st in (None, UNKNOWN): continue
        b = float(byte_cost(f)); mal_bytes += b; flow_bytes.append(b)
        if anchors[i]: anchor_bytes += b
    return _sparsity_stats(gt, flows, anchors, mal_bytes, anchor_bytes, flow_bytes, "edge-iiot")


def sparsity_iot23(flows):
    from hardening_suite import iot23_value_labels
    gt = I23.host_ground_truth(flows)
    _, anchors = iot23_value_labels(flows)
    mal_bytes = 0.0; anchor_bytes = 0.0; flow_bytes = []
    for i, f in enumerate(flows):
        st, _ = I23.map_label(f["dlabel"])
        if not st: continue
        b = max(1.0, f["bytes_tot"]); mal_bytes += b; flow_bytes.append(b)
        if anchors[i]: anchor_bytes += b
    return _sparsity_stats(gt, flows, anchors, mal_bytes, anchor_bytes, flow_bytes, "iot23")


def _sparsity_stats(gt, flows, anchors, mal_bytes, anchor_bytes, flow_bytes, ds):
    fb = np.array(sorted(flow_bytes, reverse=True))
    top1 = fb[:max(1, len(fb)//100)].sum() / max(1e-9, fb.sum())
    # median flows per campaign-stage
    perstage = []
    is_host = (ds == "iot23")
    stage_of = (lambda f: I23.map_label(f["dlabel"])[0]) if is_host else (lambda f: map_attack(f["attack"], "edge-iiot")[0])
    camp_key = (lambda f: f["src"]) if is_host else None
    counts = defaultdict(lambda: defaultdict(int))
    for f in flows:
        st = stage_of(f)
        if st in (None, UNKNOWN): continue
        key = f["src"] if is_host else tuple(sorted((f["src"], f["dst"])))
        counts[key][st] += 1
    for k, sd in counts.items():
        for st, n in sd.items(): perstage.append(n)
    return {
        "n_campaigns": len(gt),
        "median_flows_per_stage": float(np.median(perstage)) if perstage else 0.0,
        "anchor_byte_share": float(anchor_bytes / max(1e-9, mal_bytes)),
        "byte_concentration_top1pct": float(top1),
    }


# ---------------- Controlled stress test: sparse-vs-dense controlled stress ----------------
def part_stress(flows, X, seed=0):
    """Hold attack flows fixed; vary stage redundancy by SUBSAMPLING each campaign-stage's
    non-anchor flows to a target keep-rate r (r=1 dense ... r small sparse-but-not-anchor),
    while INFLATING benign+flood background to keep total volume ~constant. Measure
    FERN-Threshold vs random advantage as a function of the resulting anchor-byte-share."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    anchors = anchor_labels(flows, "edge-iiot")
    mal_idx = [i for i, f in enumerate(flows) if map_attack(f["attack"], "edge-iiot")[0] not in (None, UNKNOWN)]
    ben_idx = [i for i, f in enumerate(flows) if map_attack(f["attack"], "edge-iiot")[0] is None]
    # rank malicious non-anchor flows by bytes (flood-like = big) to drop them first at low r
    nonanchor = [i for i in mal_idx if not anchors[i]]
    curves = []
    for r in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02]:
        keep_na = set(rng.choice(nonanchor, size=int(r * len(nonanchor)), replace=False).tolist())
        sub = [i for i in range(len(flows)) if (anchors[i] or i in keep_na or
               map_attack(flows[i]["attack"], "edge-iiot")[0] is None)]
        sub_flows = [flows[i] for i in sub]; sub_X = X[sub]
        # recompute split + scorer on the subsampled corpus
        rng2 = np.random.default_rng(0)
        tr, te = strat_split([f["attack"] for f in sub_flows], rng2)
        ftr = [sub_flows[i] for i in tr]; fte = [sub_flows[i] for i in te]
        Xtr, Xte = sub_X[tr], sub_X[te]
        ce = np.array([float(byte_cost(f)) for f in fte]); full = float(ce.sum())
        gt = build_ground_truth(fte, "edge-iiot")
        if not gt: continue
        v = forensic_value_labels(ftr, "edge-iiot", n_rare_stages=3, anchor_focused=True)
        s_te, _ = train_fern_scorer(Xtr, v, Xte, dev=dev)
        sm = train_stage_clf(Xtr, ftr, "edge-iiot"); pst = predict_stages(sm, Xte)
        # anchor byte share of THIS subsample
        ab = sum(float(byte_cost(f)) for i, f in enumerate(fte)
                 if anchor_labels(fte, "edge-iiot")[i])
        mb = sum(float(byte_cost(f)) for f in fte if map_attack(f["attack"], "edge-iiot")[0] not in (None, UNKNOWN))
        share = ab / max(1e-9, mb)
        b = 0.02
        bb = b * full
        m_fern = keep_threshold_offline(s_te, ce, bb)
        m_rand = keep_under_budget(rng.permutation(len(fte)), ce, bb)
        def sr(mask):
            return forensic_fidelity(fte, (pst != "Normal").astype(int) * mask.astype(int),
                                     "edge-iiot", gt=gt, pred_stages=pst)["stage_recall"]
        srf, srr = sr(m_fern), sr(m_rand)
        curves.append({"keep_rate": r, "anchor_byte_share": float(share),
                       "fern": srf, "random": srr, "advantage": srf - srr})
        print(f"  r={r}: anchor_byte_share={share:.4f} fern={srf:.3f} random={srr:.3f} adv={srf-srr:.3f}")
    return curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge-flows", default="data/processed/edge_flows_full.jsonl")
    ap.add_argument("--iot23-dir", default="data/raw/iot23")
    ap.add_argument("--out", default="outputs/evaluation_suite.json")
    ap.add_argument("--parts", default="A,B,C,D")
    args = ap.parse_args()
    parts = set(args.parts.split(","))
    eflows = load_flows(args.edge_flows); eX, _ = make_xy(eflows, "edge-iiot")
    iflows = []
    if {"B", "C", "D"} & parts:
        for fn in sorted(os.listdir(args.iot23_dir)):
            if not fn.endswith(".csv"): continue
            fl = I23.parse_csv(os.path.join(args.iot23_dir, fn), 300000)
            tag = fn.replace("dataset", "s").replace(".csv", "")
            for f in fl: f["src"] = f"{tag}:{f['src']}"; f["dst"] = f"{tag}:{f['dst']}"
            iflows += fl
    iX = I23.featmat(iflows) if iflows else None
    import sys
    out = {}
    def flush(): json.dump(out, open(args.out, "w"), indent=2)
    if "A" in parts:
        print("== Part A: Edge unified retention =="); sys.stdout.flush()
        out["edge"] = part_edge(eflows, eX); print("A done"); flush(); sys.stdout.flush()
    if "B" in parts:
        print("== Part B: IoT-23 unified retention =="); sys.stdout.flush()
        out["iot23"] = part_iot23(iflows, iX); print("B done"); flush(); sys.stdout.flush()
    if "C" in parts:
        print("== Part C: sparsity =="); sys.stdout.flush()
        out["sparsity"] = {"edge-iiot": sparsity_edge(eflows),
                           "iot23": sparsity_iot23(iflows) if iflows else None}
        print(json.dumps(out["sparsity"], indent=1)); flush()
    if "D" in parts:
        print("== Controlled stress test: sparse-vs-dense stress =="); sys.stdout.flush()
        out["stress"] = part_stress(eflows, eX); print("D done"); flush()
    flush()
    print("wrote", args.out)


if __name__ == "__main__":
    main()



