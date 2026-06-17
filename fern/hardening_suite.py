"""Additional validation suite for retention and ablations.

IoT-23 forensic rate-distortion
  - in-dataset FERN + 7 baselines + first-anchor reference, budgets {0.1..10}%, 3 seeds (mean/std)
  - cross-dataset: FERN scorer TRAINED ON EDGE-IIOTSET, applied to IoT-23 retention
  - leave-one-scenario-out (LOSO): scorer+investigator trained on 15 scenarios, retention on the held-out one
  Byte cost = orig_ip_bytes + resp_ip_bytes (missing -> 0, stated). Evidence units are Zeek
  flow records; per-policy costs mirror the Edge setup (netflow record=80B, head-of-flow=200B
  with masked investigator features, raw policies pay full represented bytes).

Edge-IIoTset ablations
  - teacher variants: rare_only / anchor_only / rare+anchor (FERN) / all_malicious / random_positive
  - value-per-byte exponent alpha in {0,0.25,0.5,0.75,1.0}
  - feature-group drops: ports / volume / timing / proto / netflow_like_only

bounded revision buffer online (Edge + IoT-23)
  - two-tier causal: permanent keep above thr; bounded buffer (0.1% bytes) for borderline flows,
    evict-lowest-score within budget; buffered flows finalize as retained. Goal: chain>0 online.

Output: one JSON with every block. All decisions causal/train-only as in the audited harness.
"""
import argparse, json, os
import numpy as np
from collections import defaultdict

# ---------- shared helpers (Edge side) ----------
from pilot_k1 import load_flows, FEATS, make_xy
from retention import (forensic_value_labels, anchor_labels, train_fern_scorer,
                       byte_cost, keep_under_budget)
from fidelity import build_ground_truth, forensic_fidelity
from stage_clf import train_stage_clf, predict_stages
from stage_mapping import map_attack, UNKNOWN

# ---------- IoT-23 side ----------
import iot23_pipeline as I23

BUDGETS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]


def split_strat(keys, rng, frac=0.7):
    buckets = defaultdict(list)
    for i, k in enumerate(keys): buckets[k].append(i)
    tr, te = [], []
    for k, idxs in buckets.items():
        idxs = list(idxs); rng.shuffle(idxs)
        c = int(frac * len(idxs)); tr += idxs[:c]; te += idxs[c:]
    return tr, te


# ===================== IoT-23 retention =====================
def iot23_value_labels(flows, n_rare=3):
    """FERN teacher on IoT-23: malicious flows of the n_rare rarest stages + per-(host,stage)
    first flows, with the same >=2-stage campaign filter (host-centric)."""
    cnt = defaultdict(int); stages = []
    for f in flows:
        st, _ = I23.map_label(f["dlabel"]); stages.append(st)
        if st: cnt[st] += 1
    rare = set(sorted(cnt, key=lambda s: cnt[s])[:n_rare])
    # host-centric anchors (>=2 stages)
    host_items = defaultdict(list)
    for i, f in enumerate(flows):
        if stages[i]: host_items[f["src"]].append((i, stages[i], f["t_first"]))
    anchors = np.zeros(len(flows), dtype=np.int64)
    for h, items in host_items.items():
        if len({st for _, st, _ in items}) < 2: continue
        first = {}
        for i, st, ts in items:
            if st not in first or ts < first[st][1]: first[st] = (i, ts)
        for st, (i, _) in first.items(): anchors[i] = 1
    val = np.zeros(len(flows), dtype=np.int64)
    for i, st in enumerate(stages):
        if st and (st in rare or anchors[i]): val[i] = 1
    return val, anchors


def iot23_first_anchor_mask_order(flows_te, anchors_te, ce):
    return np.argsort(-(anchors_te.astype(float)) - 1e-9 * ce)


def iot23_retention(flows, X, seeds=(0, 1, 2), cross_scorer=None, tag=""):
    """Run the budget x policy grid on IoT-23. cross_scorer: optional (mu,sd,net) trained on
    Edge -used as the 'fern_cross' policy. Returns aggregate dict."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ys = I23.stage_y(flows)
    runs = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        scen = [f["src"].split(":")[0] for f in flows]
        keys = [f"{s}|{y}" for s, y in zip(scen, ys)]
        tr, te = split_strat(keys, rng)
        Xtr = X[tr]; fte = [flows[i] for i in te]; Xte = X[te]
        ce = np.array([max(1.0, f["bytes_tot"]) for f in fte])
        full = float(ce.sum())
        gt = I23.host_ground_truth(fte)
        if not gt: continue
        # teachers/models on TRAIN only
        val_tr, _ = iot23_value_labels([flows[i] for i in tr])
        fern_te, _ = train_fern_scorer(Xtr, val_tr, Xte, dev=dev)
        # train-only multiclass investigator
        import lightgbm as lgb
        sm = lgb.train({"objective": "multiclass", "num_class": len(I23.CLASSES),
                        "verbose": -1, "num_leaves": 31, "learning_rate": 0.1},
                       lgb.Dataset(Xtr, label=ys[tr]), 80)
        pstage_full = np.array([I23.CLASSES[i] for i in np.argmax(sm.predict(Xte), 1)], dtype=object)
        # head-of-flow masked investigator (totals unknown from 200B)
        mu_tr = Xtr.mean(0)
        HEAD_OK = {"sport", "dport", "proto_tcp", "proto_udp", "proto_icmp", "ratio_fwd", "mean_pkt"}
        Xm = Xte.copy()
        for j, k in enumerate(I23.FEATS):
            if k not in HEAD_OK: Xm[:, j] = mu_tr[j]
        pstage_head = np.array([I23.CLASSES[i] for i in np.argmax(sm.predict(Xm), 1)], dtype=object)
        # scores
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        anomaly = np.linalg.norm((Xte - mu) / sd, axis=1)
        ids_b = lgb.train({"objective": "binary", "verbose": -1, "num_leaves": 31, "learning_rate": 0.1},
                          lgb.Dataset(Xtr, label=(ys[tr] != 0).astype(int)), 60).predict(Xte)
        _, anchors_te = iot23_value_labels(fte)
        cross_te = None
        if cross_scorer is not None:
            cmu, csd, cnet = cross_scorer
            import torch as T
            with T.no_grad():
                cross_te = T.sigmoid(cnet(T.tensor((Xte - cmu) / csd, dtype=T.float32,
                                                   device=dev)).squeeze(-1)).cpu().numpy()
        POL = ["random", "recency", "reservoir", "netflow_only", "head_of_flow",
               "anomaly_topk", "alert_triggered", "fern", "first_anchor_ref"]
        if cross_te is not None: POL.append("fern_cross")
        res = {"n_gt": len(gt)}
        for pol in POL:
            pc = ce.copy()
            if pol == "netflow_only": pc = np.minimum(ce, 80.0)
            if pol == "head_of_flow": pc = np.minimum(ce, 200.0)
            pst = pstage_head if pol == "head_of_flow" else pstage_full
            curve = []
            for b in BUDGETS:
                bb = b * full
                if pol == "random": o = rng.permutation(len(fte))
                elif pol == "recency": o = np.argsort([-f["t_last"] for f in fte])
                elif pol == "reservoir": o = rng.permutation(len(fte))
                elif pol == "netflow_only": o = np.argsort(ce)
                elif pol == "head_of_flow": o = np.argsort([-f["n_tot"] for f in fte])
                elif pol == "anomaly_topk": o = np.argsort(-anomaly)
                elif pol == "alert_triggered": o = np.argsort(-ids_b)
                elif pol == "fern": o = np.argsort(-(fern_te / np.sqrt(ce)))
                elif pol == "fern_cross": o = np.argsort(-(cross_te / np.sqrt(ce)))
                else: o = iot23_first_anchor_mask_order(fte, anchors_te, ce)
                mask = keep_under_budget(o, pc, bb)
                mal = np.zeros(len(fte), dtype=int)
                kept = np.where(mask)[0]
                mal[kept] = (pst[kept] != "Normal").astype(int)
                fid = I23.fidelity(fte, mal, pst, gt)
                curve.append({"budget": b, "stage_recall": fid["stage_recall"],
                              "chain": fid["chain_completeness"]})
            res[pol] = curve
        runs.append(res)
    # aggregate
    agg = {"budgets": BUDGETS, "n_seeds": len(runs),
           "n_gt_mean": float(np.mean([r["n_gt"] for r in runs]))}
    pols = [k for k in runs[0] if k != "n_gt"]
    for pol in pols:
        agg[pol] = {"stage_recall_mean": [float(np.mean([r[pol][i]["stage_recall"] for r in runs])) for i in range(len(BUDGETS))],
                    "stage_recall_std": [float(np.std([r[pol][i]["stage_recall"] for r in runs])) for i in range(len(BUDGETS))],
                    "chain_mean": [float(np.mean([r[pol][i]["chain"] for r in runs])) for i in range(len(BUDGETS))]}
    return agg


def iot23_loso(flows, X, max_scen=16):
    """Leave-one-scenario-out: train scorer+investigator on other scenarios, retention on the
    held-out one (only scenarios that contain >=1 host campaign count). FERN vs random/anomaly @5%."""
    import torch, lightgbm as lgb
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ys = I23.stage_y(flows)
    scen = np.array([f["src"].split(":")[0] for f in flows])
    out = {}
    for s in sorted(set(scen)):
        te = np.where(scen == s)[0]; tr = np.where(scen != s)[0]
        fte = [flows[i] for i in te]
        gt = I23.host_ground_truth(fte)
        if not gt: continue
        Xtr, Xte = X[tr], X[te]
        ce = np.array([max(1.0, f["bytes_tot"]) for f in fte]); full = float(ce.sum())
        val_tr, _ = iot23_value_labels([flows[i] for i in tr])
        fern_te, _ = train_fern_scorer(Xtr, val_tr, Xte, dev=dev)
        sm = lgb.train({"objective": "multiclass", "num_class": len(I23.CLASSES), "verbose": -1,
                        "num_leaves": 31, "learning_rate": 0.1}, lgb.Dataset(Xtr, label=ys[tr]), 80)
        pst = np.array([I23.CLASSES[i] for i in np.argmax(sm.predict(Xte), 1)], dtype=object)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        anomaly = np.linalg.norm((Xte - mu) / sd, axis=1)
        rng = np.random.default_rng(0)
        row = {}
        for pol, order in [("fern", np.argsort(-(fern_te / np.sqrt(ce)))),
                           ("anomaly", np.argsort(-anomaly)),
                           ("random", rng.permutation(len(fte)))]:
            mask = keep_under_budget(order, ce, 0.05 * full)
            mal = np.zeros(len(fte), dtype=int); kept = np.where(mask)[0]
            mal[kept] = (pst[kept] != "Normal").astype(int)
            row[pol] = I23.fidelity(fte, mal, pst, gt)["stage_recall"]
        row["n_campaigns"] = len(gt)
        out[s] = row
    means = {p: float(np.mean([v[p] for v in out.values()])) for p in ["fern", "anomaly", "random"]}
    return {"per_scenario": out, "mean": means, "n_scenarios": len(out)}


# ===================== Edge ablations =====================
def edge_ablations(flows, X, seed=0):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    keys = [f["attack"] for f in flows]
    tr, te = split_strat(keys, rng)
    flows_tr = [flows[i] for i in tr]; fte = [flows[i] for i in te]
    Xtr, Xte = X[tr], X[te]
    ce = np.array([float(byte_cost(f)) for f in fte]); full = float(ce.sum())
    gt = build_ground_truth(fte, "edge-iiot")
    sm = train_stage_clf(Xtr, flows_tr, "edge-iiot")
    pst = predict_stages(sm, Xte)

    def run_order(order, budgets=(0.05, 0.10)):
        outs = {}
        for b in budgets:
            mask = keep_under_budget(order, ce, b * full)
            mal = np.zeros(len(fte), dtype=int); kept = np.where(mask)[0]
            mal[kept] = (pst[kept] != "Normal").astype(int)
            outs[b] = forensic_fidelity(fte, mal, "edge-iiot", gt=gt, pred_stages=pst)["stage_recall"]
        return outs

    # ---- teacher ablation ----
    def teacher_variant(kind):
        anchors = anchor_labels(flows_tr, "edge-iiot")
        cnt = defaultdict(int); stages = []
        for f in flows_tr:
            st, _ = map_attack(f["attack"], "edge-iiot"); stages.append(st)
            if st not in (None, UNKNOWN): cnt[st] += 1
        rare = set(sorted(cnt, key=lambda s: cnt[s])[:3])
        val = np.zeros(len(flows_tr), dtype=np.int64)
        if kind == "rare_only":
            for i, st in enumerate(stages):
                if st in rare: val[i] = 1
        elif kind == "anchor_only":
            val = anchors.copy()
        elif kind == "rare_plus_anchor":
            for i, st in enumerate(stages):
                if (st in rare) or anchors[i]: val[i] = 1
        elif kind == "all_malicious":
            for i, st in enumerate(stages):
                if st not in (None, UNKNOWN): val[i] = 1
        elif kind == "random_positive":
            base = forensic_value_labels(flows_tr, "edge-iiot", n_rare_stages=3)
            k = int(base.sum())
            idx = rng.choice(len(flows_tr), size=k, replace=False)
            val[idx] = 1
        return val
    teacher = {}
    for kind in ["rare_only", "anchor_only", "rare_plus_anchor", "all_malicious", "random_positive"]:
        v = teacher_variant(kind)
        s_te, _ = train_fern_scorer(Xtr, v, Xte, dev=dev)
        teacher[kind] = run_order(np.argsort(-(s_te / np.sqrt(ce))))

    # ---- alpha sweep (uses the standard teacher) ----
    v_std = forensic_value_labels(flows_tr, "edge-iiot", n_rare_stages=3)
    s_std, _ = train_fern_scorer(Xtr, v_std, Xte, dev=dev)
    alphas = {}
    for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
        alphas[str(a)] = run_order(np.argsort(-(s_std / np.power(ce, a))))

    # ---- feature-group ablation (mask in BOTH scorer training and scoring) ----
    GROUPS = {
        "no_ports": {"sport", "dport"},
        "no_volume": {"n_fwd", "n_bwd", "n_tot", "bytes_fwd", "bytes_bwd", "bytes_tot", "mean_pkt"},
        "no_timing": {"dur", "pps", "bps"},
        "no_proto": {"proto_tcp", "proto_udp", "proto_icmp"},
        "netflow_like_only": set(FEATS) - {"dur", "n_tot", "bytes_tot", "pps", "bps", "mean_pkt",
                                           "proto_tcp", "proto_udp", "proto_icmp"},
    }
    feats_ab = {}
    for name, drop in GROUPS.items():
        keep_idx = [j for j, k in enumerate(FEATS) if k not in drop]
        s_te, _ = train_fern_scorer(Xtr[:, keep_idx], v_std, Xte[:, keep_idx], dev=dev)
        feats_ab[name] = run_order(np.argsort(-(s_te / np.sqrt(ce))))
    feats_ab["all_features"] = run_order(np.argsort(-(s_std / np.sqrt(ce))))
    return {"teacher": teacher, "alpha": alphas, "features": feats_ab}


# ===================== bounded revision buffer online =====================
def buffered_online(scores, costs, budget_bytes, buffer_frac=0.001, n_windows=20, init_thr=None):
    """Two-tier causal online: permanent keep if score>=thr (adaptive, prev-window quantile);
    else flow enters a bounded candidate buffer (<= buffer_frac of full bytes) that keeps the
    HIGHEST-scoring borderline flows seen so far, evicting the lowest (eviction = irrevocable
    drop of the evicted flow only). Final retained set = permanent + buffer (within budget+buffer).
    Causality: every decision uses only current/past scores; no future information."""
    n = len(scores); w = max(1, n // n_windows)
    thr = init_thr if init_thr is not None else np.inf
    bpw = budget_bytes / n_windows
    avail = 0.0
    permanent = np.zeros(n, dtype=bool)
    buf = []  # list of (score, idx, cost), kept sorted asc by score
    buf_cap = buffer_frac * (costs.sum())
    buf_bytes = 0.0
    for wi in range(n_windows):
        lo, hi = wi * w, (n if wi == n_windows - 1 else (wi + 1) * w)
        avail += bpw
        spent = 0.0
        for i in range(lo, hi):
            c = costs[i]
            if scores[i] >= thr and spent + c <= avail:
                permanent[i] = True; spent += c
            else:
                # buffer admission: keep if room, or better than current worst
                if buf_bytes + c <= buf_cap:
                    buf.append((scores[i], i, c)); buf_bytes += c; buf.sort()
                elif buf and scores[i] > buf[0][0]:
                    while buf and buf_bytes + c > buf_cap:
                        s0, i0, c0 = buf.pop(0); buf_bytes -= c0
                    if buf_bytes + c <= buf_cap:
                        buf.append((scores[i], i, c)); buf_bytes += c; buf.sort()
        avail -= spent
        ws, wc = scores[lo:hi], costs[lo:hi]
        if len(ws):
            order = np.argsort(-ws); cum, t_new = 0.0, thr
            for idx in order:
                if cum + wc[idx] > bpw: break
                cum += wc[idx]; t_new = ws[idx]
            thr = 0.7 * thr + 0.3 * t_new if np.isfinite(thr) else t_new
    mask = permanent.copy()
    for s0, i0, c0 in buf: mask[i0] = True
    return mask


def edge_buffer_online(flows, X, seed=0):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    keys = [f["attack"] for f in flows]
    tr, te = split_strat(keys, rng)
    te = sorted(te, key=lambda i: flows[i]["t_first"])
    flows_tr = [flows[i] for i in tr]; fte = [flows[i] for i in te]
    Xtr, Xte = X[tr], X[te]
    ce = np.array([float(byte_cost(f)) for f in fte]); full = float(ce.sum())
    gt = build_ground_truth(fte, "edge-iiot")
    v = forensic_value_labels(flows_tr, "edge-iiot", n_rare_stages=3)
    s_tr, _ = train_fern_scorer(Xtr, v, Xtr, dev=dev)
    s_te, _ = train_fern_scorer(Xtr, v, Xte, dev=dev)
    sm = train_stage_clf(Xtr, flows_tr, "edge-iiot")
    pst = predict_stages(sm, Xte)
    ctr = np.array([float(byte_cost(flows[i])) for i in tr])
    from streaming_eval import calib_threshold, stream_keep, stream_keep_adaptive
    out = {}
    for b in [0.01, 0.02, 0.05, 0.10]:
        bb = b * full
        thr = calib_threshold(s_tr, ctr, b * float(ctr.sum()))
        rows = {}
        for name, mask in [
            ("fixed", stream_keep(s_te, ce, thr, bb)),
            ("adaptive", stream_keep_adaptive(s_te, ce, bb, 20, thr)),
            ("adaptive_buffer", buffered_online(s_te, ce, bb, 0.001, 20, thr)),
        ]:
            mal = np.zeros(len(fte), dtype=int); kept = np.where(mask)[0]
            mal[kept] = (pst[kept] != "Normal").astype(int)
            fid = forensic_fidelity(fte, mal, "edge-iiot", gt=gt, pred_stages=pst)
            rows[name] = {"stage_recall": fid["stage_recall"], "chain": fid["chain_completeness"]}
        out[str(b)] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge-flows", default="data/processed/edge_flows_full.jsonl")
    ap.add_argument("--iot23-dir", default="data/raw/iot23")
    ap.add_argument("--out", default="outputs/hardening.json")
    ap.add_argument("--parts", default="1,2,3")
    args = ap.parse_args()
    parts = set(args.parts.split(","))
    out = {}

    # load Edge
    eflows = load_flows(args.edge_flows)
    eX, _ = make_xy(eflows, "edge-iiot")
    # load IoT-23
    files = sorted(os.listdir(args.iot23_dir))
    iflows = []
    for fn in files:
        if not fn.endswith(".csv"): continue
        fl = I23.parse_csv(os.path.join(args.iot23_dir, fn), 300000)
        tag = fn.replace("dataset", "s").replace(".csv", "")
        for f in fl: f["src"] = f"{tag}:{f['src']}"; f["dst"] = f"{tag}:{f['dst']}"
        iflows += fl
    iX = I23.featmat(iflows)
    print(f"edge {len(eflows)} flows | iot23 {len(iflows)} flows")

    if "1" in parts:
        # cross-dataset scorer trained on FULL Edge train side
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        rng = np.random.default_rng(0)
        tr, _ = split_strat([f["attack"] for f in eflows], rng)
        v = forensic_value_labels([eflows[i] for i in tr], "edge-iiot", n_rare_stages=3)
        # train and capture the net for reuse
        import torch.nn as nn
        Xtr = eX[tr]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xt = torch.tensor((Xtr - mu) / sd, device=dev); at = torch.tensor(v, device=dev).float()
        net = nn.Sequential(nn.Linear(Xtr.shape[1], 32), nn.ReLU(), nn.Linear(32, 1)).to(dev)
        opt = torch.optim.Adam(net.parameters(), 2e-3)
        pw = torch.tensor([(len(v) - v.sum()) / max(1, v.sum())], device=dev)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        for _ in range(12):
            perm = torch.randperm(len(Xt), device=dev)
            for i in range(0, len(Xt), 4096):
                idx = perm[i:i+4096]; opt.zero_grad()
                lossf(net(Xt[idx]).squeeze(-1), at[idx]).backward(); opt.step()
        out["iot23_retention"] = iot23_retention(iflows, iX, cross_scorer=(mu, sd, net))
        print("iot23 retention done")
        out["iot23_loso"] = iot23_loso(iflows, iX)
        print("loso done")
    if "2" in parts:
        out["edge_ablations"] = edge_ablations(eflows, eX)
        print("ablations done")
    if "3" in parts:
        out["edge_buffer_online"] = edge_buffer_online(eflows, eX)
        print("buffer online done")
    json.dump(out, open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()



