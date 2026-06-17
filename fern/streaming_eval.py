"""Chronological streaming retention evaluation.

Flows arrive in t_first order. Each policy makes an IRREVOCABLE keep/drop decision using only
the current flow + a threshold calibrated on TRAIN (never future test flows). A running byte
budget is enforced; once exhausted, no more can be kept (budget-exhaustion behavior is explicit).

This contrasts with the offline budgeted top-k in retention.py and answers "does FERN work under
online gateway constraints?". FERN keeps a flow if its forensic-importance score exceeds a
train-calibrated threshold AND budget remains. Compares FERN vs random/recency/anomaly/netflow.
"""
import argparse, json
import numpy as np
from collections import defaultdict
from stage_mapping import map_attack, UNKNOWN
from fidelity import build_ground_truth, forensic_fidelity
from pilot_k1 import load_flows, FEATS, make_xy
from retention import forensic_value_labels, train_fern_scorer, byte_cost
from stage_clf import train_stage_clf, predict_stages

BUDGETS = [0.005, 0.01, 0.02, 0.05, 0.10]


def calib_threshold(scores, costs, budget_bytes):
    """Train-side: highest threshold whose kept (score>=thr) cost <= budget. Calibrated on
    TRAIN scores only, then applied irrevocably online to test."""
    order = np.argsort(-scores)
    spent = 0.0
    thr = np.inf
    for idx in order:
        if spent + costs[idx] > budget_bytes:
            break
        spent += costs[idx]; thr = scores[idx]
    return thr


def stream_keep(scores, costs, thr, budget_bytes):
    """Online irrevocable: walk flows in ARRIVAL order, keep if score>=thr and budget remains."""
    mask = np.zeros(len(scores), dtype=bool)
    spent = 0.0
    for i in range(len(scores)):          # arrival order (already time-sorted)
        if scores[i] >= thr and spent + costs[i] <= budget_bytes:
            mask[i] = True; spent += costs[i]
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows", required=True)
    ap.add_argument("--dataset", default="edge-iiot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/streaming.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    flows = load_flows(args.flows)
    X, y = make_xy(flows, args.dataset)
    costs = np.array([byte_cost(f) for f in flows], dtype=float)

    # stratified train/test (train = calibrate scorer+threshold; test = stream chronologically)
    buckets = defaultdict(list)
    for i, f in enumerate(flows):
        buckets[f["attack"]].append(i)
    tr, te = [], []
    for a, idxs in buckets.items():
        idxs = list(idxs); rng.shuffle(idxs); k = int(0.7*len(idxs)); tr += idxs[:k]; te += idxs[k:]
    # TEST must be in chronological arrival order for true streaming
    te = sorted(te, key=lambda i: flows[i]["t_first"])

    fvalue_tr = forensic_value_labels([flows[i] for i in tr], args.dataset, n_rare_stages=3, anchor_focused=True)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fern_tr, _ = train_fern_scorer(X[tr], fvalue_tr, X[tr], dev=dev)       # train scores (calib)
    fern_te, _ = train_fern_scorer(X[tr], fvalue_tr, X[te], dev=dev)       # test scores
    stage_model = train_stage_clf(X[tr], [flows[i] for i in tr], args.dataset)
    pred_stages_te = predict_stages(stage_model, X[te])

    fe = [flows[i] for i in te]; ce = costs[te]
    gt = build_ground_truth(fe, args.dataset)
    full_bytes = float(ce.sum())
    mu, sd = X[tr].mean(0), X[tr].std(0)+1e-6
    anomaly_tr = np.linalg.norm((X[tr]-mu)/sd, axis=1)
    anomaly_te = np.linalg.norm((X[te]-mu)/sd, axis=1)

    POL = {
        "fern": (fern_tr, fern_te),
        "anomaly": (anomaly_tr, anomaly_te),
        "random": (rng.random(len(tr)), rng.random(len(te))),
        "recency": (np.arange(len(tr)), np.arange(len(te))),   # later arrival = higher (recency)
    }
    results = {"budgets": BUDGETS, "policies": {}}
    # adaptive variants use the SAME scores; only the (causal) thresholding differs
    POL["fern_adaptive"] = POL["fern"]
    for name, (s_tr, s_te) in POL.items():
        curve = []
        for b in BUDGETS:
            bb = b*full_bytes
            thr = calib_threshold(s_tr, costs[tr], b*float(costs[tr].sum()))  # train-calibrated
            if name.endswith("_adaptive"):
                mask = stream_keep_adaptive(s_te, ce, bb, n_windows=20, init_thr=thr)
            else:
                mask = stream_keep(s_te, ce, thr, bb)
            preds = np.zeros(len(fe), dtype=int)
            kept = np.where(mask)[0]
            preds[kept] = (pred_stages_te[kept] != "Normal").astype(int)
            fid = forensic_fidelity(fe, preds, args.dataset, gt=gt, pred_stages=pred_stages_te)
            curve.append({"budget": b, "stage_recall": fid["stage_recall"],
                          "chain": fid["chain_completeness"], "kept_frac": float(ce[mask].sum()/full_bytes)})
        results["policies"][name] = curve
        print(f"[{name:8s}] stage_recall " + " ".join(f"{c['stage_recall']:.2f}" for c in curve))
    json.dump(results, open(args.out, "w"), indent=2)
    print("wrote", args.out, "| ONLINE: irrevocable, train-calibrated threshold, no future info")


def stream_keep_adaptive(scores, costs, budget_bytes, n_windows=20, init_thr=None):
    """Adaptive online retention (strictly causal): the stream is split into equal-duration
    windows; each window gets a prorated byte budget (unused budget rolls over). Within a
    window, a flow is kept if its score exceeds the threshold adapted from the PREVIOUS
    window's observed scores (the quantile that would have met the byte rate), so every
    decision uses only past information. First window uses the train-calibrated threshold.
    """
    n = len(scores)
    w = max(1, n // n_windows)
    mask = np.zeros(n, dtype=bool)
    thr = init_thr if init_thr is not None else np.inf
    budget_per_window = budget_bytes / n_windows
    avail = 0.0
    for wi in range(n_windows):
        lo, hi = wi * w, (n if wi == n_windows - 1 else (wi + 1) * w)
        avail += budget_per_window
        spent = 0.0
        for i in range(lo, hi):
            if scores[i] >= thr and spent + costs[i] <= avail:
                mask[i] = True; spent += costs[i]
        avail -= spent
        # adapt: choose the quantile of THIS window's scores that would have spent the
        # window budget exactly (used only for FUTURE windows -> causal)
        ws, wc = scores[lo:hi], costs[lo:hi]
        if len(ws):
            order = np.argsort(-ws)
            cum, t_new = 0.0, thr
            for idx in order:
                if cum + wc[idx] > budget_per_window: break
                cum += wc[idx]; t_new = ws[idx]
            thr = 0.7 * thr + 0.3 * t_new if np.isfinite(thr) else t_new
    return mask


if __name__ == "__main__":
    main()



