"""Baseline diagnostic: detection F1 does not predict forensic fidelity.

Train a small IDS family to *matched* detection F1, then measure how far apart they
land on the 4 forensic-fidelity metrics. If high-F1 models that miss low-volume
early-stage (Recon/Exploit) flows have much lower stage_recall/chain_completeness,
K1 holds.

Models (flow-level binary malicious/benign):
  - LightGBM (gradient boosting)
  - Logistic Regression
  - MLP (torch)
  - "Majority-volume" model: strong on high-volume DDoS, weak on rare Recon (stress case)

Reports per-model detection F1 + fidelity, and the Spearman correlation across models.
A LOW correlation + LARGE fidelity spread among matched-F1 models = K1 confirmed.
"""
import argparse
import json
import numpy as np
from collections import defaultdict
from stage_mapping import map_attack, UNKNOWN
from fidelity import build_ground_truth, forensic_fidelity

FEATS = ["dur", "n_fwd", "n_bwd", "n_tot", "bytes_fwd", "bytes_bwd", "bytes_tot",
         "pps", "bps", "mean_pkt", "ratio_fwd", "sport", "dport",
         "proto_tcp", "proto_udp", "proto_icmp"]


def load_flows(path, max_flows=0):
    flows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            flows.append(r)
            if max_flows and len(flows) >= max_flows:
                break
    return flows


def make_xy(flows, dataset, mask_ids=False):
    X = np.array([[float(fl.get(k, 0) or 0) for k in FEATS] for fl in flows], dtype=np.float32)
    if mask_ids:
        # identifier-masking probe: zero out sport/dport (identifier shortcut)
        for j, k in enumerate(FEATS):
            if k in ("sport", "dport"):
                X[:, j] = 0.0
    y = np.array([0 if map_attack(fl["attack"], dataset)[0] in (None,) else 1
                  for fl in flows], dtype=np.int64)
    return X, y


def calibrate_threshold(y, score, target_recall):
    """Pick the threshold whose recall is closest to target_recall (matched operating
    point so K1 compares models at the SAME detection level)."""
    order = np.argsort(-score)
    P = max(1, int((y == 1).sum()))
    tp = 0
    best_thr, best_gap = 0.5, 1e9
    for k, idx in enumerate(order, 1):
        if y[idx] == 1:
            tp += 1
        rec = tp / P
        gap = abs(rec - target_recall)
        if gap < best_gap:
            best_gap, best_thr = gap, score[idx]
    return best_thr


def detection_metrics(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    acc = (tp + tn) / max(1, len(y))
    return {"f1": f1, "precision": prec, "recall": rec, "accuracy": acc}


def train_lgbm(Xtr, ytr, Xte):
    import lightgbm as lgb
    d = lgb.Dataset(Xtr, label=ytr)
    m = lgb.train({"objective": "binary", "verbose": -1, "num_leaves": 31,
                   "learning_rate": 0.1}, d, num_boost_round=60)
    return (m.predict(Xte) > 0.5).astype(int), m.predict(Xte)


def train_logreg(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=200, C=1.0).fit(sc.transform(Xtr), ytr)
    p = m.predict_proba(sc.transform(Xte))[:, 1]
    return (p > 0.5).astype(int), p


def train_mlp(Xtr, ytr, Xte, epochs=8, dev="cuda"):
    import torch, torch.nn as nn
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_n = (Xtr - mu) / sd; Xte_n = (Xte - mu) / sd
    Xt = torch.tensor(Xtr_n, device=dev); yt = torch.tensor(ytr, device=dev).float()
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 64), nn.ReLU(),
                        nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    lossf = nn.BCEWithLogitsLoss()
    bs = 4096
    for ep in range(epochs):
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            out = net(Xt[idx]).squeeze(-1)
            loss = lossf(out, yt[idx]); loss.backward(); opt.step()
    with torch.no_grad():
        p = torch.sigmoid(net(torch.tensor(Xte_n, device=dev)).squeeze(-1)).cpu().numpy()
    return (p > 0.5).astype(int), p


def volume_biased(Xtr, ytr, Xte, flows_te, dataset):
    """Stress model: only flags high-volume flows (mimics a detector that learns DDoS
    but ignores low-volume Recon/Exploit). High overall F1 if DDoS dominates, but
    forensically blind to early stages."""
    n_tot = Xte[:, FEATS.index("n_tot")]
    thr = np.percentile(n_tot, 60)
    return (n_tot > thr).astype(int), n_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows", required=True)
    ap.add_argument("--dataset", default="edge-iiot")
    ap.add_argument("--max-flows", type=int, default=0)
    ap.add_argument("--mask-ids", action="store_true")
    ap.add_argument("--target-recall", type=float, default=0.90,
                    help="common operating recall to match detectors for the K1 test")
    ap.add_argument("--out", default="pilot_k1_results.json")
    args = ap.parse_args()

    flows = load_flows(args.flows, args.max_flows)
    print(f"loaded {len(flows)} flows")
    X, y = make_xy(flows, args.dataset, mask_ids=args.mask_ids)
    print(f"malicious rate {y.mean():.3f}")

    # stratified-by-attack 60/20/20 train/val/test split (threshold on val, metrics on
    # test; no test-label leakage). Stratified because Edge-IIoTset attacks are separate
    # captures and a global-time split induces distribution shift.
    from collections import defaultdict as _dd
    rng = np.random.default_rng(0)
    buckets = _dd(list)
    for i, fl in enumerate(flows):
        buckets[fl["attack"]].append(i)
    tr, va, te = [], [], []
    for atk, idxs in buckets.items():
        idxs = list(idxs); rng.shuffle(idxs)
        n1 = int(0.6 * len(idxs)); n2 = int(0.8 * len(idxs))
        tr += idxs[:n1]; va += idxs[n1:n2]; te += idxs[n2:]
    order = tr + va + te
    flows = [flows[i] for i in order]; X = X[order]; y = y[order]
    ntr, nva = len(tr), len(va)
    Xtr, ytr = X[:ntr], y[:ntr]
    Xva, yva = X[ntr:ntr+nva], y[ntr:ntr+nva]
    Xte, yte = X[ntr+nva:], y[ntr+nva:]
    flows_tr = flows[:ntr]; flows_va = flows[ntr:ntr+nva]; flows_te = flows[ntr+nva:]
    print(f"train {ntr} val {nva} test {len(te)}")

    gt = build_ground_truth(flows_te, args.dataset)
    print(f"GT campaigns (>=2 stages) in test: {len(gt)}")

    # train-only multi-class stage classifier -> predicted stages on TEST
    from stage_clf import train_stage_clf, predict_stages
    stage_model = train_stage_clf(Xtr, flows_tr, args.dataset)
    pred_stages_te = predict_stages(stage_model, Xte)

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xeval = np.vstack([Xva, Xte])      # score val+test together, then split
    nv = len(Xva)
    raw = {}
    try: raw["lgbm"] = train_lgbm(Xtr, ytr, Xeval)
    except Exception as e: print("lgbm fail", e)
    try: raw["logreg"] = train_logreg(Xtr, ytr, Xeval)
    except Exception as e: print("logreg fail", e)
    try: raw["mlp"] = train_mlp(Xtr, ytr, Xeval, dev=dev)
    except Exception as e: print("mlp fail", e)
    raw["volume_biased"] = volume_biased(Xtr, ytr, Xeval, None, args.dataset)

    # calibrate each detector's threshold on VAL to a common operating recall, then
    # evaluate ONCE on untouched TEST. Any fidelity gap among matched detectors is K1.
    target_recall = args.target_recall
    models = {}
    for name, (_, score) in raw.items():
        score = np.asarray(score, dtype=float)
        sv, stx = score[:nv], score[nv:]
        if name == "volume_biased":
            thr = np.percentile(sv, 60)            # stress case: fixed high-volume cut on val
        else:
            thr = calibrate_threshold(yva, sv, target_recall)
        models[name] = ((stx >= thr).astype(int), stx)

    results = {}
    for name, (pred, score) in models.items():
        det = detection_metrics(yte, pred)
        fid = forensic_fidelity(flows_te, pred, args.dataset, gt=gt, pred_stages=pred_stages_te)
        results[name] = {"detection": det, "fidelity": fid}
        print(f"\n[{name}] det F1={det['f1']:.4f} acc={det['accuracy']:.4f} | "
              f"stage_recall={fid['stage_recall']:.3f} order={fid['ordering_consistency']:.3f} "
              f"entityF1={fid['entity_attribution_f1']:.3f} chain={fid['chain_completeness']:.3f}")

    # K1 evidence: among the MATCHED detectors (similar detection F1), does forensic
    # fidelity still diverge? volume_biased is excluded -it is a separate stress case.
    mm = [m for m in results if m != "volume_biased"]
    f1s = np.array([results[m]["detection"]["f1"] for m in mm])
    chains = np.array([results[m]["fidelity"]["chain_completeness"] for m in mm])
    recalls = np.array([results[m]["fidelity"]["stage_recall"] for m in mm])
    det_spread = float(f1s.max() - f1s.min()) if len(f1s) else 0.0
    chain_spread = float(chains.max() - chains.min()) if len(chains) else 0.0
    recall_spread = float(recalls.max() - recalls.min()) if len(recalls) else 0.0
    summary = {
        "matched_models": mm,
        "det_f1_range": [float(f1s.min()), float(f1s.max())] if len(f1s) else None,
        "det_f1_spread": det_spread,
        "chain_completeness_spread": chain_spread,
        "stage_recall_spread": recall_spread,
        # K1 holds if detectors at near-identical F1 (spread<0.05) still diverge on fidelity
        "k1_signal": bool(det_spread < 0.05 and (chain_spread > 0.15 or recall_spread > 0.10)),
    }
    results["_K1_summary"] = summary
    print(f"\n=== K1 SUMMARY ===\n det-F1 spread={summary['det_f1_spread']:.4f} | "
          f"chain-completeness spread={summary['chain_completeness_spread']:.4f} | "
          f"stage-recall spread={summary['stage_recall_spread']:.4f}")
    print(f" K1 signal (fidelity diverges >> detection): {summary['k1_signal']}")
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()



