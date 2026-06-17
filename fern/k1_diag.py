"""Diagnostic evaluation: detection F1 does not predict forensic fidelity.

Honest design (addresses known leakage checks):
  * Each MODEL is its OWN multi-class stage predictor (benign + 5 kill-chain stages),
    trained on the TRAIN split only. No shared stage oracle, no label leakage.
  * Detection F1 = binary (predicted-stage != Normal) vs (true != Normal) on TEST.
  * Forensic fidelity = the 4 metrics computed by feeding EACH model's PREDICTED per-flow
    stages through the fixed reconstructor (fidelity.reconstruct), scored vs full-data GT.
  * K1 holds if, across architectures, detection F1 and forensic fidelity are weakly
    correlated -i.e. a model can detect well yet reconstruct chains poorly (it misses the
    low-volume early stages that anchor chains).

Models: lgbm / logreg / mlp multiclass, plus a 'volume_biased' detector that only labels
high-volume flows as ExfilImpact (a DDoS-focused detector -high detection F1 where DDoS
dominates, but forensically blind to Recon/Exploit/C2).
"""
import argparse, json
import numpy as np
from collections import defaultdict
from stage_mapping import map_attack, UNKNOWN
from stage_clf import STAGE_CLASSES
from fidelity import build_ground_truth, forensic_fidelity
from pilot_k1 import load_flows, FEATS, make_xy

CIDX = {c: i for i, c in enumerate(STAGE_CLASSES)}


def stage_y(flows, dataset):
    out = []
    for f in flows:
        s, _ = map_attack(f["attack"], dataset)
        out.append(CIDX.get(s if s not in (None, UNKNOWN) else "Normal", 0))
    return np.array(out, dtype=np.int64)


def m_lgbm(Xtr, ytr, Xte):
    import lightgbm as lgb
    d = lgb.Dataset(Xtr, label=ytr)
    m = lgb.train({"objective": "multiclass", "num_class": len(STAGE_CLASSES),
                   "verbose": -1, "num_leaves": 31, "learning_rate": 0.1}, d, 80)
    return np.argmax(m.predict(Xte), axis=1)


def m_logreg(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=300).fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))


def m_mlp(Xtr, ytr, Xte, dev="cuda", epochs=12):
    import torch, torch.nn as nn
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xt = torch.tensor((Xtr - mu) / sd, device=dev, dtype=torch.float32)
    yt = torch.tensor(ytr, device=dev)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 64), nn.ReLU(),
                        nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, len(STAGE_CLASSES))).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3); lf = nn.CrossEntropyLoss()
    bs = 4096
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i+bs]; opt.zero_grad()
            loss = lf(net(Xt[idx]), yt[idx]); loss.backward(); opt.step()
    with torch.no_grad():
        p = net(torch.tensor((Xte - mu) / sd, device=dev, dtype=torch.float32)).argmax(1).cpu().numpy()
    return p


def m_cnn1d(Xtr, ytr, Xte, dev="cuda", epochs=12):
    """1D-CNN over the flow-feature vector (treats the 16 features as a 1-channel signal)."""
    import torch, torch.nn as nn
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xt = torch.tensor((Xtr - mu) / sd, device=dev, dtype=torch.float32).unsqueeze(1)
    yt = torch.tensor(ytr, device=dev)
    net = nn.Sequential(
        nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv1d(16, 16, 3, padding=1), nn.ReLU(),
        nn.Flatten(), nn.Linear(16 * Xtr.shape[1], 64), nn.ReLU(),
        nn.Linear(64, len(STAGE_CLASSES))).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3); lf = nn.CrossEntropyLoss(); bs = 4096
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i+bs]; opt.zero_grad()
            loss = lf(net(Xt[idx]), yt[idx]); loss.backward(); opt.step()
    with torch.no_grad():
        Xe = torch.tensor((Xte - mu) / sd, device=dev, dtype=torch.float32).unsqueeze(1)
        return net(Xe).argmax(1).cpu().numpy()


def m_tinyml(Xtr, ytr, Xte, dev="cuda", epochs=15):
    """TinyML-class detector: a tiny 2-layer net (~few-hundred params) for MCU/gateway."""
    import torch, torch.nn as nn
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xt = torch.tensor((Xtr - mu) / sd, device=dev, dtype=torch.float32)
    yt = torch.tensor(ytr, device=dev)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 8), nn.ReLU(),
                        nn.Linear(8, len(STAGE_CLASSES))).to(dev)
    opt = torch.optim.Adam(net.parameters(), 2e-3); lf = nn.CrossEntropyLoss(); bs = 4096
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i+bs]; opt.zero_grad()
            loss = lf(net(Xt[idx]), yt[idx]); loss.backward(); opt.step()
    with torch.no_grad():
        return net(torch.tensor((Xte - mu) / sd, device=dev, dtype=torch.float32)).argmax(1).cpu().numpy()


def m_volume_biased(Xtr, ytr, Xte):
    """DDoS-focused: label top-volume flows ExfilImpact, rest Normal. Forensically blind."""
    nt = Xte[:, FEATS.index("n_tot")]
    thr = np.percentile(Xtr[:, FEATS.index("n_tot")], 70)
    pred = np.where(nt > thr, CIDX["ExfilImpact"], CIDX["Normal"])
    return pred


def to_stage_strings(pred_idx):
    return np.array([STAGE_CLASSES[i] for i in pred_idx], dtype=object)


def det_f1(true_idx, pred_idx):
    yt = (true_idx != 0).astype(int); yp = (pred_idx != 0).astype(int)
    tp = int(((yp == 1) & (yt == 1)).sum()); fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    return 2 * p * r / max(1e-9, p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows", required=True)
    ap.add_argument("--dataset", default="edge-iiot")
    ap.add_argument("--out", default="k1_diag.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    flows = load_flows(args.flows)
    X, _ = make_xy(flows, args.dataset)
    ys = stage_y(flows, args.dataset)

    rng = np.random.default_rng(args.seed)
    buckets = defaultdict(list)
    for i, f in enumerate(flows):
        buckets[f["attack"]].append(i)
    tr, te = [], []
    for atk, idxs in buckets.items():
        idxs = list(idxs); rng.shuffle(idxs)
        k = int(0.7 * len(idxs)); tr += idxs[:k]; te += idxs[k:]
    Xtr, ytr = X[tr], ys[tr]
    Xte = X[te]; fte = [flows[i] for i in te]
    true_te = ys[te]
    gt = build_ground_truth(fte, args.dataset)
    print(f"train {len(tr)} test {len(te)} | GT campaigns {len(gt)}")

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    preds = {}
    preds["lgbm"] = m_lgbm(Xtr, ytr, Xte)
    preds["logreg"] = m_logreg(Xtr, ytr, Xte)
    preds["mlp"] = m_mlp(Xtr, ytr, Xte, dev=dev)
    preds["cnn1d"] = m_cnn1d(Xtr, ytr, Xte, dev=dev)
    preds["tinyml"] = m_tinyml(Xtr, ytr, Xte, dev=dev)
    preds["volume_biased"] = m_volume_biased(Xtr, ytr, Xte)

    rows = {}
    for name, pidx in preds.items():
        f1 = det_f1(true_te, pidx)
        ps = to_stage_strings(pidx)
        mal = (pidx != 0).astype(int)
        fid = forensic_fidelity(fte, mal, args.dataset, gt=gt, pred_stages=ps)
        rows[name] = {"det_f1": f1, **{k: fid[k] for k in
                      ["stage_recall", "ordering_consistency", "entity_attribution_f1", "chain_completeness"]}}
        print(f"[{name:14s}] detF1={f1:.4f} | stage_recall={fid['stage_recall']:.3f} "
              f"order={fid['ordering_consistency']:.3f} entityF1={fid['entity_attribution_f1']:.3f} "
              f"chain={fid['chain_completeness']:.3f}")

    # K1 signal: among the real detectors (high detF1), does fidelity diverge a lot?
    real = ["lgbm", "logreg", "mlp", "cnn1d", "tinyml"]
    f1s = np.array([rows[m]["det_f1"] for m in real])
    srs = np.array([rows[m]["stage_recall"] for m in real])
    ccs = np.array([rows[m]["chain_completeness"] for m in real])
    # correlation across ALL models (incl volume_biased) between detF1 and fidelity
    allm = list(rows)
    af1 = np.array([rows[m]["det_f1"] for m in allm])
    asr = np.array([rows[m]["stage_recall"] for m in allm])
    # spearman by hand
    def spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        if np.std(ra) < 1e-9 or np.std(rb) < 1e-9: return 0.0
        return float(np.corrcoef(ra, rb)[0, 1])
    summary = {
        "matched_detectors": real,
        "det_f1_spread_matched": float(f1s.max() - f1s.min()),
        "stage_recall_spread_matched": float(srs.max() - srs.min()),
        "chain_spread_matched": float(ccs.max() - ccs.min()),
        "spearman_detf1_vs_stage_recall_all": spearman(af1, asr),
        "k1_signal": bool((f1s.max() - f1s.min()) < 0.06 and
                          (srs.max() - srs.min() > 0.10 or ccs.max() - ccs.min() > 0.15)
                          or spearman(af1, asr) < 0.5),
    }
    rows["_K1_summary"] = summary
    print(f"\n=== K1 ===\n matched detF1 spread={summary['det_f1_spread_matched']:.4f} | "
          f"stage_recall spread={summary['stage_recall_spread_matched']:.4f} | "
          f"chain spread={summary['chain_spread_matched']:.4f}")
    print(f" Spearman(detF1, stage_recall) over all models = {summary['spearman_detf1_vs_stage_recall_all']:.3f}")
    print(f" K1 signal: {summary['k1_signal']}")
    json.dump(rows, open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()



