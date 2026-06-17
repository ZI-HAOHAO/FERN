"""IoT-23 chain-level validation utilities.

Parses Zeek conn.log.labeled CSVs (ts, id.orig_h/p, id.resp_h/p, proto, duration,
orig/resp pkts+ip_bytes, label, detailed-label) into the FERN flow schema, builds
HOST-CENTRIC forensic ground truth (campaign = infected device's incident timeline:
all its malicious activity, stages ordered by first-seen -the DFIR unit "which device
was compromised and what did it do"), then runs the K1 diagnostic and M2 retention.

Stage map (IoT-23 detailed-labels -> kill chain), confidence flags:
  PartOfAHorizontalPortScan -> Recon (exact)
  Attack                    -> Exploit (exact: telnet brute/cmd injection per IoT-23 docs)
  FileDownload              -> Exploit (derived: payload retrieval)
  C&C* (HeartBeat/Torii/FileDownload/Mirai variants) -> C2 (exact)
  Okiru*                    -> Lateral (derived: botnet propagation/scanning for victims)
  DDoS                      -> ExfilImpact (exact)
  Benign/-                  -> None

Host-centric campaign: for each source host with malicious flows in >=2 stages,
chain = stages ordered by first-seen time. Reconstruction (FIXED, same rule): from a
model's alert stream + its own predicted stages, group flagged flows by source host,
order predicted stages by first-seen; score with the same 4 fidelity metrics
(entity = the infected host identity).
"""
import argparse, json, os, sys
import numpy as np
from collections import defaultdict

STAGES = ["Recon", "Exploit", "C2", "Lateral", "ExfilImpact"]
CLASSES = ["Normal"] + STAGES
CIDX = {c: i for i, c in enumerate(CLASSES)}

IOT23_MAP = {
    "PartOfAHorizontalPortScan": ("Recon", "exact"),
    "Attack": ("Exploit", "exact"),
    "FileDownload": ("Exploit", "derived"),
    "DDoS": ("ExfilImpact", "exact"),
    "Okiru": ("Lateral", "derived"),
    "Okiru-Attack": ("Lateral", "derived"),
    "-": (None, "exact"),
}
def map_label(dl):
    if dl is None: return (None, "exact")
    dl = str(dl).strip()
    if dl in IOT23_MAP: return IOT23_MAP[dl]
    if dl.startswith("C&C"): return ("C2", "exact")
    if dl.startswith("PartOfAHorizontal"): return ("Recon", "exact")
    if dl.lower() in ("benign", "-", "nan", ""): return (None, "exact")
    return (None, "exact")   # unknown -> treat as benign, conservative

FEATS = ["dur","n_fwd","n_bwd","n_tot","bytes_fwd","bytes_bwd","bytes_tot",
         "pps","bps","mean_pkt","ratio_fwd","sport","dport",
         "proto_tcp","proto_udp","proto_icmp"]

def fnum(x):
    try:
        v = float(x); return v if np.isfinite(v) else 0.0
    except Exception:
        return 0.0

def parse_csv(path, max_rows=0):
    """Stream-parse a conn.log.labeled CSV. Last comma-field = 'tunnel label dlabel'."""
    flows = []
    with open(path) as f:
        header = f.readline()
        for ln, line in enumerate(f):
            if max_rows and len(flows) >= max_rows: break
            parts = line.rstrip("\n").split(",")
            if len(parts) < 21: continue
            tail = parts[20].split()
            dl = tail[2] if len(tail) >= 3 else "-"
            ts = fnum(parts[0]); dur = fnum(parts[8])
            opk = fnum(parts[16]); ob = fnum(parts[17])
            rpk = fnum(parts[18]); rb = fnum(parts[19])
            proto = parts[6]
            n_tot = opk + rpk; b_tot = ob + rb
            d = max(1e-6, dur)
            flows.append({
                "src": parts[2], "sport": fnum(parts[3]),
                "dst": parts[4], "dport": fnum(parts[5]),
                "proto": proto, "t_first": ts, "t_last": ts + dur,
                "dur": dur, "n_fwd": opk, "n_bwd": rpk, "n_tot": n_tot,
                "bytes_fwd": ob, "bytes_bwd": rb, "bytes_tot": b_tot,
                "pps": n_tot / d, "bps": b_tot / d,
                "mean_pkt": b_tot / max(1, n_tot), "ratio_fwd": opk / max(1, n_tot),
                "proto_tcp": int(proto == "tcp"), "proto_udp": int(proto == "udp"),
                "proto_icmp": int(proto == "icmp"),
                "dlabel": dl,
            })
    return flows

def host_ground_truth(flows, min_stages=2):
    """Campaign = source host with >=2 malicious stages. Returns {host: ordered stages}."""
    host_stage_first = defaultdict(dict)
    for f in flows:
        st, conf = map_label(f["dlabel"])
        if st is None: continue
        h = f["src"]
        if st not in host_stage_first[h] or f["t_first"] < host_stage_first[h][st]:
            host_stage_first[h][st] = f["t_first"]
    gt = {}
    for h, sf in host_stage_first.items():
        if len(sf) < min_stages: continue
        gt[h] = sorted(sf, key=lambda s: sf[s])
    return gt

def host_reconstruct(flows, mal_pred, pred_stages, min_stages=2):
    host_stage_first = defaultdict(dict)
    for i, f in enumerate(flows):
        if not mal_pred[i]: continue
        st = pred_stages[i]
        if st in (None, "Normal"): continue
        h = f["src"]
        if st not in host_stage_first[h] or f["t_first"] < host_stage_first[h][st]:
            host_stage_first[h][st] = f["t_first"]
    pred = {}
    for h, sf in host_stage_first.items():
        if len(sf) < min_stages: continue
        pred[h] = sorted(sf, key=lambda s: sf[s])
    return pred

def order_agreement(gt_stages, pr_stages):
    common = [s for s in gt_stages if s in pr_stages]
    if len(common) < 2: return 1.0
    gr = {s: i for i, s in enumerate(gt_stages)}; pr = {s: i for i, s in enumerate(pr_stages)}
    agree = tot = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            a, b = common[i], common[j]; tot += 1
            agree += int((gr[a] < gr[b]) == (pr[a] < pr[b]))
    return agree / max(1, tot)

def fidelity(flows, mal_pred, pred_stages, gt):
    pred = host_reconstruct(flows, mal_pred, pred_stages)
    srs, orders, chains = [], [], []
    tp = fn = 0
    for h, gst in gt.items():
        p = pred.get(h)
        if p is None:
            srs.append(0.0); orders.append(0.0); chains.append(0.0); fn += 1; continue
        tp += 1
        srs.append(sum(1 for s in gst if s in p) / len(gst))
        orders.append(order_agreement(gst, p))
        chains.append(1.0 if (set(gst) == set(p) and p[:len(gst)] == gst) else 0.0)
    fp = sum(1 for h in pred if h not in gt)
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    ef1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"stage_recall": float(np.mean(srs)) if srs else 0.0,
            "ordering_consistency": float(np.mean(orders)) if orders else 0.0,
            "entity_attribution_f1": float(ef1),
            "chain_completeness": float(np.mean(chains)) if chains else 0.0,
            "n_gt": len(gt), "n_pred": len(pred)}

def stage_y(flows):
    return np.array([CIDX[map_label(f["dlabel"])[0] or "Normal"] for f in flows], dtype=np.int64)

def featmat(flows):
    return np.array([[fnum(f.get(k, 0)) for k in FEATS] for f in flows], dtype=np.float32)

def det_f1(yt, yp):
    t = (yt != 0).astype(int); p = (yp != 0).astype(int)
    tp = int(((p == 1) & (t == 1)).sum()); fp = int(((p == 1) & (t == 0)).sum()); fn = int(((p == 0) & (t == 1)).sum())
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
    return 2 * pr * rc / max(1e-9, pr + rc)

# ---- models (same family as k1_diag) ----
def m_lgbm(Xtr, ytr, Xte):
    import lightgbm as lgb
    m = lgb.train({"objective": "multiclass", "num_class": len(CLASSES), "verbose": -1,
                   "num_leaves": 31, "learning_rate": 0.1}, lgb.Dataset(Xtr, label=ytr), 80)
    return np.argmax(m.predict(Xte), 1)

def m_logreg(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    return LogisticRegression(max_iter=300).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))

def m_net(Xtr, ytr, Xte, hidden, dev="cuda", epochs=12):
    import torch, torch.nn as nn
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xt = torch.tensor((Xtr - mu) / sd, device=dev); yt = torch.tensor(ytr, device=dev)
    layers = []; d = Xtr.shape[1]
    for h in hidden: layers += [nn.Linear(d, h), nn.ReLU()]; d = h
    layers += [nn.Linear(d, len(CLASSES))]
    net = nn.Sequential(*layers).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3); lf = nn.CrossEntropyLoss(); bs = 8192
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i+bs]; opt.zero_grad(); lf(net(Xt[idx]), yt[idx]).backward(); opt.step()
    import torch as T
    with T.no_grad():
        return net(T.tensor((Xte - mu) / sd, device=dev)).argmax(1).cpu().numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--files", default="dataset5.csv,dataset13.csv,dataset17.csv,dataset19.csv,dataset21.csv,dataset23.csv,dataset10.csv")
    ap.add_argument("--max-rows-per-file", type=int, default=400000)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="outputs/iot23_chain.json")
    args = ap.parse_args()

    flows = []
    for fn in args.files.split(","):
        p = os.path.join(args.csv_dir, fn.strip())
        if not os.path.exists(p): print("skip missing", fn); continue
        fl = parse_csv(p, args.max_rows_per_file)
        # prefix host with scenario id so identical private IPs in different captures stay distinct
        tag = fn.replace("dataset", "s").replace(".csv", "")
        for f in fl: f["src"] = f"{tag}:{f['src']}"; f["dst"] = f"{tag}:{f['dst']}"
        flows += fl
        print(f"  {fn}: {len(fl)} flows")
    print(f"total {len(flows)} flows")
    ys = stage_y(flows)
    print("stage dist:", {CLASSES[i]: int((ys == i).sum()) for i in range(len(CLASSES)) if (ys == i).sum()})
    X = featmat(flows)

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_runs = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        rng = np.random.default_rng(seed)
        # stratify by (scenario, stage) to keep all stages in both splits
        buckets = defaultdict(list)
        for i, f in enumerate(flows):
            buckets[(f["src"].split(":")[0], int(ys[i]))].append(i)
        tr, te = [], []
        for k, idxs in buckets.items():
            idxs = list(idxs); rng.shuffle(idxs); cut = int(0.7 * len(idxs))
            tr += idxs[:cut]; te += idxs[cut:]
        Xtr, ytr = X[tr], ys[tr]; Xte, yte = X[te], ys[te]
        fte = [flows[i] for i in te]
        gt = host_ground_truth(fte)
        res = {"n_gt_campaigns": len(gt)}
        preds = {
            "lgbm": m_lgbm(Xtr, ytr, Xte),
            "logreg": m_logreg(Xtr, ytr, Xte),
            "mlp": m_net(Xtr, ytr, Xte, [64, 32], dev),
            "cnn-like": m_net(Xtr, ytr, Xte, [64, 64, 32], dev),
            "tinyml": m_net(Xtr, ytr, Xte, [8], dev),
        }
        for name, pidx in preds.items():
            ps = np.array([CLASSES[i] for i in pidx], dtype=object)
            mal = (pidx != 0).astype(int)
            fid = fidelity(fte, mal, ps, gt)
            res[name] = {"det_f1": det_f1(yte, pidx), **fid}
        out_runs.append(res)
        print(f"seed {seed}: GT campaigns={len(gt)} | " + " | ".join(
            f"{m}: F1={res[m]['det_f1']:.3f} SR={res[m]['stage_recall']:.3f} chain={res[m]['chain_completeness']:.3f}"
            for m in preds))

    # aggregate
    models = ["lgbm", "logreg", "mlp", "cnn-like", "tinyml"]
    agg = {}
    for m in models:
        agg[m] = {k: {"mean": float(np.mean([r[m][k] for r in out_runs])),
                      "std": float(np.std([r[m][k] for r in out_runs]))}
                  for k in ["det_f1", "stage_recall", "ordering_consistency",
                            "entity_attribution_f1", "chain_completeness"]}
    f1s = [agg[m]["det_f1"]["mean"] for m in models]
    srs = [agg[m]["stage_recall"]["mean"] for m in models]
    summary = {"det_f1_spread": max(f1s) - min(f1s), "stage_recall_spread": max(srs) - min(srs),
               "n_gt_campaigns_mean": float(np.mean([r["n_gt_campaigns"] for r in out_runs]))}
    json.dump({"runs": out_runs, "agg": agg, "summary": summary}, open(args.out, "w"), indent=2)
    print("\n=== IoT-23 K1 (host-centric chains, incl. C2) ===")
    for m in models:
        a = agg[m]
        print(f"  {m:9s} detF1={a['det_f1']['mean']:.3f}+/-{a['det_f1']['std']:.3f} "
              f"SR={a['stage_recall']['mean']:.3f}+/-{a['stage_recall']['std']:.3f} "
              f"order={a['ordering_consistency']['mean']:.3f} "
              f"entity={a['entity_attribution_f1']['mean']:.3f} "
              f"chain={a['chain_completeness']['mean']:.3f}")
    print(f"  detF1 spread={summary['det_f1_spread']:.4f}  stage-recall spread={summary['stage_recall_spread']:.3f}  "
          f"campaigns={summary['n_gt_campaigns_mean']:.0f}")
    print("wrote", args.out)

if __name__ == "__main__":
    main()



