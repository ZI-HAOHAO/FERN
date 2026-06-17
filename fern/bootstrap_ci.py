"""Bootstrap confidence intervals over campaign-level stage-recall gaps.

Edge-IIoTset: per-campaign stage recall via fidelity (seed 0 split, same as k1_diag).
Reports: per-model mean stage-recall + percentile bootstrap CI of (lgbm - tinyml) gap.
"""
import json, sys
import numpy as np
from collections import defaultdict
from stage_mapping import map_attack, UNKNOWN
from stage_clf import STAGE_CLASSES
from fidelity import build_ground_truth, forensic_fidelity
from pilot_k1 import load_flows, make_xy
from k1_diag import stage_y, m_lgbm, m_tinyml, to_stage_strings

def main(flows_path, out):
    flows = load_flows(flows_path)
    X, _ = make_xy(flows, "edge-iiot")
    ys = stage_y(flows, "edge-iiot")
    rng = np.random.default_rng(0)
    buckets = defaultdict(list)
    for i, f in enumerate(flows): buckets[f["attack"]].append(i)
    tr, te = [], []
    for a, idxs in buckets.items():
        idxs = list(idxs); rng.shuffle(idxs); k = int(0.7*len(idxs)); tr += idxs[:k]; te += idxs[k:]
    Xtr, ytr = X[tr], ys[tr]; Xte = X[te]
    fte = [flows[i] for i in te]
    gt = build_ground_truth(fte, "edge-iiot")
    per = {}
    for name, fn in [("lgbm", m_lgbm), ("tinyml", lambda a,b,c: m_tinyml(a,b,c,dev="cuda"))]:
        pidx = fn(Xtr, ytr, Xte)
        ps = to_stage_strings(pidx); mal = (pidx != 0).astype(int)
        fid = forensic_fidelity(fte, mal, "edge-iiot", gt=gt, pred_stages=ps)
        per[name] = np.array(fid["per_campaign_stage_recall"])
        print(f"{name}: mean SR {per[name].mean():.3f} over {len(per[name])} campaigns")
    a, b = per["lgbm"], per["tinyml"]
    n = len(a); B = 10000
    idx = np.random.default_rng(1).integers(0, n, size=(B, n))
    gaps = a[idx].mean(1) - b[idx].mean(1)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    res = {"n_campaigns": n, "lgbm_mean": float(a.mean()), "tinyml_mean": float(b.mean()),
           "gap_mean": float((a-b).mean()), "gap_ci95": [float(lo), float(hi)],
           "ci_excludes_zero": bool(lo > 0)}
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res, indent=1))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])



