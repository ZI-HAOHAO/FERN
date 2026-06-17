"""Controlled sparse-vs-dense stress test.

Hold attack anchors fixed; vary stage redundancy by subsampling each campaign-stage's
non-anchor malicious flows to keep-rate r (r=1 dense ... r small sparse), keeping all benign
background. Recompute split + anchor-trained scorer on the subsampled corpus and measure
FERN-Threshold vs random stage-recall at a fixed 2% budget as a function of the resulting
anchor-byte-share. anchor_labels is computed ONCE per subsample (the original part_stress
recomputed it per-flow -> O(n^2)).
"""
import json
import numpy as np
import torch
from pilot_k1 import load_flows, make_xy
from retention import forensic_value_labels, anchor_labels, train_fern_scorer, byte_cost, keep_under_budget
from fidelity import build_ground_truth, forensic_fidelity
from stage_clf import train_stage_clf, predict_stages
from stage_mapping import map_attack, UNKNOWN
from evaluation_suite import strat_split, keep_threshold_offline

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def run(flows, X, seed=0):
    rng = np.random.default_rng(seed)
    anchors_all = anchor_labels(flows, "edge-iiot")
    stage_all = [map_attack(f["attack"], "edge-iiot")[0] for f in flows]
    mal_idx = [i for i in range(len(flows)) if stage_all[i] not in (None, UNKNOWN)]
    nonanchor = [i for i in mal_idx if not anchors_all[i]]
    curves = []
    for r in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02]:
        keep_na = set(rng.choice(nonanchor, size=int(r * len(nonanchor)), replace=False).tolist())
        sub = [i for i in range(len(flows))
               if (anchors_all[i] or i in keep_na or stage_all[i] is None)]
        sub_flows = [flows[i] for i in sub]
        sub_X = X[sub]
        sub_anc = anchors_all[sub]
        sub_stage = [stage_all[i] for i in sub]
        rng2 = np.random.default_rng(0)
        tr, te = strat_split([f["attack"] for f in sub_flows], rng2)
        ftr = [sub_flows[i] for i in tr]
        fte = [sub_flows[i] for i in te]
        Xtr, Xte = sub_X[tr], sub_X[te]
        anc_te = sub_anc[te]
        stage_te = [sub_stage[i] for i in te]
        ce = np.array([float(byte_cost(f)) for f in fte])
        full = float(ce.sum())
        gt = build_ground_truth(fte, "edge-iiot")
        if not gt:
            continue
        v = forensic_value_labels(ftr, "edge-iiot", n_rare_stages=3, anchor_focused=True)
        s_te, _ = train_fern_scorer(Xtr, v, Xte, dev=DEV)
        sm = train_stage_clf(Xtr, ftr, "edge-iiot")
        pst = predict_stages(sm, Xte)
        # anchor byte share of THIS subsample's test set (computed once)
        ab = float(ce[anc_te.astype(bool)].sum())
        mb = float(sum(ce[i] for i in range(len(fte)) if stage_te[i] not in (None, UNKNOWN)))
        share = ab / max(1e-9, mb)
        bb = 0.02 * full
        m_fern = keep_threshold_offline(s_te, ce, bb)
        m_rand = keep_under_budget(rng.permutation(len(fte)), ce, bb)

        def sr(mask):
            return forensic_fidelity(fte, (pst != "Normal").astype(int) * mask.astype(int),
                                     "edge-iiot", gt=gt, pred_stages=pst)["stage_recall"]
        srf, srr = sr(m_fern), sr(m_rand)
        row = {"keep_rate": r, "anchor_byte_share": share, "fern": srf, "random": srr,
               "advantage": srf - srr, "n_mal_test": int(sum(1 for s in stage_te if s not in (None, UNKNOWN)))}
        curves.append(row)
        print(f"  r={r}: share={share:.4f} fern={srf:.3f} random={srr:.3f} adv={srf-srr:.3f}", flush=True)
    return curves


def aggregate(all_runs):
    """Average fern/random/advantage/share per keep_rate across seeds."""
    rates = [d["keep_rate"] for d in all_runs[0]]
    out = []
    for r in rates:
        rows = [next(d for d in run if d["keep_rate"] == r) for run in all_runs]
        out.append({
            "keep_rate": r,
            "anchor_byte_share": float(np.mean([x["anchor_byte_share"] for x in rows])),
            "fern": float(np.mean([x["fern"] for x in rows])),
            "fern_std": float(np.std([x["fern"] for x in rows])),
            "random": float(np.mean([x["random"] for x in rows])),
            "random_std": float(np.std([x["random"] for x in rows])),
            "advantage": float(np.mean([x["fern"] - x["random"] for x in rows])),
        })
    return out


if __name__ == "__main__":
    flows = load_flows("data/processed/edge_flows_full.jsonl")
    X, _ = make_xy(flows, "edge-iiot")
    all_runs = []
    for sd in (0, 1, 2):
        print(f"=== seed {sd} ===", flush=True)
        all_runs.append(run(flows, X, sd))
    agg = aggregate(all_runs)
    json.dump({"stress": agg, "n_seeds": len(all_runs)}, open("outputs/stress.json", "w"), indent=2)
    print("STRESS DONE", flush=True)
    for d in agg:
        print(f"  r={d['keep_rate']}: share={d['anchor_byte_share']:.4f} fern={d['fern']:.3f}+/-{d['fern_std']:.3f} random={d['random']:.3f} adv={d['advantage']:.3f}", flush=True)



