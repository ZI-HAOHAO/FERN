"""Unified FERN evaluation: FERN = context-aware (20-feat) scorer + submodular forensic
coverage. Compares against mechanism baselines that isolate each component, on Edge-IIoTset,
3 seeds, one harness. Also produces the online FERN coverage-fill budget audit.

Selectors:
  fern            : context scorer + submodular coverage (the main method)
  vpb_variant     : context scorer + independent value-per-byte greedy (coverage ablation)
  stage_balanced  : context scorer, round-robin across predicted cells (naive diversification)
  coverage_only   : submodular coverage with p_i = normalized anomaly score (no learned scorer)
  first_anchor    : keep highest-score first-seen flow per predicted cell (anchor heuristic)
  online_fern     : context scorer + causal coverage-fill rule (for the budget audit)
"""
import json
import numpy as np
from collections import defaultdict

from pilot_k1 import load_flows, make_xy
from retention import forensic_value_labels, train_fern_scorer, byte_cost, keep_under_budget
from fidelity import build_ground_truth, forensic_fidelity
from stage_clf import train_stage_clf, predict_stages
from stage_mapping import UNKNOWN
from evaluation_suite import strat_split, calib_thr
from fern_submod import submod_offline, submod_online, calib_tau_online, _cells
from fern_context import causal_context

EDGE_BUDGETS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]


def stage_balanced(scores, costs, flows, pst, budget):
    _, members, _ = _cells(flows, pst)
    for k in members:
        members[k].sort(key=lambda i: -scores[i])
    ptr = {k: 0 for k in members}
    cells = list(members.keys())
    mask = np.zeros(len(flows), bool); spent = 0.0
    progress = True
    while progress:
        progress = False
        for k in cells:
            p = ptr[k]
            while p < len(members[k]):
                i = members[k][p]
                if not mask[i] and spent + costs[i] <= budget:
                    mask[i] = True; spent += costs[i]; ptr[k] = p + 1; progress = True; break
                p += 1
            ptr[k] = max(ptr[k], p)
    return mask


def first_anchor_pred(scores, costs, flows, pst, budget):
    _, members, _ = _cells(flows, pst)
    firsts = []
    for k, mem in members.items():
        i0 = min(mem, key=lambda i: flows[i]["t_first"])
        firsts.append((scores[i0], i0))
    firsts.sort(reverse=True)
    mask = np.zeros(len(flows), bool); spent = 0.0
    for _, i in firsts:
        if spent + costs[i] <= budget:
            mask[i] = True; spent += costs[i]
    return mask


def audit_row(mask, costs, flows, pst, gt, target_bytes, full):
    fid = forensic_fidelity(flows, (pst != "Normal").astype(int) * mask.astype(int),
                            "edge-iiot", gt=gt, pred_stages=pst)
    return {"kept_byte_frac": float(costs[mask].sum() / full),
            "violation": bool(costs[mask].sum() > target_bytes + 1),
            "n_kept": int(mask.sum()),
            "stage_recall": fid["stage_recall"], "chain": fid["chain_completeness"]}


def run(seeds=(0, 1, 2)):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    flows = load_flows("data/processed/edge_flows_full.jsonl")
    X16, _ = make_xy(flows, "edge-iiot")
    X = X16                                                     # base 16-feat scorer = FERN default
    sels = ["fern", "vpb_variant", "stage_balanced", "coverage_only", "first_anchor"]
    agg = {s: {"sr": defaultdict(list), "chain": defaultdict(list)} for s in sels}
    audit = {b: defaultdict(list) for b in EDGE_BUDGETS}    # online_fern audit
    for sd in seeds:
        rng = np.random.default_rng(sd)
        tr, te = strat_split([f["attack"] for f in flows], rng)
        te = sorted(te, key=lambda i: flows[i]["t_first"])
        ftr = [flows[i] for i in tr]; fte = [flows[i] for i in te]
        Xtr, Xte = X[tr], X[te]
        ce = np.array([float(byte_cost(f)) for f in fte]); full = float(ce.sum())
        ctr = np.array([float(byte_cost(flows[i])) for i in tr])
        gt = build_ground_truth(fte, "edge-iiot")
        v = forensic_value_labels(ftr, "edge-iiot", n_rare_stages=3, anchor_focused=True)
        s_te, _ = train_fern_scorer(Xtr, v, Xte, dev=dev)
        s_tr, _ = train_fern_scorer(Xtr, v, Xtr, dev=dev)
        sm = train_stage_clf(X16[tr], ftr, "edge-iiot")     # stage head on static features
        pst = predict_stages(sm, X16[te]); pst_tr = predict_stages(sm, X16[tr])
        # anomaly score for coverage_only (train-centroid distance, normalized to [0,1])
        mu, sd_ = Xtr.mean(0), Xtr.std(0) + 1e-6
        anom = np.linalg.norm((Xte - mu) / sd_, axis=1)
        panom = (anom - anom.min()) / (anom.max() - anom.min() + 1e-9)

        def ev(mask):
            fid = forensic_fidelity(fte, (pst != "Normal").astype(int) * mask.astype(int),
                                    "edge-iiot", gt=gt, pred_stages=pst)
            return fid["stage_recall"], fid["chain_completeness"]

        for b in EDGE_BUDGETS:
            bb = b * full
            masks = {
                "fern": submod_offline(s_te, ce, fte, pst, bb, alpha=0.5, rare_weight=False),
                "vpb_variant": keep_under_budget(np.argsort(-(s_te / np.sqrt(ce))), ce, bb),
                "stage_balanced": stage_balanced(s_te, ce, fte, pst, bb),
                "coverage_only": submod_offline(panom, ce, fte, pst, bb, alpha=0.5, rare_weight=False),
                "first_anchor": first_anchor_pred(s_te, ce, fte, pst, bb),
            }
            for s, mk in masks.items():
                sr, ch = ev(mk)
                agg[s]["sr"][b].append(sr); agg[s]["chain"][b].append(ch)
            # online FERN: pure causal coverage tier (no fill) -the clean tight-budget rule
            tau = calib_tau_online(s_tr, ctr, ftr, pst_tr, b * float(ctr.sum()), alpha=1.0)
            mo = submod_online(s_te, ce, fte, pst, bb, tau, tau_fill=None, alpha=1.0)
            r = audit_row(mo, ce, fte, pst, gt, bb, full)
            for k, val in r.items():
                audit[b][k].append(val)
        print(f"seed {sd} done", flush=True)
    out = {"budgets": EDGE_BUDGETS, "selectors": {}, "online_audit": {}}
    for s in sels:
        out["selectors"][s] = {
            "stage_recall": [round(float(np.mean(agg[s]["sr"][b])), 4) for b in EDGE_BUDGETS],
            "stage_recall_std": [round(float(np.std(agg[s]["sr"][b])), 4) for b in EDGE_BUDGETS],
            "chain": [round(float(np.mean(agg[s]["chain"][b])), 4) for b in EDGE_BUDGETS]}
    for b in EDGE_BUDGETS:
        a = audit[b]
        out["online_audit"][f"{b}"] = {
            "kept_byte_frac": round(float(np.mean(a["kept_byte_frac"])), 4),
            "violation_any": bool(any(a["violation"])),
            "n_kept": int(round(np.mean(a["n_kept"]))),
            "stage_recall": round(float(np.mean(a["stage_recall"])), 4),
            "chain": round(float(np.mean(a["chain"])), 4)}
    return out


if __name__ == "__main__":
    out = run()
    json.dump(out, open("outputs/fern_main.json", "w"), indent=2)
    print(json.dumps(out, indent=1)); print("FERN_MAIN DONE", flush=True)



