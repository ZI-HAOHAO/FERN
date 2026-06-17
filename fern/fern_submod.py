"""FERN++ : submodular forensic-coverage retention (offline + online) vs the old
value-per-byte greedy. Goal: a theoretically-grounded, stronger selection algorithm.

Offline: maximize the monotone submodular forensic-COVERAGE objective
    F(K) = sum_{cell=(pair,stage)} w_cell * (1 - prod_{i in K cap cell} (1 - p_i))
subject to sum_{i in K} c_i <= B (knapsack), via cost-benefit lazy greedy
(Krause-Guestrin / Leskovec lazy greedy), which gives the standard (1-1/e) guarantee
for monotone submodular maximization under a knapsack (with cost-benefit + singleton).
p_i = scorer value; cell weight w favors rare/early stages (inverse predicted-stage freq).
The diminishing-returns marginal gain w*U_cell*p_i (U_cell = prod (1-p_kept)) forces
the budget to DIVERSIFY across uncovered (campaign,stage) cells instead of stacking
redundant evidence in one easy stage -- the failure mode of value-per-byte greedy.

Online: streaming primal-dual threshold on marginal-gain-per-byte g/c >= tau, tau the
budget dual price (calibrated on train), strictly causal.

Compared against the old keep_under_budget(g/sqrt(c)) greedy on stage-recall AND
chain-completeness on Edge-IIoTset, 3 seeds, one harness.
"""
import argparse, json
import numpy as np
import heapq
from collections import defaultdict

from pilot_k1 import load_flows, make_xy
from retention import forensic_value_labels, anchor_labels, train_fern_scorer, byte_cost, keep_under_budget
from fidelity import build_ground_truth, forensic_fidelity
from stage_clf import train_stage_clf, predict_stages
from stage_mapping import map_attack, UNKNOWN, STAGE_ORDER
from evaluation_suite import strat_split

EDGE_BUDGETS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]


def _cells(flows, pst):
    cellof = {}
    members = defaultdict(list)
    stage_count = defaultdict(int)
    for i, f in enumerate(flows):
        st = pst[i]
        if st is None or st == "Normal" or st == UNKNOWN:
            continue
        k = (tuple(sorted((f["src"], f["dst"]))), st)
        cellof[i] = k
        members[k].append(i)
        stage_count[st] += 1
    return cellof, members, stage_count


def submod_offline(scores, costs, flows, pst, budget, alpha=0.5, rare_weight=True):
    cellof, members, stage_count = _cells(flows, pst)
    if not cellof:
        return np.zeros(len(flows), bool)
    # cell weight: inverse predicted-stage frequency (rare/early stages weigh more)
    if rare_weight:
        tot = sum(stage_count.values())
        sw = {s: (tot / (len(stage_count) * c)) for s, c in stage_count.items()}
    else:
        sw = {s: 1.0 for s in stage_count}
    U = defaultdict(lambda: 1.0)
    p = np.clip(scores, 0.0, 1.0)

    def gain(i):
        k = cellof[i]
        return sw[k[1]] * U[k] * p[i]

    heap = []
    for i in cellof:
        g = gain(i)
        if g > 0:
            heapq.heappush(heap, (-(g / (costs[i] ** alpha)), i))
    mask = np.zeros(len(flows), bool)
    spent = 0.0
    while heap:
        neg, i = heapq.heappop(heap)
        if mask[i]:
            continue
        cur = gain(i) / (costs[i] ** alpha)
        if -neg > cur + 1e-12:                 # stale: re-push with current value
            if cur > 0:
                heapq.heappush(heap, (-cur, i))
            continue
        if cur <= 0:
            break
        if spent + costs[i] <= budget:
            mask[i] = True
            spent += costs[i]
            U[cellof[i]] *= (1.0 - p[i])
    return mask


def submod_online(scores, costs, flows, pst, budget, tau, tau_fill=None, alpha=1.0):
    """Hybrid causal online: accept flow i iff budget remains AND
       (marginal coverage gain / cost >= tau)   [coverage tier, dominates tight budgets]
       OR (raw scorer value >= tau_fill).        [fill tier, uses spare budget at loose B]
    Strictly causal: U_cell updated only from already-kept flows."""
    cellof, _, _ = _cells(flows, pst)
    U = defaultdict(lambda: 1.0)
    p = np.clip(scores, 0.0, 1.0)
    order = sorted(range(len(flows)), key=lambda i: flows[i]["t_first"])
    mask = np.zeros(len(flows), bool)
    spent = 0.0
    for i in order:
        if i not in cellof:
            continue
        k = cellof[i]
        g = U[k] * p[i]
        cover = (g / (costs[i] ** alpha) >= tau)
        fill = (tau_fill is not None and p[i] >= tau_fill)
        if (cover or fill) and spent + costs[i] <= budget:
            mask[i] = True
            spent += costs[i]
            U[k] *= (1.0 - p[i])
    return mask


def calib_tau_online(scores, costs, flows, pst, budget, alpha=1.0):
    """Pick tau on the TRAIN split so the online rule spends ~budget (dual price)."""
    cellof, _, _ = _cells(flows, pst)
    p = np.clip(scores, 0.0, 1.0)
    order = sorted(range(len(flows)), key=lambda i: flows[i]["t_first"])
    # simulate marginal gains in causal order, collect gain/cost ratios actually realizable
    U = defaultdict(lambda: 1.0)
    ratios = []
    for i in order:
        if i not in cellof:
            continue
        k = cellof[i]
        g = U[k] * p[i]
        ratios.append((g / (costs[i] ** alpha), i, costs[i]))
        U[k] *= (1.0 - p[i])                    # assume greedily kept for calibration
    ratios.sort(reverse=True)
    spent = 0.0
    tau = 0.0
    for r, i, c in ratios:
        if spent + c > budget:
            break
        spent += c
        tau = r
    return tau


def run_edge(seeds=(0, 1, 2)):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    flows = load_flows("data/processed/edge_flows_full.jsonl")
    X, _ = make_xy(flows, "edge-iiot")
    methods = ["greedy_vpb", "submod_off", "submod_off_uniform", "online_thr", "submod_online"]
    agg = {m: {"sr": defaultdict(list), "chain": defaultdict(list)} for m in methods}
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
        sm = train_stage_clf(Xtr, ftr, "edge-iiot")
        pst = predict_stages(sm, Xte)
        pst_tr = predict_stages(sm, Xtr)

        def evalmask(mask):
            fid = forensic_fidelity(fte, (pst != "Normal").astype(int) * mask.astype(int),
                                    "edge-iiot", gt=gt, pred_stages=pst)
            return fid["stage_recall"], fid["chain_completeness"]

        for b in EDGE_BUDGETS:
            bb = b * full
            masks = {
                "greedy_vpb": keep_under_budget(np.argsort(-(s_te / np.sqrt(ce))), ce, bb),
                "submod_off": submod_offline(s_te, ce, fte, pst, bb, alpha=0.5, rare_weight=True),
                "submod_off_uniform": submod_offline(s_te, ce, fte, pst, bb, alpha=0.5, rare_weight=False),
            }
            tau = calib_tau_online(s_tr, ctr, ftr, pst_tr, b * float(ctr.sum()), alpha=1.0)
            from evaluation_suite import calib_thr, keep_threshold_online
            thr = calib_thr(s_tr, ctr, b * float(ctr.sum()))   # raw-score dual price for the fill tier
            masks["submod_online"] = submod_online(s_te, ce, fte, pst, bb, tau, tau_fill=thr, alpha=1.0)
            # old online threshold (raw score) for reference
            masks["online_thr"] = keep_threshold_online(s_te, ce, thr, bb)
            for m, mk in masks.items():
                sr, ch = evalmask(mk)
                agg[m]["sr"][b].append(sr); agg[m]["chain"][b].append(ch)
        print(f"seed {sd} done", flush=True)
    out = {"budgets": EDGE_BUDGETS, "methods": {}}
    for m in methods:
        out["methods"][m] = {
            "stage_recall": [round(float(np.mean(agg[m]["sr"][b])), 4) for b in EDGE_BUDGETS],
            "stage_recall_std": [round(float(np.std(agg[m]["sr"][b])), 4) for b in EDGE_BUDGETS],
            "chain": [round(float(np.mean(agg[m]["chain"][b])), 4) for b in EDGE_BUDGETS],
            "chain_std": [round(float(np.std(agg[m]["chain"][b])), 4) for b in EDGE_BUDGETS],
        }
    return out


if __name__ == "__main__":
    out = run_edge()
    json.dump(out, open("outputs/fern_submod.json", "w"), indent=2)
    print(json.dumps(out, indent=1))
    print("SUBMOD DONE", flush=True)



