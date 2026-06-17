"""FERN++ context-aware forensic scorer: augment the 16 per-flow features with
CAUSAL per-host-pair context (computable online, no future info) and a deeper net,
so the scorer learns anchor-ness (first-of-stage evidence) directly rather than relying
only on static flow stats.

Causal context features per flow i, within its (src,dst) pair, using only flows seen
at-or-before i (sorted by t_first):
  ctx_rank    : index of i among the pair's flows so far (0,1,2,...), log1p-scaled
  ctx_is_first: 1.0 if i is the first flow seen for this pair (a campaign opener)
  ctx_dt      : inter-arrival gap to the previous flow of this pair (log1p seconds)
  ctx_newproto: 1.0 if i's (proto,dport) bucket is unseen for this pair so far
                (a new behavior -> likely a new kill-chain stage = an anchor)

These are exactly the signals that distinguish a stage's FIRST flow (anchor) from its
redundant followers, and all are causal, so the same features feed the online rule.

Compared head-to-head with the baseline 16-feature scorer on Edge-IIoTset retention
(stage-recall, 3 seeds), holding the selection rule fixed.
"""
import argparse, json
import numpy as np
from collections import defaultdict

from pilot_k1 import load_flows, make_xy
from retention import forensic_value_labels, train_fern_scorer, byte_cost, keep_under_budget
from fidelity import build_ground_truth, forensic_fidelity
from stage_clf import train_stage_clf, predict_stages
from evaluation_suite import strat_split

EDGE_BUDGETS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]


def causal_context(flows):
    """Return (N,4) causal context features. Pure function of arrival order within pairs."""
    order = sorted(range(len(flows)), key=lambda i: flows[i]["t_first"])
    seen = defaultdict(int)
    last_t = {}
    seen_bucket = defaultdict(set)
    ctx = np.zeros((len(flows), 4), dtype=np.float32)
    for i in order:
        f = flows[i]
        key = tuple(sorted((f["src"], f["dst"])))
        n = seen[key]
        ctx[i, 0] = np.log1p(n)
        ctx[i, 1] = 1.0 if n == 0 else 0.0
        dt = 0.0 if key not in last_t else max(0.0, f["t_first"] - last_t[key])
        ctx[i, 2] = np.log1p(dt)
        bucket = (int(f.get("proto_tcp", 0)), int(f.get("proto_udp", 0)),
                  int(f.get("proto_icmp", 0)), int(f.get("dport", 0)) // 1024)
        ctx[i, 3] = 1.0 if bucket not in seen_bucket[key] else 0.0
        seen[key] = n + 1
        last_t[key] = f["t_first"]
        seen_bucket[key].add(bucket)
    return ctx


def run_edge(seeds=(0, 1, 2)):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    flows = load_flows("data/processed/edge_flows_full.jsonl")
    X, _ = make_xy(flows, "edge-iiot")
    ctx = causal_context(flows)
    Xc = np.concatenate([X, ctx], axis=1).astype(np.float32)   # 16 + 4 = 20 features
    variants = {"base16": X, "context20": Xc}
    agg = {v: {"sr": defaultdict(list), "chain": defaultdict(list)} for v in variants}
    for sd in seeds:
        rng = np.random.default_rng(sd)
        tr, te = strat_split([f["attack"] for f in flows], rng)
        te = sorted(te, key=lambda i: flows[i]["t_first"])
        ftr = [flows[i] for i in tr]; fte = [flows[i] for i in te]
        ce = np.array([float(byte_cost(f)) for f in fte]); full = float(ce.sum())
        gt = build_ground_truth(fte, "edge-iiot")
        v = forensic_value_labels(ftr, "edge-iiot", n_rare_stages=3, anchor_focused=True)
        sm = train_stage_clf(X[tr], ftr, "edge-iiot")
        pst = predict_stages(sm, X[te])
        for name, XX in variants.items():
            s_te, _ = train_fern_scorer(XX[tr], v, XX[te], dev=dev)
            for b in EDGE_BUDGETS:
                mask = keep_under_budget(np.argsort(-(s_te / np.sqrt(ce))), ce, b * full)
                fid = forensic_fidelity(fte, (pst != "Normal").astype(int) * mask.astype(int),
                                        "edge-iiot", gt=gt, pred_stages=pst)
                agg[name]["sr"][b].append(fid["stage_recall"])
                agg[name]["chain"][b].append(fid["chain_completeness"])
        print(f"seed {sd} done", flush=True)
    out = {"budgets": EDGE_BUDGETS, "variants": {}}
    for name in variants:
        out["variants"][name] = {
            "stage_recall": [round(float(np.mean(agg[name]["sr"][b])), 4) for b in EDGE_BUDGETS],
            "stage_recall_std": [round(float(np.std(agg[name]["sr"][b])), 4) for b in EDGE_BUDGETS],
            "chain": [round(float(np.mean(agg[name]["chain"][b])), 4) for b in EDGE_BUDGETS],
        }
    return out


if __name__ == "__main__":
    out = run_edge()
    json.dump(out, open("outputs/fern_context.json", "w"), indent=2)
    print(json.dumps(out, indent=1))
    print("CONTEXT DONE", flush=True)



