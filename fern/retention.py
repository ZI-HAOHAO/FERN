"""Forensic rate-distortion: learned evidence retention under byte budget.

Setting (online, anti-gaming per the project data contract):
  * flows arrive in chronological order (no future lookahead for online policies);
  * a retention POLICY decides which flows to KEEP under a fixed retained-BYTE budget
    (budget = kept_bytes / total_bytes); a flow's byte cost = its bytes_tot;
  * forensic fidelity is then measured by running the FIXED reconstructor over the
    KEPT evidence only, scored against the FULL-data ground-truth chains.

A good policy preserves the low-volume early-stage anchors (Recon/Exploit/C2) that
define chain completeness -these are cheap in bytes but forensically decisive, which
is exactly why anomaly/volume-triggered capture (which favors big DDoS flows) loses them.

Policies:
  random, recency, reservoir, netflow_only (flow-record cost), head_of_flow (first-K-bytes
  cost), anomaly_topk (unsupervised score), alert_triggered (IDS-flagged), FERN (learned
  forensic-importance scorer). The FERN reward proxy: predict whether a flow is the FIRST
  flow of a stage in its campaign (the reconstruction anchor) -teacher labels from
  full-data ground truth; the scorer sees ONLY flow features, never labels, at test time.
"""
import argparse
import json
import numpy as np
from collections import defaultdict
from stage_mapping import map_attack, UNKNOWN
from fidelity import build_ground_truth, forensic_fidelity, reconstruct
from pilot_k1 import load_flows, FEATS, make_xy

BUDGETS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]


def byte_cost(f):
    return max(1, int(f.get("bytes_tot", 0) or 0))


def forensic_value_labels(flows, dataset, n_rare_stages=2, anchor_focused=False):
    """FERN teacher: a flow is forensically valuable (1) if it is malicious AND belongs to
    one of the `n_rare_stages` RAREST stages (fewest flows -the early Recon/Exploit/C2/
    Lateral evidence that byte-driven policies drop), OR it is a stage anchor. Benign -> 0.

    Rare-stage membership IS a function of flow features (a port-scan flow looks unlike a
    DDoS flood), so the scorer can learn it -unlike the positional 'anchor' identity.
    """
    stage_count = defaultdict(int)
    stages = []
    for f in flows:
        s, _ = map_attack(f["attack"], dataset)
        stages.append(s)
        if s not in (None, UNKNOWN):
            stage_count[s] += 1
    rare = set(sorted(stage_count, key=lambda s: stage_count[s])[:n_rare_stages])
    anchors = anchor_labels(flows, dataset)
    val = np.zeros(len(flows), dtype=np.int64)
    if anchor_focused:
        # ablation-driven: per-(campaign,stage) first flows only -directly targets the
        # evidence stage-recall rewards. Found best in the teacher ablation (App. C).
        return anchors.astype(np.int64)
    for i, s in enumerate(stages):
        if s in (None, UNKNOWN):
            continue
        if s in rare or anchors[i]:
            val[i] = 1
    return val


def anchor_labels(flows, dataset, min_stages=2):
    """Teacher signal for FERN: 1 if flow is the first flow of its stage within a REAL
    (>=min_stages) (attacker,victim) campaign -the forensic anchor.

    Spoofed-source floods (UDP/ICMP) create one singleton (src,victim) pair per packet;
    those are NOT real campaigns and must not each contribute an anchor. We therefore use
    the same >=min_stages campaign filter as build_ground_truth before assigning anchors.
    """
    pair_items = defaultdict(list)   # pair -> list of (idx, stage, ts)
    for i, f in enumerate(flows):
        stage, _ = map_attack(f["attack"], dataset)
        if stage in (None, UNKNOWN):
            continue
        key = tuple(sorted((f["src"], f["dst"])))
        pair_items[key].append((i, stage, f["t_first"]))
    anchors = np.zeros(len(flows), dtype=np.int64)
    for key, items in pair_items.items():
        stages_present = {st for _, st, _ in items}
        if len(stages_present) < min_stages:
            continue                                  # singleton/flood pair, not a campaign
        stage_first = {}
        for idx, st, ts in items:
            if st not in stage_first or ts < stage_first[st][1]:
                stage_first[st] = (idx, ts)
        for st, (idx, _) in stage_first.items():
            anchors[idx] = 1
    return anchors


def keep_under_budget(order, costs, budget_bytes):
    """Greedily keep flows in priority `order` until byte budget exhausted. Returns mask."""
    mask = np.zeros(len(costs), dtype=bool)
    spent = 0
    for idx in order:
        c = costs[idx]
        if spent + c > budget_bytes:
            continue
        mask[idx] = True
        spent += c
    return mask


def policy_order(name, flows, X, costs, rng, ids_score=None, anomaly_score=None, fern_score=None):
    n = len(flows)
    if name == "random":
        o = rng.permutation(n)
    elif name == "recency":
        o = np.argsort([-f["t_last"] for f in flows])           # newest first
    elif name == "reservoir":
        o = rng.permutation(n)                                   # uniform == random order
    elif name == "netflow_only":
        # NetFlow keeps cheap flow records: prioritize SMALL-byte flows (fit many)
        o = np.argsort(costs)
    elif name == "head_of_flow":
        # truncate濮ｅ桓low to first ~200B: cost is min(200, bytes); prioritize by pkt count
        o = np.argsort([-f["n_tot"] for f in flows])
    elif name == "anomaly_topk":
        o = np.argsort(-anomaly_score)
    elif name == "alert_triggered":
        # keep IDS-flagged malicious first (high score), then by score
        o = np.argsort(-ids_score)
    elif name == "fern":
        o = np.argsort(-fern_score)
    else:
        raise ValueError(name)
    return np.asarray(o)


def train_fern_scorer(Xtr, atr, Xte, dev="cuda"):
    """Lightweight forensic-importance scorer (gateway-deployable MLP). Predicts anchor."""
    import torch, torch.nn as nn
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xt = torch.tensor((Xtr - mu) / sd, device=dev)
    at = torch.tensor(atr, device=dev).float()
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 32), nn.ReLU(), nn.Linear(32, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    # class imbalance: anchors are rare -> pos_weight
    pw = torch.tensor([(len(atr) - atr.sum()) / max(1, atr.sum())], device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    bs = 4096
    for ep in range(12):
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            out = net(Xt[idx]).squeeze(-1)
            loss = lossf(out, at[idx]); loss.backward(); opt.step()
    with torch.no_grad():
        s = torch.sigmoid(net(torch.tensor((Xte - mu) / sd, device=dev)).squeeze(-1)).cpu().numpy()
    return s, net


def unsupervised_anomaly(X):
    """Simple anomaly score = distance from benign-ish centroid in standardized space."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    return np.linalg.norm(Z, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows", required=True)
    ap.add_argument("--dataset", default="edge-iiot")
    ap.add_argument("--out", default="m2_rate_distortion.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anchor-focused", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    flows = load_flows(args.flows)
    # Edge-IIoTset attack pcaps are SEPARATE captures with independent clocks, so a global
    # absolute-time split induces train/test distribution shift (DDoS lands in one side).
    # Use a seeded stratified-by-attack split for train/eval; within-flow t_first is still
    # used by temporal policies (recency) as the arrival proxy.
    by_attack = defaultdict(list)
    for i, f in enumerate(flows):
        by_attack[f["attack"]].append(i)
    tr_idx, te_idx = [], []
    for atk, idxs in by_attack.items():
        idxs = list(idxs); rng.shuffle(idxs)
        k = int(0.7 * len(idxs))
        tr_idx += idxs[:k]; te_idx += idxs[k:]
    order = tr_idx + te_idx          # train block then test block
    flows = [flows[i] for i in order]
    _cut_strat = len(tr_idx)
    X, y = make_xy(flows, args.dataset)
    costs = np.array([byte_cost(f) for f in flows])
    total_bytes = int(costs.sum())
    print(f"{len(flows)} flows, {total_bytes/1e6:.1f} MB total, mal-rate {y.mean():.3f}")

    # stratified split boundary (train block | test block) from above
    cut = _cut_strat
    flows_tr = flows[:cut]
    # FERN teacher built on TRAIN ONLY (rare-stage counts + anchors from train flows).
    fvalue_tr = forensic_value_labels(flows_tr, args.dataset, n_rare_stages=3,
                                      anchor_focused=getattr(args, "anchor_focused", False))
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fern_score_te, _ = train_fern_scorer(X[:cut], fvalue_tr, X[cut:], dev=dev)
    print(f"FERN teacher positives (train): {int(fvalue_tr.sum())} ({100*fvalue_tr.mean():.2f}%)")

    # eval on the test segment only (held-out)
    fe = flows[cut:]; Xe = X[cut:]; ye = y[cut:]; ce = costs[cut:]
    gt = build_ground_truth(fe, args.dataset)
    anchors_te = anchor_labels(fe, args.dataset)         # oracle ceiling uses test GT (allowed)
    full_bytes = int(ce.sum())
    print(f"test: {len(fe)} flows, {full_bytes/1e6:.1f} MB, GT campaigns {len(gt)}")

    # train-only multiclass stage classifier -> predicted stages for the investigator.
    from stage_clf import train_stage_clf, predict_stages
    from pilot_k1 import FEATS
    stage_model = train_stage_clf(X[:cut], flows_tr, args.dataset)

    # the investigator can only use features its policy's RETAINED bytes support.
    #  * raw-packet & netflow policies keep the whole flow / full flow record -> all
    #    flow-stat features (NetFlow record == our aggregate features) are available.
    #  * head_of_flow keeps only the first ~200B (first 1-3 packets): total counts/bytes/
    #    duration/rate of the FULL flow are UNKNOWN -> those features are masked to the
    #    train mean (information the truncated capture genuinely does not contain).
    train_mean = X[:cut].mean(0)
    HEADFLOW_OK = {"sport", "dport", "proto_tcp", "proto_udp", "proto_icmp", "ratio_fwd", "mean_pkt"}
    def masked_eval(avail_set):
        Xm = Xe.copy()
        for j, k in enumerate(FEATS):
            if k not in avail_set:
                Xm[:, j] = train_mean[j]
        return predict_stages(stage_model, Xm)
    pred_stages_full = predict_stages(stage_model, Xe)          # whole-flow / netflow record
    pred_stages_head = masked_eval(HEADFLOW_OK)                 # 200B truncation
    def stages_for(pol):
        return pred_stages_head if pol == "head_of_flow" else pred_stages_full

    # IDS score (lgbm) + anomaly score on test. Anomaly centroid fit on TRAIN only (train-only).
    from pilot_k1 import train_lgbm
    ids_pred, ids_score = train_lgbm(X[:cut], y[:cut], Xe)
    mu_tr, sd_tr = X[:cut].mean(0), X[:cut].std(0) + 1e-6
    anomaly_score = np.linalg.norm((Xe - mu_tr) / sd_tr, axis=1)
    ids_priority = ids_score.copy()

    POLICIES = ["random", "recency", "reservoir", "netflow_only", "head_of_flow",
                "anomaly_topk", "alert_triggered", "fern"]
    # oracle upper bound: keep anchors first (perfect forensic-importance)
    POLICIES.append("oracle_anchor")

    def policy_cost(pol):
        # per-policy retained-byte cost. NetFlow keeps a tiny flow record; head-of-flow
        # keeps only the first 200B; raw-packet policies pay full bytes.
        if pol == "netflow_only":
            return np.minimum(ce, 80).astype(float)
        if pol == "head_of_flow":
            return np.minimum(ce, 200).astype(float)
        return ce.astype(float)

    results = {"budgets": BUDGETS, "policies": {}, "total_test_bytes": full_bytes,
               "n_gt_campaigns": len(gt)}
    for pol in POLICIES:
        pc = policy_cost(pol)
        pstages = stages_for(pol)        # per-policy investigator features 
        curve = []
        for b in BUDGETS:
            bb = int(b * full_bytes)
            if pol == "oracle_anchor":
                o = np.argsort(-anchors_te.astype(float) - 1e-6 * ce)  # anchors first, cheap first
            elif pol == "fern":
                # value-PER-BYTE: keep many cheap high-value flows (like the oracle does
                # with cheap anchors) rather than a few expensive ones.
                o = np.argsort(-(fern_score_te / np.sqrt(ce.astype(float))))
            elif pol == "anomaly_topk":
                o = np.argsort(-anomaly_score)
            elif pol == "alert_triggered":
                o = np.argsort(-ids_priority)
            else:
                o = policy_order(pol, fe, Xe, ce, rng)
            mask = keep_under_budget(o, pc, bb)
            # realistic investigator -the analyst reads retained evidence and the
            # train-only stage classifier labels each KEPT flow's stage. No true-label use.
            preds = np.zeros(len(fe), dtype=int)
            kept_idx = np.where(mask)[0]
            preds[kept_idx] = (pstages[kept_idx] != "Normal").astype(int)
            fid = forensic_fidelity(fe, preds, args.dataset, gt=gt, pred_stages=pstages)
            curve.append({"budget": b, "kept_bytes_frac": float(ce[mask].sum() / full_bytes),
                          "chain_completeness": fid["chain_completeness"],
                          "stage_recall": fid["stage_recall"],
                          "entity_f1": fid["entity_attribution_f1"]})
        results["policies"][pol] = curve
        cc = [f"{c['chain_completeness']:.2f}" for c in curve]
        sr = [f"{c['stage_recall']:.2f}" for c in curve]
        print(f"[{pol:15s}] chain@budgets {cc}  stage_recall {sr}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")

    # Summary check: does FERN dominate anomaly_triggered & best baseline at 1-5%?
    def at(pol, b):
        for c in results["policies"][pol]:
            if abs(c["budget"] - b) < 1e-9:
                return c["chain_completeness"]
        return 0.0
    print("\n=== K2 SUMMARY (chain_completeness) ===")
    for b in [0.01, 0.05]:
        fern = at("fern", b); anom = at("anomaly_topk", b); alert = at("alert_triggered", b)
        base = max(at("random", b), at("recency", b), at("netflow_only", b), at("head_of_flow", b))
        print(f" @{b*100:.0f}% budget: FERN={fern:.2f} vs anomaly={anom:.2f} "
              f"alert={alert:.2f} bestbaseline={base:.2f} oracle={at('oracle_anchor', b):.2f}"
              f"  -> {'FERN wins' if fern >= max(anom, alert, base) and fern>0 else 'no'}")


if __name__ == "__main__":
    main()



