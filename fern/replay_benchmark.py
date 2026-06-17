"""Controlled replay benchmark: controlled storage-stress replay benchmark.

A semi-synthetic benchmark that raises the campaign count from 30 to N=80 with proper
episode-level statistics, while staying grounded in REAL Edge-IIoTset flows and reproducing
the storage-stress that the storage-stress setting is about:

  * REAL flows only -- every attack anchor and every background flow is a real Edge-IIoTset
    capture; we only re-compose them into controlled campaigns.
  * Each episode = one (attacker,victim) host pair, a sampled kill-chain template, 1-2 CHEAP
    real flows per stage (drawn from the low-byte half of that stage's pool, so anchors are
    byte-cheap, matching the real Edge regime).
  * Storage stress = real flood-heavy background: bulky DDoS flows + benign flows at random
    singleton host pairs (never a campaign), injected so the anchors are only ~0.5% of bytes
    and the budget is dominated by redundant flood evidence -- exactly what defeats byte- and
    anomaly-driven retention on real Edge-IIoTset.
  * Baselines use the same evaluation protocol: Random, and centroid-distance Anomaly-top-k (keep most
    anomalous by raw score, cost-unaware), versus FERN-Greedy and the causal online rule.
  * No leakage: scorer, stage classifier and online threshold are trained on the REAL train
    split; episodes are built from the disjoint REAL test donors.

Reports per-episode forensic stage-recall and chain-completeness at 0.5/2/5% byte budgets,
mean over N episodes with bootstrap 95% CIs, plus the FERN-random gap CI.
"""
import json
import numpy as np
from collections import defaultdict
from pilot_k1 import load_flows, make_xy
from retention import forensic_value_labels, train_fern_scorer, byte_cost, keep_under_budget
from stage_clf import train_stage_clf, predict_stages
from stage_mapping import map_attack, UNKNOWN, STAGE_RANK
from evaluation_suite import strat_split, keep_threshold_online, calib_thr
from fidelity import build_ground_truth, reconstruct

N_EPISODES = 80
TARGET_ANCHOR_FRAC = 0.005         # episode anchors ~0.5% of bytes -> flood-dominated stress
FLOOD_BYTE_SHARE = 0.75            # of the injected background bytes, 75% from bulky floods
BUDGETS = [0.005, 0.02, 0.05]
TEMPLATES = [
    (["Recon", "Exploit", "ExfilImpact"], 3),
    (["Recon", "Exploit", "Lateral", "ExfilImpact"], 3),
    (["Recon", "Exploit"], 2),
    (["Recon", "Lateral", "ExfilImpact"], 2),
    (["Recon", "ExfilImpact"], 2),
]


def stage_of(f):
    s, _ = map_attack(f["attack"], "edge-iiot")
    return s if s not in (None, UNKNOWN) else None


def build_episodes(donor_flows, donor_X, rng):
    pool = defaultdict(list)       # stage -> donor indices (sorted cheap->expensive below)
    benign, mal = [], []
    for i, f in enumerate(donor_flows):
        s = stage_of(f)
        if s is None:
            benign.append(i)
        else:
            pool[s].append(i); mal.append(i)
    for s in pool:                 # cheap half = lower byte_cost
        pool[s] = sorted(pool[s], key=lambda i: byte_cost(donor_flows[i]))
    flood = sorted(mal, key=lambda i: -byte_cost(donor_flows[i]))   # bulky distractors
    flood = flood[:max(1, len(flood) // 2)]
    tmpl_idx = list(range(len(TEMPLATES)))
    tmpl_w = np.array([w for _, w in TEMPLATES], float); tmpl_w /= tmpl_w.sum()
    rep_flows, rep_rows, ep_id = [], [], []
    W = 10_000.0

    def add(di, src, dst, t):
        f = dict(donor_flows[di]); f["src"], f["dst"], f["t_first"] = src, dst, t
        rep_flows.append(f); rep_rows.append(donor_X[di])
        return byte_cost(f)

    for e in range(N_EPISODES):
        atk, vic = f"10.90.{e // 256}.{e % 256}", f"10.80.{e // 256}.{e % 256}"
        chain = [s for s in TEMPLATES[rng.choice(tmpl_idx, p=tmpl_w)][0] if pool.get(s)]
        if len(set(chain)) < 2:
            chain = ["Recon", "Exploit"]
        t0 = e * W; anchor_bytes = 0
        n_anchor_start = len(rep_flows)
        for p, s in enumerate(chain):
            cheap = pool[s][:max(1, len(pool[s]) // 2)]            # low-byte half
            k = int(rng.integers(1, 3))                           # 1-2 cheap anchors
            for j, di in enumerate(rng.choice(cheap, size=min(k, len(cheap)), replace=False)):
                anchor_bytes += add(di, atk, vic, t0 + p * 100.0 + j + rng.random())
        for _ in range(len(rep_flows) - n_anchor_start):
            ep_id.append(e)
        # flood-heavy background to push anchors down to TARGET_ANCHOR_FRAC of episode bytes
        need = anchor_bytes * (1.0 / TARGET_ANCHOR_FRAC - 1.0)
        flood_need, benign_need = need * FLOOD_BYTE_SHARE, need * (1.0 - FLOOD_BYTE_SHARE)
        for src_pool, budget_bytes in ((flood, flood_need), (benign, benign_need)):
            got = 0.0
            order = rng.permutation(len(src_pool))
            oi = 0
            while got < budget_bytes and oi < len(order):
                di = src_pool[order[oi]]; oi += 1
                rs = f"10.{rng.integers(1, 60)}.{rng.integers(0, 256)}.{rng.integers(0, 256)}"
                rd = f"10.{rng.integers(1, 60)}.{rng.integers(0, 256)}.{rng.integers(0, 256)}"
                got += add(di, rs, rd, t0 + rng.random() * (len(chain) * 100.0))
                ep_id.append(-1)
    return rep_flows, np.array(rep_rows, dtype=np.float32), np.array(ep_id)


def per_episode_metrics(rep_flows, mask, pst, gt):
    pred = reconstruct(rep_flows, (pst != "Normal").astype(int) * mask.astype(int),
                       "edge-iiot", pred_stages=pst)
    srs, chains = [], []
    for cid, d in gt.items():
        gts = d["stages_ordered"]
        p = pred.get(cid); ps = p["stages_ordered"] if p else []
        srs.append(sum(1 for s in gts if s in ps) / len(gts))
        got_order = [s for s in ps if s in gts]
        ok = (set(gts) <= set(ps)) and got_order == sorted(set(gts), key=lambda s: STAGE_RANK[s])
        chains.append(1.0 if ok else 0.0)
    return np.array(srs), np.array(chains)


def boot_ci(x, reps=10000, seed=0):
    rng = np.random.default_rng(seed); n = len(x)
    means = [x[rng.integers(0, n, n)].mean() for _ in range(reps)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run(seed=0):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    flows = load_flows("data/processed/edge_flows_full.jsonl")
    X, _ = make_xy(flows, "edge-iiot")
    rng = np.random.default_rng(seed)
    tr, te = strat_split([f["attack"] for f in flows], rng)
    ftr = [flows[i] for i in tr]; Xtr = X[tr]
    donor = [flows[i] for i in te]; Xd = X[te]
    v = forensic_value_labels(ftr, "edge-iiot", n_rare_stages=3, anchor_focused=True)
    s_tr = train_fern_scorer(Xtr, v, Xtr, dev=dev)[0]
    sm = train_stage_clf(Xtr, ftr, "edge-iiot")
    ctr = np.array([float(byte_cost(flows[i])) for i in tr])
    # the anomaly baseline: distance from TRAIN centroid in standardized space (cost-unaware top-k)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6

    rep_flows, Xr, ep = build_episodes(donor, Xd, np.random.default_rng(seed + 1))
    gt = build_ground_truth(rep_flows, "edge-iiot")
    n_lat = sum(1 for d in gt.values() if "Lateral" in d["stages_ordered"])
    ce = np.array([float(byte_cost(f)) for f in rep_flows]); full = float(ce.sum())
    anchor_frac = float(ce[ep >= 0].sum() / full)
    s_re, _ = train_fern_scorer(Xtr, v, Xr, dev=dev)
    pst = predict_stages(sm, Xr)
    anom = np.linalg.norm((Xr - mu) / sd, axis=1)

    out = {"n_episodes": len(gt), "n_with_lateral": n_lat, "anchor_byte_frac": anchor_frac,
           "flood_byte_share": FLOOD_BYTE_SHARE, "budgets": BUDGETS, "policies": {}}
    rngp = np.random.default_rng(seed + 7)
    for b in BUDGETS:
        bb = b * full
        masks = {
            "fern_greedy": keep_under_budget(np.argsort(-(s_re / np.sqrt(ce))), ce, bb),
            "fern_online": keep_threshold_online(s_re, ce, calib_thr(s_tr, ctr, b * float(ctr.sum())), bb),
            "random": keep_under_budget(rngp.permutation(len(rep_flows)), ce, bb),
            "anomaly": keep_under_budget(np.argsort(-anom), ce, bb),     # baseline-compatible: raw desc
        }
        for name, mask in masks.items():
            sr, ch = per_episode_metrics(rep_flows, mask, pst, gt)
            lo, hi = boot_ci(sr)
            out["policies"].setdefault(name, {})[f"{b}"] = {
                "stage_recall": float(sr.mean()), "sr_ci": [lo, hi],
                "chain": float(ch.mean()), "kept_byte_frac": float(ce[mask].sum() / full)}
        srf, _ = per_episode_metrics(rep_flows, masks["fern_greedy"], pst, gt)
        srr, _ = per_episode_metrics(rep_flows, masks["random"], pst, gt)
        glo, ghi = boot_ci(srf - srr)
        out["policies"].setdefault("_gap", {})[f"{b}"] = {"gap": float((srf - srr).mean()), "ci": [glo, ghi]}
    return out


if __name__ == "__main__":
    out = run(0)
    json.dump(out, open("outputs/replay_benchmark.json", "w"), indent=2)
    print(json.dumps(out, indent=1))
    print("REPLAY DONE", flush=True)



