"""Fixed reconstruction procedure and forensic-fidelity metrics.

Ground-truth model (scenario-grounded, derived from public-capture metadata):
  * A CAMPAIGN is the set of malicious flows sharing (attacker_ip, victim_ip).
  * The campaign's GT chain = the stages present, ordered by each stage's first-seen
    timestamp. (attacker_ip = the endpoint that is the src of the majority of the
    campaign's malicious flows; victim_ip = the other endpoint.)
  * Confidence flags come from stage_mapping; campaigns with <2 ordered stages are
    excluded from chain-level metrics (kept for detection only) -never padded.

A model's ALERT STREAM is: for each flow, (is_malicious_pred, pred_stage|None).
The FIXED reconstructor turns that stream into a predicted chain per campaign using
ONLY the flows the model flagged malicious -no oracle access to labels.

Metrics (all in [0,1], higher=better), distinct from detection F1:
  * stage_recall      -frac of GT stages whose flows the model flagged (>=1)
  * ordering_consistency -Kendall-tau-like pairwise order agreement of recovered stages
  * entity_attribution_f1 -F1 of identifying (attacker_ip, victim_ip) per campaign
  * chain_completeness -frac of campaigns whose full stage set is recovered in correct order
"""
import numpy as np
from collections import defaultdict
from stage_mapping import map_attack, STAGE_RANK, STAGE_ORDER, UNKNOWN


def build_ground_truth(flows, dataset="edge-iiot", min_stages=2):
    """flows: list of dicts with src,dst,attack,t_first. Returns campaigns dict.

    campaign_id = (attacker_ip, victim_ip). Returns {cid: {stages_ordered, stage_first_ts,
    flow_idx, attacker, victim}}.
    """
    # group malicious flows by unordered ip-pair, decide attacker by src-majority
    pair_flows = defaultdict(list)
    for i, f in enumerate(flows):
        stage, conf = map_attack(f["attack"], dataset)
        if stage is None or stage == UNKNOWN:
            continue
        a, b = f["src"], f["dst"]
        key = tuple(sorted((a, b)))
        pair_flows[key].append((i, f, stage))

    campaigns = {}
    for key, items in pair_flows.items():
        # attacker = endpoint that is src most often
        src_count = defaultdict(int)
        for _, f, _ in items:
            src_count[f["src"]] += 1
        attacker = max(src_count, key=src_count.get)
        victim = key[1] if key[0] == attacker else key[0]
        stage_first = {}
        flow_idx = []
        for i, f, stage in items:
            flow_idx.append(i)
            ts = f["t_first"]
            if stage not in stage_first or ts < stage_first[stage]:
                stage_first[stage] = ts
        stages_ordered = sorted(stage_first, key=lambda s: stage_first[s])
        if len(stages_ordered) < min_stages:
            continue
        campaigns[(attacker, victim)] = {
            "stages_ordered": stages_ordered,
            "stage_first_ts": stage_first,
            "flow_idx": flow_idx,
            "attacker": attacker, "victim": victim,
        }
    return campaigns


def reconstruct(flows, preds, dataset="edge-iiot", min_stages=2, pred_stages=None):
    """FIXED reconstruction from model alert stream.

    preds: array len(flows), predicted malicious(1)/benign(0).
    pred_stages: REQUIRED array len(flows) of PREDICTED stage strings (from a train-only
      multi-class classifier). A flagged flow contributes its PREDICTED stage -never the
      true label (that would be oracle leakage). pred_stages[i] may be None/benign.
    Benign-predicted or None-stage flagged flows are dropped from chain assembly but still
    count as false-positive campaigns for entity precision.
    """
    if pred_stages is None:
        raise ValueError("reconstruct requires pred_stages (predicted), not true labels")
    pair_flows = defaultdict(list)
    for i, f in enumerate(flows):
        if not preds[i]:
            continue
        stage = pred_stages[i]
        if stage is None or stage == UNKNOWN or stage == "Normal":
            continue
        a, b = f["src"], f["dst"]
        key = tuple(sorted((a, b)))
        pair_flows[key].append((i, f, stage))
    pred_campaigns = {}
    for key, items in pair_flows.items():
        src_count = defaultdict(int)
        for _, f, _ in items:
            src_count[f["src"]] += 1
        attacker = max(src_count, key=src_count.get)
        victim = key[1] if key[0] == attacker else key[0]
        stage_first = {}
        for i, f, stage in items:
            ts = f["t_first"]
            if stage not in stage_first or ts < stage_first[stage]:
                stage_first[stage] = ts
        stages_ordered = sorted(stage_first, key=lambda s: stage_first[s])
        # hold predicted campaigns to the SAME >=min_stages bar as the GT builder,
        # so benign false-positive pairs (0 stages) don't swamp entity attribution.
        if len(stages_ordered) < min_stages:
            continue
        pred_campaigns[(attacker, victim)] = {
            "stages_ordered": stages_ordered,
            "stage_first_ts": stage_first,
        }
    return pred_campaigns


def _pairwise_order_agreement(gt_stages, pred_stages):
    """Kendall-tau-like: over GT stage pairs both recovered, frac in correct order."""
    common = [s for s in gt_stages if s in pred_stages]
    if len(common) < 2:
        return 1.0 if len(common) >= 0 else 0.0
    gt_rank = {s: i for i, s in enumerate(gt_stages)}
    pr_rank = {s: i for i, s in enumerate(pred_stages)}
    agree = tot = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            a, b = common[i], common[j]
            tot += 1
            if (gt_rank[a] < gt_rank[b]) == (pr_rank[a] < pr_rank[b]):
                agree += 1
    return agree / max(1, tot)


def forensic_fidelity(flows, preds, dataset="edge-iiot", gt=None, pred_stages=None):
    """Return dict of the 4 fidelity metrics + support counts.

    pred_stages: predicted stages (train-only classifier) -passed to reconstruct so no
    true-label leakage. Ground truth (gt) legitimately uses true labels.
    """
    if gt is None:
        gt = build_ground_truth(flows, dataset)
    pred = reconstruct(flows, preds, dataset, pred_stages=pred_stages)

    stage_recalls, orderings, completeness = [], [], []
    ent_tp = ent_fp = ent_fn = 0
    for cid, g in gt.items():
        p = pred.get(cid)
        gt_stages = g["stages_ordered"]
        if p is None:
            stage_recalls.append(0.0); orderings.append(0.0); completeness.append(0.0)
            ent_fn += 1
            continue
        ent_tp += 1  # campaign (attacker,victim) correctly identified
        pred_stages = p["stages_ordered"]
        rec = sum(1 for s in gt_stages if s in pred_stages) / len(gt_stages)
        stage_recalls.append(rec)
        orderings.append(_pairwise_order_agreement(gt_stages, pred_stages))
        completeness.append(1.0 if (set(gt_stages) == set(pred_stages) and
                                    pred_stages[:len(gt_stages)] == gt_stages) else 0.0)
    # false-positive campaigns (model produced an attacker/victim pair not in GT)
    ent_fp = sum(1 for cid in pred if cid not in gt)
    ent_prec = ent_tp / max(1, ent_tp + ent_fp)
    ent_rec = ent_tp / max(1, ent_tp + ent_fn)
    ent_f1 = 2 * ent_prec * ent_rec / max(1e-9, ent_prec + ent_rec)
    return {
        "per_campaign_stage_recall": [float(x) for x in stage_recalls],
        "stage_recall": float(np.mean(stage_recalls)) if stage_recalls else 0.0,
        "ordering_consistency": float(np.mean(orderings)) if orderings else 0.0,
        "entity_attribution_f1": float(ent_f1),
        "chain_completeness": float(np.mean(completeness)) if completeness else 0.0,
        "n_campaigns_gt": len(gt),
        "n_campaigns_pred": len(pred),
    }




