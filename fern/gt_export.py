"""Export forensic ground-truth files.

Emits GT_VALIDITY.md + gt_campaigns.json documenting how the scenario-grounded attack-chain
ground truth is constructed, with: construction rules, campaign count, stage distribution,
confidence-flag breakdown, excluded-case counts + reasons, and representative example chains.
This documents the benchmark construction rules and summary statistics.
"""
import argparse, json
import numpy as np
from collections import defaultdict, Counter
from stage_mapping import map_attack, UNKNOWN, STAGE_ORDER
from pilot_k1 import load_flows


def build_full(flows, dataset, min_stages=2):
    pair = defaultdict(list)
    excluded = Counter()
    for i, f in enumerate(flows):
        stage, conf = map_attack(f["attack"], dataset)
        if stage is None:
            continue
        if stage == UNKNOWN:
            excluded["unmapped_attack_name"] += 1
            continue
        key = tuple(sorted((f["src"], f["dst"])))
        pair[key].append((i, f, stage, conf))
    campaigns, conf_hist, stage_hist = {}, Counter(), Counter()
    for key, items in pair.items():
        stages_present = {st for _, _, st, _ in items}
        if len(stages_present) < min_stages:
            excluded["singleton_or_single_stage_pair"] += 1
            continue
        src_count = Counter(f["src"] for _, f, _, _ in items)
        attacker = src_count.most_common(1)[0][0]
        victim = key[1] if key[0] == attacker else key[0]
        stage_first = {}
        for _, f, st, conf in items:
            ts = f["t_first"]
            if st not in stage_first or ts < stage_first[st][0]:
                stage_first[st] = (ts, conf)
        ordered = sorted(stage_first, key=lambda s: stage_first[s][0])
        confs = [stage_first[s][1] for s in ordered]
        for cf in confs: conf_hist[cf] += 1
        for s in ordered: stage_hist[s] += 1
        campaigns[f"{attacker}->{victim}"] = {
            "attacker": attacker, "victim": victim,
            "stages_ordered": ordered, "stage_confidence": dict(zip(ordered, confs)),
            "n_flows": len(items),
        }
    return campaigns, conf_hist, stage_hist, excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows", required=True)
    ap.add_argument("--dataset", default="edge-iiot")
    ap.add_argument("--out-md", default="outputs/GT_VALIDITY.md")
    ap.add_argument("--out-json", default="outputs/gt_campaigns.json")
    args = ap.parse_args()
    flows = load_flows(args.flows)
    camps, conf_hist, stage_hist, excl = build_full(flows, args.dataset)
    json.dump(camps, open(args.out_json, "w"), indent=2)

    multi = [c for c in camps.values() if len(c["stages_ordered"]) >= 3]
    examples = sorted(camps.values(), key=lambda c: -len(c["stages_ordered"]))[:8]
    lines = []
    lines.append("# Forensic Ground-Truth Validity (scenario-grounded, checkable)\n")
    lines.append(f"Dataset: {args.dataset} | flows: {len(flows)} | campaigns (>=2 ordered stages): {len(camps)}\n")
    lines.append("## Construction rules (deterministic)\n")
    lines.append("1. PCAP -> bidirectional flows (fixed timeout) with packet-index back-pointers.\n"
                 "2. Each labeled malicious flow -> kill-chain stage via a published attack-name->stage table (ambiguous -> unknown-stage, excluded, never force-labeled).\n"
                 "3. Campaign = (attacker,victim) IP pair; attacker = src-majority endpoint.\n"
                 "4. Stage event time = first flow of that stage; chain order = stage first-seen time.\n"
                 "5. Confidence per stage = exact|derived|ambiguous from the mapping table.\n"
                 "6. Campaigns with <2 ordered stages excluded (singleton/spoofed-flood pairs); never padded.\n")
    lines.append(f"\n## Campaign statistics\n- total campaigns: {len(camps)}\n- multi-stage (>=3): {len(multi)}\n")
    lines.append(f"- stage occurrence: {dict(stage_hist)}\n")
    lines.append(f"- stage-confidence breakdown: {dict(conf_hist)}\n")
    lines.append(f"\n## Excluded cases (transparency)\n")
    for k, v in excl.items():
        lines.append(f"- {k}: {v}\n")
    lines.append(f"\n## Representative reconstructed chains (longest)\n")
    for c in examples:
        chain = " -> ".join(c["stages_ordered"])
        lines.append(f"- {c['attacker']} 閳?{c['victim']}: {chain}  (flows={c['n_flows']}, conf={c['stage_confidence']})\n")
    lines.append("\n## Limitation (stated)\n"
                 "Ground truth is scenario-grounded -composed from public per-attack captures sharing testbed "
                 "attacker/victim IPs -NOT real DFIR investigations. Confidence-labeled, deterministic, and "
                 "included in the release (gt_campaigns.json + this file + construction scripts) for audit.\n")
    open(args.out_md, "w").write("".join(lines))
    print(f"campaigns={len(camps)} multi>=3={len(multi)} stages={dict(stage_hist)} conf={dict(conf_hist)} excluded={dict(excl)}")
    print("wrote", args.out_md, "and", args.out_json)


if __name__ == "__main__":
    main()



