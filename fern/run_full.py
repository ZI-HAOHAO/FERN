"""Orchestrate multi-seed diagnostics, retention, and deployability profiling.

Usage: python3 run_full.py --flows <jsonl> --dataset edge-iiot --seeds 0,1,2 --out-dir <dir>
"""
import argparse, json, os, subprocess, time
import numpy as np


def run(cmd):
    print("+", cmd)
    subprocess.run(cmd, shell=True, check=True)


def agg(vals):
    a = np.array(vals, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std()), "n": len(a)}


def profile_fern(flows_path, dataset, out_dir):
    """CPU inference latency and memory of the FERN forensic-importance scorer."""
    import torch, torch.nn as nn
    from pilot_k1 import load_flows, FEATS, make_xy
    flows = load_flows(flows_path, max_flows=60000)
    X, _ = make_xy(flows, dataset)
    dev = "cpu"   # gateway proxy: CPU-only
    net = nn.Sequential(nn.Linear(X.shape[1], 32), nn.ReLU(), nn.Linear(32, 1)).to(dev)
    net.eval()
    nparams = sum(p.numel() for p in net.parameters())
    mem_kb = nparams * 4 / 1024.0          # float32
    Xt = torch.tensor((X - X.mean(0)) / (X.std(0) + 1e-6), dtype=torch.float32)
    # warmup
    with torch.no_grad():
        net(Xt[:1000])
    N = min(50000, len(Xt))
    t0 = time.perf_counter()
    with torch.no_grad():
        net(Xt[:N])
    dt = time.perf_counter() - t0
    per_flow_us = dt / N * 1e6
    throughput = N / dt
    return {"params": int(nparams), "model_mem_kb": round(mem_kb, 2),
            "cpu_latency_us_per_flow": round(per_flow_us, 3),
            "cpu_throughput_flows_per_s": int(throughput)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows", required=True)
    ap.add_argument("--dataset", default="edge-iiot")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    k1_runs, m2_runs = [], []
    for sd in seeds:
        k1o = f"{args.out_dir}/k1_s{sd}.json"
        m2o = f"{args.out_dir}/m2_s{sd}.json"
        run(f"python3 k1_diag.py --flows {args.flows} --dataset {args.dataset} --seed {sd} --out {k1o}")
        run(f"python3 retention.py --flows {args.flows} --dataset {args.dataset} --seed {sd} --out {m2o}")
        k1_runs.append(json.load(open(k1o)))
        m2_runs.append(json.load(open(m2o)))

    # ---- aggregate K1 ----
    models = ["lgbm", "logreg", "mlp", "cnn1d", "tinyml", "volume_biased"]
    k1_agg = {}
    for m in models:
        k1_agg[m] = {
            "det_f1": agg([r[m]["det_f1"] for r in k1_runs]),
            "stage_recall": agg([r[m]["stage_recall"] for r in k1_runs]),
            "chain_completeness": agg([r[m]["chain_completeness"] for r in k1_runs]),
        }
    k1_agg["matched_stage_recall_spread"] = agg([r["_K1_summary"]["stage_recall_spread_matched"] for r in k1_runs])
    k1_agg["matched_detf1_spread"] = agg([r["_K1_summary"]["det_f1_spread_matched"] for r in k1_runs])

    # ---- aggregate M2 ----
    budgets = m2_runs[0]["budgets"]
    pols = list(m2_runs[0]["policies"].keys())

    def val_at(run, pol, b, metric):
        for c in run["policies"][pol]:
            if abs(c["budget"] - b) < 1e-9:
                return c[metric]
        return 0.0

    m2_agg = {"budgets": budgets, "policies": {}}
    for pol in pols:
        chain = [agg([val_at(r, pol, b, "chain_completeness") for r in m2_runs]) for b in budgets]
        srec = [agg([val_at(r, pol, b, "stage_recall") for r in m2_runs]) for b in budgets]
        m2_agg["policies"][pol] = {"chain": chain, "stage_recall": srec}

    k4 = profile_fern(args.flows, args.dataset, args.out_dir)

    summary = {"seeds": seeds, "K1": k1_agg, "M2": m2_agg, "K4_deployability": k4}
    out = f"{args.out_dir}/CONSOLIDATED.json"
    json.dump(summary, open(out, "w"), indent=2)

    # human-readable
    print("\n================ CONSOLIDATED (mean+/-std over seeds) ================")
    print("K1 (detection F1 vs forensic stage-recall):")
    for m in models:
        a = k1_agg[m]
        print(f"  {m:14s} detF1={a['det_f1']['mean']:.3f}+/-{a['det_f1']['std']:.3f}  "
              f"stage_recall={a['stage_recall']['mean']:.3f}+/-{a['stage_recall']['std']:.3f}")
    print(f"  >> matched-detector stage-recall spread = {k1_agg['matched_stage_recall_spread']['mean']:.3f}"
          f" (detF1 spread {k1_agg['matched_detf1_spread']['mean']:.3f})")
    print("\nM2 stage_recall @ budgets", budgets)
    for pol in pols:
        sr = m2_agg["policies"][pol]["stage_recall"]
        print(f"  {pol:14s} " + " ".join(f"{x['mean']:.2f}" for x in sr))
    print(f"\nK4 deployability (CPU/gateway proxy): {k4}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()



