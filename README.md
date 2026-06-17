# FERN

FERN is a Python toolkit for evaluating forensic utility in IoT network traffic and for retaining flow evidence under byte budgets.

The code supports three main workflows:

- Convert packet captures or labeled flow logs into normalized flow records.
- Evaluate whether detector outputs support attack-chain reconstruction, not just binary detection.
- Train and test evidence-retention policies that decide which flows to keep under storage constraints.

## Repository Layout

```text
fern/
  pcap_to_flow.py          # PCAP -> bidirectional JSONL flow records
  stage_mapping.py         # attack-name -> kill-chain stage mapping
  fidelity.py              # campaign reconstruction and forensic-fidelity metrics
  stage_clf.py             # train-only stage classifier
  k1_diag.py               # detector-vs-forensic-fidelity diagnostic
  retention.py             # byte-budget retention baselines and scorer
  streaming_eval.py        # chronological online retention
  fern_submod.py           # submodular coverage selection
  fern_main.py             # mechanism comparison for FERN selectors
  iot23_pipeline.py        # IoT-23 parsing and host-centric evaluation
  cic_k1.py                # CIC-IoT-2023 per-flow transfer check
  evaluation_suite.py      # combined evaluation utilities
data/
  raw/                     # put downloaded datasets here
  processed/               # generated flow JSONL files
  examples/                # tiny synthetic schema example
outputs/                   # generated JSON summaries
```

## Installation

```bash
python -m venv .venv
.venv/Scripts/activate      # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Preparation

Raw datasets and experiment-result JSON files are not bundled in this code-only repository. Place downloaded data under `data/raw/`.

For a tiny schema example, see `data/examples/mini_flows.jsonl`. It is synthetic and only meant to show the expected JSONL fields.

For Edge-IIoTset PCAPs:

```bash
python fern/pcap_to_flow.py \
  --pcap-dir data/raw/edge-iiotset \
  --out data/processed/edge_flows_full.jsonl
```

For IoT-23 Zeek logs, put CSV files under `data/raw/iot23/`.

For CIC-IoT-2023, pass train/test CSV paths directly to `fern/cic_k1.py`.

## Example Commands

Detector fidelity diagnostic:

```bash
python fern/k1_diag.py \
  --flows data/processed/edge_flows_full.jsonl \
  --dataset edge-iiot \
  --seed 0 \
  --out outputs/k1_diag_seed0.json
```

Retention rate-distortion evaluation:

```bash
python fern/retention.py \
  --flows data/processed/edge_flows_full.jsonl \
  --dataset edge-iiot \
  --seed 0 \
  --anchor-focused \
  --out outputs/retention_seed0.json
```

Chronological online retention:

```bash
python fern/streaming_eval.py \
  --flows data/processed/edge_flows_full.jsonl \
  --dataset edge-iiot \
  --seed 0 \
  --out outputs/streaming_seed0.json
```

IoT-23 chain-level evaluation:

```bash
python fern/iot23_pipeline.py \
  --csv-dir data/raw/iot23 \
  --seeds 0,1,2 \
  --out outputs/iot23_chain.json
```

Ground-truth audit export:

```bash
python fern/gt_export.py \
  --flows data/processed/edge_flows_full.jsonl \
  --dataset edge-iiot \
  --out-md outputs/GT_VALIDITY.md \
  --out-json outputs/gt_campaigns.json
```

## Notes

- The evaluation code intentionally separates detection accuracy from forensic-fidelity metrics.
- Stage labels used by the investigator are predicted by train-only classifiers; test labels are not used as an oracle.
- Retention policies are charged by retained-byte cost, with policy-specific handling for raw flow, NetFlow-like, and head-of-flow evidence.
