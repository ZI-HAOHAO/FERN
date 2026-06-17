# Data Layout

This repository does not bundle raw datasets or experiment-result JSON files.

Why:

- Raw datasets are large and should be downloaded from the original public providers.
- Processed flow files and result JSON files can encode the exact experimental outputs; they are intentionally not included in this code-only GitHub version.
- `examples/mini_flows.jsonl` is a tiny synthetic file included only to document the expected flow schema.

Expected local layout:

```text
data/raw/
  edge-iiotset/      # downloaded PCAP files
  iot23/             # IoT-23 labeled Zeek CSV files
  cic-iot-2023/      # CIC-IoT-2023 CSV files

data/processed/
  edge_flows_full.jsonl
data/examples/
  mini_flows.jsonl
```

Use `fern/pcap_to_flow.py` to create `data/processed/edge_flows_full.jsonl` from Edge-IIoTset PCAPs.
