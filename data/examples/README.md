# Example Data

`mini_flows.jsonl` is a tiny synthetic flow file for checking field names and basic script wiring.

It is not an experimental dataset and should not be used for evaluation claims.

Each line is one bidirectional flow record. Required fields used by the core scripts include:

- `t_first`, `t_last`: first and last timestamps.
- `src`, `dst`: endpoint identifiers.
- `sport`, `dport`: source and destination ports.
- `proto`: protocol string such as `TCP`, `UDP`, or `ICMP`.
- `duration`: flow duration.
- `pkts_fwd`, `pkts_bwd`: forward/backward packet counts.
- `bytes_fwd`, `bytes_bwd`: forward/backward byte counts.
- `label`: `benign` or `malicious`.
- `attack`: attack name mapped by `fern/stage_mapping.py`.

Real experiments require full public datasets under `data/raw/` and processed flow JSONL files under `data/processed/`.

