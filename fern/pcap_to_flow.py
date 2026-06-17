"""Convert PCAP files to bidirectional flow records with packet back-pointers.

Each flow keeps:
  - 5-tuple key (canonicalized so A->B and B->A share a flow)
  - first/last timestamp
  - per-direction packet count + byte count
  - packet_indices: list of (global_pkt_id, ts, size, dir) so retention/reconstruction
    can point back to raw evidence
  - label / attack name carried from the source pcap file (Edge-IIoTset is per-attack file)

Parsing uses scapy's PcapReader (streaming, low memory). For very large pcaps we
cap packets via --max-packets to keep pilots fast; full runs remove the cap.

Output: a parquet/JSONL of flow records + a sidecar npy of packet sizes for byte-budget
accounting. Designed to be deterministic (no randomness) for reproducible ground truth.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

try:
    from scapy.all import PcapReader, IP, IPv6, TCP, UDP, ICMP
except Exception as e:  # pragma: no cover
    print(f"scapy import failed: {e}", file=sys.stderr)
    raise


def canon_key(src, sport, dst, dport, proto):
    """Direction-independent 5-tuple key. dir=0 means packet goes src->dst as keyed."""
    a = (src, sport)
    b = (dst, dport)
    if a <= b:
        return (src, sport, dst, dport, proto), 0
    return (dst, dport, src, sport, proto), 1


def parse_pcap(path, attack_name, max_packets=0, base_pkt_id=0):
    """Yield nothing; return (flows dict, n_packets). flows keyed by canon 5-tuple."""
    flows = {}
    pkt_id = base_pkt_id
    n = 0
    reader = PcapReader(path)
    for pkt in reader:
        n += 1
        if max_packets and n > max_packets:
            break
        ts = float(pkt.time)
        size = len(pkt)
        if IP in pkt:
            ipl = pkt[IP]
            src, dst = ipl.src, ipl.dst
        elif IPv6 in pkt:
            ipl = pkt[IPv6]
            src, dst = ipl.src, ipl.dst
        else:
            continue
        if TCP in pkt:
            proto, sport, dport = "tcp", int(pkt[TCP].sport), int(pkt[TCP].dport)
        elif UDP in pkt:
            proto, sport, dport = "udp", int(pkt[UDP].sport), int(pkt[UDP].dport)
        elif ICMP in pkt:
            proto, sport, dport = "icmp", 0, 0
        else:
            proto, sport, dport = "other", 0, 0
        key, direction = canon_key(src, sport, dst, dport, proto)
        f = flows.get(key)
        if f is None:
            f = {
                "src": key[0], "sport": key[1], "dst": key[2], "dport": key[3],
                "proto": key[4], "t_first": ts, "t_last": ts,
                "n_fwd": 0, "n_bwd": 0, "bytes_fwd": 0, "bytes_bwd": 0,
                "pkts": [], "attack": attack_name,
            }
            flows[key] = f
        f["t_last"] = ts if ts > f["t_last"] else f["t_last"]
        f["t_first"] = ts if ts < f["t_first"] else f["t_first"]
        if direction == 0:
            f["n_fwd"] += 1; f["bytes_fwd"] += size
        else:
            f["n_bwd"] += 1; f["bytes_bwd"] += size
        # packet back-pointer: (global id, ts, size, dir)
        f["pkts"].append((pkt_id, round(ts, 6), size, direction))
        pkt_id += 1
    reader.close()
    return flows, (pkt_id - base_pkt_id)


def flow_features(f):
    """Compact numeric feature vector for IDS training (flow-level)."""
    dur = max(1e-6, f["t_last"] - f["t_first"])
    n_tot = f["n_fwd"] + f["n_bwd"]
    b_tot = f["bytes_fwd"] + f["bytes_bwd"]
    return {
        "dur": dur,
        "n_fwd": f["n_fwd"], "n_bwd": f["n_bwd"], "n_tot": n_tot,
        "bytes_fwd": f["bytes_fwd"], "bytes_bwd": f["bytes_bwd"], "bytes_tot": b_tot,
        "pps": n_tot / dur, "bps": b_tot / dur,
        "mean_pkt": b_tot / max(1, n_tot),
        "ratio_fwd": f["n_fwd"] / max(1, n_tot),
        "sport": f["sport"], "dport": f["dport"],
        "proto_tcp": int(f["proto"] == "tcp"), "proto_udp": int(f["proto"] == "udp"),
        "proto_icmp": int(f["proto"] == "icmp"),
        # identifiers kept separately so masking probes can drop them
        "src": f["src"], "dst": f["dst"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap-dir", required=True, help="dir with .pcap files (recursive)")
    ap.add_argument("--out", required=True, help="output JSONL of flows")
    ap.add_argument("--max-packets", type=int, default=0, help="per-file cap (0=all)")
    ap.add_argument("--max-files", type=int, default=0)
    args = ap.parse_args()

    pcaps = []
    for root, _, files in os.walk(args.pcap_dir):
        for fn in files:
            if fn.lower().endswith(".pcap") or fn.lower().endswith(".pcapng"):
                pcaps.append(os.path.join(root, fn))
    # NOTE: files under a "Normal traffic" directory are benign regardless of the
    # sensor-named filename (Distance, Flame_Sensor, ...). Labeling is by directory.
    pcaps.sort()
    if args.max_files:
        pcaps = pcaps[:args.max_files]
    print(f"found {len(pcaps)} pcap files")

    base = 0
    n_flows = 0
    with open(args.out, "w") as out:
        for p in pcaps:
            if "normal traffic" in p.lower():
                attack = "Normal"          # benign, labeled by directory
            else:
                attack = os.path.splitext(os.path.basename(p))[0]
                attack = attack.replace("_attack", "").replace(" Attacks", "").replace(" Attack", "")
            try:
                flows, npkt = parse_pcap(p, attack, args.max_packets, base)
            except Exception as e:
                print(f"  ! {p}: {e}", file=sys.stderr); continue
            base += npkt
            for key, f in flows.items():
                rec = flow_features(f)
                rec["attack"] = f["attack"]
                rec["t_first"] = f["t_first"]; rec["t_last"] = f["t_last"]
                rec["pkts"] = f["pkts"]
                out.write(json.dumps(rec) + "\n")
                n_flows += 1
            print(f"  {os.path.basename(p):40s} {npkt:>8d} pkts -> {len(flows):>6d} flows [{attack}]")
    print(f"TOTAL: {n_flows} flows, {base} packets -> {args.out}")


if __name__ == "__main__":
    main()




