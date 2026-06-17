"""Attack-name to forensic kill-chain stage mapping.

Stages are dataset-agnostic and operational (see the project data contract):
  Recon, Exploit, C2, Lateral, ExfilImpact, unknown-stage (never force-labeled).

The mapping is included in the release. Each entry carries a confidence flag so
the downstream ground-truth builder can report uncertainty rather than hide it.
"""

# canonical stage order used for ordering metrics
STAGE_ORDER = ["Recon", "Exploit", "C2", "Lateral", "ExfilImpact"]
STAGE_RANK = {s: i for i, s in enumerate(STAGE_ORDER)}
UNKNOWN = "unknown-stage"

# Edge-IIoTset attack classes -> (stage, confidence)
# confidence: exact (clear semantic match) | derived (reasoned) | ambiguous
EDGE_IIOT_MAP = {
    # Recon / information gathering
    "Port_Scanning": ("Recon", "exact"),
    "Port Scanning": ("Recon", "exact"),
    "OS_Fingerprinting": ("Recon", "exact"),
    "OS Fingerprinting": ("Recon", "exact"),
    "Vulnerability_scanner": ("Recon", "exact"),
    "Vulnerability scanner": ("Recon", "exact"),
    # Exploit / gaining access
    "Password": ("Exploit", "exact"),
    "Password attacks": ("Exploit", "exact"),
    "Backdoor": ("Exploit", "derived"),       # plants a foothold -> access stage
    "SQL_injection": ("Exploit", "exact"),
    "SQL injection": ("Exploit", "exact"),
    "Uploading": ("Exploit", "derived"),      # malicious file upload -> access
    "XSS": ("Exploit", "exact"),
    "Ransomware": ("ExfilImpact", "exact"),   # impact stage
    # C2 / control
    "MITM": ("Lateral", "derived"),           # ARP/DNS spoof to pivot/intercept
    "MITM (ARP spoofing + DNS)": ("Lateral", "derived"),
    # Exfil / Impact (DoS/DDoS = availability impact)
    "DDoS_HTTP": ("ExfilImpact", "exact"),
    "DDoS HTTP Flood": ("ExfilImpact", "exact"),
    "DDoS_ICMP": ("ExfilImpact", "exact"),
    "DDoS ICMP Flood": ("ExfilImpact", "exact"),
    "DDoS_TCP": ("ExfilImpact", "exact"),
    "DDoS TCP SYN Flood": ("ExfilImpact", "exact"),
    "DDoS_UDP": ("ExfilImpact", "exact"),
    "DDoS UDP Flood": ("ExfilImpact", "exact"),
    "Fingerprinting": ("Recon", "derived"),
    "Normal": (None, "exact"),                # benign
    "Benign": (None, "exact"),
}

# CIC-IoT-2023 (33 attacks) -> stage
CICIOT_MAP = {
    "BenignTraffic": (None, "exact"),
    "Recon-PingSweep": ("Recon", "exact"),
    "Recon-OSScan": ("Recon", "exact"),
    "Recon-PortScan": ("Recon", "exact"),
    "Recon-HostDiscovery": ("Recon", "exact"),
    "VulnerabilityScan": ("Recon", "exact"),
    "DictionaryBruteForce": ("Exploit", "exact"),
    "BrowserHijacking": ("Exploit", "derived"),
    "CommandInjection": ("Exploit", "exact"),
    "SqlInjection": ("Exploit", "exact"),
    "XSS": ("Exploit", "exact"),
    "Backdoor_Malware": ("Exploit", "derived"),
    "Uploading_Attack": ("Exploit", "derived"),
    "DDoS-RSTFINFlood": ("ExfilImpact", "exact"),
    "DDoS-PSHACK_Flood": ("ExfilImpact", "exact"),
    "DDoS-SYN_Flood": ("ExfilImpact", "exact"),
    "DDoS-UDP_Flood": ("ExfilImpact", "exact"),
    "DDoS-TCP_Flood": ("ExfilImpact", "exact"),
    "DDoS-ICMP_Flood": ("ExfilImpact", "exact"),
    "DDoS-SynonymousIP_Flood": ("ExfilImpact", "exact"),
    "DDoS-ACK_Fragmentation": ("ExfilImpact", "exact"),
    "DDoS-UDP_Fragmentation": ("ExfilImpact", "exact"),
    "DDoS-ICMP_Fragmentation": ("ExfilImpact", "exact"),
    "DDoS-HTTP_Flood": ("ExfilImpact", "exact"),
    "DDoS-SlowLoris": ("ExfilImpact", "exact"),
    "DoS-UDP_Flood": ("ExfilImpact", "exact"),
    "DoS-SYN_Flood": ("ExfilImpact", "exact"),
    "DoS-TCP_Flood": ("ExfilImpact", "exact"),
    "DoS-HTTP_Flood": ("ExfilImpact", "exact"),
    "Mirai-greeth_flood": ("ExfilImpact", "derived"),
    "Mirai-greip_flood": ("ExfilImpact", "derived"),
    "Mirai-udpplain": ("ExfilImpact", "derived"),
    "DNS_Spoofing": ("Lateral", "derived"),
    "MITM-ArpSpoofing": ("Lateral", "derived"),
}


def _norm(s):
    """Robust normalization so 'Port Scanning attack', 'Port_Scanning', 'PortScanning'
    all collapse to one key. Strips attack/attacks suffix, separators, case."""
    s = s.lower().strip()
    for suf in ("attacks", "attack"):
        if s.endswith(suf):
            s = s[: -len(suf)].strip()
    for ch in ("_", "-", "(", ")", "+", ".", ","):
        s = s.replace(ch, " ")
    return " ".join(s.split())


# precompute normalized lookup for both tables
_NORM_EDGE = {_norm(k): v for k, v in EDGE_IIOT_MAP.items()}
_NORM_CIC = {_norm(k): v for k, v in CICIOT_MAP.items()}


def map_attack(name, dataset="edge-iiot"):
    """Return (stage, confidence). Unknown attacks -> (UNKNOWN, 'ambiguous')."""
    table = EDGE_IIOT_MAP if dataset.startswith("edge") else CICIOT_MAP
    if name in table:
        return table[name]
    norm_table = _NORM_EDGE if dataset.startswith("edge") else _NORM_CIC
    nk = _norm(name)
    if nk in norm_table:
        return norm_table[nk]
    # prefix/substring fallback (e.g. 'ddos http flood' vs 'ddos http')
    for k, v in norm_table.items():
        if nk.startswith(k) or k.startswith(nk):
            return v
    return (UNKNOWN, "ambiguous")


if __name__ == "__main__":
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 else "edge-iiot"
    tbl = EDGE_IIOT_MAP if ds.startswith("edge") else CICIOT_MAP
    n = sum(1 for v in tbl.values() if v[0] not in (None, UNKNOWN))
    print(f"{ds}: {len(tbl)} classes mapped, {n} malicious stage assignments")
    for k, v in tbl.items():
        print(f"  {k:32s} -> {v[0]} ({v[1]})")



