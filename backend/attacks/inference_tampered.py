"""
attacks/inference_tampered.py  (Person 3)

Benchmark scenario generator for inference-stage attacks. Produces a stream of 'delivered'
(passport, input, output) records, applies ONE attack, runs PassportVerifier, reports detection.
  python -m attacks.inference_tampered
"""
from __future__ import annotations
import copy, random, json
from ..provenance.passport import PassportIssuer, PassportVerifier, sha256_bytes, passport_hash, hash_obj

KEY = b"trustx-demo-key-change-me"
GOOD = dict(model_hash=sha256_bytes(b"model_v1"), preproc_hash=sha256_bytes(b"pp1"), config_hash=sha256_bytes(b"cfg1"))
ATTACKS = ["none", "output_tamper", "input_tamper", "replay", "model_swap",
           "config_tamper", "drop_records", "forged_signature"]

def _stream(n, rng, model_hash=None, config_hash=None, start=None):
    iss = PassportIssuer(KEY, "D7", model_hash or GOOD["model_hash"], GOOD["preproc_hash"], config_hash or GOOD["config_hash"])
    recs, t0 = [], 1_700_000_000.0
    for i in range(n):
        img = bytes(rng.getrandbits(8) for _ in range(64))
        out = {"detections": [{"cls": rng.choice(["vehicle", "person", "building"]),
                               "conf": round(rng.uniform(.5, .99), 2),
                               "box": [rng.randint(0, 600) for _ in range(4)]}]}
        recs.append({"p": iss.issue(img, out, now=t0 + i), "img": img, "out": out, "t": t0 + i})
    return recs, iss

def build(attack: str, n: int = 50, seed: int = 0, at: int = 25):
    rng = random.Random(seed)
    recs, _ = _stream(n, rng)
    if attack == "none":
        pass
    elif attack == "output_tamper":          # change delivered output, passport untouched
        recs[at]["out"]["detections"][0].update(cls="building", conf=0.01)
    elif attack == "input_tamper":           # image swapped after passport was issued
        recs[at]["img"] = recs[at]["img"][::-1] + b"x"
    elif attack == "replay":                 # resend an old valid record later in the stream
        recs.insert(at + 5, copy.deepcopy(recs[at - 10]))
    elif attack == "model_swap":             # host swaps model; passports are genuinely signed
        recs = _resign_tail(recs, at, model_hash=sha256_bytes(b"EVIL_model"))
    elif attack == "config_tamper":          # e.g. confidence threshold changed
        recs = _resign_tail(recs, at, config_hash=sha256_bytes(b"cfg_conf_thresh_0.99"))
    elif attack == "drop_records":           # suppress 4 predictions
        del recs[at:at + 4]
    elif attack == "forged_signature":       # attacker edits output AND rewrites output_hash, no key
        recs[at]["out"]["detections"][0].update(cls="building", conf=0.01)
        recs[at]["p"].output_hash = hash_obj(recs[at]["out"])
    else:
        raise ValueError(attack)
    return recs

def _resign_tail(recs, at, model_hash=None, config_hash=None):
    """Compromised host re-issues the remaining passports with its own (real-key) signatures."""
    iss = PassportIssuer(KEY, "D7", model_hash or GOOD["model_hash"], GOOD["preproc_hash"], config_hash or GOOD["config_hash"])
    iss.seq, iss.prev = recs[at - 1]["p"].sequence, passport_hash(recs[at - 1]["p"])
    out = recs[:at]
    for r in recs[at:]:
        out.append({**r, "p": iss.issue(r["img"], r["out"], now=r["t"])})
    return out

def run(attack: str, n=50, seed=0, at=25) -> dict:
    recs = build(attack, n, seed, at)
    ver = PassportVerifier(KEY, GOOD["model_hash"], GOOD["preproc_hash"], GOOD["config_hash"])
    evidence, first, fp = [], None, 0
    for idx, r in enumerate(recs):
        ev = ver.verify(r["p"], r["img"], r["out"], now=r["t"])
        if ev:
            if first is None: first = idx
            evidence += ev
    fp = 1 if (first is not None and first < at) else 0   # flagged before the attack point
    if attack == "none": fp = len(evidence)
    if not evidence: evidence = ver.positive_evidence("D7")
    return {"attack": attack, "detected": bool(first is not None), "first_flagged_index": first,
            "types": sorted({e["type"] for e in evidence}), "false_positives_before_attack": fp,
            "evidence": evidence}

def evaluate_all(n=50, seeds=range(5)):
    rows = []
    for a in ATTACKS:
        res = [run(a, n, s) for s in seeds]
        rows.append({"attack": a, "detection_rate": sum(r["detected"] for r in res) / len(res),
                     "false_positives": sum(r["false_positives_before_attack"] for r in res),
                     "evidence_types": sorted({t for r in res for t in r["types"]})})
    return rows

if __name__ == "__main__":
    print(f"{'attack':18} {'det.rate':>8} {'FP':>3}  evidence types")
    for r in evaluate_all():
        print(f"{r['attack']:18} {r['detection_rate']:8.0%} {r['false_positives']:3d}  {', '.join(r['evidence_types'])}")
