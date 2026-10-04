"""
provenance/passport.py  (Person 3)

Prediction Passport = tamper-evident receipt for ONE prediction.
Fields: input/output/model/preprocessing/config hashes, timestamp, sequence, nonce,
        prev_hash (hash chain over passports), HMAC-SHA256 signature.

Signature is HMAC (symmetric, offline, no PKI). Threat model: attacker who tampers with
data/output in transit or replays records WITHOUT the signing key. Key compromise is out of
scope (upgrade path: Ed25519 + HSM). A host-level attacker who signs with the real key but
runs a different model/config is still caught by comparing the signed hashes to the
authorised ones (MODEL_ARTIFACT_MISMATCH / CONFIG_MISMATCH).
"""
from __future__ import annotations
import hashlib, hmac, json, secrets, time
from dataclasses import dataclass, asdict, field
from typing import Any

def sha256_bytes(b: bytes) -> str: return hashlib.sha256(b).hexdigest()
def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
def hash_obj(obj: Any) -> str: return sha256_bytes(canonical(obj))

SIGNED_FIELDS = ("deployment_id", "inference_id", "input_hash", "output_hash", "model_hash",
                 "preproc_hash", "config_hash", "timestamp", "sequence", "nonce", "prev_hash")

@dataclass
class Passport:
    deployment_id: str; inference_id: str
    input_hash: str; output_hash: str
    model_hash: str; preproc_hash: str; config_hash: str
    timestamp: float; sequence: int; nonce: str; prev_hash: str
    signature: str = ""
    def to_dict(self): return asdict(self)
    @classmethod
    def from_dict(cls, d): return cls(**{k: d[k] for k in cls.__dataclass_fields__})

def _sign(key: bytes, p: Passport) -> str:
    body = {k: getattr(p, k) for k in SIGNED_FIELDS}
    return hmac.new(key, canonical(body), hashlib.sha256).hexdigest()

def passport_hash(p: Passport) -> str: return hash_obj(p.to_dict())


class PassportIssuer:
    """Runs next to the model. Call issue() once per prediction."""
    def __init__(self, key: bytes, deployment_id: str, model_hash: str, preproc_hash: str, config_hash: str):
        self.key, self.dep = key, deployment_id
        self.model_hash, self.preproc_hash, self.config_hash = model_hash, preproc_hash, config_hash
        self.seq, self.prev = 0, "GENESIS"

    def issue(self, input_bytes: bytes, output: Any, now: float | None = None) -> Passport:
        self.seq += 1
        p = Passport(self.dep, f"I{self.seq:04d}", sha256_bytes(input_bytes), hash_obj(output),
                     self.model_hash, self.preproc_hash, self.config_hash,
                     now if now is not None else time.time(), self.seq, secrets.token_hex(12), self.prev)
        p.signature = _sign(self.key, p)
        self.prev = passport_hash(p)
        return p


class PassportVerifier:
    """Runs in the TRUST-X analyst plane. Stateful: tracks nonces, sequence, hash chain."""
    DETECTOR = "passport_verifier"

    def __init__(self, key: bytes, expected_model_hash: str, expected_preproc_hash: str,
                 expected_config_hash: str, max_skew_s: float = 300.0):
        self.key, self.exp = key, dict(model_hash=expected_model_hash, preproc_hash=expected_preproc_hash,
                                       config_hash=expected_config_hash)
        self.max_skew = max_skew_s
        self.seen: set[str] = set()
        self.last_seq, self.last_hash = 0, "GENESIS"
        self.n_ok, self.n_total, self._n = 0, 0, 0
        self.reanchor = False      # set after an invalid passport so one forgery doesn't cascade into fake gaps

    def _ev(self, typ, p: Passport, sev, conf, detail):
        self._n += 1
        return {"id": f"E-INF-{self._n:04d}", "type": typ, "source": p.inference_id, "stage": "inference",
                "severity": sev, "confidence": conf, "detector": self.DETECTOR,
                "meta": {"deployment": p.deployment_id, "sequence": p.sequence, "detail": detail}}

    def verify(self, p: Passport, input_bytes: bytes | None = None, output: Any = None,
               now: float | None = None) -> list[dict]:
        self.n_total += 1
        ev: list[dict] = []
        now = now if now is not None else time.time()

        if not hmac.compare_digest(p.signature, _sign(self.key, p)):
            ev.append(self._ev("SIGNATURE_INVALID", p, 1.0, 1.0, "passport fields do not match signature"))
            self.reanchor = True
            return ev                                   # nothing else in the passport is trustworthy

        if output is not None and hash_obj(output) != p.output_hash:
            ev.append(self._ev("OUTPUT_HASH_MISMATCH", p, 1.0, 1.0, "delivered output != signed output hash"))
        if input_bytes is not None and sha256_bytes(input_bytes) != p.input_hash:
            ev.append(self._ev("INPUT_HASH_MISMATCH", p, 1.0, 1.0, "processed input != signed input hash"))
        if p.model_hash != self.exp["model_hash"]:
            ev.append(self._ev("MODEL_ARTIFACT_MISMATCH", p, 1.0, 1.0,
                               f"running model {p.model_hash[:10]} != authorised {self.exp['model_hash'][:10]}"))
        if p.config_hash != self.exp["config_hash"] or p.preproc_hash != self.exp["preproc_hash"]:
            ev.append(self._ev("CONFIG_MISMATCH", p, 0.9, 1.0, "config/preprocessing hash differs from authorised"))
        if abs(now - p.timestamp) > self.max_skew:
            ev.append(self._ev("TIMESTAMP_SKEW", p, 0.5, 0.8, f"|now-timestamp| > {self.max_skew:.0f}s"))

        if p.nonce in self.seen:                        # replay: same nonce seen before
            ev.append(self._ev("REPLAY_DETECTED", p, 1.0, 1.0, f"nonce reused (sequence {p.sequence})"))
            return ev
        self.seen.add(p.nonce)

        slack = 1 if self.reanchor else 0
        if p.sequence > self.last_seq + 1 + slack:
            ev.append(self._ev("SEQUENCE_GAP", p, min(1.0, 0.4 + 0.2 * (p.sequence - self.last_seq - 1)), 0.9,
                               f"expected {self.last_seq + 1}, got {p.sequence}"))
        if p.prev_hash != self.last_hash and not self.reanchor:
            ev.append(self._ev("CHAIN_BREAK", p, 0.9, 0.9, "prev_hash does not link to last verified passport"))
        if p.sequence > self.last_seq:
            self.last_seq, self.last_hash = p.sequence, passport_hash(p)
        self.reanchor = False
        if not ev: self.n_ok += 1
        return ev

    def positive_evidence(self, deployment_id: str) -> list[dict]:
        """One aggregated PASSPORT_VERIFIED evidence if a clean stream was observed (contradicts tampering)."""
        if self.n_total == 0 or self.n_ok != self.n_total: return []
        return [{"id": "E-INF-OK", "type": "PASSPORT_VERIFIED", "source": deployment_id, "stage": "inference",
                 "severity": 1.0, "confidence": min(1.0, self.n_total / 20), "detector": self.DETECTOR,
                 "meta": {"verified_predictions": self.n_total}}]
