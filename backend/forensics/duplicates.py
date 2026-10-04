"""Near-duplicate detector using perceptual hashing (pHash).

ImageHash is preferred when installed. A small SciPy/PIL pHash fallback keeps
TRUST-X importable and usable in constrained/offline environments as well.
"""
from __future__ import annotations
import numpy as np
from PIL import Image
from sqlalchemy.orm import Session
from .. import models_db as m
from ..evidence.schema import EvidenceRecord, EvidenceType, SourceNodeType

try:
    import imagehash  # type: ignore
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent
    imagehash = None

DUPLICATE_CLUSTER_THRESHOLD = 0.15
HASH_HAMMING_DISTANCE = 5


def _fallback_phash(image_path: str) -> str:
    """64-bit pHash compatible with our own hex parser (PIL + SciPy only)."""
    from scipy.fftpack import dct
    with Image.open(image_path) as img:
        gray = np.asarray(img.convert("L").resize((32, 32)), dtype=np.float32)
    coeff = dct(dct(gray, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    low = coeff.flatten()[1:]
    med = float(np.median(low))
    bits = (low > med).astype(np.uint8)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def compute_phash(image_path: str) -> str:
    with Image.open(image_path) as img:
        if imagehash is not None:
            return str(imagehash.phash(img))
    return _fallback_phash(image_path)


def _hash_distance(a: str, b: str) -> int:
    if imagehash is not None:
        return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)
    return (int(a, 16) ^ int(b, 16)).bit_count()


def cluster_by_phash(samples):
    clusters = []
    for sample in samples:
        if not sample.phash:
            continue
        placed = False
        for cluster in clusters:
            rep = cluster[0].phash
            if _hash_distance(sample.phash, rep) <= HASH_HAMMING_DISTANCE:
                cluster.append(sample)
                placed = True
                break
        if not placed:
            clusters.append([sample])
    return clusters


def detect_near_duplicate_flooding(db: Session, batch_id: str):
    batch = db.query(m.Batch).filter(m.Batch.id == batch_id).first()
    if not batch or not batch.samples:
        return []
    samples = [s for s in batch.samples if s.phash]
    if len(samples) < 5:
        return []
    clusters = cluster_by_phash(samples)
    total = len(samples)
    findings = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        fraction = len(cluster) / total
        if fraction >= DUPLICATE_CLUSTER_THRESHOLD:
            severity = min(1.0, fraction * 1.5)
            findings.append(EvidenceRecord(
                type=EvidenceType.NEAR_DUPLICATE_CLUSTER,
                source_node_type=SourceNodeType.batch,
                source_node_id=batch_id,
                severity=round(severity, 3),
                confidence=0.9,
                detector="duplicate_detector",
                details={
                    "cluster_size": len(cluster), "batch_total": total,
                    "fraction": round(fraction, 3),
                    "sample_ids": [s.id for s in cluster[:50]],
                },
            ))
    return findings
