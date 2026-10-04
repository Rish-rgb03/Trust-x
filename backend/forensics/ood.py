"""
Detector #3 — Out-of-distribution samples (Section 16).

v0.1 embedding: a cheap, dependency-light feature vector (color
histogram + grayscale histogram + edge-density proxy) computed with
PIL/numpy only — no GPU, no model download, runs instantly on CPU.
This is intentionally a PROXY embedding so the detector is runnable
from day one. Swap `compute_embedding()` for a CLIP or ResNet18
penultimate-layer embedding later (Tier 2 upgrade) — nothing else in
this file needs to change, since everything downstream just consumes
a fixed-length float vector.

OOD score = Euclidean distance from a sample's embedding to the
centroid of a trusted reference batch's embeddings, expressed in
units of the reference batch's own standard deviation (a cheap
z-score-style distance, not true Mahalanobis, but doesn't require
inverting a covariance matrix on small sample counts).
"""
import numpy as np
from PIL import Image
from sqlalchemy.orm import Session
from .. import models_db as m
from ..evidence.schema import EvidenceRecord, EvidenceType, SourceNodeType

EMBED_SIZE = 64          # resize target before feature extraction
COLOR_BINS = 16
GRAY_BINS = 32
OOD_Z_THRESHOLD = 3.0     # distance in std-devs from reference centroid
MIN_REFERENCE_SAMPLES = 8


def compute_embedding(image_path: str) -> np.ndarray:
    with Image.open(image_path) as img:
        img = img.convert("RGB").resize((EMBED_SIZE, EMBED_SIZE))
        arr = np.asarray(img).astype(np.float32)

    # color histogram per channel
    hist_feats = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=COLOR_BINS, range=(0, 255))
        hist_feats.append(hist / hist.sum())

    # grayscale histogram
    gray = arr.mean(axis=2)
    gray_hist, _ = np.histogram(gray, bins=GRAY_BINS, range=(0, 255))
    gray_hist = gray_hist / gray_hist.sum()

    # edge-density proxy via simple gradient magnitude (no scipy filter needed)
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    edge_density = np.array([
        np.abs(gx).mean() / 255.0,
        np.abs(gy).mean() / 255.0,
        gray.std() / 255.0,
    ])

    return np.concatenate([*hist_feats, gray_hist, edge_density]).astype(np.float32)


def _embedding_for_sample(sample: m.Sample) -> np.ndarray | None:
    if sample.embedding:
        return np.array(sample.embedding, dtype=np.float32)
    try:
        return compute_embedding(sample.file_path)
    except Exception:
        return None


def detect_ood_samples(db: Session, batch_id: str, reference_batch_id: str | None = None):
    """
    Flags samples in `batch_id` that sit far from the embedding centroid
    of a trusted reference batch. If no reference_batch_id is given, uses
    every OTHER batch in the same dataset as the reference pool.
    """
    batch = db.query(m.Batch).filter(m.Batch.id == batch_id).first()
    if not batch or not batch.samples:
        return []

    if reference_batch_id:
        ref_batches = [db.query(m.Batch).get(reference_batch_id)]
    else:
        ref_batches = (
            db.query(m.Batch)
            .filter(m.Batch.dataset_id == batch.dataset_id, m.Batch.id != batch_id)
            .all()
        )
    ref_samples = [s for b in ref_batches if b for s in b.samples]
    if len(ref_samples) < MIN_REFERENCE_SAMPLES:
        return []  # not enough reference signal yet

    ref_embeddings = []
    for s in ref_samples:
        emb = _embedding_for_sample(s)
        if emb is not None:
            ref_embeddings.append(emb)
    if len(ref_embeddings) < MIN_REFERENCE_SAMPLES:
        return []

    ref_matrix = np.stack(ref_embeddings)
    centroid = ref_matrix.mean(axis=0)
    # per-dim std, floor to avoid divide-by-zero on constant dims
    std = ref_matrix.std(axis=0)
    std = np.where(std < 1e-6, 1e-6, std)

    findings = []
    ood_sample_ids = []
    max_z = 0.0
    for s in batch.samples:
        emb = _embedding_for_sample(s)
        if emb is None:
            continue
        z = float(np.sqrt((((emb - centroid) / std) ** 2).mean()))
        max_z = max(max_z, z)
        if z >= OOD_Z_THRESHOLD:
            ood_sample_ids.append({"sample_id": s.id, "z_distance": round(z, 2)})

    if not ood_sample_ids:
        return []

    fraction = len(ood_sample_ids) / len(batch.samples)
    severity = min(1.0, (max_z / (OOD_Z_THRESHOLD * 2)))
    confidence = 0.7  # proxy embedding — lower confidence than a learned one

    findings.append(EvidenceRecord(
        type=EvidenceType.OOD_ANOMALY,
        source_node_type=SourceNodeType.batch,
        source_node_id=batch_id,
        severity=round(severity, 3),
        confidence=confidence,
        detector="ood_detector",
        details={
            "flagged_samples": ood_sample_ids[:50],
            "flagged_fraction": round(fraction, 3),
            "max_z_distance": round(max_z, 2),
            "reference_pool_size": len(ref_embeddings),
            "embedding_type": "proxy_histogram_v0.1",
        },
    ))
    return findings
