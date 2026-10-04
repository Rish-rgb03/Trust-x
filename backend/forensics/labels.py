"""
Detector #2 — Label distribution anomaly (Section 15).

Compares a batch's class distribution against the dataset's global/
historical distribution using Jensen-Shannon divergence (bounded 0..1,
symmetric — nicer than raw KL for this). A batch that suddenly skews
hard toward one class relative to history is evidence, not proof.
"""
from collections import Counter
import numpy as np
from scipy.spatial.distance import jensenshannon
from sqlalchemy.orm import Session
from .. import models_db as m
from ..evidence.schema import EvidenceRecord, EvidenceType, SourceNodeType

JS_DIVERGENCE_THRESHOLD = 0.25   # above this, flag as anomalous
MIN_SAMPLES_FOR_SIGNAL = 10


def _class_distribution(samples) -> dict:
    counts = Counter(s.label_class for s in samples if s.label_class)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {cls: count / total for cls, count in counts.items()}


def _to_aligned_vectors(dist_a: dict, dist_b: dict):
    classes = sorted(set(dist_a) | set(dist_b))
    vec_a = np.array([dist_a.get(c, 0.0) for c in classes])
    vec_b = np.array([dist_b.get(c, 0.0) for c in classes])
    # avoid all-zero vectors breaking jensenshannon
    if vec_a.sum() == 0 or vec_b.sum() == 0:
        return None, None
    return vec_a, vec_b


def detect_label_distribution_anomaly(db: Session, batch_id: str):
    """
    Compares this batch's label distribution against every OTHER batch
    in the same dataset (treated as the historical/global baseline).
    """
    batch = db.query(m.Batch).filter(m.Batch.id == batch_id).first()
    if not batch or not batch.samples:
        return []

    batch_samples = [s for s in batch.samples if s.label_class]
    if len(batch_samples) < MIN_SAMPLES_FOR_SIGNAL:
        return []

    other_batches = (
        db.query(m.Batch)
        .filter(m.Batch.dataset_id == batch.dataset_id, m.Batch.id != batch_id)
        .all()
    )
    baseline_samples = [s for b in other_batches for s in b.samples if s.label_class]
    if len(baseline_samples) < MIN_SAMPLES_FOR_SIGNAL:
        return []  # no history yet to compare against

    batch_dist = _class_distribution(batch_samples)
    baseline_dist = _class_distribution(baseline_samples)

    vec_a, vec_b = _to_aligned_vectors(batch_dist, baseline_dist)
    if vec_a is None:
        return []

    # jensenshannon() returns the *distance* (sqrt of JS divergence), already in [0,1]
    js_distance = float(jensenshannon(vec_a, vec_b, base=2))
    if np.isnan(js_distance):
        return []

    if js_distance < JS_DIVERGENCE_THRESHOLD:
        return []

    severity = min(1.0, js_distance / 0.6)  # normalize so ~0.6 distance -> full severity
    return [EvidenceRecord(
        type=EvidenceType.LABEL_DISTRIBUTION_ANOMALY,
        source_node_type=SourceNodeType.batch,
        source_node_id=batch_id,
        severity=round(severity, 3),
        confidence=0.85,
        detector="label_detector",
        details={
            "js_distance": round(js_distance, 4),
            "batch_distribution": {k: round(v, 3) for k, v in batch_dist.items()},
            "baseline_distribution": {k: round(v, 3) for k, v in baseline_dist.items()},
            "batch_sample_count": len(batch_samples),
            "baseline_sample_count": len(baseline_samples),
        },
    )]
