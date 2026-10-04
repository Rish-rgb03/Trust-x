"""
Attack #1 — Label flipping, for ground-truth benchmarking of
forensics/labels.py.

Takes an existing clean batch, copies a fraction of its samples into a
NEW batch but with their label_class reassigned to a target class. This
is what "Contributor B suddenly contributes 82% person" (Section 15)
looks like as reproducible ground truth: you know exactly which samples
were flipped, so you can measure whether the detector actually fires on
them.
"""
import random
from sqlalchemy.orm import Session
from .. import models_db as m


def create_label_poisoned_batch(
    db: Session,
    source_batch_id: str,
    target_class: str,
    flip_fraction: float = 0.6,
    batch_label: str = "label_poisoned",
    seed: int | None = 42,
) -> dict:
    source = db.query(m.Batch).get(source_batch_id)
    if not source or not source.samples:
        raise ValueError("source batch not found or empty")

    rng = random.Random(seed)
    samples = list(source.samples)
    rng.shuffle(samples)
    n_flip = int(len(samples) * flip_fraction)

    new_batch = m.Batch(
        dataset_id=source.dataset_id, contributor_id=source.contributor_id,
        label=batch_label,
    )
    db.add(new_batch)
    db.commit()
    db.refresh(new_batch)

    flipped_ids = []
    for i, s in enumerate(samples):
        new_class = target_class if i < n_flip else s.label_class
        new_sample = m.Sample(
            batch_id=new_batch.id, file_path=s.file_path, sha256=s.sha256,
            label_class=new_class, phash=s.phash,
        )
        db.add(new_sample)
        if i < n_flip:
            flipped_ids.append(new_sample.id)

    new_batch.sample_count = len(samples)
    db.commit()

    return {
        "poisoned_batch_id": new_batch.id,
        "source_batch_id": source_batch_id,
        "total_samples": len(samples),
        "flipped_to": target_class,
        "flipped_count": n_flip,
        "ground_truth_flipped_fraction": round(n_flip / len(samples), 3),
    }
