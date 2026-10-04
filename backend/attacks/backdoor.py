"""
Controlled backdoor benchmark generator (Person 2).

This is intentionally a small, dependency-light attack laboratory helper.
It creates image/label pairs with a visible trigger patch so the team can
build a known-ground-truth scenario.

It does NOT train a model. Training a real backdoored model depends on the
CV model/framework selected for the benchmark. The generated dataset is the
ground-truth input for that training step.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw
from sqlalchemy.orm import Session

from .. import models_db as m
from ..forensics.triggers import corner_patch
from ..provenance.hashing import sha256_file
from ..forensics.duplicates import compute_phash


OUTPUT_ROOT = Path("./attack_artifacts/backdoor")


def create_triggered_dataset(
    db: Session,
    source_batch_id: str,
    *,
    trigger_fraction: float = 0.3,
    target_class: Optional[str] = None,
    batch_label: str = "backdoor_triggered",
    seed: int = 42,
) -> dict:
    """Create a new batch where a known fraction receives a trigger.

    Ground truth is recorded in the returned dictionary:
    `triggered_sample_ids` identifies exactly which samples were changed.
    The caller can then train/evaluate a backdoor model using these samples.
    """
    source = db.query(m.Batch).filter(m.Batch.id == source_batch_id).first()
    if not source or not source.samples:
        raise ValueError("source batch not found or empty")

    rng = random.Random(seed)
    source_samples = list(source.samples)
    rng.shuffle(source_samples)
    n_trigger = max(1, int(len(source_samples) * trigger_fraction))

    new_batch = m.Batch(
        dataset_id=source.dataset_id,
        contributor_id=source.contributor_id,
        label=batch_label,
    )
    db.add(new_batch)
    db.commit()
    db.refresh(new_batch)

    out_dir = OUTPUT_ROOT / new_batch.id
    out_dir.mkdir(parents=True, exist_ok=True)

    triggered_ids = []
    for index, sample in enumerate(source_samples):
        should_trigger = index < n_trigger

        if not should_trigger:
            new_sample = m.Sample(
                batch_id=new_batch.id,
                file_path=sample.file_path,
                sha256=sample.sha256,
                label_class=sample.label_class,
                phash=sample.phash,
            )
        else:
            with Image.open(sample.file_path) as image:
                triggered = corner_patch(image, value=(255, 0, 0))
                out_path = out_dir / f"triggered_{index}.png"
                triggered.save(out_path)

            label = target_class if target_class else sample.label_class
            new_sample = m.Sample(
                batch_id=new_batch.id,
                file_path=str(out_path),
                sha256=sha256_file(out_path),
                label_class=label,
                phash=compute_phash(str(out_path)),
            )
            db.add(new_sample)
            db.flush()
            triggered_ids.append(new_sample.id)
            continue

        db.add(new_sample)

    new_batch.sample_count = len(source_samples)
    db.commit()

    return {
        "backdoor_batch_id": new_batch.id,
        "source_batch_id": source_batch_id,
        "total_samples": len(source_samples),
        "triggered_count": n_trigger,
        "ground_truth_trigger_fraction": round(n_trigger / len(source_samples), 3),
        "triggered_sample_ids": triggered_ids,
        "target_class": target_class,
    }
