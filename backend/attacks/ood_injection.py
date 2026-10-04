"""
Attack #3 — OOD injection, for ground-truth benchmarking of
forensics/ood.py.

Generates synthetic images that are clearly outside a normal photo
distribution (solid colors, pure random noise, extreme gradients) and
injects them into a batch at a known fraction. This gives you ground
truth for "did the OOD detector actually flag the injected samples,"
without needing a second real dataset lying around.
"""
import random
from pathlib import Path

import numpy as np
from PIL import Image
from sqlalchemy.orm import Session

from .. import models_db as m
from ..provenance.hashing import sha256_file
from ..forensics.duplicates import compute_phash

OUTPUT_ROOT = Path("./attack_artifacts/ood_injection")
IMG_SIZE = 128


def _random_noise_image(rng: random.Random) -> Image.Image:
    arr = np.random.RandomState(rng.randint(0, 2**31 - 1)).randint(
        0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8
    )
    return Image.fromarray(arr, mode="RGB")


def _solid_color_image(rng: random.Random) -> Image.Image:
    color = tuple(rng.randint(0, 255) for _ in range(3))
    return Image.new("RGB", (IMG_SIZE, IMG_SIZE), color)


def _gradient_image(rng: random.Random) -> Image.Image:
    arr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    direction = rng.choice(["h", "v"])
    ramp = np.linspace(0, 255, IMG_SIZE, dtype=np.uint8)
    if direction == "h":
        arr[:, :, :] = ramp[np.newaxis, :, np.newaxis]
    else:
        arr[:, :, :] = ramp[:, np.newaxis, np.newaxis]
    return Image.fromarray(arr, mode="RGB")


GENERATORS = [_random_noise_image, _solid_color_image, _gradient_image]


def create_ood_injected_batch(
    db: Session,
    source_batch_id: str,
    injection_fraction: float = 0.15,
    batch_label: str = "ood_injected",
    seed: int | None = 42,
) -> dict:
    source = db.query(m.Batch).get(source_batch_id)
    if not source or not source.samples:
        raise ValueError("source batch not found or empty")

    rng = random.Random(seed)
    samples = list(source.samples)
    n_inject = max(1, int(len(samples) * injection_fraction))

    new_batch = m.Batch(
        dataset_id=source.dataset_id, contributor_id=source.contributor_id,
        label=batch_label,
    )
    db.add(new_batch)
    db.commit()
    db.refresh(new_batch)

    out_dir = OUTPUT_ROOT / new_batch.id
    out_dir.mkdir(parents=True, exist_ok=True)

    # carry over the normal samples unchanged
    for s in samples:
        new_sample = m.Sample(
            batch_id=new_batch.id, file_path=s.file_path, sha256=s.sha256,
            label_class=s.label_class, phash=s.phash,
        )
        db.add(new_sample)

    injected_ids = []
    for i in range(n_inject):
        gen = rng.choice(GENERATORS)
        img = gen(rng)
        out_path = out_dir / f"ood_{i}.png"
        img.save(out_path)
        sha = sha256_file(out_path)
        try:
            phash = compute_phash(str(out_path))
        except Exception:
            phash = None
        new_sample = m.Sample(
            batch_id=new_batch.id, file_path=str(out_path), sha256=sha,
            label_class="__injected_ood__", phash=phash,
        )
        db.add(new_sample)
        db.flush()
        injected_ids.append(new_sample.id)

    total = len(samples) + n_inject
    new_batch.sample_count = total
    db.commit()

    return {
        "ood_batch_id": new_batch.id,
        "source_batch_id": source_batch_id,
        "total_samples": total,
        "injected_count": n_inject,
        "injected_sample_ids": injected_ids,
        "ground_truth_injected_fraction": round(n_inject / total, 3),
    }
