"""
Attack #2 — Near-duplicate flooding, for ground-truth benchmarking of
forensics/duplicates.py.

Takes a handful of source images and floods a new batch with lightly
perturbed copies (small brightness/contrast jitter + re-save), which is
exactly what "resubmit slightly-edited copies" looks like in pHash
terms: Hamming distance small enough to cluster, large enough that
naive exact-duplicate (file-hash) checking would miss it.
"""
import random
from pathlib import Path

from PIL import Image, ImageEnhance
from sqlalchemy.orm import Session

from .. import models_db as m
from ..provenance.hashing import sha256_file
from ..forensics.duplicates import compute_phash

OUTPUT_ROOT = Path("./attack_artifacts/duplicate_flood")


def _jitter_image(src_path: str, out_path: Path, rng: random.Random):
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.9, 1.1))
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.9, 1.1))
        img.save(out_path, quality=rng.randint(85, 95))


def create_duplicate_flood_batch(
    db: Session,
    source_batch_id: str,
    flood_fraction: float = 0.4,
    copies_per_source: int = 6,
    batch_label: str = "duplicate_flooded",
    seed: int | None = 42,
) -> dict:
    source = db.query(m.Batch).get(source_batch_id)
    if not source or not source.samples:
        raise ValueError("source batch not found or empty")

    rng = random.Random(seed)
    samples = list(source.samples)
    rng.shuffle(samples)
    n_source_for_flood = max(1, int(len(samples) * flood_fraction / copies_per_source))
    flood_sources = samples[:n_source_for_flood]
    untouched = samples[n_source_for_flood:]

    new_batch = m.Batch(
        dataset_id=source.dataset_id, contributor_id=source.contributor_id,
        label=batch_label,
    )
    db.add(new_batch)
    db.commit()
    db.refresh(new_batch)

    out_dir = OUTPUT_ROOT / new_batch.id
    out_dir.mkdir(parents=True, exist_ok=True)

    flooded_ids = []
    # untouched originals carried over unchanged (the "normal" portion of the batch)
    for s in untouched:
        new_sample = m.Sample(
            batch_id=new_batch.id, file_path=s.file_path, sha256=s.sha256,
            label_class=s.label_class, phash=s.phash,
        )
        db.add(new_sample)

    # flooded near-duplicates
    for s in flood_sources:
        ext = Path(s.file_path).suffix or ".jpg"
        for copy_idx in range(copies_per_source):
            out_path = out_dir / f"{Path(s.file_path).stem}_dup{copy_idx}{ext}"
            try:
                _jitter_image(s.file_path, out_path, rng)
            except Exception:
                continue
            sha = sha256_file(out_path)
            phash = compute_phash(str(out_path))
            new_sample = m.Sample(
                batch_id=new_batch.id, file_path=str(out_path), sha256=sha,
                label_class=s.label_class, phash=phash,
            )
            db.add(new_sample)
            flooded_ids.append(new_sample.id)

    db.commit()
    total = len(untouched) + len(flooded_ids)
    new_batch.sample_count = total
    db.commit()

    return {
        "flooded_batch_id": new_batch.id,
        "source_batch_id": source_batch_id,
        "total_samples": total,
        "flooded_count": len(flooded_ids),
        "ground_truth_flooded_fraction": round(len(flooded_ids) / total, 3) if total else 0,
    }
