"""
Bulk dataset ingestion.

Expected directory layout (the common "folder-per-class" convention —
also what you get if you export a YOLO-classification-style dataset):

    my_dataset/
        person/
            img001.jpg
            img002.jpg
        vehicle/
            img045.jpg
        building/
            img099.jpg

Each subfolder name becomes `label_class`. Every image gets hashed
(SHA-256) and perceptually hashed (pHash) on ingest, so the duplicate
detector has data to work with immediately — no separate pass needed.

Usage (direct DB, no HTTP — fast for bulk loads):

    from ..database import SessionLocal, init_db
    from ..ingestion.dataset import ingest_directory

    init_db()
    db = SessionLocal()
    result = ingest_directory(
        db, dataset_name="SatelliteDS", directory="./my_dataset",
        contributor_name="ContribB", batch_label="clean",
    )
    print(result)

Or from the command line:

    python -m backend.ingestion.dataset --dir ./my_dataset --dataset-name SatelliteDS --contributor ContribB
"""
import argparse
from pathlib import Path

from sqlalchemy.orm import Session

from .. import models_db as m
from ..provenance.hashing import sha256_file
from ..forensics.duplicates import compute_phash

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _get_or_create_contributor(db: Session, name: str | None) -> str | None:
    if not name:
        return None
    existing = db.query(m.Contributor).filter(m.Contributor.name == name).first()
    if existing:
        return existing.id
    c = m.Contributor(name=name)
    db.add(c)
    db.commit()
    db.refresh(c)
    return c.id


def _get_or_create_dataset(db: Session, name: str, fmt: str | None) -> str:
    existing = db.query(m.Dataset).filter(m.Dataset.name == name).first()
    if existing:
        return existing.id
    ds = m.Dataset(name=name, format=fmt)
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return ds.id


def ingest_directory(
    db: Session,
    dataset_name: str,
    directory: str,
    contributor_name: str | None = None,
    batch_label: str | None = None,
    dataset_format: str | None = "folder-per-class",
) -> dict:
    """
    Walks `directory`, treats each immediate subfolder as a class, creates
    one Batch for this ingest call, and inserts one Sample per image with
    sha256 + phash already computed. Returns a summary dict.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} is not a directory")

    dataset_id = _get_or_create_dataset(db, dataset_name, dataset_format)
    contributor_id = _get_or_create_contributor(db, contributor_name)

    batch = m.Batch(dataset_id=dataset_id, contributor_id=contributor_id, label=batch_label)
    db.add(batch)
    db.commit()
    db.refresh(batch)

    count = 0
    skipped = 0
    per_class_counts: dict[str, int] = {}

    for class_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
        label_class = class_dir.name
        for img_path in sorted(class_dir.rglob("*")):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                sha = sha256_file(img_path)
                phash = compute_phash(str(img_path))
            except Exception:
                skipped += 1
                continue

            sample = m.Sample(
                batch_id=batch.id, file_path=str(img_path), sha256=sha,
                label_class=label_class, phash=phash,
            )
            db.add(sample)
            count += 1
            per_class_counts[label_class] = per_class_counts.get(label_class, 0) + 1

    batch.sample_count = count
    db.commit()

    return {
        "dataset_id": dataset_id,
        "batch_id": batch.id,
        "contributor_id": contributor_id,
        "samples_ingested": count,
        "samples_skipped": skipped,
        "per_class_counts": per_class_counts,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk-ingest a folder-per-class image dataset into TRUST-X")
    parser.add_argument("--dir", required=True, help="Path to the dataset directory")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--contributor", default=None)
    parser.add_argument("--batch-label", default=None)
    args = parser.parse_args()

    from ..database import SessionLocal, init_db
    init_db()
    session = SessionLocal()
    try:
        result = ingest_directory(
            session, dataset_name=args.dataset_name, directory=args.dir,
            contributor_name=args.contributor, batch_label=args.batch_label,
        )
        print(result)
    finally:
        session.close()
