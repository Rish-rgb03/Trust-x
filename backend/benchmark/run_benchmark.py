"""
Self-contained benchmark for Person 1's detectors.

Since you don't have a real satellite-imagery corpus yet, this script
GENERATES a small synthetic "clean" dataset (simple colored-shape images
across 3 classes) so the whole pipeline — ingest -> attack -> detect ->
score — is runnable and testable today, with zero external data
dependency. Swap in a real dataset later via ingestion/dataset.py; the
detector code doesn't change.

Run:
    cd backend
    python -m benchmark.run_benchmark

What it does:
    1. Generates ~90 synthetic clean images across 3 classes, ingests them
    2. Runs the duplicate detector on the clean batch -> expect 0 findings
    3. Runs attacks/duplicate_flood.py -> runs duplicate detector -> expect a hit
    4. Runs attacks/label_flip.py -> runs label detector -> expect a hit
    5. Runs attacks/ood_injection.py -> runs OOD detector -> expect a hit
    6. Prints a pass/fail summary per scenario
"""
import shutil
import random
from pathlib import Path

from PIL import Image, ImageDraw


SYNTH_DIR = Path("./benchmark_synthetic_data")
CLASSES = {
    "person": (220, 80, 80),
    "vehicle": (80, 140, 220),
    "building": (120, 180, 100),
}
IMAGES_PER_CLASS = 30


def generate_synthetic_clean_dataset(rng: random.Random):
    if SYNTH_DIR.exists():
        shutil.rmtree(SYNTH_DIR)
    for cls, base_color in CLASSES.items():
        cls_dir = SYNTH_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(IMAGES_PER_CLASS):
            img = Image.new("RGB", (128, 128), tuple(
                max(0, min(255, c + rng.randint(-15, 15))) for c in base_color
            ))
            draw = ImageDraw.Draw(img)
            # a few random shapes so pHash values actually differ across images
            for _ in range(3):
                x0, y0 = rng.randint(0, 100), rng.randint(0, 100)
                x1, y1 = x0 + rng.randint(5, 25), y0 + rng.randint(5, 25)
                shape_color = tuple(rng.randint(0, 255) for _ in range(3))
                draw.rectangle([x0, y0, x1, y1], fill=shape_color)
            img.save(cls_dir / f"{cls}_{i}.png")
    return SYNTH_DIR


def result_line(name: str, passed: bool, detail: str):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name} — {detail}")


def main():
    rng = random.Random(7)
    print("=== TRUST-X Data Forensics Benchmark ===\n")

    print("1. Generating synthetic clean dataset...")
    generate_synthetic_clean_dataset(rng)

    db_path = Path("./benchmark.db")
    if db_path.exists():
        db_path.unlink()
    import os
    os.environ["TRUSTX_DATABASE_URL"] = f"sqlite:///{db_path.resolve()}"
    # Import project modules only after the test database URL is set.
    from ..database import SessionLocal, init_db
    from ..ingestion.dataset import ingest_directory
    from ..forensics.duplicates import detect_near_duplicate_flooding
    from ..forensics.labels import detect_label_distribution_anomaly
    from ..forensics.ood import detect_ood_samples
    from ..attacks.duplicate_flood import create_duplicate_flood_batch
    from ..attacks.label_flip import create_label_poisoned_batch
    from ..attacks.ood_injection import create_ood_injected_batch
    init_db()
    db = SessionLocal()

    print("2. Ingesting clean batch...")
    clean_result = ingest_directory(
        db, dataset_name="BenchmarkDS", directory=str(SYNTH_DIR),
        contributor_name="BaselineContributor", batch_label="clean",
    )
    clean_batch_id = clean_result["batch_id"]
    print(f"   -> {clean_result['samples_ingested']} samples ingested into batch {clean_batch_id}\n")

    results = []

    # --- Scenario: clean batch should NOT trigger duplicate detector ---
    print("3. Scenario: clean batch (expect no findings)")
    findings = detect_near_duplicate_flooding(db, clean_batch_id)
    passed = len(findings) == 0
    result_line("duplicate_detector on clean batch", passed, f"{len(findings)} findings")
    results.append(passed)

    # --- Scenario: duplicate flooding ---
    print("\n4. Scenario: duplicate flooding attack")
    flood = create_duplicate_flood_batch(db, clean_batch_id, flood_fraction=0.4, copies_per_source=20, seed=7)
    findings = detect_near_duplicate_flooding(db, flood["flooded_batch_id"])
    passed = len(findings) > 0
    detail = f"ground truth flooded_fraction={flood['ground_truth_flooded_fraction']}, detector findings={len(findings)}"
    if findings:
        detail += f", top severity={findings[0].severity}"
    result_line("duplicate_detector on flooded batch", passed, detail)
    results.append(passed)

    # --- Scenario: label flipping ---
    print("\n5. Scenario: label-flip poisoning attack")
    # need baseline history to compare against -> ingest a second clean-ish batch first
    ingest_directory(db, dataset_name="BenchmarkDS", directory=str(SYNTH_DIR),
                      contributor_name="SecondContributor", batch_label="clean_2")
    flip = create_label_poisoned_batch(db, clean_batch_id, target_class="person", flip_fraction=0.7, seed=7)
    findings = detect_label_distribution_anomaly(db, flip["poisoned_batch_id"])
    passed = len(findings) > 0
    detail = f"ground truth flipped_fraction={flip['ground_truth_flipped_fraction']}, detector findings={len(findings)}"
    if findings:
        detail += f", js_distance={findings[0].details['js_distance']}"
    result_line("label_detector on poisoned batch", passed, detail)
    results.append(passed)

    # --- Scenario: OOD injection ---
    print("\n6. Scenario: OOD injection attack")
    ood = create_ood_injected_batch(db, clean_batch_id, injection_fraction=0.15, seed=7)
    findings = detect_ood_samples(db, ood["ood_batch_id"], reference_batch_id=clean_batch_id)
    passed = len(findings) > 0
    flagged = findings[0].details["flagged_fraction"] if findings else 0
    detail = f"ground truth injected_fraction={ood['ground_truth_injected_fraction']}, detector flagged_fraction={flagged}"
    result_line("ood_detector on injected batch", passed, detail)
    results.append(passed)

    db.close()

    print(f"\n=== {sum(results)}/{len(results)} scenarios passed ===")
    if all(results):
        print("All detectors are firing correctly against known ground truth.")
    else:
        print("Some detectors need threshold tuning — check the FAIL lines above.")


if __name__ == "__main__":
    main()
