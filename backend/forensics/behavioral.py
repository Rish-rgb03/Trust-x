"""
TRUST-X Model Behavioral Forensics (Person 2).

Core idea:
    model -> controlled probes -> response measurements -> fingerprint
    -> reference/candidate comparison -> EvidenceRecord

This module deliberately does NOT make an attack verdict. It produces
structured evidence that Person 3 can fuse with data/provenance evidence.

The default probes are model-agnostic image transformations:
brightness, contrast, blur, noise, rotation, crop, occlusion and JPEG
compression. The response summarizer is intentionally conservative and
works with common scalar/classification/detection-style outputs.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from ..evidence.schema import EvidenceRecord, EvidenceType, SourceNodeType
from ..ingestion.model import ModelAdapter, to_jsonable


@dataclass
class Probe:
    name: str
    transform: Callable[[Image.Image], Image.Image]


def _brightness(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(factor)


def _contrast(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(image).enhance(factor)


def _add_noise(image: Image.Image, sigma: float = 0.05) -> Image.Image:
    arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    noise = np.random.default_rng(42).normal(0.0, sigma, arr.shape)
    out = np.clip((arr + noise) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def _crop_and_restore(image: Image.Image, fraction: float = 0.9) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    nw, nh = max(1, int(w * fraction)), max(1, int(h * fraction))
    left = (w - nw) // 2
    top = (h - nh) // 2
    cropped = image.crop((left, top, left + nw, top + nh))
    return cropped.resize((w, h))


def _occlude(image: Image.Image, fraction: float = 0.2) -> Image.Image:
    image = image.convert("RGB").copy()
    w, h = image.size
    ow, oh = max(1, int(w * fraction)), max(1, int(h * fraction))
    left = (w - ow) // 2
    top = (h - oh) // 2
    patch = Image.new("RGB", (ow, oh), (0, 0, 0))
    image.paste(patch, (left, top))
    return image


def _jpeg_roundtrip(image: Image.Image, quality: int = 45) -> Image.Image:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    with Image.open(buf) as compressed:
        return compressed.convert("RGB").copy()


def default_probe_suite() -> list[Probe]:
    return [
        Probe("original", lambda im: im.copy()),
        Probe("brightness_up", lambda im: _brightness(im, 1.20)),
        Probe("brightness_down", lambda im: _brightness(im, 0.80)),
        Probe("contrast_up", lambda im: _contrast(im, 1.20)),
        Probe("contrast_down", lambda im: _contrast(im, 0.80)),
        Probe("blur", lambda im: im.convert("RGB").filter(ImageFilter.GaussianBlur(radius=1.5))),
        Probe("noise", lambda im: _add_noise(im)),
        Probe("rotation_plus_5", lambda im: im.convert("RGB").rotate(5, expand=False)),
        Probe("rotation_minus_5", lambda im: im.convert("RGB").rotate(-5, expand=False)),
        Probe("crop", lambda im: _crop_and_restore(im)),
        Probe("occlusion", lambda im: _occlude(im)),
        Probe("jpeg_compression", lambda im: _jpeg_roundtrip(im)),
    ]


def _flatten_numeric(value: Any) -> np.ndarray:
    """Extract numeric values from arbitrary JSON-safe model output."""
    if value is None:
        return np.array([], dtype=np.float64)
    if isinstance(value, bool):
        return np.array([float(value)])
    if isinstance(value, (int, float)):
        return np.array([float(value)], dtype=np.float64)
    if isinstance(value, dict):
        chunks = [_flatten_numeric(v) for v in value.values()]
        return np.concatenate([x for x in chunks if x.size]) if any(x.size for x in chunks) else np.array([])
    if isinstance(value, (list, tuple)):
        chunks = [_flatten_numeric(v) for v in value]
        return np.concatenate([x for x in chunks if x.size]) if any(x.size for x in chunks) else np.array([])
    return np.array([], dtype=np.float64)


def summarize_prediction(prediction: Any) -> dict[str, Any]:
    """Turn a prediction into a stable, compact behavioral measurement.

    For detection-style outputs we explicitly retain detection count, classes,
    class distribution, confidence statistics and bounding boxes. A small fixed
    vector is used for model-to-model distance calculations.
    """
    p = to_jsonable(prediction)
    numeric = _flatten_numeric(p)
    summary: dict[str, Any] = {
        "raw": p,
        "numeric_count": int(numeric.size),
        "numeric_mean": float(numeric.mean()) if numeric.size else 0.0,
        "numeric_std": float(numeric.std()) if numeric.size else 0.0,
    }

    detections = p.get("detections") if isinstance(p, dict) else None
    if isinstance(detections, list):
        classes: list[str] = []
        confidences: list[float] = []
        boxes: list[list[float]] = []
        for det in detections:
            if not isinstance(det, dict):
                continue
            for key in ("class", "label", "cls", "predicted_class"):
                if key in det:
                    classes.append(str(det[key]))
                    break
            for key in ("confidence", "score", "probability", "conf"):
                val = det.get(key)
                if isinstance(val, (int, float)):
                    confidences.append(float(val))
                    break
            for key in ("bbox", "box", "bounding_box"):
                val = det.get(key)
                if isinstance(val, (list, tuple)) and len(val) >= 4:
                    try:
                        boxes.append([float(x) for x in val[:4]])
                    except (TypeError, ValueError):
                        pass
                    break
        summary.update({
            "detection_count": len(detections),
            "classes": classes,
            "class_distribution": ({c: classes.count(c) / len(classes) for c in sorted(set(classes))}
                                    if classes else {}),
            "bounding_boxes": boxes[:50],
            "detection_confidence_mean": float(np.mean(confidences)) if confidences else 0.0,
            "detection_confidence_std": float(np.std(confidences)) if confidences else 0.0,
        })
    elif isinstance(p, dict):
        for key in ("class", "label", "predicted_class"):
            if key in p:
                summary["class"] = p[key]
                break
        for key in ("confidence", "score", "probability"):
            if isinstance(p.get(key), (int, float)):
                summary["confidence"] = float(p[key])
                break

    response = [
        summary["numeric_mean"],
        summary["numeric_std"],
        float(summary.get("detection_count", summary["numeric_count"])),
        float(summary.get("detection_confidence_mean", summary.get("confidence", 0.0))),
    ]
    summary["response_vector"] = response
    return summary


def run_probe_suite(
    adapter: ModelAdapter,
    image: Image.Image,
    probes: Optional[Iterable[Probe]] = None,
) -> dict[str, dict[str, Any]]:
    """Run all probes and return measured responses."""
    probes = list(probes or default_probe_suite())
    results: dict[str, dict[str, Any]] = {}

    for probe in probes:
        transformed = probe.transform(image.convert("RGB"))
        prediction = adapter.predict(transformed)
        results[probe.name] = summarize_prediction(prediction)

    return results


def build_behavioral_fingerprint(
    adapter: ModelAdapter,
    image: Image.Image,
    probes: Optional[Iterable[Probe]] = None,
) -> dict[str, Any]:
    results = run_probe_suite(adapter, image, probes)
    names = list(results)
    vector: list[float] = []
    for name in names:
        vector.extend(results[name]["response_vector"])

    original = results.get("original", {})
    stability_values = []
    base = np.asarray(original.get("response_vector", []), dtype=float)

    if base.size:
        for name, result in results.items():
            if name == "original":
                continue
            current = np.asarray(result["response_vector"], dtype=float)
            if current.shape == base.shape:
                denom = np.linalg.norm(base) + 1e-8
                stability_values.append(
                    max(0.0, 1.0 - float(np.linalg.norm(current - base) / denom))
                )

    return {
        "probe_names": names,
        "response_vector": vector,
        "responses": results,
        "stability_score": float(np.mean(stability_values)) if stability_values else 1.0,
    }


def l2_distance(vector_a: Iterable[float], vector_b: Iterable[float]) -> float:
    a = np.asarray(list(vector_a), dtype=float)
    b = np.asarray(list(vector_b), dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Fingerprint shapes differ: {a.shape} vs {b.shape}")
    return float(np.linalg.norm(a - b))


def cosine_similarity(vector_a: Iterable[float], vector_b: Iterable[float]) -> float:
    a = np.asarray(list(vector_a), dtype=float)
    b = np.asarray(list(vector_b), dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Fingerprint shapes differ: {a.shape} vs {b.shape}")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.dot(a, b) / denom)


def compare_fingerprints(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float]:
    """Return raw comparison metrics; thresholds should be benchmark-derived."""
    a = reference["response_vector"]
    b = candidate["response_vector"]
    return {
        "l2_distance": l2_distance(a, b),
        "cosine_similarity": cosine_similarity(a, b),
    }


def _deviation_severity(l2: float, cosine: float) -> float:
    """Conservative v0.1 normalization, NOT a calibrated probability.

    The benchmark should later replace these constants with empirical
    thresholds derived from clean-vs-clean/fine-tune/attack experiments.
    """
    l2_component = min(1.0, l2 / 2.0)
    cosine_component = min(1.0, max(0.0, (1.0 - cosine) / 0.5))
    return round(max(l2_component, cosine_component), 3)


def behavioral_evidence(
    *,
    source_node_id: str,
    reference_fingerprint: dict[str, Any],
    candidate_fingerprint: dict[str, Any],
    detector: str = "behavior_detector",
) -> Optional[EvidenceRecord]:
    metrics = compare_fingerprints(reference_fingerprint, candidate_fingerprint)
    severity = _deviation_severity(metrics["l2_distance"], metrics["cosine_similarity"])

    # v0.1 evidence threshold. It is deliberately not described as a
    # universal compromise threshold; benchmark calibration should replace it.
    if severity < 0.25:
        return None

    confidence = min(
        0.99,
        0.65
        + 0.20 * min(1.0, metrics["l2_distance"])
        + 0.15 * min(1.0, max(0.0, 1.0 - metrics["cosine_similarity"])),
    )

    return EvidenceRecord(
        type=EvidenceType.BEHAVIORAL_DEVIATION,
        source_node_type=SourceNodeType.model,
        source_node_id=source_node_id,
        severity=severity,
        confidence=round(confidence, 3),
        detector=detector,
        details={
            "l2_distance": round(metrics["l2_distance"], 6),
            "cosine_similarity": round(metrics["cosine_similarity"], 6),
            "reference_stability": reference_fingerprint.get("stability_score"),
            "candidate_stability": candidate_fingerprint.get("stability_score"),
            "threshold_note": "v0.1 heuristic; calibrate from benchmark scenarios",
        },
    )


def structural_evidence(
    *,
    source_node_id: str,
    differences: dict[str, Any],
    detector: str = "structural_detector",
) -> Optional[EvidenceRecord]:
    if not differences:
        return None

    severity = min(1.0, 0.25 * len(differences))
    return EvidenceRecord(
        type=EvidenceType.STRUCTURAL_DEVIATION,
        source_node_type=SourceNodeType.model,
        source_node_id=source_node_id,
        severity=round(severity, 3),
        confidence=0.95,
        detector=detector,
        details={"differences": differences},
    )
