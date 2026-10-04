"""
Controlled model-substitution benchmark helper (Person 2).

The function registers a candidate model as a separate MLModel record and
returns the structural/artifact comparison inputs. It does not modify or
replace the authorized model on disk.

Use it to create a known-ground-truth substitution scenario:
    authorized model -> candidate model
Then run structural/behavioral checks against the two artifacts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from .. import models_db as m
from ..ingestion.model import (
    ModelMetadata,
    extract_structural_metadata,
    structural_difference,
)
from ..provenance.hashing import sha256_file


def register_candidate_model(
    db: Session,
    *,
    authorized_model_id: str,
    candidate_path: str,
    name: Optional[str] = None,
    version: str = "candidate",
) -> dict:
    authorized = db.query(m.MLModel).filter(m.MLModel.id == authorized_model_id).first()
    if not authorized:
        raise ValueError("authorized model not found")

    candidate_path = str(candidate_path)
    if not Path(candidate_path).is_file():
        raise FileNotFoundError(candidate_path)

    candidate = m.MLModel(
        training_run_id=authorized.training_run_id,
        name=name or Path(candidate_path).stem,
        version=version,
        format=Path(candidate_path).suffix.lstrip(".") or "unknown",
        sha256=sha256_file(candidate_path),
    )
    db.add(candidate)
    db.commit()
    db.refresh(candidate)

    return {
        "authorized_model_id": authorized.id,
        "candidate_model_id": candidate.id,
        "authorized_sha256": authorized.sha256,
        "candidate_sha256": candidate.sha256,
        "artifact_mismatch": authorized.sha256 != candidate.sha256,
    }


def artifact_mismatch_evidence(
    *,
    source_node_id: str,
    authorized_sha256: str,
    candidate_sha256: str,
):
    from ..evidence.schema import EvidenceRecord, EvidenceType, SourceNodeType

    if not authorized_sha256 or not candidate_sha256:
        return None
    if authorized_sha256 == candidate_sha256:
        return None

    return EvidenceRecord(
        type=EvidenceType.MODEL_ARTIFACT_MISMATCH,
        source_node_type=SourceNodeType.model,
        source_node_id=source_node_id,
        severity=1.0,
        confidence=0.99,
        detector="model_artifact_hash_checker",
        details={
            "authorized_sha256": authorized_sha256,
            "candidate_sha256": candidate_sha256,
            "note": "Hash mismatch proves artifact difference, not maliciousness by itself.",
        },
    )
