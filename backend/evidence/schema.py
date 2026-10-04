"""Shared evidence contract used by data, model, provenance and inference detectors."""
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field

class EvidenceType(str, Enum):
    NEAR_DUPLICATE_CLUSTER = "NEAR_DUPLICATE_CLUSTER"
    LABEL_DISTRIBUTION_ANOMALY = "LABEL_DISTRIBUTION_ANOMALY"
    OOD_ANOMALY = "OOD_ANOMALY"
    TRIGGER_SENSITIVITY = "TRIGGER_SENSITIVITY"
    MODEL_ARTIFACT_MISMATCH = "MODEL_ARTIFACT_MISMATCH"
    STRUCTURAL_DEVIATION = "STRUCTURAL_DEVIATION"
    STRUCTURAL_FINGERPRINT_DEVIATION = "STRUCTURAL_FINGERPRINT_DEVIATION"
    BEHAVIORAL_DEVIATION = "BEHAVIORAL_DEVIATION"
    BEHAVIORAL_FINGERPRINT_DEVIATION = "BEHAVIORAL_FINGERPRINT_DEVIATION"
    PROVENANCE_MISMATCH = "PROVENANCE_MISMATCH"
    REPLAY_SUSPECTED = "REPLAY_SUSPECTED"
    REPLAY_DETECTED = "REPLAY_DETECTED"
    CONTRIBUTOR_ANOMALY = "CONTRIBUTOR_ANOMALY"
    BENIGN_ENVIRONMENT_SHIFT = "BENIGN_ENVIRONMENT_SHIFT"
    ENVIRONMENT_METADATA_CHANGE = "ENVIRONMENT_METADATA_CHANGE"
    TEMPORAL_ANOMALY = "TEMPORAL_ANOMALY"
    CLASS_IMBALANCE = "CLASS_IMBALANCE"
    OUTPUT_HASH_MISMATCH = "OUTPUT_HASH_MISMATCH"
    INPUT_HASH_MISMATCH = "INPUT_HASH_MISMATCH"
    SIGNATURE_INVALID = "SIGNATURE_INVALID"
    SEQUENCE_GAP = "SEQUENCE_GAP"
    CHAIN_BREAK = "CHAIN_BREAK"
    TIMESTAMP_SKEW = "TIMESTAMP_SKEW"
    CONFIG_MISMATCH = "CONFIG_MISMATCH"
    MODEL_HASH_VERIFIED = "MODEL_HASH_VERIFIED"
    DATA_PROVENANCE_VERIFIED = "DATA_PROVENANCE_VERIFIED"
    TRIGGER_PROBE_CLEAN = "TRIGGER_PROBE_CLEAN"
    PASSPORT_VERIFIED = "PASSPORT_VERIFIED"

class SourceNodeType(str, Enum):
    contributor = "contributor"
    dataset = "dataset"
    batch = "batch"
    sample = "sample"
    training_run = "training_run"
    model = "model"
    deployment = "deployment"
    inference = "inference"
    input = "input"
    output = "output"
    configuration = "configuration"

class EvidenceRecord(BaseModel):
    id: Optional[str] = None
    type: EvidenceType | str
    source_node_type: SourceNodeType | str
    source_node_id: str
    severity: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    detector: str
    details: Optional[dict[str, Any]] = None
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
