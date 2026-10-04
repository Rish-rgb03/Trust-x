"""
ORM models for the AI pipeline lifecycle.
Contributor -> Dataset -> Batch -> Sample
                               -> TrainingRun -> Model -> ModelFingerprint
                                                        -> Deployment -> Inference
Plus Evidence (every detector writes here) and Hypothesis (scored explanations).
"""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from .database import Base


def gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now():
    return datetime.now(timezone.utc)


class Contributor(Base):
    __tablename__ = "contributors"
    id = Column(String, primary_key=True, default=lambda: gen_id("contrib"))
    name = Column(String, nullable=False)
    org = Column(String, nullable=True)
    created_at = Column(DateTime, default=now)
    batches = relationship("Batch", back_populates="contributor")


class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(String, primary_key=True, default=lambda: gen_id("ds"))
    name = Column(String, nullable=False)
    format = Column(String, nullable=True)
    file_path = Column(String, nullable=True)
    architecture = Column(String, nullable=True)
    sha256 = Column(String, nullable=True)
    created_at = Column(DateTime, default=now)
    batches = relationship("Batch", back_populates="dataset")


class Batch(Base):
    __tablename__ = "batches"
    id = Column(String, primary_key=True, default=lambda: gen_id("batch"))
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=False)
    contributor_id = Column(String, ForeignKey("contributors.id"), nullable=True)
    label = Column(String, nullable=True)   # human tag e.g. "clean", "label_poisoned_20pct"
    sha256 = Column(String, nullable=True)
    sample_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=now)
    dataset = relationship("Dataset", back_populates="batches")
    contributor = relationship("Contributor", back_populates="batches")
    samples = relationship("Sample", back_populates="batch")


class Sample(Base):
    __tablename__ = "samples"
    id = Column(String, primary_key=True, default=lambda: gen_id("sample"))
    batch_id = Column(String, ForeignKey("batches.id"), nullable=False)
    file_path = Column(String, nullable=False)
    sha256 = Column(String, nullable=True)
    label_class = Column(String, nullable=True)
    phash = Column(String, nullable=True)
    embedding = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=now)
    batch = relationship("Batch", back_populates="samples")


class TrainingRun(Base):
    __tablename__ = "training_runs"
    id = Column(String, primary_key=True, default=lambda: gen_id("train"))
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=False)
    batch_id = Column(String, ForeignKey("batches.id"), nullable=True)
    config_sha256 = Column(String, nullable=True)
    started_at = Column(DateTime, default=now)
    finished_at = Column(DateTime, nullable=True)
    models = relationship("MLModel", back_populates="training_run")
    dataset = relationship("Dataset")
    batch = relationship("Batch")


class MLModel(Base):
    __tablename__ = "models"
    id = Column(String, primary_key=True, default=lambda: gen_id("model"))
    training_run_id = Column(String, ForeignKey("training_runs.id"), nullable=True)
    name = Column(String, nullable=False)
    version = Column(String, nullable=True)
    format = Column(String, nullable=True)
    file_path = Column(String, nullable=True)
    architecture = Column(String, nullable=True)
    sha256 = Column(String, nullable=True)
    param_count = Column(Integer, nullable=True)
    layer_count = Column(Integer, nullable=True)
    input_shape = Column(String, nullable=True)
    created_at = Column(DateTime, default=now)
    training_run = relationship("TrainingRun", back_populates="models")
    fingerprints = relationship("ModelFingerprint", back_populates="model")
    deployments = relationship("Deployment", back_populates="model")


class ModelFingerprint(Base):
    __tablename__ = "model_fingerprints"
    id = Column(String, primary_key=True, default=lambda: gen_id("fp"))
    model_id = Column(String, ForeignKey("models.id"), nullable=False)
    probe_name = Column(String, nullable=False)
    response_vector = Column(JSON, nullable=False)
    stability_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=now)
    model = relationship("MLModel", back_populates="fingerprints")


class Deployment(Base):
    __tablename__ = "deployments"
    id = Column(String, primary_key=True, default=lambda: gen_id("deploy"))
    model_id = Column(String, ForeignKey("models.id"), nullable=False)
    environment = Column(String, nullable=True)
    config_sha256 = Column(String, nullable=True)
    deployed_at = Column(DateTime, default=now)
    model = relationship("MLModel", back_populates="deployments")
    inferences = relationship("Inference", back_populates="deployment")


class Inference(Base):
    __tablename__ = "inferences"
    id = Column(String, primary_key=True, default=lambda: gen_id("infer"))
    deployment_id = Column(String, ForeignKey("deployments.id"), nullable=False)
    input_sha256 = Column(String, nullable=True)
    output_sha256 = Column(String, nullable=True)
    model_sha256_at_inference = Column(String, nullable=True)
    preprocessing_sha256 = Column(String, nullable=True)
    config_sha256 = Column(String, nullable=True)
    sequence_number = Column(Integer, nullable=True)
    nonce = Column(String, nullable=True)
    output_json = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=now)
    deployment = relationship("Deployment", back_populates="inferences")


class Evidence(Base):
    __tablename__ = "evidence"
    id = Column(String, primary_key=True, default=lambda: gen_id("ev"))
    type = Column(String, nullable=False)
    source_node_type = Column(String, nullable=False)
    source_node_id = Column(String, nullable=False)
    severity = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    detector = Column(String, nullable=False)
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=now)


class Inspection(Base):
    __tablename__ = "inspections"
    id = Column(String, primary_key=True, default=lambda: gen_id("insp"))
    node_type = Column(String, nullable=False)
    node_id = Column(String, nullable=False)
    detector = Column(String, nullable=False)
    inspected_at = Column(DateTime, default=now)


class Hypothesis(Base):
    __tablename__ = "hypotheses"
    id = Column(String, primary_key=True, default=lambda: gen_id("hyp"))
    investigation_id = Column(String, nullable=False)
    label = Column(String, nullable=False)
    support_score = Column(Float, nullable=False)
    contributing_evidence_ids = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=now)
