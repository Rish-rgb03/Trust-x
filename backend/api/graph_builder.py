"""Build the canonical IntegrityGraph from the SQL database."""
from __future__ import annotations
from datetime import datetime
from sqlalchemy.orm import Session
from .. import models_db as m
from ..evidence.graph import EdgeType as E, IntegrityGraph, NodeType as N

def _attrs(**kwargs):
    return {k: v for k, v in kwargs.items() if v is not None}

def build_graph(db: Session) -> IntegrityGraph:
    g = IntegrityGraph()
    contributors = db.query(m.Contributor).all()
    datasets = db.query(m.Dataset).all()
    batches = db.query(m.Batch).all()
    samples = db.query(m.Sample).all()
    runs = db.query(m.TrainingRun).all()
    models = db.query(m.MLModel).all()
    deployments = db.query(m.Deployment).all()
    inferences = db.query(m.Inference).all()
    inspections = db.query(m.Inspection).all()
    evidence = db.query(m.Evidence).all()

    for x in contributors:
        g.add_node(x.id, N.CONTRIBUTOR, x.name, **_attrs(org=x.org))
    for x in datasets:
        g.add_node(x.id, N.DATASET, x.name, **_attrs(format=x.format, sha256=x.sha256))
    for x in batches:
        g.add_node(x.id, N.BATCH, f"Batch {x.label or x.id}", **_attrs(dataset_id=x.dataset_id, contributor_id=x.contributor_id, sha256=x.sha256, sample_count=x.sample_count))
    for x in samples:
        g.add_node(x.id, N.SAMPLE, f"Sample {x.id}", **_attrs(batch_id=x.batch_id, sha256=x.sha256, label_class=x.label_class))
    for x in runs:
        g.add_node(x.id, N.TRAINING_RUN, f"Training Run {x.id}", **_attrs(dataset_id=x.dataset_id, batch_id=x.batch_id, config_sha256=x.config_sha256, started_at=x.started_at.isoformat() if isinstance(x.started_at, datetime) else x.started_at))
    for x in models:
        g.add_node(x.id, N.MODEL, f"{x.name} {x.version or ''}".strip(), **_attrs(training_run_id=x.training_run_id, format=x.format, file_path=x.file_path, architecture=x.architecture, sha256=x.sha256, param_count=x.param_count, layer_count=x.layer_count, input_shape=x.input_shape))
    for x in deployments:
        g.add_node(x.id, N.DEPLOYMENT, f"Deployment {x.id}", **_attrs(model_id=x.model_id, environment=x.environment, config_sha256=x.config_sha256))
        if x.config_sha256:
            cid=f"config_{x.id}"
            g.add_node(cid, N.CONFIGURATION, f"Configuration {x.id}", sha256=x.config_sha256)
            g.add_edge(cid, x.id, E.CONFIGURES)
    for x in inferences:
        input_id=f"input_{x.id}"
        output_id=f"output_{x.id}"
        g.add_node(x.id, N.INFERENCE, f"Inference {x.id}", **_attrs(deployment_id=x.deployment_id, timestamp=x.timestamp.isoformat() if isinstance(x.timestamp, datetime) else x.timestamp, sequence_number=x.sequence_number))
        g.add_node(input_id, N.INPUT, f"Input {x.id}", **_attrs(sha256=x.input_sha256))
        g.add_node(output_id, N.OUTPUT, f"Output {x.id}", **_attrs(sha256=x.output_sha256, output_json=x.output_json))
        g.add_edge(input_id, x.id, E.INFERRED_ON)
        g.add_edge(x.id, output_id, E.GENERATED)

    for x in batches:
        g.add_edge(x.dataset_id, x.id, E.CONTAINS) if x.dataset_id in g else None
        if x.contributor_id and x.contributor_id in g: g.add_edge(x.contributor_id, x.id, E.CONTRIBUTED)
    for x in samples:
        if x.batch_id in g: g.add_edge(x.batch_id, x.id, E.CONTAINS)
    for x in runs:
        if x.batch_id and x.batch_id in g: g.add_edge(x.batch_id, x.id, E.TRAINED_ON)
        elif x.dataset_id in g: g.add_edge(x.dataset_id, x.id, E.TRAINED_ON)
    for x in models:
        if x.training_run_id and x.training_run_id in g: g.add_edge(x.training_run_id, x.id, E.PRODUCED)
    for x in deployments:
        if x.model_id in g: g.add_edge(x.model_id, x.id, E.DEPLOYED_AS)
    for x in inferences:
        if x.deployment_id in g: g.add_edge(x.deployment_id, x.id, E.INFERRED_ON)

    for x in inspections:
        if x.node_id in g: g.record_inspection(x.node_id, x.detector)
    for x in evidence:
        if x.source_node_id not in g:
            continue
        g.attach_evidence({
            "evidence_id": x.id,
            "source_id": x.source_node_id,
            "evidence_type": x.type,
            "severity": x.severity,
            "confidence": x.confidence,
            "detector": x.detector,
            "payload": {"source_node_type": x.source_node_type, **(x.details or {})},
        })
    return g
