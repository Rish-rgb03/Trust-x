import os

import numpy as np
from PIL import Image

os.environ["TRUSTX_DATABASE_URL"] = "sqlite:////tmp/trustx_integration_pytest.db"

from backend.database import SessionLocal, init_db
from backend import models_db as m
from backend.evidence.schema import EvidenceRecord, EvidenceType, SourceNodeType
from backend.evidence.graph import IntegrityGraph, NodeType
from backend.reasoning.investigator import investigate_pipeline
from backend.forensics.behavioral import build_behavioral_fingerprint, behavioral_evidence, compare_fingerprints
from backend.ingestion.model import make_callable_adapter
from backend.forensics.triggers import run_trigger_suite, trigger_evidence
from backend.provenance.passport import PassportIssuer, PassportVerifier, sha256_bytes, hash_obj
from backend.api.graph_builder import build_graph


def test_shared_evidence_record_is_accepted_by_canonical_graph():
    g = IntegrityGraph()
    g.add_node("B1", NodeType.BATCH, "Batch 1")
    ev = EvidenceRecord(type=EvidenceType.LABEL_DISTRIBUTION_ANOMALY, source_node_type=SourceNodeType.batch,
                        source_node_id="B1", severity=.7, confidence=.9, detector="label_detector")
    rec = g.attach_evidence(ev)
    assert rec.source_id == "B1"
    assert rec.evidence_id.startswith("E-")


def test_behavioral_self_comparison_is_exact():
    image = Image.new("RGB", (64, 64), "gray")
    def predict(im):
        arr = np.asarray(im)
        return {"detections": [{"cls": "vehicle", "conf": float(arr.mean()/255), "box": [1,2,3,4]}]}
    adapter = make_callable_adapter(predict, name="toy")
    fp = build_behavioral_fingerprint(adapter, image)
    metrics = compare_fingerprints(fp, fp)
    assert metrics["l2_distance"] == 0.0
    assert abs(metrics["cosine_similarity"] - 1.0) < 1e-12
    assert fp["responses"]["original"]["detection_count"] == 1
    assert fp["responses"]["original"]["classes"] == ["vehicle"]


def test_trigger_delta_creates_evidence_for_trigger_sensitive_adapter():
    image = Image.new("RGB", (64, 64), "gray")
    def predict(im):
        a = np.asarray(im.convert("RGB"))
        red = float(a[0,0,0] > 200 and a[0,0,1] < 80)
        return {"detections": [{"cls": "vehicle", "conf": .5 + .49*red, "box": [1,2,3,4]}]}
    adapter = make_callable_adapter(predict, name="trigger-toy")
    results = run_trigger_suite(adapter, image)
    ev = trigger_evidence(source_node_id="M1", results=results, delta_threshold=.5)
    assert ev is not None
    assert ev.type == EvidenceType.TRIGGER_SENSITIVITY


def test_passport_detects_output_tamper():
    key = b"integration-key"
    model_hash = sha256_bytes(b"model")
    issuer = PassportIssuer(key, "D1", model_hash, sha256_bytes(b"pp"), sha256_bytes(b"cfg"))
    img = b"image-bytes"
    out = {"detections": [{"cls":"vehicle","conf":.9}]}
    passport = issuer.issue(img, out, now=1700000000.0)
    verifier = PassportVerifier(key, model_hash, passport.preproc_hash, passport.config_hash)
    tampered = {"detections": [{"cls":"building","conf":.1}]}
    events = verifier.verify(passport, img, tampered, now=1700000000.0)
    assert any(e["type"] == "OUTPUT_HASH_MISMATCH" for e in events)


def test_db_graph_builder_connects_lifecycle_and_evidence():
    init_db()
    db = SessionLocal()
    c=m.Contributor(name="C"); ds=m.Dataset(name="DS"); db.add_all([c,ds]); db.commit()
    b=m.Batch(dataset_id=ds.id, contributor_id=c.id, label="clean"); db.add(b); db.commit()
    tr=m.TrainingRun(dataset_id=ds.id,batch_id=b.id); db.add(tr); db.commit()
    model=m.MLModel(training_run_id=tr.id,name="M",sha256="h"); db.add(model); db.commit()
    dep=m.Deployment(model_id=model.id); db.add(dep); db.commit()
    ev=m.Evidence(type="LABEL_DISTRIBUTION_ANOMALY",source_node_type="batch",source_node_id=b.id,severity=.8,confidence=.9,detector="label_detector"); db.add(ev)
    db.add(m.Inspection(node_type="batch",node_id=b.id,detector="label_detector")); db.commit()
    graph=build_graph(db)
    assert graph.node_type(b.id) is NodeType.BATCH
    assert graph.node_type(model.id) is NodeType.MODEL
    assert ev.id in graph.g.nodes[b.id]["evidence_ids"]
    assert investigate_pipeline(graph).top_hypothesis == "DATASET_POISONING"
    db.close()


def test_grounded_report_writer_is_deterministic_and_cited():
    from backend.evidence.graph import build_demo_graph
    from backend.reasoning.report_writer import generate_grounded_report
    report = investigate_pipeline(build_demo_graph()).to_dict()
    written = generate_grounded_report(report)
    assert written["generator"] == "template"
    assert written["grounded"] is True
    assert "[F1]" in written["narrative"]
    assert written["facts"]
