"""TRUST-X integrated FastAPI application.

This is the single application entry point for the four contributors' work.
The data/model detectors emit the shared EvidenceRecord contract; the canonical
IntegrityGraph + fusion + assurance engine consumes persisted evidence and the
React frontend reads these APIs.
"""
from __future__ import annotations

import base64
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from . import models_db as m
from .api.graph_builder import build_graph
from .api.routes_reasoning import router as reasoning_router
from .database import get_db, init_db
from .evidence.schema import EvidenceRecord
from .evidence.graph import build_demo_graph
from .forensics.duplicates import compute_phash, detect_near_duplicate_flooding
from .forensics.labels import detect_label_distribution_anomaly
from .forensics.ood import detect_ood_samples
from .forensics.behavioral import build_behavioral_fingerprint, behavioral_evidence, structural_evidence, compare_fingerprints
from .forensics.triggers import run_trigger_suite, trigger_evidence
from .ingestion.model import detect_model_format, load_pytorch_adapter
from .attacks.model_substitution import artifact_mismatch_evidence
from .provenance.hashing import sha256_file
from .provenance.passport import Passport, PassportVerifier

STORAGE_ROOT = Path(os.environ.get("TRUSTX_STORAGE", "./storage"))
MODEL_ROOT = STORAGE_ROOT / "models"
UPLOAD_ROOT = STORAGE_ROOT / "uploads"
MODEL_ROOT.mkdir(parents=True, exist_ok=True)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="TRUST-X", version="0.1.0", description="Evidence-driven AI integrity assurance")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in os.environ.get("TRUSTX_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if x.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(reasoning_router)

@app.on_event("startup")
def startup() -> None:
    init_db()


def _json_model(ev: EvidenceRecord) -> dict[str, Any]:
    return ev.model_dump(mode="json")


def _store_findings(db: Session, findings: list[Any], inspected_node: str, detector: str, node_type: str = "batch") -> list[dict[str, Any]]:
    db.add(m.Inspection(node_type=node_type, node_id=inspected_node, detector=detector))
    stored = []
    for ev in findings:
        data = ev.model_dump(mode="json") if hasattr(ev, "model_dump") else dict(ev)
        if data.get("id") is None:
            data.pop("id", None)
        row = m.Evidence(
            id=data.get("id") or m.gen_id("ev"),
            type=str(data["type"]),
            source_node_type=str(data["source_node_type"]),
            source_node_id=data["source_node_id"],
            severity=float(data["severity"]),
            confidence=float(data["confidence"]),
            detector=data["detector"],
            details=data.get("details"),
        )
        db.add(row)
        stored.append({"id": row.id, **{k: v for k, v in data.items() if k != "id"}})
    db.commit()
    return stored

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "trust-x"}

@app.post("/contributors")
def create_contributor(name: str = Form(...), org: Optional[str] = Form(None), db: Session = Depends(get_db)):
    row = m.Contributor(name=name.strip(), org=org)
    db.add(row); db.commit(); db.refresh(row)
    return {"id": row.id, "name": row.name, "org": row.org}

@app.post("/datasets")
def create_dataset(name: str = Form(...), format: Optional[str] = Form(None), db: Session = Depends(get_db)):
    row = m.Dataset(name=name.strip(), format=format)
    db.add(row); db.commit(); db.refresh(row)
    return {"id": row.id, "name": row.name, "format": row.format}

@app.post("/batches")
def create_batch(dataset_id: str = Form(...), contributor_id: Optional[str] = Form(None), label: Optional[str] = Form(None), db: Session = Depends(get_db)):
    if not db.get(m.Dataset, dataset_id):
        raise HTTPException(404, "dataset not found")
    if contributor_id and not db.get(m.Contributor, contributor_id):
        raise HTTPException(404, "contributor not found")
    row = m.Batch(dataset_id=dataset_id, contributor_id=contributor_id, label=label)
    db.add(row); db.commit(); db.refresh(row)
    return {"id": row.id, "dataset_id": row.dataset_id, "contributor_id": row.contributor_id, "label": row.label}

@app.post("/batches/{batch_id}/samples")
def upload_sample(batch_id: str, file: UploadFile = File(...), label_class: Optional[str] = Form(None), db: Session = Depends(get_db)):
    batch = db.get(m.Batch, batch_id)
    if not batch:
        raise HTTPException(404, "batch not found")
    name = Path(file.filename or "").name
    if not name or Path(name).suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        raise HTTPException(400, "only supported image files are accepted")
    target = UPLOAD_ROOT / f"{m.gen_id('sample')}{Path(name).suffix.lower()}"
    content = file.file.read()
    target.write_bytes(content)
    try:
        sha = sha256_file(target)
        ph = compute_phash(str(target))
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(400, f"invalid image: {exc}") from exc
    row = m.Sample(batch_id=batch_id, file_path=str(target.resolve()), sha256=sha, label_class=label_class, phash=ph)
    db.add(row); batch.sample_count = (batch.sample_count or 0) + 1; db.commit(); db.refresh(row)
    return {"id": row.id, "file_path": row.file_path, "sha256": row.sha256, "phash": row.phash}

@app.post("/batches/{batch_id}/detect/duplicates")
def detect_duplicates(batch_id: str, db: Session = Depends(get_db)):
    findings = detect_near_duplicate_flooding(db, batch_id)
    stored = _store_findings(db, findings, batch_id, "duplicate_detector", "batch")
    return {"findings": stored}

@app.post("/batches/{batch_id}/detect/labels")
def detect_labels(batch_id: str, db: Session = Depends(get_db)):
    findings = detect_label_distribution_anomaly(db, batch_id)
    stored = _store_findings(db, findings, batch_id, "label_detector", "batch")
    return {"findings": stored}

@app.post("/batches/{batch_id}/detect/ood")
def detect_ood(batch_id: str, reference_batch_id: Optional[str] = None, db: Session = Depends(get_db)):
    findings = detect_ood_samples(db, batch_id, reference_batch_id)
    stored = _store_findings(db, findings, batch_id, "ood_detector", "batch")
    return {"findings": stored}

@app.post("/batches/{batch_id}/detect/all")
def detect_all(batch_id: str, db: Session = Depends(get_db)):
    findings = []
    findings.extend(detect_near_duplicate_flooding(db, batch_id))
    findings.extend(detect_label_distribution_anomaly(db, batch_id))
    findings.extend(detect_ood_samples(db, batch_id, None))
    # Persist each detector inspection even when it returns no evidence.
    for det in ("duplicate_detector", "label_detector", "ood_detector"):
        db.add(m.Inspection(node_type="batch", node_id=batch_id, detector=det))
    stored=[]
    for ev in findings:
        data=ev.model_dump(mode="json")
        row=m.Evidence(id=data.get("id") or m.gen_id("ev"), type=str(data["type"]), source_node_type=str(data["source_node_type"]), source_node_id=data["source_node_id"], severity=data["severity"], confidence=data["confidence"], detector=data["detector"], details=data.get("details"))
        db.add(row); stored.append({"id":row.id, **{k:v for k,v in data.items() if k!="id"}})
    db.commit()
    return {"findings": stored}

class TrainingRunRequest(BaseModel):
    dataset_id: str
    batch_id: Optional[str] = None
    config_sha256: Optional[str] = None

@app.post("/training-runs")
def create_training_run(req: TrainingRunRequest, db: Session = Depends(get_db)):
    if not db.get(m.Dataset, req.dataset_id): raise HTTPException(404, "dataset not found")
    if req.batch_id:
        batch=db.get(m.Batch, req.batch_id)
        if not batch or batch.dataset_id != req.dataset_id: raise HTTPException(400, "batch does not belong to dataset")
    row=m.TrainingRun(dataset_id=req.dataset_id, batch_id=req.batch_id, config_sha256=req.config_sha256)
    db.add(row); db.commit(); db.refresh(row)
    return {"id":row.id, "dataset_id":row.dataset_id, "batch_id":row.batch_id, "config_sha256":row.config_sha256}

@app.post("/models")
def register_model(training_run_id: Optional[str] = Form(None), name: Optional[str] = Form(None), version: Optional[str] = Form(None), file: UploadFile = File(...), db: Session = Depends(get_db)):
    if training_run_id and not db.get(m.TrainingRun, training_run_id): raise HTTPException(404, "training run not found")
    original = Path(file.filename or "model").name
    suffix=Path(original).suffix.lower()
    target=MODEL_ROOT / f"{m.gen_id('model')}{suffix}"
    target.write_bytes(file.file.read())
    row=m.MLModel(training_run_id=training_run_id, name=name or Path(original).stem, version=version, format=detect_model_format(target), file_path=str(target.resolve()), sha256=sha256_file(target))
    db.add(row); db.commit(); db.refresh(row)
    return {"id":row.id, "name":row.name, "version":row.version, "format":row.format, "file_path":row.file_path, "sha256":row.sha256}

@app.get("/models")
def list_models(db: Session = Depends(get_db)):
    return [{"id":x.id,"name":x.name,"version":x.version,"format":x.format,"file_path":x.file_path,"sha256":x.sha256,"param_count":x.param_count,"layer_count":x.layer_count,"input_shape":x.input_shape,"architecture":x.architecture} for x in db.query(m.MLModel).order_by(m.MLModel.created_at.desc()).all()]

@app.post("/models/{model_id}/inspect")
def inspect_model(model_id: str, db: Session = Depends(get_db)):
    model_row=db.get(m.MLModel, model_id)
    if not model_row or not model_row.file_path: raise HTTPException(404,"model artifact not found")
    if model_row.format not in {"pytorch","torchscript"}: raise HTTPException(409,"detailed structural inspection currently supports PyTorch/TorchScript artifacts")
    try:
        adapter=load_pytorch_adapter(model_row.file_path)
    except Exception as exc: raise HTTPException(503, str(exc)) from exc
    meta=adapter.metadata
    model_row.param_count=meta.parameter_count; model_row.layer_count=meta.layer_count; model_row.input_shape=str(meta.input_shape) if meta.input_shape else None; model_row.architecture=meta.architecture; db.commit()
    db.add(m.Inspection(node_type="model",node_id=model_id,detector="structural_detector")); db.commit()
    return {"model_id":model_id,"metadata":{k:v for k,v in meta.__dict__.items()},"sha256":model_row.sha256}

@app.post("/models/{candidate_id}/compare/{reference_id}")
def compare_models(candidate_id: str, reference_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    cand=db.get(m.MLModel,candidate_id); ref=db.get(m.MLModel,reference_id)
    if not cand or not ref or not cand.file_path or not ref.file_path: raise HTTPException(404,"model not found")
    if cand.format not in {"pytorch","torchscript"} or ref.format not in {"pytorch","torchscript"}: raise HTTPException(409,"behavioral comparison currently requires PyTorch/TorchScript")
    tmp=UPLOAD_ROOT / f"{m.gen_id('probe')}{Path(file.filename or 'probe.jpg').suffix.lower()}"; tmp.write_bytes(file.file.read())
    try:
        from PIL import Image
        image=Image.open(tmp).convert("RGB")
        ref_adapter=load_pytorch_adapter(ref.file_path); cand_adapter=load_pytorch_adapter(cand.file_path)
        ref_fp=build_behavioral_fingerprint(ref_adapter,image); cand_fp=build_behavioral_fingerprint(cand_adapter,image)
        ev=behavioral_evidence(source_node_id=candidate_id,reference_fingerprint=ref_fp,candidate_fingerprint=cand_fp)
        differences={}
        # structural comparison from stored DB values, including those loaded here
        differences={k:(getattr(ref_adapter.metadata,k),getattr(cand_adapter.metadata,k)) for k in ("format","architecture","parameter_count","layer_count","input_shape") if getattr(ref_adapter.metadata,k)!=getattr(cand_adapter.metadata,k)}
        sev=structural_evidence(source_node_id=candidate_id,differences=differences)
        findings=[x for x in (ev,sev) if x is not None]
        for det in ("behavior_detector","structural_detector"): db.add(m.Inspection(node_type="model",node_id=candidate_id,detector=det))
        for x in findings:
            data=x.model_dump(mode="json"); db.add(m.Evidence(id=m.gen_id("ev"),type=str(data["type"]),source_node_type=str(data["source_node_type"]),source_node_id=data["source_node_id"],severity=data["severity"],confidence=data["confidence"],detector=data["detector"],details=data.get("details")))
        db.add(m.ModelFingerprint(model_id=candidate_id,probe_name="default_suite",response_vector=cand_fp["response_vector"],stability_score=cand_fp["stability_score"]))
        db.commit()
        return {"reference_model_id":reference_id,"candidate_model_id":candidate_id,"comparison":compare_fingerprints(ref_fp,cand_fp),"behavioral":cand_fp,"findings":[x.model_dump(mode="json") for x in findings]}
    finally: tmp.unlink(missing_ok=True)

@app.post("/models/{model_id}/trigger-test")
def trigger_test(model_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    model_row=db.get(m.MLModel,model_id)
    if not model_row or not model_row.file_path: raise HTTPException(404,"model not found")
    if model_row.format not in {"pytorch","torchscript"}: raise HTTPException(409,"trigger testing currently requires PyTorch/TorchScript")
    tmp=UPLOAD_ROOT / f"{m.gen_id('trigger')}{Path(file.filename or 'probe.jpg').suffix.lower()}"; tmp.write_bytes(file.file.read())
    try:
        from PIL import Image
        adapter=load_pytorch_adapter(model_row.file_path); results=run_trigger_suite(adapter,Image.open(tmp).convert("RGB")); ev=trigger_evidence(source_node_id=model_id,results=results)
        db.add(m.Inspection(node_type="model",node_id=model_id,detector="trigger_detector"))
        stored=[]
        if ev:
            data=ev.model_dump(mode="json"); row=m.Evidence(id=m.gen_id("ev"),type=str(data["type"]),source_node_type=str(data["source_node_type"]),source_node_id=data["source_node_id"],severity=data["severity"],confidence=data["confidence"],detector=data["detector"],details=data.get("details")); db.add(row); stored.append(data)
        db.commit(); return {"model_id":model_id,"results":results,"finding":stored}
    finally: tmp.unlink(missing_ok=True)

class DeploymentRequest(BaseModel):
    model_id: str
    environment: Optional[str] = None
    config_sha256: Optional[str] = None

@app.post("/deployments")
def create_deployment(req: DeploymentRequest, db: Session = Depends(get_db)):
    if not db.get(m.MLModel,req.model_id): raise HTTPException(404,"model not found")
    row=m.Deployment(model_id=req.model_id,environment=req.environment,config_sha256=req.config_sha256); db.add(row); db.commit(); db.refresh(row)
    return {"id":row.id,"model_id":row.model_id,"environment":row.environment,"config_sha256":row.config_sha256}

@app.get("/graph")
def get_graph(db: Session = Depends(get_db)):
    return build_graph(db).to_dict()

@app.get("/graph/node-link")
def get_node_link(db: Session = Depends(get_db)):
    return build_graph(db).to_node_link()

@app.get("/investigate")
def investigate_db(db: Session = Depends(get_db)):
    from .reasoning.investigator import investigate_pipeline
    return investigate_pipeline(build_graph(db)).to_dict()

@app.get("/report")
def grounded_report(db: Session = Depends(get_db)):
    """Return a deterministic report, optionally upgraded by a loopback local LLM.

    The LLM receives only numbered facts derived from the deterministic report;
    it never changes the assurance state or hypothesis scores.
    """
    from .reasoning.investigator import investigate_pipeline
    from .reasoning.report_writer import LocalLLM, generate_grounded_report
    report = investigate_pipeline(build_graph(db)).to_dict()
    llm = None
    if os.environ.get("TRUSTX_USE_LLM", "0") == "1":
        llm = LocalLLM(os.environ.get("TRUSTX_LLM_URL", "http://127.0.0.1:11434/v1/chat/completions"), os.environ.get("TRUSTX_LLM_MODEL", "llama3.1:8b"))
    return {"investigation": report, "report": generate_grounded_report(report, llm)}

@app.get("/demo/graph")
def demo_graph():
    return build_demo_graph().to_dict()

@app.get("/demo/investigate")
def demo_investigate():
    from .reasoning.investigator import investigate_pipeline
    return investigate_pipeline(build_demo_graph()).to_dict()

@app.post("/evidence")
def submit_evidence(ev: EvidenceRecord, db: Session = Depends(get_db)):
    source_type=str(ev.source_node_type); source_id=ev.source_node_id
    source_models = {
        "contributor": m.Contributor, "dataset": m.Dataset, "batch": m.Batch, "sample": m.Sample,
        "training_run": m.TrainingRun, "model": m.MLModel, "deployment": m.Deployment, "inference": m.Inference,
    }
    model_cls = source_models.get(source_type)
    if model_cls is None:
        raise HTTPException(400, f"unsupported persisted source_node_type: {source_type}")
    if db.get(model_cls, source_id) is None:
        raise HTTPException(404, f"source node not found: {source_type}/{source_id}")
    row=m.Evidence(id=ev.id or m.gen_id("ev"),type=str(ev.type),source_node_type=source_type,source_node_id=source_id,severity=ev.severity,confidence=ev.confidence,detector=ev.detector,details=ev.details)
    db.add(row); db.add(m.Inspection(node_type=source_type,node_id=source_id,detector=ev.detector)); db.commit(); return {"id":row.id,"stored":True}

class PassportVerifyRequest(BaseModel):
    deployment_id: str
    passport: dict[str, Any]
    image_b64: Optional[str] = None
    output: Optional[Any] = None
    now: Optional[float] = None


_PASSPORT_VERIFIERS: dict[str, tuple[tuple[str, str, str], PassportVerifier]] = {}

def _get_passport_verifier(deployment: m.Deployment) -> PassportVerifier:
    key = os.environ.get("TRUSTX_PASSPORT_KEY", "trustx-development-key").encode()
    preproc = os.environ.get("TRUSTX_PREPROC_SHA256", "")
    expected = (deployment.model.sha256 or "", preproc, deployment.config_sha256 or "")
    cached = _PASSPORT_VERIFIERS.get(deployment.id)
    if cached and cached[0] == expected:
        return cached[1]
    verifier = PassportVerifier(key, *expected)
    _PASSPORT_VERIFIERS[deployment.id] = (expected, verifier)
    return verifier

@app.post("/api/v1/inference/verify")
def verify_passport(req: PassportVerifyRequest, db: Session = Depends(get_db)):
    deployment=db.get(m.Deployment,req.deployment_id)
    if not deployment or not deployment.model: raise HTTPException(404,"deployment not found")
    if not deployment.model.sha256: raise HTTPException(409,"deployed model has no registered hash")
    verifier=_get_passport_verifier(deployment)
    p=Passport.from_dict(req.passport)
    input_bytes=base64.b64decode(req.image_b64) if req.image_b64 else None
    anomalies=verifier.verify(p,input_bytes,req.output,now=req.now)
    positive = verifier.positive_evidence(deployment.id) if not anomalies else []
    ts=datetime.fromtimestamp(p.timestamp,tz=timezone.utc)
    if db.get(m.Inference, p.inference_id) is None:
        inf=m.Inference(id=p.inference_id,deployment_id=deployment.id,input_sha256=p.input_hash,output_sha256=p.output_hash,model_sha256_at_inference=p.model_hash,preprocessing_sha256=p.preproc_hash,config_sha256=p.config_hash,sequence_number=p.sequence,nonce=p.nonce,output_json=req.output,timestamp=ts)
        db.add(inf)
    db.add(m.Inspection(node_type="inference",node_id=p.inference_id,detector="passport_verifier"))
    for x in anomalies:
        db.add(m.Evidence(id=x["id"],type=x["type"],source_node_type="inference",source_node_id=p.inference_id,severity=x["severity"],confidence=x["confidence"],detector=x["detector"],details=x.get("meta")))
    for x in positive:
        db.add(m.Evidence(id=m.gen_id("ev"),type=x["type"],source_node_type="deployment",source_node_id=deployment.id,severity=x["severity"],confidence=x["confidence"],detector=x["detector"],details=x.get("meta")))
    db.commit()
    return {"verified": not anomalies, "inference_id":p.inference_id,"anomalies":anomalies,"positive_evidence":positive}
