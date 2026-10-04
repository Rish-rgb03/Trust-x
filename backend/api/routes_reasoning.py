"""TRUST-X reasoning API.

Exposes the evidence/reasoning engine over REST:

    POST /api/v1/investigate                  -> InvestigationReport (dict)
    POST /api/v1/counterfactual/{node_id}     -> CounterfactualResult (dict)

Both endpoints take the same body: a serialised ``IntegrityGraph`` (the output
of ``IntegrityGraph.to_dict()``)::

    {
      "nodes":    [{"id": "B17", "type": "BATCH", "label": "...", "attrs": {}}, ...],
      "edges":    [{"source": "B17", "target": "TR42", "type": "TRAINED_ON"}, ...],
      "evidence": [{"evidence_id": "E001", "source_id": "B17", ...}, ...]
    }

Error contract
--------------
400  the graph payload is not a valid IntegrityGraph (unknown node/edge type,
     edge to a missing node, schema violation, cycle, bad evidence, ...)
404  (counterfactual only) ``node_id`` is not a node of the supplied graph
422  the body is not shaped like a graph at all (FastAPI/Pydantic validation)
500  unexpected failure inside the engine; details are logged, not leaked

The engine is stateless and read-only on the graph, so every request builds its
own graph from the payload. Routes are plain ``def`` (not ``async def``) because
fusion is CPU-bound; FastAPI runs them in its threadpool and the event loop
stays responsive.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from ..counterfactual.analysis import CounterfactualConfig, simulate_node_removal
from ..evidence.graph import IntegrityGraph
from ..reasoning.investigator import investigate_pipeline

logger = logging.getLogger("trustx.api.reasoning")

router = APIRouter(prefix="/api/v1", tags=["reasoning"])


# --------------------------------------------------------------------------- #
# Request model
# --------------------------------------------------------------------------- #
class GraphPayload(BaseModel):
    """JSON form of an ``IntegrityGraph`` (what ``IntegrityGraph.to_dict()`` emits).

    Items are kept as free-form dicts on purpose: the graph itself is the single
    source of truth for validation (node/edge types, schema, cycles, evidence
    ranges), and duplicating those rules here would let the two drift apart.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "nodes": [
                    {"id": "B17", "type": "BATCH", "label": "Batch B17"},
                    {"id": "TR42", "type": "TRAINING_RUN", "label": "Training Run 42"},
                ],
                "edges": [{"source": "B17", "target": "TR42", "type": "TRAINED_ON"}],
                "evidence": [{
                    "evidence_id": "E001", "source_id": "B17",
                    "evidence_type": "NEAR_DUPLICATE_CLUSTER",
                    "severity": 0.81, "confidence": 0.94,
                    "detector": "duplicate_detector",
                }],
            }
        },
    )

    nodes: list[dict[str, Any]] = Field(..., description="Lifecycle nodes (id, type, label?, attrs?, inspected_by?).")
    edges: list[dict[str, Any]] = Field(default_factory=list, description="Edges (source, target, type, attrs?).")
    evidence: list[dict[str, Any]] = Field(default_factory=list, description="Evidence attached to nodes.")

    def as_graph_dict(self) -> dict[str, Any]:
        return {"nodes": self.nodes, "edges": self.edges, "evidence": self.evidence}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _build_graph(payload: GraphPayload, strict: bool) -> IntegrityGraph:
    """Payload -> IntegrityGraph, translating construction failures into HTTP 400."""
    try:
        return IntegrityGraph.from_dict(payload.as_graph_dict(), strict=strict)
    except KeyError as exc:
        # Missing required field, edge/evidence pointing at an unknown node, ...
        msg = str(exc.args[0]) if exc.args else "missing key"
        # A bare key name (e.g. 'type') means a required field is absent.
        detail = f"missing required field '{msg}'" if msg.isidentifier() else msg
        raise HTTPException(status_code=400, detail=f"Invalid graph payload: {detail}") from exc
    except (ValueError, TypeError) as exc:
        # Covers NodeType/EdgeType errors, SchemaViolationError, CycleError,
        # out-of-range severity/confidence, duplicate evidence ids, ...
        raise HTTPException(status_code=400, detail=f"Invalid graph payload: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - last line of defence
        logger.exception("Unexpected error while building IntegrityGraph")
        raise HTTPException(status_code=500, detail="Internal error while building the graph.") from exc


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@router.post(
    "/investigate",
    summary="Investigate an AI pipeline's integrity graph",
    response_description="InvestigationReport: assurance state, risk, hypotheses, attack path, explanation.",
)
def investigate(
    payload: GraphPayload,
    strict: bool = Query(True, description="Enforce the edge-type schema when building the graph."),
) -> dict[str, Any]:
    graph = _build_graph(payload, strict)
    try:
        return investigate_pipeline(graph).to_dict()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("investigate_pipeline failed")
        raise HTTPException(status_code=500, detail="Internal error during investigation.") from exc


@router.post(
    "/counterfactual/{node_id}",
    summary="Simulate removing a node and re-run the investigation",
    response_description="CounterfactualResult: original vs ablated risk/state and a plain-language summary.",
)
def counterfactual(
    node_id: str,
    payload: GraphPayload,
    strict: bool = Query(True, description="Enforce the edge-type schema when building the graph."),
    simulate_retraining: bool = Query(
        True,
        description="Assume the model is retrained without the removed data "
                    "(only applies to data-stage targets; the assumption is stated in the result).",
    ),
) -> dict[str, Any]:
    graph = _build_graph(payload, strict)

    # Checked up front so a bad target is a clean 404, distinct from the 400s above.
    if node_id not in graph:
        raise HTTPException(status_code=404, detail=f"Node {node_id!r} not found in the supplied graph.")

    try:
        config = CounterfactualConfig(simulate_retraining=simulate_retraining)
        return simulate_node_removal(graph, node_id, config=config).to_dict()
    except HTTPException:
        raise
    except KeyError as exc:  # defensive: node vanished / inconsistent graph
        detail = exc.args[0] if exc.args else f"Node {node_id!r} not found"
        raise HTTPException(status_code=404, detail=str(detail)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("simulate_node_removal failed for node %r", node_id)
        raise HTTPException(status_code=500, detail="Internal error during counterfactual analysis.") from exc


# --------------------------------------------------------------------------- #
# Mounting in backend/main.py
# --------------------------------------------------------------------------- #
#
#   from fastapi import FastAPI
#   from fastapi.middleware.cors import CORSMiddleware
#   from backend.api.routes_reasoning import router as reasoning_router
#
#   app = FastAPI(title="TRUST-X")
#
#   # The React dev server runs on a different origin.
#   app.add_middleware(
#       CORSMiddleware,
#       allow_origins=["http://localhost:5173", "http://localhost:3000"],
#       allow_methods=["*"],
#       allow_headers=["*"],
#   )
#
#   app.include_router(reasoning_router)      # already prefixed with /api/v1
#
# Run:  uvicorn backend.main:app --reload     (interactive docs at /docs)
