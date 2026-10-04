"""TRUST-X counterfactual analysis: "what if node X had never been there?"

    simulate_node_removal(graph, "B17") -> CounterfactualResult

Design doc, sections 36-37::

    Original  (B17 included) -> Risk 91
    Counterfactual (excluded) -> Risk 24
    "Removing B17 substantially reduces the observed integrity anomalies,
     strengthening the hypothesis that B17 is implicated."

Why this is more than ``graph.without_nodes()``
-----------------------------------------------
``IntegrityGraph.without_nodes`` is purely structural: it deletes the node and
its *own* evidence. Evidence further downstream (a trigger-sensitive model, a
provenance mismatch on an output) is kept, because whether it would disappear
is an experimental question the graph cannot answer. On the demo graph a purely
structural removal of B17 leaves the pipeline at RED. This module supplies the
missing step explicitly, in two parts, and reports both so nothing is hidden:

1. **Cascade.** Removing a data object takes along what only existed because of
   it: a removed batch's samples (those not also in a surviving batch), for a
   removed CONTRIBUTOR everything they contributed, and for a removed DATASET
   everything it contains.

2. **Simulated retraining (``CounterfactualConfig.simulate_retraining``).**
   If a *data-stage* node is removed, every model trained downstream of it is
   assumed to be retrained without it, so that model's data-dependent findings
   (``retrain_sensitive_types``: behavioural deviation, trigger sensitivity)
   are cleared. Two guards keep this honest:

   * it only clears a model whose *remaining* training data carries no
     material integrity evidence of its own. If another batch is also
     suspicious, retraining without the removed one would not fix the model;
   * it never clears evidence that retraining cannot affect: artifact/hash
     mismatches, passport/replay/output-hash failures, environment changes.

   This is an assumption (the MVP does not actually retrain), so it is listed in
   ``assumptions``, named in the summary, and the result also carries the
   structural-only risk (``structural_ablated_risk``) as the conservative
   bound. Replace the simulation with a measured retrain on benchmark scenarios
   by passing the real post-retraining graph to ``investigate_pipeline`` and
   comparing, exactly as ``simulate_node_removal`` does with its graphs.

Reading the result: ``risk_delta = original_risk - ablated_risk``. Positive
means removing the node lowers risk. A large positive delta is corroborating
evidence that the node is implicated, not proof of causation (design doc, 36).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from ..evidence.fusion import FusionConfig
from ..evidence.graph import EdgeType, IntegrityGraph, NodeType
from ..reasoning.investigator import (
    AssuranceConfig,
    AssuranceState,
    InvestigationReport,
    investigate_pipeline,
)

__all__ = [
    "CounterfactualConfig",
    "CounterfactualResult",
    "simulate_node_removal",
    "DEFAULT_RETRAIN_SENSITIVE_TYPES",
]

#: Evidence that a model acquires from its training data, and would lose if it
#: were retrained without the offending data. Deliberately excludes artifact,
#: hash, structural, passport and environment evidence.
DEFAULT_RETRAIN_SENSITIVE_TYPES = frozenset({"BEHAVIORAL_DEVIATION", "TRIGGER_SENSITIVITY"})

_DATA_STAGE = "data"


# --------------------------------------------------------------------------- #
# Configuration and result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CounterfactualConfig:
    simulate_retraining: bool = True
    retrain_sensitive_types: frozenset[str] = DEFAULT_RETRAIN_SENSITIVE_TYPES
    #: remaining upstream data evidence at/above this weight (severity x
    #: confidence) blocks the retraining assumption. Matches fusion's
    #: ``check_fail_threshold``.
    residual_evidence_threshold: float = 0.15
    # wording thresholds, in risk points (0-100)
    material_delta: float = 1.0       # |delta| below this => "no material effect"
    moderate_delta: float = 10.0
    substantial_delta: float = 30.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.residual_evidence_threshold <= 1.0:
            raise ValueError("residual_evidence_threshold must be in [0, 1]")
        if not 0.0 <= self.material_delta <= self.moderate_delta <= self.substantial_delta <= 100.0:
            raise ValueError("need 0 <= material_delta <= moderate_delta <= substantial_delta <= 100")


@dataclass(frozen=True)
class CounterfactualResult:
    # ---- the schema you asked for ---- #
    target_node_id: str
    original_risk: float
    ablated_risk: float
    risk_delta: float                         # original - ablated (positive = removal helps)
    original_state: AssuranceState
    ablated_state: AssuranceState
    human_readable_summary: str
    # ---- context for the UI and for auditing the assumptions ---- #
    original_top_hypothesis: str = ""
    ablated_top_hypothesis: str = ""
    original_implicated_source: Optional[str] = None
    ablated_implicated_source: Optional[str] = None
    effect: str = "NONE"                      # SUBSTANTIAL / MODERATE / SLIGHT / NONE / INCREASED
    removed_nodes: tuple[str, ...] = ()       # target plus cascaded nodes
    removed_evidence: tuple[str, ...] = ()    # evidence that sat on removed nodes
    cleared_evidence: tuple[str, ...] = ()    # evidence assumed gone after retraining
    retained_downstream_evidence: tuple[str, ...] = ()  # model evidence that retraining would NOT clear
    assumptions: tuple[str, ...] = ()
    structural_ablated_risk: float = 0.0      # risk with NO retraining assumption
    structural_ablated_state: AssuranceState = AssuranceState.AMBER
    original_report: Optional[InvestigationReport] = field(default=None, repr=False, compare=False)
    ablated_report: Optional[InvestigationReport] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe. The two full reports are left out; use ``original_report.to_dict()``."""
        return {
            "target_node_id": self.target_node_id,
            "original_risk": self.original_risk,
            "ablated_risk": self.ablated_risk,
            "risk_delta": self.risk_delta,
            "original_state": self.original_state.value,
            "ablated_state": self.ablated_state.value,
            "effect": self.effect,
            "original_top_hypothesis": self.original_top_hypothesis,
            "ablated_top_hypothesis": self.ablated_top_hypothesis,
            "original_implicated_source": self.original_implicated_source,
            "ablated_implicated_source": self.ablated_implicated_source,
            "removed_nodes": list(self.removed_nodes),
            "removed_evidence": list(self.removed_evidence),
            "cleared_evidence": list(self.cleared_evidence),
            "retained_downstream_evidence": list(self.retained_downstream_evidence),
            "assumptions": list(self.assumptions),
            "structural_ablated_risk": self.structural_ablated_risk,
            "structural_ablated_state": self.structural_ablated_state.value,
            "human_readable_summary": self.human_readable_summary,
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


# --------------------------------------------------------------------------- #
# The simulation
# --------------------------------------------------------------------------- #
def simulate_node_removal(
    graph: IntegrityGraph,
    target_node_id: str,
    config: Optional[CounterfactualConfig] = None,
    fusion_config: Optional[FusionConfig] = None,
    assurance_config: Optional[AssuranceConfig] = None,
) -> CounterfactualResult:
    """Re-run the investigation with ``target_node_id`` excised and compare.

    Read-only on ``graph``. Raises ``KeyError`` for an unknown node.
    """
    if not isinstance(graph, IntegrityGraph):
        raise TypeError(f"simulate_node_removal expects an IntegrityGraph, got {type(graph).__name__}")
    if target_node_id not in graph:
        raise KeyError(f"unknown node {target_node_id!r}")
    cfg = config or CounterfactualConfig()

    original = investigate_pipeline(graph, fusion_config, assurance_config)

    removal = _removal_set(graph, target_node_id)
    removed_evidence = tuple(sorted(
        e for n in removal for e in graph.g.nodes[n]["evidence_ids"]))

    # Structural ablation: exactly graph.without_nodes([target]) plus the cascade.
    ablated = graph.without_nodes(removal)
    structural = investigate_pipeline(ablated, fusion_config, assurance_config)

    cleared: tuple[str, ...] = ()
    retained: tuple[str, ...] = ()
    assumptions: list[str] = []
    if cfg.simulate_retraining and graph.stage_of(target_node_id) == _DATA_STAGE:
        cleared, retained, notes = _simulate_retraining(graph, ablated, target_node_id, cfg)
        assumptions += notes
        if cleared:
            ablated = ablated.without_evidence(cleared)
    final = (investigate_pipeline(ablated, fusion_config, assurance_config)
             if cleared else structural)

    delta = round(original.risk_score - final.risk_score, 1)
    effect = _classify(delta, cfg)
    summary = _summarise(graph, target_node_id, original, final, delta, effect, cleared, assumptions)

    return CounterfactualResult(
        target_node_id=target_node_id,
        original_risk=original.risk_score,
        ablated_risk=final.risk_score,
        risk_delta=delta,
        original_state=original.assurance_state,
        ablated_state=final.assurance_state,
        human_readable_summary=summary,
        original_top_hypothesis=original.top_hypothesis,
        ablated_top_hypothesis=final.top_hypothesis,
        original_implicated_source=original.implicated_source,
        ablated_implicated_source=final.implicated_source,
        effect=effect,
        removed_nodes=tuple(sorted(removal)),
        removed_evidence=removed_evidence,
        cleared_evidence=cleared,
        retained_downstream_evidence=retained,
        assumptions=tuple(assumptions),
        structural_ablated_risk=structural.risk_score,
        structural_ablated_state=structural.assurance_state,
        original_report=original,
        ablated_report=final,
    )


# --------------------------------------------------------------------------- #
# Cascade
# --------------------------------------------------------------------------- #
def _removal_set(graph: IntegrityGraph, target: str) -> set[str]:
    """The target plus whatever existed only because of it."""
    removal = {target}
    # Removing a contributor removes what they contributed; removing a dataset
    # removes what it contains.
    cascade = {NodeType.CONTRIBUTOR: EdgeType.CONTRIBUTED, NodeType.DATASET: EdgeType.CONTAINS}
    edge_type = cascade.get(graph.node_type(target))
    if edge_type is not None:
        for _, dst, d in graph.g.out_edges(target, data=True):
            if d["edge_type"] is edge_type:
                removal.add(dst)
    # Samples that now belong to no surviving batch/dataset go too.
    changed = True
    while changed:
        changed = False
        for n in list(removal):
            for _, dst, d in graph.g.out_edges(n, data=True):
                if (d["edge_type"] is EdgeType.CONTAINS and dst not in removal
                        and graph.node_type(dst) is NodeType.SAMPLE
                        and set(graph.g.predecessors(dst)) <= removal):
                    removal.add(dst)
                    changed = True
    return removal


# --------------------------------------------------------------------------- #
# Simulated retraining
# --------------------------------------------------------------------------- #
def _simulate_retraining(original: IntegrityGraph, ablated: IntegrityGraph, target: str,
                         cfg: CounterfactualConfig) -> tuple[tuple[str, ...], tuple[str, ...], list[str]]:
    """Which model-level findings would plausibly vanish if the models trained
    downstream of ``target`` were retrained without it."""
    cleared: list[str] = []
    retained: list[str] = []
    notes: list[str] = []
    for model in original.downstream(target, [NodeType.MODEL]):
        if model not in ablated:
            continue
        sensitive = [r for r in ablated.evidence_for(model)
                     if r.evidence_type in cfg.retrain_sensitive_types]
        if not sensitive:
            continue
        label = ablated.g.nodes[model]["label"]
        residual = [r for r in ablated.evidence_for(model, include_upstream=True)
                    if r.stage == _DATA_STAGE and r.weight >= cfg.residual_evidence_threshold]
        if residual:
            retained += [r.evidence_id for r in sensitive]
            notes.append(f"{label} not assumed fixed: its remaining training data still carries "
                         f"integrity evidence ({', '.join(sorted(r.evidence_id for r in residual))})")
        else:
            cleared += [r.evidence_id for r in sensitive]
            notes.append(f"{label} assumed retrained without {target}; its data-dependent findings "
                         f"({', '.join(sorted(r.evidence_id for r in sensitive))}) are cleared")
    return tuple(sorted(cleared)), tuple(sorted(retained)), notes


# --------------------------------------------------------------------------- #
# Text
# --------------------------------------------------------------------------- #
def _classify(delta: float, cfg: CounterfactualConfig) -> str:
    if abs(delta) < cfg.material_delta:
        return "NONE"
    if delta < 0:
        return "INCREASED"
    if delta >= cfg.substantial_delta:
        return "SUBSTANTIAL"
    return "MODERATE" if delta >= cfg.moderate_delta else "SLIGHT"


def _summarise(graph: IntegrityGraph, target: str, original: InvestigationReport,
               final: InvestigationReport, delta: float, effect: str,
               cleared: tuple[str, ...], assumptions: list[str]) -> str:
    risk = f"Risk: {original.risk_score:.0f} -> {final.risk_score:.0f}"
    state = (f"; state {original.assurance_state.value} -> {final.assurance_state.value}"
             if original.assurance_state is not final.assurance_state else "")
    detail = f"({risk}{state})"
    if effect == "NONE":
        text = (f"Removing {target} had no material effect on the observed integrity anomalies "
                f"{detail}, so this simulation gives no evidence that {target} is implicated.")
    elif effect == "INCREASED":
        text = (f"Removing {target} increases the observed risk {detail}, which does not "
                f"support the hypothesis that {target} is implicated.")
    else:
        verb, tail = {
            "SUBSTANTIAL": ("substantially reduces", "strengthening the hypothesis that {t} is implicated"),
            "MODERATE": ("reduces", "partly supporting the hypothesis that {t} is implicated"),
            "SLIGHT": ("slightly reduces", "giving only weak support to the hypothesis that {t} is implicated"),
        }[effect]
        text = (f"Removing {target} {verb} the observed integrity anomalies {detail}, "
                f"{tail.format(t=target)}.")
    if cleared:
        text += (" This assumes the affected downstream models would be retrained without it "
                 "(simulated, not measured).")
    return text


if __name__ == "__main__":  # python -m backend.counterfactual.analysis
    from ..evidence.graph import build_demo_graph

    demo = build_demo_graph()
    for node in ("B17", "B03"):
        r = simulate_node_removal(demo, node)
        print(r.human_readable_summary)
        print(f"   structural-only: {r.structural_ablated_risk:.0f} ({r.structural_ablated_state.value})")
