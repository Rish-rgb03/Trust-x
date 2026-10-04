"""TRUST-X investigator: the final intelligence layer before the UI.

    IntegrityGraph --fuse()--> FusionResult --AssuranceEngine--> state
                   --candidate_origins()/attack_path()--> where + how
                                                       --> InvestigationReport

``investigate_pipeline(graph)`` is the single entry point. It never mutates the
graph, never raises on an empty or unlucky pipeline (that is an AMBER result,
not an error), and every number in the report can be traced to fusion output.

Assurance states (design doc, section 33)
-----------------------------------------
RED   / QUARANTINE   any attack hypothesis has support >= ``red_support`` (0.6).
GREEN / ACCEPT       CLEAN is the top hypothesis, with a margin >= ``green_min_margin``
                     over everything else, coverage >= ``green_min_coverage``, no
                     attack hypothesis worth reviewing and no broken provenance.
AMBER / REVIEW       everything else, and the reason is always spelled out in
                     ``reasons``: uninspected/inconclusive, low coverage, benign
                     shift, an attack with low/medium support, provenance gaps,
                     or a CLEAN lead that is not decisive.

RED is checked first: evidence of compromise is never downgraded because
coverage happens to be low (low coverage only lowers *confidence*).

Risk, confidence, coverage: three different things (design doc, section 33)
---------------------------------------------------------------------------
risk        100 x support of the best attack hypothesis. How concerning the
            evidence is. 0 for a clean pipeline, whatever the coverage.
coverage    100 x ``FusionResult.evidence_coverage``. How much of the lifecycle
            some detector actually looked at.
confidence  How strongly the evidence backs the *decision*, tempered by coverage::

                separation  = min(1, decision_margin / margin_full)
                strength    = w * separation + (1 - w) * leader_support
                confidence  = strength * (floor + (1 - floor) * coverage)

            ``decision_margin`` compares the leader with the rivals it must
            beat for the decision to hold, not with a sibling hypothesis:
              RED    best attack            vs best of {CLEAN, BENIGN_SHIFT}
              GREEN  CLEAN                  vs best of everything else
              AMBER  top hypothesis         vs runner-up
            This matters: dataset poisoning and a backdoor routinely score
            within a few points of each other on the same evidence, and that
            must not read as "we are unsure the pipeline is compromised".

Which node is "implicated"
--------------------------
``graph.candidate_origins()`` ranks nodes by how much of the evidence weight
they explain, most specific first. The investigator asks it for nodes that make
sense for the hypothesis being implicated (a poisoning verdict names a
contributor/dataset/batch/sample, a substitution verdict names a model, a
tampering verdict names an inference-stage node; BACKDOOR may start anywhere)
and falls back to the unfiltered ranking if the filtered best explains less
than ``origin_min_fraction`` of the evidence. Benign-shift, inconclusive and
GREEN reports deliberately implicate nothing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

from ..evidence.fusion import (
    ATTACK_HYPOTHESES,
    CheckStatus,
    FusionConfig,
    FusionResult,
    Hypothesis,
    fuse,
)
from ..evidence.graph import STAGES, IntegrityGraph, NodeType, OriginCandidate

__all__ = [
    "AssuranceState",
    "AssuranceConfig",
    "AssuranceDecision",
    "AssuranceEngine",
    "RankedHypothesis",
    "InvestigationReport",
    "investigate_pipeline",
    "INCONCLUSIVE",
]

H = Hypothesis
N = NodeType

INCONCLUSIVE = "INCONCLUSIVE"   # top_hypothesis when nothing is supported at all


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
class AssuranceState(str, Enum):
    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"


ACTIONS: Mapping[AssuranceState, str] = {
    AssuranceState.GREEN: "ACCEPT",
    AssuranceState.AMBER: "REVIEW",
    AssuranceState.RED: "QUARANTINE",
}

#: Noun phrases for sentences. Lower case, no article.
HYPOTHESIS_LABELS: Mapping[Hypothesis, str] = {
    H.CLEAN: "a clean pipeline",
    H.DATASET_POISONING: "dataset poisoning",
    H.MODEL_SUBSTITUTION: "model substitution",
    H.BACKDOOR: "a backdoor",
    H.INFERENCE_TAMPERING: "inference tampering",
    H.BENIGN_DISTRIBUTION_SHIFT: "a benign distribution shift",
}

#: Which node types can plausibly be the origin of each attack. None = anywhere.
_ORIGIN_TYPES: Mapping[Hypothesis, Optional[frozenset[NodeType]]] = {
    H.DATASET_POISONING: frozenset({N.CONTRIBUTOR, N.DATASET, N.BATCH, N.SAMPLE}),
    H.MODEL_SUBSTITUTION: frozenset({N.MODEL}),
    H.INFERENCE_TAMPERING: frozenset({N.DEPLOYMENT, N.CONFIGURATION, N.INPUT,
                                      N.INFERENCE, N.OUTPUT}),
    H.BACKDOOR: None,
}

_DATA_STAGE = "data"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AssuranceConfig:
    """Every threshold of the assurance decision. Tune from the benchmark."""

    red_support: float = 0.6            # best attack support >= this  => RED
    amber_attack_support: float = 0.3   # best attack support >= this  => at least AMBER
    green_min_coverage: float = 0.75    # need this much inspected to say GREEN
    green_min_margin: float = 0.5       # CLEAN must lead everything by this much
    # confidence
    margin_full: float = 0.5            # a decision margin this big = full separation
    margin_weight: float = 0.5          # weight of separation vs leader support
    coverage_floor: float = 0.5         # confidence factor at 0% coverage
    # origin attribution
    origin_min_fraction: float = 0.25
    top_origins: int = 5                # how many global candidates to list

    def __post_init__(self) -> None:
        unit = ("red_support", "amber_attack_support", "green_min_coverage",
                "green_min_margin", "margin_weight", "coverage_floor",
                "origin_min_fraction")
        for name in unit:
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {getattr(self, name)}")
        if self.margin_full <= 0:
            raise ValueError("margin_full must be > 0")
        if self.amber_attack_support > self.red_support:
            raise ValueError("amber_attack_support must be <= red_support")
        if self.top_origins < 0:
            raise ValueError("top_origins must be >= 0")


# --------------------------------------------------------------------------- #
# Assurance engine
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AssuranceDecision:
    state: AssuranceState
    risk_score: float                       # 0..100
    confidence: float                       # 0..100
    top_hypothesis: str                     # Hypothesis value or INCONCLUSIVE
    focus_hypothesis: Optional[Hypothesis]  # attack hypothesis to attribute, if any
    decision_margin: float
    reason_codes: tuple[str, ...]
    reasons: tuple[str, ...]

    @property
    def action(self) -> str:
        return ACTIONS[self.state]


class AssuranceEngine:
    """FusionResult -> GREEN / AMBER / RED, with the reasons written down."""

    def __init__(self, config: Optional[AssuranceConfig] = None) -> None:
        self.config = config or AssuranceConfig()

    def decide(self, fusion: FusionResult) -> AssuranceDecision:
        cfg = self.config
        scores = fusion.scores
        ranked = fusion.ranked()
        top = fusion.top                                     # None if inconclusive
        best_attack = next(s for s in ranked if s.hypothesis in ATTACK_HYPOTHESES)
        coverage = fusion.evidence_coverage
        codes: list[str] = []
        reasons: list[str] = []

        def add(code: str, text: str) -> None:
            codes.append(code)
            reasons.append(text)

        attack_name = HYPOTHESIS_LABELS[best_attack.hypothesis]

        if best_attack.support >= cfg.red_support:
            state = AssuranceState.RED
            add("ATTACK_HIGH",
                f"{attack_name} has support {best_attack.support:.2f}, at or above the "
                f"RED threshold of {cfg.red_support:.2f}")
        else:
            top_is_attack = top is not None and top.hypothesis in ATTACK_HYPOTHESES
            if best_attack.support >= cfg.amber_attack_support:
                add("ATTACK_MODERATE",
                    f"{attack_name} has moderate support ({best_attack.support:.2f}), above "
                    f"the review threshold {cfg.amber_attack_support:.2f} but below the "
                    f"RED threshold {cfg.red_support:.2f}")
            elif top_is_attack:
                add("ATTACK_LOW",
                    f"{HYPOTHESIS_LABELS[top.hypothesis]} is the leading hypothesis but "
                    f"with low support ({top.support:.2f})")
            if top is not None and top.hypothesis is H.BENIGN_DISTRIBUTION_SHIFT:
                add("BENIGN_SHIFT",
                    f"a benign distribution shift is the most supported explanation "
                    f"({top.support:.2f}); attack hypotheses are suppressed by passing "
                    f"integrity checks")
            if fusion.inconclusive:
                add("INCONCLUSIVE",
                    "no hypothesis has positive support: the pipeline is effectively "
                    "uninspected, so no conclusion can be drawn")
            elif coverage < cfg.green_min_coverage:
                add("LOW_COVERAGE",
                    f"only {coverage:.0%} of key pipeline nodes were inspected "
                    f"({cfg.green_min_coverage:.0%} needed to accept)")
            if fusion.provenance_gaps:
                node, issue = fusion.provenance_gaps[0]
                add("PROVENANCE_GAPS",
                    f"{_plural(len(fusion.provenance_gaps), 'broken provenance link')} "
                    f"(e.g. {node}: {issue})")
            if (top is not None and top.hypothesis is H.CLEAN
                    and coverage >= cfg.green_min_coverage
                    and fusion.margin < cfg.green_min_margin):
                add("CLEAN_NOT_DECISIVE",
                    f"CLEAN leads by only {fusion.margin:.2f} "
                    f"({cfg.green_min_margin:.2f} needed to accept)")
            if codes:
                state = AssuranceState.AMBER
            else:  # top is CLEAN, covered, decisive, nothing worth reviewing
                state = AssuranceState.GREEN
                minor = scores[H.CLEAN].contradicting_evidence
                tail = ("no integrity evidence against it" if not minor else
                        f"{_plural(len(minor), 'minor anomaly')} recorded "
                        f"({', '.join(minor[:5])}), none material")
                add("CLEAN_CONFIRMED",
                    f"a clean pipeline is the most supported explanation "
                    f"({top.support:.2f}, margin {fusion.margin:.2f}) with "
                    f"{coverage:.0%} inspection coverage and {tail}")

        # ---- what the decision rests on, for confidence ------------------ #
        if state is AssuranceState.RED:
            leader = best_attack.support
            rival = max(scores[H.CLEAN].support, scores[H.BENIGN_DISTRIBUTION_SHIFT].support)
        elif top is None:
            leader, rival = 0.0, 0.0
        else:
            leader = top.support
            rival = ranked[1].support
        decision_margin = max(0.0, leader - rival)
        separation = min(1.0, decision_margin / cfg.margin_full)
        strength = cfg.margin_weight * separation + (1.0 - cfg.margin_weight) * leader
        confidence = strength * (cfg.coverage_floor + (1.0 - cfg.coverage_floor) * coverage)

        focus = best_attack.hypothesis if any(c.startswith("ATTACK_") for c in codes) else None
        if state is AssuranceState.RED:
            top_hypothesis = best_attack.hypothesis.value
        else:
            top_hypothesis = top.hypothesis.value if top is not None else INCONCLUSIVE

        return AssuranceDecision(
            state=state,
            risk_score=_pct(best_attack.support),
            confidence=_pct(confidence),
            top_hypothesis=top_hypothesis,
            focus_hypothesis=focus,
            decision_margin=decision_margin,
            reason_codes=tuple(codes),
            reasons=tuple(reasons),
        )


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RankedHypothesis:
    hypothesis: str
    support: float
    level: str
    supporting_evidence: tuple[str, ...]
    contradicting_evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "support": round(self.support, 4),
            "level": self.level,
            "supporting_evidence": list(self.supporting_evidence),
            "contradicting_evidence": list(self.contradicting_evidence),
        }


@dataclass(frozen=True)
class InvestigationReport:
    # ---- the eight fields the UI headline needs ---- #
    assurance_state: AssuranceState
    risk_score: float                        # 0..100
    confidence: float                        # 0..100
    coverage: float                          # 0..100
    top_hypothesis: str                      # e.g. "DATASET_POISONING" or "INCONCLUSIVE"
    implicated_source: Optional[str]         # node id, None if nothing is implicated
    attack_path: tuple[str, ...]             # lifecycle-ordered node ids, () if none
    human_readable_summary: str
    # ---- detail for the investigation / evidence screens ---- #
    recommended_action: str = "REVIEW"       # ACCEPT / REVIEW / QUARANTINE
    top_hypothesis_support: float = 0.0      # 0..1 support score (not a probability)
    implicated_source_label: Optional[str] = None
    implicated_source_type: Optional[str] = None
    origin_explained_fraction: float = 0.0   # share of evidence weight at/below the origin
    attack_path_edges: tuple[tuple[str, str], ...] = ()
    attack_path_evidence_nodes: tuple[str, ...] = ()
    converging_evidence: tuple[str, ...] = ()   # evidence ids at/below the origin
    supporting_evidence: tuple[str, ...] = ()   # evidence ids behind top_hypothesis
    hypotheses: tuple[RankedHypothesis, ...] = ()
    origin_candidates: tuple[OriginCandidate, ...] = ()
    integrity_checks: Mapping[str, str] = field(default_factory=dict)
    decision_margin: float = 0.0
    reason_codes: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    fusion: Optional[FusionResult] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict. ``fusion`` is left out: call ``report.fusion.to_dict()``."""
        return {
            "assurance_state": self.assurance_state.value,
            "recommended_action": self.recommended_action,
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "coverage": self.coverage,
            "top_hypothesis": self.top_hypothesis,
            "top_hypothesis_support": round(self.top_hypothesis_support, 4),
            "implicated_source": self.implicated_source,
            "implicated_source_label": self.implicated_source_label,
            "implicated_source_type": self.implicated_source_type,
            "origin_explained_fraction": round(self.origin_explained_fraction, 4),
            "attack_path": list(self.attack_path),
            "attack_path_edges": [list(e) for e in self.attack_path_edges],
            "attack_path_evidence_nodes": list(self.attack_path_evidence_nodes),
            "converging_evidence": list(self.converging_evidence),
            "supporting_evidence": list(self.supporting_evidence),
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "origin_candidates": [
                {"node_id": c.node_id, "node_type": c.node_type.value, "rank": c.rank,
                 "explained_fraction": round(c.explained_fraction, 4),
                 "stages": list(c.stages), "evidence_ids": list(c.evidence_ids)}
                for c in self.origin_candidates],
            "integrity_checks": dict(self.integrity_checks),
            "decision_margin": round(self.decision_margin, 4),
            "reason_codes": list(self.reason_codes),
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "human_readable_summary": self.human_readable_summary,
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


# --------------------------------------------------------------------------- #
# The investigation
# --------------------------------------------------------------------------- #
def investigate_pipeline(
    graph: IntegrityGraph,
    fusion_config: Optional[FusionConfig] = None,
    assurance_config: Optional[AssuranceConfig] = None,
) -> InvestigationReport:
    """Fuse the evidence in ``graph``, decide the assurance state, attribute the
    anomaly to its most specific origin and explain it. Read-only on ``graph``."""
    if not isinstance(graph, IntegrityGraph):
        raise TypeError(f"investigate_pipeline expects an IntegrityGraph, got {type(graph).__name__}")
    fcfg = fusion_config or FusionConfig()
    fusion = fuse(graph, fcfg)
    engine = AssuranceEngine(assurance_config)
    cfg = engine.config
    decision = engine.decide(fusion)

    all_origins = graph.candidate_origins()
    origin: Optional[OriginCandidate] = None
    if decision.focus_hypothesis is not None:
        origin = _select_origin(graph, decision.focus_hypothesis, all_origins, cfg)
    path = graph.attack_path(origin.node_id) if origin else None

    recommendation = _recommend(graph, decision, origin)
    summary = _summarise(graph, fusion, decision, origin, recommendation, cfg)

    top_h = Hypothesis(decision.top_hypothesis) if decision.top_hypothesis != INCONCLUSIVE else None
    top_score = fusion.scores[top_h] if top_h else None

    return InvestigationReport(
        assurance_state=decision.state,
        risk_score=decision.risk_score,
        confidence=decision.confidence,
        coverage=_pct(fusion.evidence_coverage),
        top_hypothesis=decision.top_hypothesis,
        implicated_source=origin.node_id if origin else None,
        attack_path=path.nodes if path else (),
        human_readable_summary=summary,
        recommended_action=decision.action,
        top_hypothesis_support=top_score.support if top_score else 0.0,
        implicated_source_label=_label(graph, origin.node_id) if origin else None,
        implicated_source_type=origin.node_type.value if origin else None,
        origin_explained_fraction=origin.explained_fraction if origin else 0.0,
        attack_path_edges=path.edges if path else (),
        attack_path_evidence_nodes=path.evidence_nodes if path else (),
        converging_evidence=origin.evidence_ids if origin else (),
        supporting_evidence=top_score.supporting_evidence if top_score else (),
        hypotheses=tuple(
            RankedHypothesis(s.hypothesis.value, s.support, s.level,
                             s.supporting_evidence, s.contradicting_evidence)
            for s in fusion.ranked()),
        origin_candidates=tuple(all_origins[:cfg.top_origins]),
        integrity_checks={n: c.status.value for n, c in fusion.checks.items()},
        decision_margin=decision.decision_margin,
        reason_codes=decision.reason_codes,
        reasons=decision.reasons,
        warnings=_warnings(graph, fusion, fcfg),
        fusion=fusion,
    )


# --------------------------------------------------------------------------- #
# Origin attribution
# --------------------------------------------------------------------------- #
def _select_origin(graph: IntegrityGraph, hypothesis: Hypothesis,
                   all_origins: list[OriginCandidate],
                   cfg: AssuranceConfig) -> Optional[OriginCandidate]:
    types = _ORIGIN_TYPES.get(hypothesis)
    if types is not None:
        scoped = graph.candidate_origins(node_types=types)
        if scoped and scoped[0].explained_fraction >= cfg.origin_min_fraction:
            return scoped[0]
    return all_origins[0] if all_origins else None


# --------------------------------------------------------------------------- #
# Text
# --------------------------------------------------------------------------- #
def _pct(x: float) -> float:
    return round(min(1.0, max(0.0, x)) * 100.0, 1)


def _label(graph: IntegrityGraph, node_id: str) -> str:
    return graph.g.nodes[node_id]["label"]


def _join(items: list[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _recommend(graph: IntegrityGraph, d: AssuranceDecision,
               origin: Optional[OriginCandidate]) -> str:
    """One imperative sentence fragment (no trailing full stop)."""
    codes = set(d.reason_codes)
    if d.state is AssuranceState.GREEN:
        return "accept the pipeline and continue routine monitoring"
    red = d.state is AssuranceState.RED

    if d.focus_hypothesis is not None and origin is not None:
        o = origin.node_id
        name = _label(graph, o)
        runs = [_label(graph, n) for n in graph.downstream(o, [N.TRAINING_RUN])]
        models = [_label(graph, n) for n in graph.downstream(o, [N.MODEL])]
        if origin.node_type is N.MODEL:
            models = [name]
        hyp = d.focus_hypothesis
        if hyp is H.MODEL_SUBSTITUTION:
            target = _join(models) or name
            return (f"{'do not deploy' if red else 'hold deployment of'} {target} and verify its "
                    f"artifact hash against the authorized registry; re-provision from a trusted source")
        if hyp is H.INFERENCE_TAMPERING:
            return (f"{'isolate' if red else 'inspect'} {name}, then audit the prediction "
                    f"passports, signing keys and deployment configuration behind it")
        # DATASET_POISONING / BACKDOOR
        if graph.stage_of(o) == _DATA_STAGE:
            if red:
                parts = [f"quarantine {name}"]
                if runs:
                    parts.append(f"review {_join(runs)}")
            else:
                parts = [f"review {_join([name] + runs)}"]
            if models:
                parts.append(f"{'do not deploy' if red else 'hold deployment of'} {_join(models)}")
            return "; ".join(parts)
        target = _join(models) or name
        return (f"{'do not deploy' if red else 'hold deployment of'} {target}; "
                f"trace and audit the training data that produced it")

    if "BENIGN_SHIFT" in codes:
        return ("confirm the environment change (sensor, lighting, season), then re-validate and "
                "if needed recalibrate or fine-tune the model; no quarantine is indicated")
    if "INCONCLUSIVE" in codes or "LOW_COVERAGE" in codes:
        return "run the missing detectors on the uninspected stages and re-run the investigation before deployment"
    if "PROVENANCE_GAPS" in codes:
        return "restore the missing provenance records before relying on this pipeline"
    return "gather more evidence before deciding; the clean verdict is not yet decisive"


def _summarise(graph: IntegrityGraph, fusion: FusionResult, d: AssuranceDecision,
               origin: Optional[OriginCandidate], recommendation: str,
               cfg: AssuranceConfig) -> str:
    head = f"{d.state.value} ({d.action})."
    action = f"Recommended action: {recommendation}."
    coverage = fusion.evidence_coverage
    caveat = (f" Note: only {coverage:.0%} of key pipeline nodes were inspected."
              if 0 < coverage < cfg.green_min_coverage and d.state is not AssuranceState.GREEN
              and "LOW_COVERAGE" not in d.reason_codes else "")

    if d.state is AssuranceState.GREEN:
        return f"{head} {d.reasons[0][0].upper() + d.reasons[0][1:]}. {action}"

    if d.focus_hypothesis is not None:
        focus = fusion.scores[d.focus_hypothesis]
        sentence = (f"Most supported explanation: {HYPOTHESIS_LABELS[focus.hypothesis]} "
                    f"(support {focus.support:.2f})"
                    if d.state is AssuranceState.RED else
                    f"{HYPOTHESIS_LABELS[focus.hypothesis].capitalize()} has "
                    f"{'moderate' if 'ATTACK_MODERATE' in d.reason_codes else 'low'} "
                    f"support ({focus.support:.2f}), not enough to quarantine but enough to review")
        # Other strongly supported attacks, e.g. the backdoor that rides on poisoned data.
        related = [s for s in fusion.ranked()
                   if s.hypothesis in ATTACK_HYPOTHESES and s.hypothesis is not focus.hypothesis
                   and s.level == "HIGH"]
        if related and d.state is AssuranceState.RED:
            names = _join([f"{HYPOTHESIS_LABELS[s.hypothesis]} ({s.support:.2f})" for s in related])
            sentence += f"; {names} {'is' if len(related) == 1 else 'are'} also strongly supported"
            if (origin is not None and graph.stage_of(origin.node_id) == _DATA_STAGE
                    and {focus.hypothesis, related[0].hypothesis} == {H.DATASET_POISONING, H.BACKDOOR}):
                sentence += ", consistent with a dataset-originated compromise"
        sentence += "."
        if origin is not None:
            n_ev = len(origin.evidence_ids)
            detectors = {graph.get_evidence(e).detector for e in origin.evidence_ids}
            sentence += (f" {_plural(n_ev, 'evidence item')} from {_plural(len(detectors), 'detector')} "
                         f"across the {_join(list(origin.stages))} "
                         f"{'stage' if len(origin.stages) == 1 else 'stages'} "
                         f"{'converges' if n_ev == 1 else 'converge'} on "
                         f"{_label(graph, origin.node_id)} "
                         f"({origin.explained_fraction:.0%} of observed anomaly weight).")
        return f"{head} {sentence} {action}{caveat}"

    # AMBER without an attack to attribute
    if "BENIGN_SHIFT" in d.reason_codes:
        passed = [n.replace("_", " ") for n, c in fusion.checks.items() if c.status is CheckStatus.PASS]
        unknown = [n.replace("_", " ") for n, c in fusion.checks.items() if c.status is CheckStatus.UNKNOWN]
        s = (f"Observed anomalies are best explained by a benign distribution shift "
             f"(support {fusion.scores[H.BENIGN_DISTRIBUTION_SHIFT].support:.2f})")
        if passed:
            s += f"; {_join(passed)} checks passed"
        if unknown:
            s += f", but {_join(unknown)} could not be verified"
        return f"{head} {s}. {action}{caveat}"
    if "INCONCLUSIVE" in d.reason_codes:
        return (f"{head} Insufficient evidence to assess this pipeline: nothing has been inspected "
                f"({coverage:.0%} coverage), so absence of anomalies proves nothing. {action}")
    why = "; ".join(d.reasons)
    return f"{head} {why[0].upper() + why[1:]}. {action}{caveat}"


def _warnings(graph: IntegrityGraph, fusion: FusionResult, fcfg: FusionConfig) -> tuple[str, ...]:
    out: list[str] = []
    for node, issue in fusion.provenance_gaps:
        out.append(f"provenance gap at {node}: {issue}")
    for name, check in fusion.checks.items():
        if check.status is CheckStatus.UNKNOWN:
            out.append(f"integrity check '{name}' is unverified (detectors not run: "
                       f"{', '.join(check.missing_detectors)})")
    for ntype in fcfg.coverage_node_types:
        nodes = graph.nodes_of_type(ntype)
        if not nodes:
            continue
        missing = [n for n in nodes
                   if not graph.g.nodes[n]["inspected_by"] and not graph.g.nodes[n]["evidence_ids"]]
        if missing:
            shown = ", ".join(missing[:5]) + (f" (+{len(missing) - 5} more)" if len(missing) > 5 else "")
            out.append(f"{len(missing)}/{len(nodes)} {ntype.value} nodes not inspected: {shown}")
    present = set(graph.stages_present())
    for stage in STAGES:
        if stage not in present:
            out.append(f"no {stage}-stage nodes registered; that stage was not assessed")
    if fusion.unmapped_evidence:
        out.append("evidence types ignored by fusion (no evidence-map row): "
                   + ", ".join(fusion.unmapped_evidence))
    return tuple(out)


if __name__ == "__main__":  # python -m backend.reasoning.investigator
    from ..evidence.graph import build_demo_graph

    report = investigate_pipeline(build_demo_graph())
    print(report.human_readable_summary)
    print(json.dumps({k: v for k, v in report.to_dict().items()
                      if k not in ("hypotheses", "origin_candidates", "human_readable_summary")},
                     indent=2))
