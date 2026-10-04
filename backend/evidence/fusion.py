"""TRUST-X evidence fusion.

Turns the evidence attached to an ``IntegrityGraph`` into a *support score* for
each competing explanation of suspicious behaviour:

    CLEAN | DATASET_POISONING | MODEL_SUBSTITUTION | BACKDOOR |
    INFERENCE_TAMPERING | BENIGN_DISTRIBUTION_SHIFT

Conceptually (design doc, section 27)::

    Score(H) = sum_i w_i E_i  +  sum_ij C_ij  -  sum_k B_k

Concretely, for every hypothesis H:

    strength_i   = reliability(detector_i) * severity_i * confidence_i
    m(t, H)      = signed coefficient in EVIDENCE_MAP: how strongly evidence of
                   type t supports (+) or contradicts (-) H

    base(H)          = sum_i strength_i * max(m(type_i, H), 0)           # w_i E_i
    corroboration(H) = min( sum_{i<j on a connected lifecycle path,
                              both supporting H}
                            gain * proximity_ij * stage_factor_ij
                                 * independence_ij * sqrt(s_i(H) * s_j(H)),
                            max_ratio * base(H) )                        # C_ij
    contradiction(H) = sum_i strength_i * max(-m(type_i, H), 0)          # B_k
    context(H)       = benign-shift adjustment (below)                   # B_k

    raw(H)     = max(0, base + corroboration - contradiction + context)
    support(H) = 1 - exp(-raw / saturation)                 # in [0, 1)

Benign-shift rule. If distribution-shift indicators (OOD, environment metadata
change) are present, we look at four integrity checks: dataset integrity,
model hashes, trigger sensitivity and provenance. A check FAILS if evidence of
its types is present, PASSES only if no such evidence exists *and* the required
detectors actually ran, and is UNKNOWN otherwise (absence of evidence from a
detector that never looked proves nothing). With zero failures,
``scale = passes / checks`` (1.0 when everything is verified clean) and::

    context(BENIGN_DISTRIBUTION_SHIFT) = + boost_rate   * shift_signal * scale
    context(each attack hypothesis)    = - penalty_rate * shift_signal * scale

With any failed check the scale is 0: a shift that coincides with real
integrity evidence is not explained away.

CLEAN has no supporting evidence types. Its base is ``clean_prior * coverage``
(you cannot call a pipeline clean if you did not inspect it); every anomaly
contradicts it (weighted by ``clean_contradiction_gain`` so that one strong
anomaly beats the prior), as does each broken provenance link.

Scores are *support scores*, not probabilities (design doc, section 32): each
hypothesis is scored independently and the values do not sum to 1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Optional

from .graph import IntegrityGraph, NodeType

__all__ = [
    "Hypothesis",
    "ATTACK_HYPOTHESES",
    "DEFAULT_EVIDENCE_MAP",
    "DEFAULT_DETECTOR_RELIABILITY",
    "DEFAULT_INTEGRITY_CHECKS",
    "IntegrityCheck",
    "CheckStatus",
    "CheckResult",
    "FusionConfig",
    "HypothesisScore",
    "FusionResult",
    "fuse",
]


# --------------------------------------------------------------------------- #
# Hypotheses
# --------------------------------------------------------------------------- #
class Hypothesis(str, Enum):
    CLEAN = "CLEAN"
    DATASET_POISONING = "DATASET_POISONING"
    MODEL_SUBSTITUTION = "MODEL_SUBSTITUTION"
    BACKDOOR = "BACKDOOR"
    INFERENCE_TAMPERING = "INFERENCE_TAMPERING"
    BENIGN_DISTRIBUTION_SHIFT = "BENIGN_DISTRIBUTION_SHIFT"


H = Hypothesis
ATTACK_HYPOTHESES = (
    H.DATASET_POISONING,
    H.MODEL_SUBSTITUTION,
    H.BACKDOOR,
    H.INFERENCE_TAMPERING,
)

UNKNOWN_DETECTOR = "unknown"

# --------------------------------------------------------------------------- #
# Evidence type -> hypothesis coefficients
# --------------------------------------------------------------------------- #
# Positive: supports the hypothesis. Negative: contradicts it. Magnitude in
# (0, 1] = how diagnostic that evidence type is. These are expert priors; the
# benchmark (design doc, sections 29 and 50-53) is where they get tuned.
#
# Rationale in one line each:
#  * Duplicate flooding / label skew / contributor oddities point at the data,
#    and a poisoned dataset is how backdoors usually arrive (smaller weight).
#  * OOD alone is the classic false-positive: mostly "the world changed".
#  * Trigger sensitivity is the strongest backdoor signal and is incompatible
#    with a benign shift.
#  * A hash/structure mismatch points at substitution, but models get updated
#    legitimately, so it also lightly points at a backdoor and never alone.
#  * Passport / replay / output-hash failures are inference-stage tampering.
DEFAULT_EVIDENCE_MAP: dict[str, dict[Hypothesis, float]] = {
    # ---- data forensics ----
    "NEAR_DUPLICATE_CLUSTER": {
        H.DATASET_POISONING: 0.8, H.BACKDOOR: 0.3,
        H.CLEAN: -1.0, H.BENIGN_DISTRIBUTION_SHIFT: -0.3,
    },
    "LABEL_DISTRIBUTION_ANOMALY": {
        H.DATASET_POISONING: 1.0, H.BACKDOOR: 0.4,
        H.CLEAN: -1.0, H.BENIGN_DISTRIBUTION_SHIFT: -0.3,
    },
    "CONTRIBUTOR_ANOMALY": {
        H.DATASET_POISONING: 0.8, H.BACKDOOR: 0.3,
        H.CLEAN: -0.9, H.BENIGN_DISTRIBUTION_SHIFT: -0.3,
    },
    "OOD_ANOMALY": {
        H.BENIGN_DISTRIBUTION_SHIFT: 0.8, H.DATASET_POISONING: 0.25,
        H.INFERENCE_TAMPERING: 0.1, H.CLEAN: -0.6,
    },
    "TEMPORAL_ANOMALY": {
        H.DATASET_POISONING: 0.4, H.BENIGN_DISTRIBUTION_SHIFT: 0.2, H.CLEAN: -0.5,
    },
    "CLASS_IMBALANCE": {
        H.DATASET_POISONING: 0.3, H.BENIGN_DISTRIBUTION_SHIFT: 0.2, H.CLEAN: -0.4,
    },
    # ---- model forensics ----
    "BEHAVIORAL_DEVIATION": {
        H.BACKDOOR: 0.7, H.MODEL_SUBSTITUTION: 0.4, H.DATASET_POISONING: 0.4,
        H.BENIGN_DISTRIBUTION_SHIFT: 0.3, H.CLEAN: -0.8,
    },
    "BEHAVIORAL_FINGERPRINT_DEVIATION": {
        H.BACKDOOR: 0.7, H.MODEL_SUBSTITUTION: 0.4, H.DATASET_POISONING: 0.4,
        H.BENIGN_DISTRIBUTION_SHIFT: 0.3, H.CLEAN: -0.8,
    },
    "TRIGGER_SENSITIVITY": {
        H.BACKDOOR: 1.0, H.DATASET_POISONING: 0.5,
        H.BENIGN_DISTRIBUTION_SHIFT: -0.9, H.CLEAN: -1.0,
    },
    "STRUCTURAL_DEVIATION": {
        H.MODEL_SUBSTITUTION: 0.9, H.BACKDOOR: 0.1,
        H.BENIGN_DISTRIBUTION_SHIFT: -0.3, H.CLEAN: -0.8,
    },
    "STRUCTURAL_FINGERPRINT_DEVIATION": {
        H.MODEL_SUBSTITUTION: 0.9, H.BACKDOOR: 0.1,
        H.BENIGN_DISTRIBUTION_SHIFT: -0.3, H.CLEAN: -0.8,
    },
    "MODEL_ARTIFACT_MISMATCH": {
        H.MODEL_SUBSTITUTION: 1.0, H.BACKDOOR: 0.2,
        H.BENIGN_DISTRIBUTION_SHIFT: -0.5, H.CLEAN: -1.0,
    },
    # ---- inference integrity ----
    "PROVENANCE_MISMATCH": {
        H.INFERENCE_TAMPERING: 0.9, H.MODEL_SUBSTITUTION: 0.3,
        H.BENIGN_DISTRIBUTION_SHIFT: -0.6, H.CLEAN: -1.0,
    },
    "OUTPUT_HASH_MISMATCH": {
        H.INFERENCE_TAMPERING: 1.0,
        H.BENIGN_DISTRIBUTION_SHIFT: -0.6, H.CLEAN: -1.0,
    },
    "REPLAY_DETECTED": {
        H.INFERENCE_TAMPERING: 1.0,
        H.BENIGN_DISTRIBUTION_SHIFT: -0.6, H.CLEAN: -1.0,
    },
    "REPLAY_SUSPECTED": {
        H.INFERENCE_TAMPERING: 1.0,
        H.BENIGN_DISTRIBUTION_SHIFT: -0.6, H.CLEAN: -1.0,
    },
    "INPUT_HASH_MISMATCH": {H.INFERENCE_TAMPERING: 1.0, H.CLEAN: -1.0},
    "SIGNATURE_INVALID": {H.INFERENCE_TAMPERING: 1.0, H.CLEAN: -1.0},
    "SEQUENCE_GAP": {H.INFERENCE_TAMPERING: 0.8, H.CLEAN: -0.8},
    "CHAIN_BREAK": {H.INFERENCE_TAMPERING: 0.9, H.CLEAN: -0.9},
    "TIMESTAMP_SKEW": {H.INFERENCE_TAMPERING: 0.6, H.CLEAN: -0.5},
    "CONFIG_MISMATCH": {H.INFERENCE_TAMPERING: 0.8, H.MODEL_SUBSTITUTION: 0.2, H.CLEAN: -0.8},
    "MODEL_HASH_VERIFIED": {H.MODEL_SUBSTITUTION: -1.0},
    "DATA_PROVENANCE_VERIFIED": {H.DATASET_POISONING: -0.6},
    "TRIGGER_PROBE_CLEAN": {H.BACKDOOR: -0.8},
    "PASSPORT_VERIFIED": {H.INFERENCE_TAMPERING: -1.0},
    # ---- environment ----
    "ENVIRONMENT_METADATA_CHANGE": {   # e.g. camera / sensor / illumination changed
        H.BENIGN_DISTRIBUTION_SHIFT: 1.0, H.CLEAN: -0.2,
    },
    "BENIGN_ENVIRONMENT_SHIFT": {
        H.BENIGN_DISTRIBUTION_SHIFT: 0.8,
    },
}

# --------------------------------------------------------------------------- #
# Detector reliability (the w_i in the formula)
# --------------------------------------------------------------------------- #
# First five are the example precisions from design doc section 29. The rest
# are placeholders. Replace ALL of them with measured benchmark precision, or
# pass your own dict via FusionConfig(reliability=...) / with_reliability().
DEFAULT_DETECTOR_RELIABILITY: dict[str, float] = {
    "duplicate_detector": 0.91,
    "label_detector": 0.86,
    "ood_detector": 0.72,
    "behavior_detector": 0.89,
    "trigger_detector": 0.94,
    # placeholders (cryptographic checks are near-deterministic)
    "hash_verifier": 0.99,
    "passport_verifier": 0.97,
    "structural_detector": 0.85,
    "metadata_monitor": 0.70,
    # compatibility aliases for earlier contributor branches
    "behavioral_fingerprint_detector": 0.89,
    "structural_fingerprint_detector": 0.85,
    "phash_duplicate_detector": 0.91,
    "label_distribution_detector": 0.86,
    "model_artifact_hash_checker": 0.99,
}
DEFAULT_UNKNOWN_RELIABILITY = 0.5


# --------------------------------------------------------------------------- #
# Integrity checks used by the benign-shift rule
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IntegrityCheck:
    name: str
    evidence_types: frozenset[str]          # any of these present => FAIL
    required_detectors: tuple[str, ...]     # all must have run => PASS possible


DEFAULT_INTEGRITY_CHECKS: tuple[IntegrityCheck, ...] = (
    IntegrityCheck(
        "dataset_integrity",
        frozenset({"NEAR_DUPLICATE_CLUSTER", "LABEL_DISTRIBUTION_ANOMALY",
                   "CONTRIBUTOR_ANOMALY", "TEMPORAL_ANOMALY", "CLASS_IMBALANCE"}),
        ("duplicate_detector", "label_detector"),
    ),
    IntegrityCheck(
        "model_integrity",
        frozenset({"MODEL_ARTIFACT_MISMATCH", "STRUCTURAL_DEVIATION"}),
        ("hash_verifier",),
    ),
    IntegrityCheck(
        "trigger_sensitivity",
        frozenset({"TRIGGER_SENSITIVITY"}),
        ("trigger_detector",),
    ),
    IntegrityCheck(
        "provenance",
        frozenset({"PROVENANCE_MISMATCH", "OUTPUT_HASH_MISMATCH", "INPUT_HASH_MISMATCH", "REPLAY_DETECTED", "REPLAY_SUSPECTED", "SIGNATURE_INVALID", "SEQUENCE_GAP", "CHAIN_BREAK", "TIMESTAMP_SKEW", "CONFIG_MISMATCH"}),
        ("passport_verifier",),
    ),
)

DEFAULT_SHIFT_INDICATORS = frozenset({"OOD_ANOMALY", "ENVIRONMENT_METADATA_CHANGE"})
DEFAULT_COVERAGE_TYPES = (
    NodeType.DATASET, NodeType.BATCH, NodeType.TRAINING_RUN,
    NodeType.MODEL, NodeType.DEPLOYMENT, NodeType.INFERENCE,
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FusionConfig:
    """Every tunable number in one place. Tests and the benchmark swap these."""

    evidence_map: Mapping[str, Mapping[Hypothesis, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_EVIDENCE_MAP.items()})
    reliability: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_DETECTOR_RELIABILITY))
    default_reliability: float = DEFAULT_UNKNOWN_RELIABILITY

    # corroboration (C_ij)
    corroboration_gain: float = 0.5
    cross_stage_factor: float = 1.0      # sources in different lifecycle stages
    same_stage_factor: float = 0.6
    same_detector_factor: float = 0.0    # same detector => not independent
    min_pair_proximity: float = 0.0
    max_corroboration_ratio: float = 1.0  # corroboration <= ratio * base

    # normalisation
    saturation: float = 1.5

    # CLEAN
    clean_prior: float = 2.0
    clean_contradiction_gain: float = 2.0  # anomalies count double against CLEAN, so one
                                           # strong anomaly (strength ~0.9) defeats the prior
    provenance_gap_penalty: float = 0.25
    coverage_node_types: tuple[NodeType, ...] = DEFAULT_COVERAGE_TYPES

    # benign-shift context rule
    shift_indicator_types: frozenset[str] = DEFAULT_SHIFT_INDICATORS
    integrity_checks: tuple[IntegrityCheck, ...] = DEFAULT_INTEGRITY_CHECKS
    check_fail_threshold: float = 0.15   # raw severity*confidence below this is noise
    benign_boost_rate: float = 1.0
    attack_penalty_rate: float = 1.0

    # human-facing labels
    level_high: float = 0.6
    level_medium: float = 0.3

    def __post_init__(self) -> None:
        for det, r in self.reliability.items():
            if not 0.0 <= r <= 1.0:
                raise ValueError(f"reliability[{det!r}] must be in [0, 1], got {r}")
        if not 0.0 <= self.default_reliability <= 1.0:
            raise ValueError("default_reliability must be in [0, 1]")
        for etype, row in self.evidence_map.items():
            for hyp, coef in row.items():
                if not isinstance(hyp, Hypothesis):
                    raise ValueError(f"evidence_map[{etype!r}] key {hyp!r} is not a Hypothesis")
                if not -1.0 <= coef <= 1.0:
                    raise ValueError(f"evidence_map[{etype!r}][{hyp.value}] must be in [-1, 1], got {coef}")
        for name in ("same_stage_factor", "cross_stage_factor", "same_detector_factor",
                     "min_pair_proximity"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("corroboration_gain", "max_corroboration_ratio", "clean_prior",
                     "clean_contradiction_gain",
                     "provenance_gap_penalty", "check_fail_threshold",
                     "benign_boost_rate", "attack_penalty_rate"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.saturation <= 0:
            raise ValueError("saturation must be > 0")

    def reliability_of(self, detector: str) -> float:
        return self.reliability.get(detector, self.default_reliability)

    def coefficient(self, evidence_type: str, hypothesis: Hypothesis) -> float:
        return self.evidence_map.get(evidence_type, {}).get(hypothesis, 0.0)

    def with_reliability(self, overrides: Mapping[str, float]) -> "FusionConfig":
        """Copy with some detector reliabilities replaced (the mocking hook)."""
        return replace(self, reliability={**self.reliability, **dict(overrides)})

    def with_mapping(self, evidence_type: str,
                     coefficients: Mapping[Hypothesis | str, float]) -> "FusionConfig":
        """Copy with one evidence type's row replaced or added."""
        row = {Hypothesis(h): c for h, c in coefficients.items()}
        return replace(self, evidence_map={**self.evidence_map, evidence_type: row})


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    evidence_ids: tuple[str, ...] = ()       # what made it FAIL
    missing_detectors: tuple[str, ...] = ()  # why it is UNKNOWN


@dataclass(frozen=True)
class HypothesisScore:
    hypothesis: Hypothesis
    support: float                  # 0..1, saturating transform of raw
    raw: float
    base: float
    corroboration: float
    contradiction: float
    context_adjustment: float       # signed: + boost for benign shift, - penalty for attacks
    level: str                      # HIGH / MEDIUM / LOW
    supporting_evidence: tuple[str, ...]
    contradicting_evidence: tuple[str, ...]
    corroborating_pairs: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        r = lambda x: round(x, 4)  # noqa: E731
        return {
            "hypothesis": self.hypothesis.value, "support": r(self.support), "raw": r(self.raw),
            "base": r(self.base), "corroboration": r(self.corroboration),
            "contradiction": r(self.contradiction),
            "context_adjustment": r(self.context_adjustment), "level": self.level,
            "supporting_evidence": list(self.supporting_evidence),
            "contradicting_evidence": list(self.contradicting_evidence),
            "corroborating_pairs": [list(p) for p in self.corroborating_pairs],
        }


@dataclass(frozen=True)
class FusionResult:
    scores: Mapping[Hypothesis, HypothesisScore]
    checks: Mapping[str, CheckResult]
    shift_signal: float               # total strength of distribution-shift evidence
    benign_context_scale: float       # 0..1, how fully the shift is "explained away"
    evidence_coverage: float          # share of key nodes some detector inspected
    provenance_gaps: tuple[tuple[str, str], ...]
    unmapped_evidence: tuple[str, ...]  # evidence types with no row in the map (ignored)

    def ranked(self) -> list[HypothesisScore]:
        """Highest support first; ties broken by raw score, then enum order."""
        order = list(Hypothesis)
        return sorted(self.scores.values(),
                      key=lambda s: (-s.support, -s.raw, order.index(s.hypothesis)))

    @property
    def inconclusive(self) -> bool:
        """True when nothing is supported at all (e.g. nothing was inspected)."""
        return all(s.raw <= 0.0 for s in self.scores.values())

    @property
    def top(self) -> Optional[HypothesisScore]:
        """Most supported hypothesis, or None if the result is inconclusive."""
        return None if self.inconclusive else self.ranked()[0]

    @property
    def margin(self) -> float:
        """Support gap between the best and second-best hypothesis."""
        r = self.ranked()
        return r[0].support - r[1].support

    def support_dict(self) -> dict[str, float]:
        return {s.hypothesis.value: s.support for s in self.ranked()}

    def shares(self) -> dict[str, float]:
        """Relative share of raw score among hypotheses (sums to 1; 0s if none).
        A convenience for bar charts; NOT a probability."""
        total = sum(s.raw for s in self.scores.values())
        return {s.hypothesis.value: (s.raw / total if total else 0.0) for s in self.ranked()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranking": [s.to_dict() for s in self.ranked()],
            "top": self.top.hypothesis.value if self.top else None,
            "inconclusive": self.inconclusive,
            "margin": round(self.margin, 4),
            "checks": {n: {"status": c.status.value, "evidence_ids": list(c.evidence_ids),
                           "missing_detectors": list(c.missing_detectors)}
                       for n, c in self.checks.items()},
            "shift_signal": round(self.shift_signal, 4),
            "benign_context_scale": round(self.benign_context_scale, 4),
            "evidence_coverage": round(self.evidence_coverage, 4),
            "provenance_gaps": [list(g) for g in self.provenance_gaps],
            "unmapped_evidence": list(self.unmapped_evidence),
        }


# --------------------------------------------------------------------------- #
# The fusion
# --------------------------------------------------------------------------- #
def fuse(graph: IntegrityGraph, config: Optional[FusionConfig] = None) -> FusionResult:
    """Score every hypothesis from the evidence in ``graph``."""
    cfg = config or FusionConfig()
    records = graph.all_evidence()
    by_id = {r.evidence_id: r for r in records}
    strength = {r.evidence_id: cfg.reliability_of(r.detector) * r.weight for r in records}

    unmapped = tuple(sorted({r.evidence_type for r in records
                             if r.evidence_type not in cfg.evidence_map}))

    # ---- base support (w_i E_i) and contradiction (part of B_k) ----------- #
    base = {h: 0.0 for h in Hypothesis}
    contra = {h: 0.0 for h in Hypothesis}
    supporters: dict[Hypothesis, list[str]] = {h: [] for h in Hypothesis}
    contradictors: dict[Hypothesis, list[str]] = {h: [] for h in Hypothesis}
    for r in records:
        for h in Hypothesis:
            m = cfg.coefficient(r.evidence_type, h)
            if m > 0:
                base[h] += strength[r.evidence_id] * m
                supporters[h].append(r.evidence_id)
            elif m < 0:
                contra[h] += strength[r.evidence_id] * -m
                contradictors[h].append(r.evidence_id)

    # ---- corroboration (C_ij) -------------------------------------------- #
    corro = {h: 0.0 for h in Hypothesis}
    corro_pairs: dict[Hypothesis, list[tuple[str, str]]] = {h: [] for h in Hypothesis}
    for pair in graph.correlation_pairs(min_proximity=cfg.min_pair_proximity):
        ra, rb = by_id[pair.evidence_a], by_id[pair.evidence_b]
        same_detector = (pair.same_detector and ra.detector != UNKNOWN_DETECTOR)
        factor = cfg.cross_stage_factor if pair.cross_stage else cfg.same_stage_factor
        if same_detector:
            factor *= cfg.same_detector_factor
        if factor <= 0:
            continue
        for h in Hypothesis:
            ma, mb = cfg.coefficient(ra.evidence_type, h), cfg.coefficient(rb.evidence_type, h)
            if ma <= 0 or mb <= 0:
                continue  # both signals must support the same hypothesis
            sa, sb = strength[ra.evidence_id] * ma, strength[rb.evidence_id] * mb
            bonus = cfg.corroboration_gain * pair.proximity * factor * math.sqrt(sa * sb)
            if bonus > 0:
                corro[h] += bonus
                corro_pairs[h].append((ra.evidence_id, rb.evidence_id))
    for h in Hypothesis:  # many weak pairs must not outweigh the evidence itself
        corro[h] = min(corro[h], cfg.max_corroboration_ratio * base[h])

    # ---- CLEAN: absence of evidence only counts where we looked ------------ #
    coverage = graph.inspection_coverage(cfg.coverage_node_types)
    gaps = tuple(graph.provenance_gaps())
    base[Hypothesis.CLEAN] = cfg.clean_prior * coverage
    contra[Hypothesis.CLEAN] = (cfg.clean_contradiction_gain * contra[Hypothesis.CLEAN]
                                + cfg.provenance_gap_penalty * len(gaps))

    # ---- benign-shift context (rest of B_k) ------------------------------- #
    checks = _evaluate_checks(graph, records, cfg)
    shift_signal = sum(strength[r.evidence_id] for r in records
                       if r.evidence_type in cfg.shift_indicator_types)
    fails = sum(c.status is CheckStatus.FAIL for c in checks.values())
    passes = sum(c.status is CheckStatus.PASS for c in checks.values())
    scale = 0.0
    if shift_signal > 0 and checks and fails == 0:
        scale = passes / len(checks)
    context = {h: 0.0 for h in Hypothesis}
    context[Hypothesis.BENIGN_DISTRIBUTION_SHIFT] = cfg.benign_boost_rate * shift_signal * scale
    for h in ATTACK_HYPOTHESES:
        context[h] = -cfg.attack_penalty_rate * shift_signal * scale

    # ---- assemble ---------------------------------------------------------- #
    scores: dict[Hypothesis, HypothesisScore] = {}
    for h in Hypothesis:
        raw = max(0.0, base[h] + corro[h] - contra[h] + context[h])
        support = 1.0 - math.exp(-raw / cfg.saturation)
        scores[h] = HypothesisScore(
            hypothesis=h, support=support, raw=raw, base=base[h],
            corroboration=corro[h], contradiction=contra[h],
            context_adjustment=context[h], level=_level(support, cfg),
            supporting_evidence=tuple(sorted(supporters[h])),
            contradicting_evidence=tuple(sorted(contradictors[h])),
            corroborating_pairs=tuple(sorted(tuple(sorted(p)) for p in corro_pairs[h])),
        )
    return FusionResult(
        scores=scores, checks=checks, shift_signal=shift_signal,
        benign_context_scale=scale, evidence_coverage=coverage,
        provenance_gaps=gaps, unmapped_evidence=unmapped,
    )


def _level(support: float, cfg: FusionConfig) -> str:
    if support >= cfg.level_high:
        return "HIGH"
    return "MEDIUM" if support >= cfg.level_medium else "LOW"


def _evaluate_checks(graph: IntegrityGraph, records, cfg: FusionConfig) -> dict[str, CheckResult]:
    inspected: set[str] = set()
    for _, data in graph.g.nodes(data=True):
        inspected |= data["inspected_by"]
    results: dict[str, CheckResult] = {}
    for check in cfg.integrity_checks:
        failing = tuple(sorted(
            r.evidence_id for r in records
            if r.evidence_type in check.evidence_types and r.weight >= cfg.check_fail_threshold))
        if failing:
            results[check.name] = CheckResult(check.name, CheckStatus.FAIL, evidence_ids=failing)
            continue
        missing = tuple(d for d in check.required_detectors if d not in inspected)
        status = CheckStatus.UNKNOWN if missing else CheckStatus.PASS
        results[check.name] = CheckResult(check.name, status, missing_detectors=missing)
    return results


if __name__ == "__main__":  # python -m backend.evidence.fusion
    from .graph import build_demo_graph

    result = fuse(build_demo_graph())
    for s in result.ranked():
        print(f"{s.hypothesis.value:27s} support={s.support:.2f} {s.level:6s} "
              f"(base {s.base:.2f} + corr {s.corroboration:.2f} "
              f"- contra {s.contradiction:.2f} {s.context_adjustment:+.2f})")
    print("checks:", {n: c.status.value for n, c in result.checks.items()})
