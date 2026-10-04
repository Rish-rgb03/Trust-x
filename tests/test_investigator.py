"""Tests for backend/reasoning/investigator.py. Run with either:

    python -m unittest discover -s tests -t . -v
    pytest tests/test_investigator.py
"""
import json
import unittest

from backend.evidence.fusion import FusionConfig, Hypothesis, fuse
from backend.evidence.graph import EdgeType, IntegrityGraph, NodeType, build_demo_graph
from backend.reasoning.investigator import (
    INCONCLUSIVE,
    AssuranceConfig,
    AssuranceEngine,
    AssuranceState,
    InvestigationReport,
    _select_origin,
    investigate_pipeline,
)

N, E, H, S = NodeType, EdgeType, Hypothesis, AssuranceState

ALL_DETECTORS = ("duplicate_detector", "label_detector", "ood_detector", "behavior_detector",
                 "trigger_detector", "hash_verifier", "passport_verifier", "structural_detector")
COVERED = ("DS", "B1", "TR", "M", "D", "I")   # the node types fusion counts for coverage


def ev(eid, source, etype, severity, confidence, detector, **extra):
    return dict(evidence_id=eid, source_id=source, evidence_type=etype,
                severity=severity, confidence=confidence, detector=detector, **extra)


def pipeline() -> IntegrityGraph:
    """C -> B1 -> TR -> M -> D -> I -> O (+ DS, S1). No evidence, nothing inspected."""
    g = IntegrityGraph()
    for nid, t in (("C", N.CONTRIBUTOR), ("DS", N.DATASET), ("B1", N.BATCH), ("S1", N.SAMPLE),
                   ("TR", N.TRAINING_RUN), ("M", N.MODEL), ("D", N.DEPLOYMENT),
                   ("I", N.INFERENCE), ("O", N.OUTPUT)):
        g.add_node(nid, t)
    for a, b, et in (("C", "B1", E.CONTRIBUTED), ("DS", "B1", E.CONTAINS),
                     ("B1", "S1", E.CONTAINS), ("B1", "TR", E.TRAINED_ON),
                     ("TR", "M", E.PRODUCED), ("M", "D", E.DEPLOYED_AS),
                     ("D", "I", E.INFERRED_ON), ("I", "O", E.GENERATED)):
        g.add_edge(a, b, et)
    return g


def inspected(g: IntegrityGraph, nodes=COVERED) -> IntegrityGraph:
    """Every detector has looked at ``nodes`` (and found nothing, unless evidence is added)."""
    for n in nodes:
        for d in ALL_DETECTORS:
            g.record_inspection(n, d)
    return g


# --------------------------------------------------------------------------- #
class DemoScenarioTests(unittest.TestCase):
    """The B17 story from the design doc (section 56)."""

    @classmethod
    def setUpClass(cls):
        cls.graph = build_demo_graph()
        cls.report = investigate_pipeline(cls.graph)

    def test_headline(self):
        r = self.report
        self.assertIsInstance(r, InvestigationReport)
        self.assertIs(r.assurance_state, S.RED)
        self.assertEqual(r.assurance_state, "RED")            # str-enum: UI/JSON friendly
        self.assertEqual(r.recommended_action, "QUARANTINE")
        self.assertEqual(r.top_hypothesis, "DATASET_POISONING")
        self.assertEqual(r.implicated_source, "B17")
        self.assertEqual(r.implicated_source_label, "Batch B17")
        self.assertEqual(r.implicated_source_type, "BATCH")

    def test_scores_are_derived_from_fusion(self):
        r, f = self.report, fuse(self.graph)
        poison = f.scores[H.DATASET_POISONING]
        self.assertAlmostEqual(r.risk_score, round(poison.support * 100, 1))
        self.assertAlmostEqual(r.top_hypothesis_support, poison.support)
        self.assertAlmostEqual(r.coverage, round(f.evidence_coverage * 100, 1))
        self.assertAlmostEqual(r.coverage, 57.1)
        for v in (r.risk_score, r.confidence, r.coverage):
            self.assertTrue(0.0 <= v <= 100.0)
        self.assertGreater(r.risk_score, 60.0)

    def test_confidence_reflects_compromise_not_poisoning_vs_backdoor_tie(self):
        # Poisoning (0.88) and backdoor (0.83) are nearly tied, but both are attacks and
        # both far above CLEAN/BENIGN. A tie between siblings must not read as doubt.
        r = self.report
        self.assertGreater(r.decision_margin, 0.5)
        self.assertGreater(r.confidence, 60.0)
        # ...but 57% coverage must cost something versus full coverage.
        full = inspected(build_demo_graph(), nodes=["DS01", "TR42", "I982"])
        self.assertGreater(investigate_pipeline(full).confidence, r.confidence)

    def test_attack_path_matches_graph(self):
        r = self.report
        ap = self.graph.attack_path("B17")
        self.assertEqual(r.attack_path, ap.nodes)
        self.assertEqual(r.attack_path[0], "B17")
        self.assertEqual(r.attack_path[-1], "O982")
        for n in ("TR42", "V4", "D7", "I982"):
            self.assertIn(n, r.attack_path)
        self.assertNotIn("B03", r.attack_path)     # the clean batch is not compromised
        self.assertNotIn("C_B", r.attack_path)
        self.assertEqual(r.attack_path_edges, ap.edges)
        self.assertEqual(set(r.attack_path_evidence_nodes), {"B17", "S184", "V4", "O982"})

    def test_implicated_source_is_top_ranked_origin(self):
        r = self.report
        ranked = self.graph.candidate_origins()
        self.assertEqual(r.implicated_source, ranked[0].node_id)
        self.assertEqual(r.origin_candidates[0].node_id, "B17")
        self.assertAlmostEqual(r.origin_explained_fraction, 1.0)
        self.assertEqual(set(r.converging_evidence),
                         {"E001", "E002", "E003", "E010", "E011", "E021"})

    def test_hypothesis_ranking_exposed(self):
        r = self.report
        names = [h.hypothesis for h in r.hypotheses]
        self.assertEqual(names[0], "DATASET_POISONING")
        self.assertEqual(set(names), {h.value for h in H})
        self.assertEqual(r.hypotheses[0].level, "HIGH")
        self.assertEqual(r.integrity_checks["dataset_integrity"], "FAIL")
        self.assertEqual(r.integrity_checks["model_integrity"], "UNKNOWN")

    def test_summary_is_dynamic_and_traceable(self):
        s = self.report.human_readable_summary
        for needle in ("RED", "QUARANTINE", "dataset poisoning", "Batch B17",
                       "Training Run 42", "Model V4", "backdoor", "dataset-originated"):
            self.assertIn(needle, s)
        self.assertIn("6 evidence items from 6 detectors", s)
        self.assertIn("57%", s)                                  # coverage caveat

    def test_warnings_flag_what_was_not_verified(self):
        w = " | ".join(self.report.warnings)
        self.assertIn("model_integrity", w)                      # hash_verifier never ran
        self.assertIn("DS01", w)                                 # dataset never inspected

    def test_json_round_trip_and_spec_keys(self):
        d = json.loads(self.report.to_json())
        for key in ("assurance_state", "risk_score", "confidence", "coverage", "top_hypothesis",
                    "implicated_source", "attack_path", "human_readable_summary"):
            self.assertIn(key, d)
        self.assertEqual(d["assurance_state"], "RED")
        self.assertEqual(d["attack_path"][0], "B17")

    def test_deterministic_and_graph_untouched(self):
        g = build_demo_graph()
        before = g.to_json()
        a, b = investigate_pipeline(g), investigate_pipeline(g)
        self.assertEqual(a.to_dict(), b.to_dict())
        self.assertEqual(g.to_json(), before)


# --------------------------------------------------------------------------- #
class AssuranceStateTests(unittest.TestCase):
    def test_green_when_inspected_and_clean(self):
        r = investigate_pipeline(inspected(pipeline()))
        self.assertIs(r.assurance_state, S.GREEN)
        self.assertEqual(r.recommended_action, "ACCEPT")
        self.assertEqual(r.top_hypothesis, "CLEAN")
        self.assertEqual(r.risk_score, 0.0)
        self.assertEqual(r.coverage, 100.0)
        self.assertGreater(r.confidence, 70.0)
        self.assertIsNone(r.implicated_source)
        self.assertEqual(r.attack_path, ())
        self.assertIn("GREEN (ACCEPT)", r.human_readable_summary)

    def test_green_summary_does_not_overstate_minor_findings(self):
        g = inspected(pipeline())
        g.attach_evidence(ev("E1", "B1", "CLASS_IMBALANCE", 0.5, 0.7, "label_detector"))
        r = investigate_pipeline(g)
        self.assertIs(r.assurance_state, S.GREEN)
        self.assertIn("1 minor anomaly", r.human_readable_summary)
        self.assertNotIn("no integrity evidence", r.human_readable_summary)

    def test_uninspected_pipeline_is_amber_not_green(self):
        r = investigate_pipeline(pipeline())              # nothing inspected, no evidence
        self.assertIs(r.assurance_state, S.AMBER)
        self.assertEqual(r.top_hypothesis, INCONCLUSIVE)
        self.assertEqual(r.coverage, 0.0)
        self.assertEqual(r.confidence, 0.0)
        self.assertIn("INCONCLUSIVE", r.reason_codes)
        self.assertIn("nothing has been inspected", r.human_readable_summary)
        self.assertIsNone(r.implicated_source)

    def test_partially_inspected_clean_pipeline_is_never_green(self):
        r = investigate_pipeline(inspected(pipeline(), nodes=("DS", "B1", "TR")))
        self.assertIs(r.assurance_state, S.AMBER)
        self.assertEqual(r.top_hypothesis, "CLEAN")        # leading, but not enough looked at
        self.assertIn("LOW_COVERAGE", r.reason_codes)
        self.assertEqual(r.coverage, 50.0)

    def test_empty_graph_does_not_crash(self):
        r = investigate_pipeline(IntegrityGraph())
        self.assertIs(r.assurance_state, S.AMBER)
        self.assertEqual(r.top_hypothesis, INCONCLUSIVE)

    def test_broken_provenance_blocks_green(self):
        g = inspected(pipeline())
        g.add_node("V9", N.MODEL)
        for d in ALL_DETECTORS:
            g.record_inspection("V9", d)
        r = investigate_pipeline(g)
        self.assertIs(r.assurance_state, S.AMBER)
        self.assertIn("PROVENANCE_GAPS", r.reason_codes)
        self.assertIn("V9", r.human_readable_summary)

    def test_clean_lead_that_is_not_decisive_is_amber(self):
        g = inspected(pipeline())
        g.attach_evidence(ev("E1", "B1", "LABEL_DISTRIBUTION_ANOMALY", 0.6, 0.7, "label_detector"))
        r = investigate_pipeline(g)
        self.assertIs(r.assurance_state, S.AMBER)
        self.assertEqual(r.top_hypothesis, "CLEAN")
        self.assertIn("CLEAN_NOT_DECISIVE", r.reason_codes)

    def test_benign_shift_is_amber_and_implicates_nothing(self):
        g = inspected(pipeline())
        g.attach_many([ev("E1", "S1", "OOD_ANOMALY", 0.7, 0.85, "ood_detector"),
                       ev("E2", "O", "ENVIRONMENT_METADATA_CHANGE", 0.9, 0.9, "metadata_monitor")])
        r = investigate_pipeline(g)
        self.assertIs(r.assurance_state, S.AMBER)
        self.assertEqual(r.recommended_action, "REVIEW")
        self.assertEqual(r.top_hypothesis, "BENIGN_DISTRIBUTION_SHIFT")
        self.assertIn("BENIGN_SHIFT", r.reason_codes)
        self.assertLess(r.risk_score, 30.0)
        self.assertIsNone(r.implicated_source)
        self.assertEqual(r.attack_path, ())
        self.assertIn("benign distribution shift", r.human_readable_summary)
        self.assertIn("checks passed", r.human_readable_summary)
        self.assertIn("no quarantine", r.human_readable_summary)

    def test_shift_coinciding_with_integrity_evidence_is_not_explained_away(self):
        g = inspected(pipeline())
        g.attach_many([ev("E1", "S1", "OOD_ANOMALY", 0.7, 0.85, "ood_detector"),
                       ev("E2", "O", "ENVIRONMENT_METADATA_CHANGE", 0.9, 0.9, "metadata_monitor"),
                       ev("E3", "B1", "LABEL_DISTRIBUTION_ANOMALY", 0.8, 0.9, "label_detector")])
        r = investigate_pipeline(g)
        self.assertNotEqual(r.top_hypothesis, "BENIGN_DISTRIBUTION_SHIFT")
        self.assertIn(r.assurance_state, (S.AMBER, S.RED))
        self.assertEqual(r.implicated_source, "B1")

    def test_moderate_attack_is_amber_with_attribution(self):
        g = inspected(pipeline())
        g.attach_evidence(ev("E1", "M", "TRIGGER_SENSITIVITY", 0.9, 0.9, "trigger_detector"))
        r = investigate_pipeline(g)
        self.assertIs(r.assurance_state, S.AMBER)
        self.assertEqual(r.top_hypothesis, "BACKDOOR")
        self.assertIn("ATTACK_MODERATE", r.reason_codes)
        self.assertTrue(0.3 * 100 <= r.risk_score < 0.6 * 100)
        self.assertEqual(r.implicated_source, "M")
        self.assertEqual(r.attack_path[0], "M")
        self.assertIn("moderate", r.human_readable_summary)

    def test_low_support_attack_as_leader_is_amber(self):
        g = inspected(pipeline())
        g.attach_evidence(ev("E1", "B1", "CLASS_IMBALANCE", 0.5, 0.7, "label_detector"))
        cfg = FusionConfig(clean_prior=0.0)   # remove CLEAN's prior so the weak signal leads
        r = investigate_pipeline(g, fusion_config=cfg)
        self.assertIs(r.assurance_state, S.AMBER)
        self.assertIn("ATTACK_LOW", r.reason_codes)
        self.assertLess(r.risk_score, 30.0)

    def test_model_substitution_is_red_and_names_the_model(self):
        g = inspected(pipeline())
        g.attach_many([ev("E1", "M", "MODEL_ARTIFACT_MISMATCH", 0.95, 0.95, "hash_verifier"),
                       ev("E2", "M", "STRUCTURAL_DEVIATION", 0.8, 0.9, "structural_detector")])
        r = investigate_pipeline(g)
        self.assertIs(r.assurance_state, S.RED)
        self.assertEqual(r.top_hypothesis, "MODEL_SUBSTITUTION")
        self.assertEqual(r.implicated_source, "M")
        self.assertEqual(r.attack_path[0], "M")
        self.assertIn("artifact hash", r.human_readable_summary)

    def test_inference_tampering_is_red_and_names_inference_node(self):
        g = inspected(pipeline())
        g.attach_many([ev("E1", "O", "OUTPUT_HASH_MISMATCH", 0.95, 0.95, "passport_verifier"),
                       ev("E2", "I", "REPLAY_DETECTED", 0.9, 0.9, "passport_verifier")])
        r = investigate_pipeline(g)
        self.assertIs(r.assurance_state, S.RED)
        self.assertEqual(r.top_hypothesis, "INFERENCE_TAMPERING")
        self.assertEqual(r.implicated_source, "I")
        self.assertEqual(r.attack_path, ("I", "O"))
        self.assertIn("passport", r.human_readable_summary)

    def test_red_is_not_downgraded_by_low_coverage(self):
        g = pipeline()   # nothing inspected except what produced evidence
        g.attach_many([ev("E1", "M", "MODEL_ARTIFACT_MISMATCH", 0.95, 0.95, "hash_verifier"),
                       ev("E2", "M", "STRUCTURAL_DEVIATION", 0.9, 0.95, "structural_detector")])
        r = investigate_pipeline(g)
        self.assertIs(r.assurance_state, S.RED)
        self.assertLess(r.coverage, 30.0)
        self.assertIn("inspected", r.human_readable_summary)    # caveat is shown


# --------------------------------------------------------------------------- #
class AttributionTests(unittest.TestCase):
    def test_origin_moves_when_data_evidence_is_ablated(self):
        g = build_demo_graph().without_evidence(predicate=lambda r: r.stage == "data")
        r = investigate_pipeline(g)
        self.assertIs(r.assurance_state, S.RED)
        self.assertEqual(r.top_hypothesis, "BACKDOOR")
        self.assertEqual(r.implicated_source, "V4")       # only model/inference evidence is left
        self.assertEqual(r.attack_path[0], "V4")
        self.assertNotIn("B17", r.attack_path)

    def test_poisoning_verdict_names_a_data_node(self):
        g = build_demo_graph().without_evidence(predicate=lambda r: r.stage == "model")
        r = investigate_pipeline(g)
        self.assertEqual(r.top_hypothesis, "DATASET_POISONING")
        self.assertEqual(r.implicated_source_type, "BATCH")
        self.assertEqual(r.implicated_source, "B17")

    def test_origin_selection_respects_hypothesis_and_falls_back(self):
        # An isolated batch carries a small finding; a separate model chain carries a big one.
        # Globally the model wins; for a poisoning verdict only data nodes are admissible.
        g = IntegrityGraph()
        g.add_node("B_iso", N.BATCH)
        g.add_node("TR", N.TRAINING_RUN)
        g.add_node("M", N.MODEL)
        g.add_edge("TR", "M", E.PRODUCED)
        g.attach_many([ev("E1", "B_iso", "CLASS_IMBALANCE", 0.2, 0.5, "label_detector"),
                       ev("E2", "M", "BEHAVIORAL_DEVIATION", 0.9, 0.9, "behavior_detector")])
        everything = g.candidate_origins()
        self.assertEqual(everything[0].node_id, "M")
        loose = AssuranceConfig(origin_min_fraction=0.0)
        self.assertEqual(_select_origin(g, H.DATASET_POISONING, everything, loose).node_id, "B_iso")
        self.assertEqual(_select_origin(g, H.BACKDOOR, everything, loose).node_id, "M")
        # B_iso explains ~11% of the weight: below the default 25% floor => global ranking.
        self.assertEqual(_select_origin(g, H.DATASET_POISONING, everything, AssuranceConfig()).node_id, "M")
        self.assertIsNone(_select_origin(g, H.BACKDOOR, [], AssuranceConfig()))

    def test_counterfactual_graph_can_be_investigated(self):
        # The counterfactual engine will feed graphs from without_nodes(); they must not crash.
        cf = build_demo_graph().without_nodes(["B17"])
        r = investigate_pipeline(cf)
        self.assertIsInstance(r, InvestigationReport)
        self.assertNotIn("B17", r.attack_path)
        self.assertLess(r.risk_score, investigate_pipeline(build_demo_graph()).risk_score + 1e-9)


# --------------------------------------------------------------------------- #
class ConfigAndApiTests(unittest.TestCase):
    def test_thresholds_are_configurable(self):
        g = build_demo_graph()
        strict = investigate_pipeline(g, assurance_config=AssuranceConfig(red_support=0.95))
        self.assertIs(strict.assurance_state, S.AMBER)
        self.assertEqual(strict.implicated_source, "B17")      # still attributed
        g2 = inspected(pipeline())
        g2.attach_evidence(ev("E1", "B1", "LABEL_DISTRIBUTION_ANOMALY", 0.6, 0.7, "label_detector"))
        lax = investigate_pipeline(g2, assurance_config=AssuranceConfig(
            red_support=0.2, amber_attack_support=0.1))
        self.assertIs(lax.assurance_state, S.RED)

    def test_engine_decides_straight_from_a_fusion_result(self):
        decision = AssuranceEngine().decide(fuse(build_demo_graph()))
        self.assertIs(decision.state, S.RED)
        self.assertEqual(decision.action, "QUARANTINE")
        self.assertIs(decision.focus_hypothesis, H.DATASET_POISONING)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            AssuranceConfig(red_support=1.5)
        with self.assertRaises(ValueError):
            AssuranceConfig(red_support=0.3, amber_attack_support=0.5)
        with self.assertRaises(ValueError):
            AssuranceConfig(margin_full=0)

    def test_rejects_non_graph_input(self):
        with self.assertRaises(TypeError):
            investigate_pipeline({"nodes": []})

    def test_graph_round_trip_gives_same_verdict(self):
        g = build_demo_graph()
        again = IntegrityGraph.from_json(g.to_json())
        a, b = investigate_pipeline(g), investigate_pipeline(again)
        self.assertEqual(a.to_dict(), b.to_dict())


if __name__ == "__main__":
    unittest.main()
