"""Tests for backend/evidence/fusion.py. Run with either:

    python -m unittest discover -s tests -t . -v
    pytest tests/test_fusion.py
"""
import json
import math
import unittest

from backend.evidence.fusion import (
    ATTACK_HYPOTHESES,
    DEFAULT_EVIDENCE_MAP,
    CheckStatus,
    FusionConfig,
    Hypothesis,
    fuse,
)
from backend.evidence.graph import IntegrityGraph, NodeType, EdgeType, build_demo_graph

H = Hypothesis
N, E = NodeType, EdgeType


def ev(eid, source, etype, severity, confidence, detector, **extra):
    return dict(evidence_id=eid, source_id=source, evidence_type=etype,
                severity=severity, confidence=confidence, detector=detector, **extra)


def small_pipeline() -> IntegrityGraph:
    """C -> B1 -> TR -> M -> D -> I -> O, nothing attached yet."""
    g = IntegrityGraph()
    g.add_node("C", N.CONTRIBUTOR)
    g.add_node("DS", N.DATASET)
    g.add_node("B1", N.BATCH)
    g.add_node("TR", N.TRAINING_RUN)
    g.add_node("M", N.MODEL)
    g.add_node("D", N.DEPLOYMENT)
    g.add_node("I", N.INFERENCE)
    g.add_node("O", N.OUTPUT)
    g.add_edge("C", "B1", E.CONTRIBUTED)
    g.add_edge("DS", "B1", E.CONTAINS)
    g.add_edge("B1", "TR", E.TRAINED_ON)
    g.add_edge("TR", "M", E.PRODUCED)
    g.add_edge("M", "D", E.DEPLOYED_AS)
    g.add_edge("D", "I", E.INFERRED_ON)
    g.add_edge("I", "O", E.GENERATED)
    return g


ALL_DETECTORS_RAN = [
    ("B1", "duplicate_detector"), ("B1", "label_detector"),
    ("M", "hash_verifier"), ("M", "trigger_detector"), ("I", "passport_verifier"),
]


def inspect_everything(g: IntegrityGraph) -> IntegrityGraph:
    for node, det in ALL_DETECTORS_RAN:
        g.record_inspection(node, det)
    for n in ("DS", "TR", "D"):
        g.record_inspection(n, "coverage_filler")
    return g


# --------------------------------------------------------------------------- #
class VocabularyTests(unittest.TestCase):
    def test_six_hypotheses(self):
        self.assertEqual({h.value for h in Hypothesis}, {
            "CLEAN", "DATASET_POISONING", "MODEL_SUBSTITUTION", "BACKDOOR",
            "INFERENCE_TAMPERING", "BENIGN_DISTRIBUTION_SHIFT"})
        self.assertEqual(set(ATTACK_HYPOTHESES), {
            H.DATASET_POISONING, H.MODEL_SUBSTITUTION, H.BACKDOOR, H.INFERENCE_TAMPERING})

    def test_map_covers_the_standard_evidence_types(self):
        for t in ("NEAR_DUPLICATE_CLUSTER", "LABEL_DISTRIBUTION_ANOMALY", "OOD_ANOMALY",
                  "BEHAVIORAL_DEVIATION", "TRIGGER_SENSITIVITY", "PROVENANCE_MISMATCH",
                  "MODEL_ARTIFACT_MISMATCH"):
            self.assertIn(t, DEFAULT_EVIDENCE_MAP)

    def test_map_is_well_formed_and_every_hypothesis_is_reachable(self):
        supported, contradicted = set(), set()
        for row in DEFAULT_EVIDENCE_MAP.values():
            for h, coef in row.items():
                self.assertIsInstance(h, Hypothesis)
                self.assertTrue(-1.0 <= coef <= 1.0 and coef != 0)
                (supported if coef > 0 else contradicted).add(h)
        # Everything except CLEAN can be supported by evidence; CLEAN can only be contradicted.
        self.assertEqual(supported, set(Hypothesis) - {H.CLEAN})
        self.assertIn(H.CLEAN, contradicted)
        # The design-doc distinctions that matter most:
        self.assertGreater(DEFAULT_EVIDENCE_MAP["TRIGGER_SENSITIVITY"][H.BACKDOOR], 0)
        self.assertLess(DEFAULT_EVIDENCE_MAP["TRIGGER_SENSITIVITY"][H.BENIGN_DISTRIBUTION_SHIFT], 0)
        self.assertGreater(DEFAULT_EVIDENCE_MAP["OOD_ANOMALY"][H.BENIGN_DISTRIBUTION_SHIFT],
                           DEFAULT_EVIDENCE_MAP["OOD_ANOMALY"][H.DATASET_POISONING])

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            FusionConfig(reliability={"x": 1.5})
        with self.assertRaises(ValueError):
            FusionConfig(evidence_map={"T": {H.BACKDOOR: 2.0}})
        with self.assertRaises(ValueError):
            FusionConfig(saturation=0)
        with self.assertRaises(ValueError):
            FusionConfig(evidence_map={"T": {"BACKDOOR": 0.5}})


# --------------------------------------------------------------------------- #
class BaseSupportTests(unittest.TestCase):
    def test_base_support_matches_hand_calculation(self):
        g = small_pipeline()
        g.attach_evidence(ev("E1", "B1", "NEAR_DUPLICATE_CLUSTER", 0.8, 0.5, "duplicate_detector"))
        cfg = FusionConfig()
        strength = 0.91 * 0.8 * 0.5                      # reliability * severity * confidence
        res = fuse(g, cfg)
        dp = res.scores[H.DATASET_POISONING]
        self.assertAlmostEqual(dp.base, strength * 0.8)  # coefficient 0.8
        self.assertAlmostEqual(res.scores[H.BACKDOOR].base, strength * 0.3)
        self.assertEqual(dp.corroboration, 0.0)          # a single item cannot corroborate itself
        self.assertAlmostEqual(dp.support, 1 - math.exp(-dp.raw / cfg.saturation))
        self.assertEqual(dp.supporting_evidence, ("E1",))
        self.assertEqual(res.scores[H.CLEAN].contradicting_evidence, ("E1",))
        self.assertEqual(res.scores[H.MODEL_SUBSTITUTION].raw, 0.0)

    def test_detector_reliability_is_mockable(self):
        g = small_pipeline()
        g.attach_evidence(ev("E1", "B1", "NEAR_DUPLICATE_CLUSTER", 0.8, 0.5, "duplicate_detector"))
        strong = fuse(g).scores[H.DATASET_POISONING].base
        weak = fuse(g, FusionConfig().with_reliability({"duplicate_detector": 0.455})).scores[
            H.DATASET_POISONING].base
        self.assertAlmostEqual(weak, strong / 2)
        # a detector with no entry gets the default, and the default is configurable
        g2 = small_pipeline()
        g2.attach_evidence(ev("E1", "B1", "NEAR_DUPLICATE_CLUSTER", 1.0, 1.0, "homebrew"))
        self.assertAlmostEqual(fuse(g2).scores[H.DATASET_POISONING].base, 0.5 * 0.8)
        self.assertAlmostEqual(
            fuse(g2, FusionConfig(default_reliability=1.0)).scores[H.DATASET_POISONING].base, 0.8)
        self.assertEqual(FusionConfig().reliability["trigger_detector"], 0.94)

    def test_more_evidence_never_lowers_a_supported_hypothesis(self):
        g = small_pipeline()
        g.attach_evidence(ev("E1", "B1", "LABEL_DISTRIBUTION_ANOMALY", 0.6, 0.7, "label_detector"))
        before = fuse(g).scores[H.DATASET_POISONING].raw
        g.attach_evidence(ev("E2", "B1", "NEAR_DUPLICATE_CLUSTER", 0.6, 0.7, "duplicate_detector"))
        self.assertGreater(fuse(g).scores[H.DATASET_POISONING].raw, before)

    def test_unmapped_evidence_is_ignored_but_reported(self):
        g = small_pipeline()
        g.attach_evidence(ev("E1", "B1", "SOMETHING_NEW", 0.9, 0.9, "future_detector"))
        res = fuse(g)
        self.assertEqual(res.unmapped_evidence, ("SOMETHING_NEW",))
        self.assertTrue(all(res.scores[h].base == 0 for h in ATTACK_HYPOTHESES))
        # ...until someone teaches the config about it
        cfg = FusionConfig().with_mapping("SOMETHING_NEW", {H.BACKDOOR: 0.5})
        res2 = fuse(g, cfg)
        self.assertEqual(res2.unmapped_evidence, ())
        self.assertGreater(res2.scores[H.BACKDOOR].base, 0)


# --------------------------------------------------------------------------- #
class CorroborationTests(unittest.TestCase):
    def two_item_graph(self, det_a, det_b, node_b="B1"):
        g = small_pipeline()
        g.attach_evidence(ev("A", "B1", "LABEL_DISTRIBUTION_ANOMALY", 0.8, 0.8, det_a))
        g.attach_evidence(ev("B", node_b, "NEAR_DUPLICATE_CLUSTER" if node_b == "B1"
                             else "TRIGGER_SENSITIVITY", 0.8, 0.8, det_b))
        return g

    def test_independent_signals_on_one_path_are_corroborated(self):
        res = fuse(self.two_item_graph("label_detector", "duplicate_detector"))
        dp = res.scores[H.DATASET_POISONING]
        self.assertGreater(dp.corroboration, 0)
        self.assertEqual(dp.corroborating_pairs, (("A", "B"),))
        self.assertLessEqual(dp.corroboration, dp.base)  # cap

    def test_same_detector_is_not_independent(self):
        res = fuse(self.two_item_graph("label_detector", "label_detector"))
        self.assertEqual(res.scores[H.DATASET_POISONING].corroboration, 0.0)
        # ...unless the config says same-detector pairs still count a little
        cfg = FusionConfig(same_detector_factor=0.5)
        res2 = fuse(self.two_item_graph("label_detector", "label_detector"), cfg)
        self.assertGreater(res2.scores[H.DATASET_POISONING].corroboration, 0)

    def test_unrelated_parts_of_the_pipeline_do_not_corroborate(self):
        g = IntegrityGraph()
        g.add_node("B1", N.BATCH)
        g.add_node("B2", N.BATCH)
        g.attach_evidence(ev("A", "B1", "LABEL_DISTRIBUTION_ANOMALY", 0.8, 0.8, "label_detector"))
        g.attach_evidence(ev("B", "B2", "NEAR_DUPLICATE_CLUSTER", 0.8, 0.8, "duplicate_detector"))
        self.assertEqual(fuse(g).scores[H.DATASET_POISONING].corroboration, 0.0)

    def test_both_signals_must_support_the_same_hypothesis(self):
        g = small_pipeline()
        g.attach_evidence(ev("A", "B1", "NEAR_DUPLICATE_CLUSTER", 0.8, 0.8, "duplicate_detector"))
        g.attach_evidence(ev("B", "I", "PROVENANCE_MISMATCH", 0.8, 0.8, "passport_verifier"))
        res = fuse(g)
        # duplicates support poisoning; provenance mismatch supports tampering. No shared H.
        self.assertEqual(res.scores[H.DATASET_POISONING].corroboration, 0.0)
        self.assertEqual(res.scores[H.INFERENCE_TAMPERING].corroboration, 0.0)

    def test_evidence_on_a_connected_path_corroborates_but_a_disconnected_model_does_not(self):
        def corroboration(model_node):
            g = small_pipeline()
            g.add_node("M2", N.MODEL)   # a model that is not wired into the pipeline
            g.attach_evidence(ev("A", "B1", "LABEL_DISTRIBUTION_ANOMALY", 0.8, 0.8, "label_detector"))
            g.attach_evidence(ev("B", model_node, "TRIGGER_SENSITIVITY", 0.8, 0.8, "trigger_detector"))
            return fuse(g).scores[H.DATASET_POISONING].corroboration
        # batch -> training run -> model: a data-stage and a model-stage signal on one path
        self.assertGreater(corroboration("M"), 0)
        self.assertEqual(corroboration("M2"), 0.0)

    def test_nearer_signals_corroborate_more_than_distant_ones(self):
        def corroboration(model_node):
            g = small_pipeline()
            g.add_node("M2", N.MODEL)
            g.add_edge("TR", "M2", E.PRODUCED)
            g.add_node("M3", N.MODEL)
            g.add_edge("M2", "M3", E.DERIVED_FROM)
            g.attach_evidence(ev("A", "B1", "LABEL_DISTRIBUTION_ANOMALY", 0.8, 0.8, "label_detector"))
            g.attach_evidence(ev("B", model_node, "TRIGGER_SENSITIVITY", 0.8, 0.8, "trigger_detector"))
            return fuse(g).scores[H.DATASET_POISONING].corroboration
        self.assertGreater(corroboration("M2"), corroboration("M3"))   # 2 hops vs 3 hops from B1

    def test_corroboration_can_be_switched_off_and_is_capped(self):
        g = build_demo_graph()
        with_c = fuse(g)
        without = fuse(g, FusionConfig(corroboration_gain=0.0))
        for h in ATTACK_HYPOTHESES:
            self.assertEqual(without.scores[h].corroboration, 0.0)
            self.assertLessEqual(with_c.scores[h].corroboration, with_c.scores[h].base + 1e-12)
        self.assertGreater(with_c.scores[H.DATASET_POISONING].raw,
                           without.scores[H.DATASET_POISONING].raw)
        tight = fuse(g, FusionConfig(max_corroboration_ratio=0.1))
        self.assertAlmostEqual(tight.scores[H.DATASET_POISONING].corroboration,
                               0.1 * tight.scores[H.DATASET_POISONING].base)


# --------------------------------------------------------------------------- #
class DemoScenarioTests(unittest.TestCase):
    """build_demo_graph(): B17 poisoned, V4 trained on it, O982 provenance mismatch."""

    def setUp(self):
        self.g = build_demo_graph()
        self.res = fuse(self.g)

    def test_dataset_originated_compromise_is_the_top_explanation(self):
        ranked = self.res.ranked()
        self.assertEqual(ranked[0].hypothesis, H.DATASET_POISONING)
        self.assertEqual(ranked[1].hypothesis, H.BACKDOOR)
        self.assertEqual(self.res.top.hypothesis, H.DATASET_POISONING)
        self.assertGreater(self.res.margin, 0.0)
        self.assertEqual(ranked[0].level, "HIGH")
        self.assertEqual(ranked[1].level, "HIGH")

    def test_the_other_explanations_are_low(self):
        s = {h: self.res.scores[h].support for h in Hypothesis}
        self.assertLess(s[H.BENIGN_DISTRIBUTION_SHIFT], 0.05)
        self.assertLess(s[H.CLEAN], 0.05)
        self.assertLess(s[H.MODEL_SUBSTITUTION], 0.3)
        self.assertLess(s[H.INFERENCE_TAMPERING], 0.35)
        for h in (H.BENIGN_DISTRIBUTION_SHIFT, H.CLEAN, H.MODEL_SUBSTITUTION, H.INFERENCE_TAMPERING):
            self.assertLess(s[h], min(s[H.DATASET_POISONING], s[H.BACKDOOR]) - 0.4)

    def test_scores_are_normalised(self):
        for sc in self.res.scores.values():
            self.assertTrue(0.0 <= sc.support < 1.0)
            self.assertGreaterEqual(sc.raw, 0.0)
        sd = self.res.support_dict()
        self.assertEqual(list(sd), [s.hypothesis.value for s in self.res.ranked()])
        self.assertEqual(list(sd.values()), sorted(sd.values(), reverse=True))
        self.assertAlmostEqual(sum(self.res.shares().values()), 1.0)

    def test_every_score_is_traceable_to_evidence(self):
        dp = self.res.scores[H.DATASET_POISONING]
        self.assertTrue({"E001", "E002", "E010", "E011"} <= set(dp.supporting_evidence))
        self.assertIn(("E001", "E010"), dp.corroborating_pairs)   # data <-> model, cross-stage
        bd = self.res.scores[H.BACKDOOR]
        self.assertIn("E011", bd.supporting_evidence)
        benign = self.res.scores[H.BENIGN_DISTRIBUTION_SHIFT]
        self.assertTrue({"E011", "E021"} <= set(benign.contradicting_evidence))
        for sc in self.res.scores.values():
            for eid in sc.supporting_evidence + sc.contradicting_evidence:
                self.assertIsNotNone(self.g.get_evidence(eid))

    def test_integrity_checks_report_what_failed(self):
        c = self.res.checks
        self.assertEqual(c["dataset_integrity"].status, CheckStatus.FAIL)
        self.assertEqual(c["trigger_sensitivity"].status, CheckStatus.FAIL)
        self.assertEqual(c["provenance"].status, CheckStatus.FAIL)
        self.assertEqual(c["model_integrity"].status, CheckStatus.UNKNOWN)
        self.assertEqual(c["model_integrity"].missing_detectors, ("hash_verifier",))
        self.assertEqual(self.res.benign_context_scale, 0.0)   # real evidence: nothing explained away

    def test_reliability_mock_changes_the_outcome(self):
        distrust = FusionConfig().with_reliability({"trigger_detector": 0.0})
        low = fuse(self.g, distrust)
        self.assertLess(low.scores[H.BACKDOOR].support, self.res.scores[H.BACKDOOR].support)
        in_pairs = {eid for pair in low.scores[H.BACKDOOR].corroborating_pairs for eid in pair}
        self.assertNotIn("E011", in_pairs)   # a zero-reliability detector corroborates nothing
        self.assertEqual(low.ranked()[0].hypothesis, H.DATASET_POISONING)

    def test_result_serialises(self):
        d = json.loads(json.dumps(self.res.to_dict()))
        self.assertEqual(d["top"], "DATASET_POISONING")
        self.assertEqual(d["ranking"][0]["hypothesis"], "DATASET_POISONING")
        self.assertEqual(len(d["ranking"]), 6)
        self.assertIn("provenance", d["checks"])

    def test_counterfactual_and_ablation_hooks(self):
        original = self.res.scores[H.DATASET_POISONING].support
        no_b17 = fuse(self.g.without_nodes(["B17"]))
        self.assertLess(no_b17.scores[H.DATASET_POISONING].support, original)
        self.assertNotIn("E001", no_b17.scores[H.DATASET_POISONING].supporting_evidence)
        no_data = fuse(self.g.without_evidence(predicate=lambda r: r.stage == "data"))
        self.assertLess(no_data.scores[H.DATASET_POISONING].support, original)
        no_model = fuse(self.g.without_evidence(predicate=lambda r: r.stage == "model"))
        self.assertLess(no_model.scores[H.BACKDOOR].support,
                        self.res.scores[H.BACKDOOR].support)
        self.assertEqual(len(self.g.all_evidence()), 6)  # original untouched


# --------------------------------------------------------------------------- #
class BenignShiftTests(unittest.TestCase):
    def shifted(self, inspected=True) -> IntegrityGraph:
        """Low-light deployment: OOD + camera metadata changed, nothing else."""
        g = small_pipeline()
        if inspected:
            inspect_everything(g)
        g.attach_evidence(ev("O1", "I", "OOD_ANOMALY", 0.8, 0.85, "ood_detector"))
        g.attach_evidence(ev("M1", "D", "ENVIRONMENT_METADATA_CHANGE", 0.9, 0.9, "metadata_monitor"))
        return g

    def test_benign_shift_is_the_most_supported_explanation(self):
        res = fuse(self.shifted())
        self.assertEqual(res.top.hypothesis, H.BENIGN_DISTRIBUTION_SHIFT)
        self.assertTrue(all(c.status is CheckStatus.PASS for c in res.checks.values()))
        self.assertEqual(res.benign_context_scale, 1.0)
        self.assertGreater(res.scores[H.BENIGN_DISTRIBUTION_SHIFT].context_adjustment, 0)
        for h in ATTACK_HYPOTHESES:
            self.assertEqual(res.scores[h].raw, 0.0, h)
            self.assertLess(res.scores[h].context_adjustment, 0)
        self.assertGreater(res.scores[H.BENIGN_DISTRIBUTION_SHIFT].support, 0.6)

    def test_clean_checks_are_what_makes_the_difference(self):
        """Same OOD evidence: with verified-clean integrity checks it is a benign
        shift; without them (detectors never ran) the system must not be as sure."""
        verified = fuse(self.shifted(inspected=True))
        blind = fuse(self.shifted(inspected=False))
        self.assertEqual(blind.benign_context_scale, 0.0)
        self.assertTrue(all(c.status is CheckStatus.UNKNOWN for c in blind.checks.values()))
        self.assertGreater(verified.scores[H.BENIGN_DISTRIBUTION_SHIFT].support,
                           blind.scores[H.BENIGN_DISTRIBUTION_SHIFT].support)
        self.assertGreater(blind.scores[H.DATASET_POISONING].raw, 0.0)   # OOD alone nudges poisoning
        self.assertEqual(verified.scores[H.DATASET_POISONING].raw, 0.0)  # ...until integrity is verified

    def test_partial_verification_gives_partial_benefit(self):
        g = self.shifted(inspected=False)
        g.record_inspection("B1", "duplicate_detector")
        g.record_inspection("B1", "label_detector")        # dataset check now PASS: 1 of 4
        res = fuse(g)
        self.assertAlmostEqual(res.benign_context_scale, 0.25)

    def test_malicious_shift_is_not_explained_away(self):
        g = self.shifted()
        g.attach_evidence(ev("L1", "B1", "LABEL_DISTRIBUTION_ANOMALY", 0.8, 0.85, "label_detector"))
        g.attach_evidence(ev("T1", "M", "TRIGGER_SENSITIVITY", 0.9, 0.9, "trigger_detector"))
        g.attach_evidence(ev("B1e", "M", "BEHAVIORAL_DEVIATION", 0.8, 0.85, "behavior_detector"))
        res = fuse(g)
        self.assertEqual(res.benign_context_scale, 0.0)
        self.assertEqual(res.checks["dataset_integrity"].status, CheckStatus.FAIL)
        self.assertEqual(res.checks["trigger_sensitivity"].status, CheckStatus.FAIL)
        self.assertIn(res.top.hypothesis, (H.DATASET_POISONING, H.BACKDOOR))
        benign = res.scores[H.BENIGN_DISTRIBUTION_SHIFT]
        # The environment really did change (metadata evidence), so benign shift keeps
        # some support, but the attack explanations must clearly beat it.
        top_attack = max(res.scores[h].support for h in ATTACK_HYPOTHESES)
        self.assertGreater(top_attack - benign.support, 0.25)
        self.assertNotEqual(benign.level, "HIGH")
        self.assertEqual(benign.context_adjustment, 0.0)
        self.assertTrue({"T1", "L1"} <= set(benign.contradicting_evidence))

    def test_tiny_evidence_does_not_veto_the_benign_conclusion(self):
        g = self.shifted()
        g.attach_evidence(ev("N1", "B1", "LABEL_DISTRIBUTION_ANOMALY", 0.1, 0.5, "label_detector"))
        res = fuse(g)   # weight 0.05 < check_fail_threshold 0.15 => noise
        self.assertEqual(res.checks["dataset_integrity"].status, CheckStatus.PASS)
        self.assertEqual(res.top.hypothesis, H.BENIGN_DISTRIBUTION_SHIFT)

    def test_cross_stage_fusion_beats_an_isolated_detector(self):
        """The thesis in miniature: an OOD detector on its own would flag an
        attack; fusion with the other stages says 'environment changed'."""
        g = self.shifted()
        ood_only = max(g.get_evidence("O1").severity, 0)          # what a lone detector reports
        self.assertGreater(ood_only, 0.5)                          # would alert on its own
        res = fuse(g)
        self.assertEqual(res.top.hypothesis, H.BENIGN_DISTRIBUTION_SHIFT)
        self.assertLess(max(res.scores[h].support for h in ATTACK_HYPOTHESES), 0.05)


# --------------------------------------------------------------------------- #
class CleanPipelineTests(unittest.TestCase):
    def test_inspected_and_clean_is_clean(self):
        res = fuse(inspect_everything(small_pipeline()))
        self.assertEqual(res.top.hypothesis, H.CLEAN)
        self.assertGreater(res.scores[H.CLEAN].support, 0.6)
        self.assertEqual(res.evidence_coverage, 1.0)
        for h in set(Hypothesis) - {H.CLEAN}:
            self.assertEqual(res.scores[h].raw, 0.0)

    def test_uninspected_pipeline_is_not_called_clean(self):
        res = fuse(small_pipeline())
        self.assertEqual(res.evidence_coverage, 0.0)
        self.assertEqual(res.scores[H.CLEAN].raw, 0.0)
        self.assertTrue(res.inconclusive)
        self.assertIsNone(res.top)

    def test_partial_inspection_gives_proportionally_less_clean_support(self):
        half = small_pipeline()
        for n in ("DS", "B1", "TR"):
            half.record_inspection(n, "x")
        full = inspect_everything(small_pipeline())
        self.assertLess(fuse(half).scores[H.CLEAN].raw, fuse(full).scores[H.CLEAN].raw)
        self.assertAlmostEqual(fuse(half).evidence_coverage, 3 / 6)

    def test_a_broken_provenance_chain_costs_clean_support(self):
        g = inspect_everything(small_pipeline())
        g.add_node("V9", N.MODEL, "Model of unknown origin")
        g.record_inspection("V9", "hash_verifier")
        res = fuse(g)
        self.assertEqual(res.provenance_gaps, (("V9", "MODEL_WITHOUT_TRAINING_RUN"),))
        self.assertLess(res.scores[H.CLEAN].raw, fuse(inspect_everything(small_pipeline())).scores[H.CLEAN].raw)

    def test_any_real_anomaly_removes_clean_support(self):
        g = inspect_everything(small_pipeline())
        g.attach_evidence(ev("E1", "M", "MODEL_ARTIFACT_MISMATCH", 0.95, 0.95, "hash_verifier"))
        res = fuse(g)
        clean_before = fuse(inspect_everything(small_pipeline())).scores[H.CLEAN].support
        self.assertGreater(clean_before, 0.6)
        self.assertLess(res.scores[H.CLEAN].support, 0.15)       # one strong anomaly defeats CLEAN
        self.assertEqual(res.top.hypothesis, H.MODEL_SUBSTITUTION)
        self.assertGreater(res.scores[H.MODEL_SUBSTITUTION].support, res.scores[H.CLEAN].support)


if __name__ == "__main__":
    unittest.main()
