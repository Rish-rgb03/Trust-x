"""Tests for backend/counterfactual/analysis.py. Run with either:

    python -m unittest discover -s tests -t . -v
    pytest tests/test_analysis.py
"""
import json
import unittest
from types import SimpleNamespace

from backend.counterfactual.analysis import (
    CounterfactualConfig,
    CounterfactualResult,
    _classify,
    _summarise,
    simulate_node_removal,
)
from backend.evidence.graph import IntegrityGraph, build_demo_graph
from backend.reasoning.investigator import AssuranceState as S, investigate_pipeline


def ev(eid, source, etype, severity, confidence, detector):
    return dict(evidence_id=eid, source_id=source, evidence_type=etype,
                severity=severity, confidence=confidence, detector=detector)


class PoisonedBatchRemovalTests(unittest.TestCase):
    """The design-doc money shot: excise B17 and the risk plummets."""

    @classmethod
    def setUpClass(cls):
        cls.graph = build_demo_graph()
        cls.r = simulate_node_removal(cls.graph, "B17")

    def test_risk_plummets_and_state_improves(self):
        r = self.r
        self.assertIsInstance(r, CounterfactualResult)
        self.assertEqual(r.target_node_id, "B17")
        self.assertIs(r.original_state, S.RED)
        self.assertIs(r.ablated_state, S.AMBER)
        self.assertAlmostEqual(r.original_risk, 87.5)
        self.assertLess(r.ablated_risk, 40.0)
        self.assertGreater(r.risk_delta, 50.0)
        self.assertEqual(r.effect, "SUBSTANTIAL")

    def test_delta_is_original_minus_ablated(self):
        r = self.r
        self.assertAlmostEqual(r.risk_delta, round(r.original_risk - r.ablated_risk, 1))

    def test_baseline_is_the_plain_investigation(self):
        base = investigate_pipeline(self.graph)
        self.assertEqual(self.r.original_risk, base.risk_score)
        self.assertEqual(self.r.original_report.to_dict(), base.to_dict())
        self.assertEqual(self.r.original_top_hypothesis, "DATASET_POISONING")
        self.assertEqual(self.r.original_implicated_source, "B17")

    def test_summary_matches_design_doc_wording(self):
        s = self.r.human_readable_summary
        self.assertTrue(s.startswith(
            "Removing B17 substantially reduces the observed integrity anomalies (Risk: 88 -> 27"), s)
        self.assertIn("RED -> AMBER", s)
        self.assertIn("strengthening the hypothesis that B17 is implicated", s)

    def test_cascade_takes_the_batchs_samples(self):
        # A purely structural removal would leave S184 (and its OOD finding) behind.
        self.assertEqual(set(self.r.removed_nodes), {"B17", "S184", "S185"})
        self.assertEqual(set(self.r.removed_evidence), {"E001", "E002", "E003"})

    def test_retraining_assumption_is_explicit(self):
        r = self.r
        self.assertEqual(set(r.cleared_evidence), {"E010", "E011"})
        self.assertEqual(r.retained_downstream_evidence, ())
        self.assertEqual(len(r.assumptions), 1)
        self.assertIn("retrained", r.assumptions[0])
        self.assertIn("simulated, not measured", r.human_readable_summary)

    def test_structural_only_bound_is_reported_and_stays_red(self):
        # Without the retraining assumption the model's findings survive: still RED.
        r = self.r
        self.assertIs(r.structural_ablated_state, S.RED)
        self.assertGreater(r.structural_ablated_risk, 60.0)
        self.assertGreater(r.structural_ablated_risk, r.ablated_risk)
        plain = investigate_pipeline(self.graph.without_nodes(["B17", "S184", "S185"]))
        self.assertEqual(r.structural_ablated_risk, plain.risk_score)

    def test_what_remains_is_the_unrelated_inference_finding(self):
        self.assertEqual(self.r.ablated_top_hypothesis, "INFERENCE_TAMPERING")
        self.assertEqual(self.r.ablated_implicated_source, "O982")

    def test_original_graph_is_untouched(self):
        g = build_demo_graph()
        before = g.to_json()
        simulate_node_removal(g, "B17")
        self.assertEqual(g.to_json(), before)


class AttributionSanityTests(unittest.TestCase):
    """Removing the guilty node must help far more than removing innocent ones."""

    @classmethod
    def setUpClass(cls):
        cls.g = build_demo_graph()

    def test_clean_batch_has_no_material_effect(self):
        r = simulate_node_removal(self.g, "B03")
        self.assertEqual(r.risk_delta, 0.0)
        self.assertEqual(r.effect, "NONE")
        self.assertIs(r.ablated_state, S.RED)
        self.assertIn("no material effect", r.human_readable_summary)
        self.assertIn("no evidence that B03 is implicated", r.human_readable_summary)
        # B17 still taints V4, so retraining without B03 must not be assumed to fix it.
        self.assertEqual(r.cleared_evidence, ())
        self.assertEqual(set(r.retained_downstream_evidence), {"E010", "E011"})
        self.assertIn("still carries integrity evidence", r.assumptions[0])
        self.assertNotIn("simulated", r.human_readable_summary)

    def test_clean_contributor_has_no_material_effect(self):
        r = simulate_node_removal(self.g, "C_A")
        self.assertEqual(r.effect, "NONE")
        self.assertEqual(set(r.removed_nodes), {"C_A", "B03", "S001"})   # cascade via CONTRIBUTED

    def test_guilty_contributor_matches_guilty_batch(self):
        b, c = simulate_node_removal(self.g, "B17"), simulate_node_removal(self.g, "C_B")
        self.assertEqual(set(c.removed_nodes), {"C_B", "B17", "S184", "S185"})
        self.assertEqual(c.ablated_risk, b.ablated_risk)
        self.assertEqual(c.effect, "SUBSTANTIAL")

    def test_removing_the_whole_dataset_removes_its_contents(self):
        r = simulate_node_removal(self.g, "DS01")
        self.assertTrue({"B03", "B17", "S001", "S184", "S185"} <= set(r.removed_nodes))
        self.assertEqual(r.effect, "SUBSTANTIAL")

    def test_b17_is_the_best_data_removal_among_batches_and_contributors(self):
        deltas = {n: simulate_node_removal(self.g, n).risk_delta for n in ("B17", "B03", "C_A")}
        self.assertEqual(max(deltas, key=deltas.get), "B17")
        self.assertEqual(deltas["B03"], 0.0)
        self.assertEqual(deltas["C_A"], 0.0)

    def test_second_suspicious_batch_blocks_the_retraining_assumption(self):
        g = build_demo_graph()
        g.attach_evidence(ev("E200", "B03", "LABEL_DISTRIBUTION_ANOMALY", 0.8, 0.9, "label_detector"))
        r = simulate_node_removal(g, "B17")
        self.assertEqual(r.cleared_evidence, ())
        self.assertEqual(set(r.retained_downstream_evidence), {"E010", "E011"})
        self.assertIs(r.ablated_state, S.RED)       # B03 is also suspicious: model not fixed
        self.assertGreater(r.ablated_risk, 55.0)
        self.assertIn(r.ablated_implicated_source, {"V4", "B03"})

    def test_shared_sample_survives_when_another_batch_still_holds_it(self):
        g = build_demo_graph()
        g.add_edge("B03", "S184", "CONTAINS")      # S184 now lives in both batches
        r = simulate_node_removal(g, "B17")
        self.assertEqual(set(r.removed_nodes), {"B17", "S185"})
        self.assertNotIn("E003", r.removed_evidence)   # its finding stays with the sample

    def test_retraining_does_not_clear_artifact_level_model_evidence(self):
        # A hash mismatch on the model points at substitution; retraining on clean data
        # would not make it go away, so it must never be "cleared".
        g = build_demo_graph()
        g.attach_evidence(ev("E300", "V4", "MODEL_ARTIFACT_MISMATCH", 0.9, 0.9, "hash_verifier"))
        r = simulate_node_removal(g, "B17")
        self.assertEqual(set(r.cleared_evidence), {"E010", "E011"})
        self.assertNotIn("E300", r.cleared_evidence)
        # The mismatch is what remains: the verdict moves to substitution of V4, and the
        # residual risk is clearly higher than when nothing artifact-level survives (27).
        self.assertEqual(r.ablated_top_hypothesis, "MODEL_SUBSTITUTION")
        self.assertEqual(r.ablated_implicated_source, "V4")
        self.assertGreater(r.ablated_risk, 45.0)

    def test_retraining_never_clears_non_training_evidence(self):
        r = simulate_node_removal(self.g, "B17")
        self.assertNotIn("E021", r.cleared_evidence)    # provenance mismatch on the output
        self.assertIn("E021", r.ablated_report.converging_evidence)   # still driving what remains

class SwitchesAndEdgeCaseTests(unittest.TestCase):
    def test_retraining_can_be_switched_off(self):
        r = simulate_node_removal(build_demo_graph(), "B17",
                                  config=CounterfactualConfig(simulate_retraining=False))
        self.assertEqual(r.cleared_evidence, ())
        self.assertEqual(r.assumptions, ())
        self.assertEqual(r.ablated_risk, r.structural_ablated_risk)
        self.assertIs(r.ablated_state, S.RED)
        self.assertNotIn("simulated", r.human_readable_summary)

    def test_non_data_targets_never_assume_retraining(self):
        g = build_demo_graph()
        for node in ("TR42", "V4", "O982"):
            r = simulate_node_removal(g, node)
            self.assertEqual(r.cleared_evidence, (), node)
            self.assertEqual(r.assumptions, (), node)
            self.assertEqual(r.ablated_risk, r.structural_ablated_risk, node)
        # Removing the evidence-bearing model is a real, but only structural, reduction.
        v4 = simulate_node_removal(g, "V4")
        self.assertGreater(v4.risk_delta, 0.0)
        self.assertIs(v4.ablated_state, S.RED)

    def test_clean_pipeline_removal_is_no_effect(self):
        g = IntegrityGraph()
        for n, t in (("B1", "BATCH"), ("S1", "SAMPLE")):
            g.add_node(n, t)
        g.add_edge("B1", "S1", "CONTAINS")
        r = simulate_node_removal(g, "B1")
        self.assertEqual((r.original_risk, r.ablated_risk, r.risk_delta), (0.0, 0.0, 0.0))
        self.assertEqual(r.effect, "NONE")

    def test_unknown_node_and_bad_input(self):
        with self.assertRaises(KeyError):
            simulate_node_removal(build_demo_graph(), "ghost")
        with self.assertRaises(TypeError):
            simulate_node_removal({"nodes": []}, "B17")

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            CounterfactualConfig(material_delta=20, moderate_delta=10)
        with self.assertRaises(ValueError):
            CounterfactualConfig(residual_evidence_threshold=2)


class WordingTests(unittest.TestCase):
    def test_classification_bands(self):
        cfg = CounterfactualConfig()
        self.assertEqual(_classify(0.0, cfg), "NONE")
        self.assertEqual(_classify(0.4, cfg), "NONE")
        self.assertEqual(_classify(5.0, cfg), "SLIGHT")
        self.assertEqual(_classify(10.0, cfg), "MODERATE")
        self.assertEqual(_classify(29.9, cfg), "MODERATE")
        self.assertEqual(_classify(30.0, cfg), "SUBSTANTIAL")
        self.assertEqual(_classify(-5.0, cfg), "INCREASED")

    def _reports(self, a, b):
        mk = lambda risk: SimpleNamespace(risk_score=risk, assurance_state=S.AMBER)  # noqa: E731
        return mk(a), mk(b)

    def test_every_effect_has_its_own_sentence(self):
        g = build_demo_graph()
        expect = {
            "SLIGHT": ("slightly reduces", "weak support"),
            "MODERATE": ("reduces the observed", "partly supporting"),
            "SUBSTANTIAL": ("substantially reduces", "strengthening"),
            "NONE": ("no material effect", "no evidence"),
            "INCREASED": ("increases the observed risk", "does not support"),
        }
        for effect, needles in expect.items():
            o, f = self._reports(50.0, 40.0)
            text = _summarise(g, "B17", o, f, 10.0, effect, (), [])
            for n in needles:
                self.assertIn(n, text, (effect, text))

    def test_zero_delta_states_no_material_effect(self):
        r = simulate_node_removal(build_demo_graph(), "S185")   # carries no evidence
        self.assertEqual(r.risk_delta, 0.0)
        self.assertIn("no material effect", r.human_readable_summary)


class SerialisationTests(unittest.TestCase):
    def test_json_has_the_requested_schema(self):
        d = json.loads(simulate_node_removal(build_demo_graph(), "B17").to_json())
        for key in ("target_node_id", "original_risk", "ablated_risk", "risk_delta",
                    "original_state", "ablated_state", "human_readable_summary"):
            self.assertIn(key, d)
        self.assertEqual((d["original_state"], d["ablated_state"]), ("RED", "AMBER"))

    def test_deterministic(self):
        g = build_demo_graph()
        a, b = simulate_node_removal(g, "B17"), simulate_node_removal(g, "B17")
        self.assertEqual(a.to_dict(), b.to_dict())


if __name__ == "__main__":
    unittest.main()
