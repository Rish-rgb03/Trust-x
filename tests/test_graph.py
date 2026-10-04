"""Tests for backend/evidence/graph.py. Run with either:

    python -m unittest discover -s tests -v
    pytest tests/test_graph.py
"""
import dataclasses
import json
import unittest

from backend.evidence.graph import (
    AttackPath,
    CycleError,
    EdgeType,
    IntegrityGraph,
    NodeType,
    Relation,
    SchemaViolationError,
    build_demo_graph,
)

N, E = NodeType, EdgeType


class StructureTests(unittest.TestCase):
    def test_cycle_is_rejected_even_when_not_strict(self):
        g = IntegrityGraph(strict=False)
        for n in "abc":
            g.add_node(n, N.MODEL)
        g.add_edge("a", "b", E.DERIVED_FROM)
        g.add_edge("b", "c", E.DERIVED_FROM)
        with self.assertRaises(CycleError):
            g.add_edge("c", "a", E.DERIVED_FROM)
        with self.assertRaises(CycleError):
            g.add_edge("a", "a", E.DERIVED_FROM)

    def test_schema_violation_in_strict_mode(self):
        g = build_demo_graph()
        with self.assertRaises(SchemaViolationError):
            g.add_edge("V4", "B17", E.PRODUCED)      # wrong direction/type
        with self.assertRaises(SchemaViolationError):
            g.add_edge("C_A", "V4", E.CONTRIBUTED)   # contributor -> model

    def test_unknown_node_and_type_conflict(self):
        g = IntegrityGraph()
        g.add_node("B1", N.BATCH)
        with self.assertRaises(KeyError):
            g.add_edge("B1", "nope", E.CONTAINS)
        with self.assertRaises(ValueError):
            g.add_node("B1", N.MODEL)

    def test_add_is_idempotent(self):
        g = IntegrityGraph()
        g.add_node("B1", N.BATCH)
        g.add_node("S1", N.SAMPLE)
        g.add_edge("B1", "S1", E.CONTAINS)
        g.add_edge("B1", "S1", E.CONTAINS)
        g.add_node("B1", N.BATCH, sha256="abc")
        self.assertEqual(g.g.number_of_edges(), 1)
        self.assertEqual(g.g.nodes["B1"]["attrs"]["sha256"], "abc")

    def test_modified_and_replayed_edges(self):
        g = IntegrityGraph()
        g.add_node("O1", N.OUTPUT)
        g.add_node("O1_mod", N.OUTPUT)
        g.add_edge("O1", "O1_mod", E.MODIFIED)
        g.add_node("I1", N.INFERENCE)
        g.add_node("I1_replay", N.INFERENCE)
        g.add_edge("I1", "I1_replay", E.REPLAYED)
        g.add_node("M", N.MODEL)
        with self.assertRaises(SchemaViolationError):
            g.add_edge("O1", "M", E.MODIFIED)

    def test_lineage_queries(self):
        g = build_demo_graph()
        self.assertEqual(g.trace_back("V4")[0], "TR42")
        self.assertEqual(set(g.upstream("V4", [N.BATCH])), {"B03", "B17"})
        self.assertEqual(set(g.upstream("V4", [N.CONTRIBUTOR])), {"C_A", "C_B"})
        self.assertEqual(g.downstream("V4"), ["D7", "I982", "O982"])
        paths = g.lineage_paths("B17", "O982")
        self.assertEqual(paths, [["B17", "TR42", "V4", "D7", "I982", "O982"]])
        self.assertEqual(g.lineage_paths("O982", "B17"), [])


class EvidenceTests(unittest.TestCase):
    def test_attach_dict_and_object(self):
        @dataclasses.dataclass
        class Ev:
            id: str
            source: str
            type: str
            severity: float
            confidence: float
            detector: str

        g = build_demo_graph()
        g.attach_evidence(Ev("E100", "B03", "OOD_ANOMALY", 0.3, 0.5, "ood_detector"))
        rec = g.get_evidence("E100")
        self.assertEqual((rec.source_id, rec.stage), ("B03", "data"))
        self.assertAlmostEqual(rec.weight, 0.15)
        self.assertIn("E100", g.g.nodes["B03"]["evidence_ids"])

    def test_extra_fields_go_to_payload(self):
        g = build_demo_graph()
        self.assertEqual(g.get_evidence("E001").payload["affected_samples"], 421)

    def test_validation(self):
        g = build_demo_graph()
        good = dict(evidence_id="X", source_id="B17", evidence_type="T",
                    severity=0.5, confidence=0.5, detector="d")
        with self.assertRaises(ValueError):
            g.attach_evidence({**good, "severity": 1.5})
        with self.assertRaises(KeyError):
            g.attach_evidence({**good, "source_id": "ghost"})
        with self.assertRaises(ValueError):
            g.attach_evidence({**good, "evidence_id": "E001"})   # duplicate id
        bad = dict(good)
        del bad["evidence_type"]
        with self.assertRaises(ValueError):
            g.attach_evidence(bad)

    def test_evidence_for_with_neighbours(self):
        g = build_demo_graph()
        own = {r.evidence_id for r in g.evidence_for("V4")}
        self.assertEqual(own, {"E010", "E011"})
        up = {r.evidence_id for r in g.evidence_for("V4", include_upstream=True)}
        self.assertTrue({"E001", "E002"} <= up)
        # Samples hang off their batch as leaves; S184 is not on V4's lineage.
        self.assertNotIn("E003", up)
        down = {r.evidence_id for r in g.evidence_for("V4", include_downstream=True)}
        self.assertIn("E021", down)
        self.assertNotIn("E001", down)


class AnalysisTests(unittest.TestCase):
    def test_relations(self):
        g = build_demo_graph()
        self.assertEqual(g.relation("V4", "V4").kind, Relation.SAME_NODE)
        up = g.relation("B17", "V4")
        self.assertEqual((up.kind, up.distance), (Relation.UPSTREAM, 2))
        self.assertEqual(g.relation("V4", "B17").kind, Relation.DOWNSTREAM)
        sib = g.relation("B03", "B17")
        self.assertEqual(sib.kind, Relation.COMMON_ORIGIN)
        self.assertEqual(sib.via, "DS01")
        self.assertEqual(g.relation("C_A", "C_B").kind, Relation.UNRELATED)
        # Closer and more direct means more structural weight.
        self.assertGreater(g.relation("V4", "D7").proximity, g.relation("B17", "O982").proximity)
        self.assertGreater(up.proximity, sib.proximity)

    def test_correlation_pairs_link_stages(self):
        g = build_demo_graph()
        pairs = {frozenset((p.evidence_a, p.evidence_b)): p for p in g.correlation_pairs()}
        # data-stage evidence on B17 and model-stage evidence on V4: connected, cross-stage.
        p = pairs[frozenset(("E001", "E010"))]
        self.assertTrue(p.cross_stage)
        self.assertFalse(p.same_detector)
        self.assertEqual(p.relation.kind, Relation.UPSTREAM)
        # E010 and E011 are both on V4: same node, same stage.
        same = pairs[frozenset(("E010", "E011"))]
        self.assertEqual(same.relation.kind, Relation.SAME_NODE)
        self.assertFalse(same.cross_stage)
        self.assertEqual(len(pairs), 15)  # all 6 items sit on one lifecycle path

    def test_unrelated_evidence_does_not_correlate(self):
        g = IntegrityGraph()
        g.add_node("C1", N.CONTRIBUTOR)
        g.add_node("C2", N.CONTRIBUTOR)
        g.attach_many([
            dict(evidence_id="A", source_id="C1", evidence_type="T", severity=.5, confidence=.5, detector="d"),
            dict(evidence_id="B", source_id="C2", evidence_type="T", severity=.5, confidence=.5, detector="d"),
        ])
        self.assertEqual(g.correlation_pairs(), [])

    def test_candidate_origins_prefers_the_specific_source(self):
        g = build_demo_graph()
        ranked = g.candidate_origins(node_types={N.BATCH, N.CONTRIBUTOR, N.DATASET})
        ids = [c.node_id for c in ranked]
        self.assertEqual(ids[0], "B17")                       # most specific full explanation
        self.assertLess(ids.index("B17"), ids.index("C_B"))   # beats its contributor
        self.assertLess(ids.index("C_B"), ids.index("DS01"))  # beats the whole dataset
        top = ranked[0]
        self.assertAlmostEqual(top.explained_fraction, 1.0)
        self.assertEqual(top.stages, ("data", "model", "inference"))
        # The clean batch only explains the downstream evidence.
        b03 = next(c for c in ranked if c.node_id == "B03")
        self.assertLess(b03.explained_fraction, 0.6)
        self.assertEqual(b03.stages, ("model", "inference"))
        self.assertEqual([c.rank for c in ranked], list(range(1, len(ranked) + 1)))

    def test_candidate_origins_empty_without_evidence(self):
        g = IntegrityGraph()
        g.add_node("B1", N.BATCH)
        self.assertEqual(g.candidate_origins(), [])

    def test_attack_path(self):
        g = build_demo_graph()
        path = g.attack_path("B17")
        self.assertIsInstance(path, AttackPath)
        self.assertEqual(path.nodes[0], "B17")
        self.assertEqual(path.nodes[-1], "O982")
        self.assertEqual(set(path.evidence_nodes), {"B17", "S184", "V4", "O982"})
        self.assertIn(("TR42", "V4"), path.edges)
        self.assertNotIn("B03", path.nodes)
        self.assertNotIn("C_B", path.nodes)

    def test_provenance_gaps_and_validate(self):
        g = build_demo_graph()
        self.assertEqual(g.provenance_gaps(), [])
        self.assertEqual(g.validate(), [])
        g.add_node("V9", N.MODEL, "Model V9 (unknown origin)")
        self.assertEqual(g.provenance_gaps(), [("V9", "MODEL_WITHOUT_TRAINING_RUN")])
        problems = g.validate()
        self.assertTrue(any("V9" in p for p in problems))

    def test_inspection_coverage(self):
        g = IntegrityGraph()
        g.add_node("B1", N.BATCH)
        g.add_node("B2", N.BATCH)
        self.assertEqual(g.inspection_coverage([N.BATCH]), 0.0)
        g.record_inspection("B1", "duplicate_detector")
        self.assertEqual(g.inspection_coverage([N.BATCH]), 0.5)
        g.attach_evidence(dict(evidence_id="X", source_id="B2", evidence_type="T",
                               severity=.1, confidence=.1, detector="d"))
        self.assertEqual(g.inspection_coverage([N.BATCH]), 1.0)
        self.assertEqual(g.inspection_coverage([N.MODEL]), 0.0)  # none exist
        self.assertEqual(g.stages_present(), ["data"])


class CounterfactualSupportTests(unittest.TestCase):
    def test_without_nodes_leaves_original_untouched(self):
        g = build_demo_graph()
        before = g.to_json()
        cf = g.without_nodes(["B17"])
        self.assertEqual(g.to_json(), before)
        self.assertNotIn("B17", cf)
        ids = {r.evidence_id for r in cf.all_evidence()}
        self.assertNotIn("E001", ids)
        self.assertNotIn("E002", ids)
        # Downstream evidence is kept for the counterfactual engine to resolve.
        self.assertTrue({"E010", "E011", "E021"} <= ids)
        # Removing B17 orphans its contributor and samples (reported, not hidden),
        # but TR42 is still fed by B03 so no provenance chain is broken.
        self.assertEqual(cf.provenance_gaps(), [])
        orphans = {p.split("'")[1] for p in cf.validate() if "not connected" in p}
        self.assertEqual(orphans, {"C_B", "S184", "S185"})
        # Mutating the copy never leaks into the original.
        cf.g.nodes["V4"]["evidence_ids"].clear()
        self.assertEqual(len(g.evidence_for("V4")), 2)

    def test_without_nodes_rejects_unknown(self):
        with self.assertRaises(KeyError):
            build_demo_graph().without_nodes(["ghost"])

    def test_without_evidence_by_id_and_predicate(self):
        g = build_demo_graph()
        cf = g.without_evidence(["E001"])
        self.assertEqual({r.evidence_id for r in cf.all_evidence()},
                         {"E002", "E003", "E010", "E011", "E021"})
        self.assertEqual(len(g.all_evidence()), 6)

        no_data = g.without_evidence(predicate=lambda r: r.stage == "data")
        self.assertEqual({r.evidence_id for r in no_data.all_evidence()},
                         {"E010", "E011", "E021"})
        self.assertEqual(no_data.g.number_of_nodes(), g.g.number_of_nodes())
        with self.assertRaises(KeyError):
            g.without_evidence(["nope"])

    def test_candidate_origin_moves_after_ablation(self):
        g = build_demo_graph().without_evidence(predicate=lambda r: r.stage == "data")
        top = g.candidate_origins(node_types={N.MODEL, N.BATCH})[0]
        self.assertEqual(top.node_id, "V4")  # only model/inference evidence is left


class SerialisationTests(unittest.TestCase):
    def test_round_trip(self):
        g = build_demo_graph()
        again = IntegrityGraph.from_json(g.to_json())
        self.assertEqual(again.to_dict(), g.to_dict())
        self.assertEqual(again.get_evidence("E001").payload["affected_samples"], 421)
        self.assertEqual(again.g.nodes["B03"]["inspected_by"],
                         {"duplicate_detector", "label_detector"})

    def test_to_dict_is_json_serialisable(self):
        json.dumps(build_demo_graph().to_dict())

    def test_node_link_export(self):
        g = build_demo_graph()
        path = g.attack_path("B17")
        out = g.to_node_link(highlight=path.nodes)
        nodes = {n["id"]: n for n in out["nodes"]}
        self.assertEqual(nodes["B17"]["evidence_count"], 2)
        self.assertAlmostEqual(nodes["B17"]["max_severity"], 0.81)
        self.assertTrue(nodes["V4"]["highlighted"])
        self.assertFalse(nodes["B03"]["highlighted"])
        self.assertLess(nodes["B17"]["rank"], nodes["O982"]["rank"])
        edge = next(e for e in out["edges"] if (e["source"], e["target"]) == ("TR42", "V4"))
        self.assertTrue(edge["highlighted"])
        self.assertEqual(edge["type"], "PRODUCED")


if __name__ == "__main__":
    unittest.main()
