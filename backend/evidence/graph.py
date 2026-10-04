"""TRUST-X Integrity Graph.

A directed acyclic graph (NetworkX) that links every object in an AI pipeline
(contributor -> dataset -> batch -> sample -> training run -> model ->
deployment -> inference -> output) and lets detectors attach *evidence* to the
nodes they are about.

Conventions
-----------
* Every edge points in the direction of the lifecycle: upstream -> downstream.
  So ``ancestors(node)`` is "where did this come from" (provenance) and
  ``descendants(node)`` is "what could this have affected" (blast radius).
* The graph does NOT decide anything. It stores structure + evidence and
  answers structural questions. The maths lives in ``fusion.py``, the
  verdicts in ``reasoning/investigator.py``, the what-ifs in
  ``counterfactual/analysis.py``.

What the other three modules call on this one
---------------------------------------------
fusion.py        -> ``correlation_pairs()``   structural weight for C_ij
investigator.py  -> ``candidate_origins()``, ``attack_path()``,
                    ``provenance_gaps()``, ``inspection_coverage()``
analysis.py      -> ``without_nodes()``, ``without_evidence()``,
                    ``downstream()``
UI / API         -> ``to_node_link()``, ``to_dict()`` / ``from_dict()``

Evidence is accepted as a dict or any object (dataclass, pydantic model, ...)
with these fields, so this file does not import ``evidence/schema.py``:

    evidence_id | id          unique id, e.g. "E001"
    source_id | source        node id the evidence is about, e.g. "B17"
    evidence_type | type      e.g. "NEAR_DUPLICATE_CLUSTER"
    severity                  float in [0, 1]
    confidence                float in [0, 1]
    detector                  e.g. "duplicate_detector"
    (anything else)           kept in ``payload``
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from enum import Enum
from itertools import islice
from typing import Any, Callable, Iterable, Mapping, Optional

import networkx as nx

__all__ = [
    "NodeType",
    "EdgeType",
    "Relation",
    "STAGES",
    "EvidenceRecord",
    "LineageRelation",
    "CorrelatedPair",
    "OriginCandidate",
    "AttackPath",
    "IntegrityGraph",
    "CycleError",
    "SchemaViolationError",
    "build_demo_graph",
]


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
class NodeType(str, Enum):
    CONTRIBUTOR = "CONTRIBUTOR"
    DATASET = "DATASET"
    BATCH = "BATCH"
    SAMPLE = "SAMPLE"
    TRAINING_RUN = "TRAINING_RUN"
    MODEL = "MODEL"
    DEPLOYMENT = "DEPLOYMENT"
    CONFIGURATION = "CONFIGURATION"
    INPUT = "INPUT"
    INFERENCE = "INFERENCE"
    OUTPUT = "OUTPUT"


class EdgeType(str, Enum):
    CONTRIBUTED = "CONTRIBUTED"
    CONTAINS = "CONTAINS"
    TRAINED_ON = "TRAINED_ON"
    PRODUCED = "PRODUCED"
    DERIVED_FROM = "DERIVED_FROM"
    DEPLOYED_AS = "DEPLOYED_AS"
    INFERRED_ON = "INFERRED_ON"
    GENERATED = "GENERATED"
    MODIFIED = "MODIFIED"
    REPLAYED = "REPLAYED"
    # Not in the original design doc: a configuration/preprocessing artifact
    # needs *some* edge into the pipeline. Remove if the team prefers.
    CONFIGURES = "CONFIGURES"


N, E = NodeType, EdgeType

#: Lifecycle stages, in order. Used for layout and for "cross-stage" tests.
STAGES = ("data", "training", "model", "deployment", "inference")

NODE_STAGE: dict[NodeType, str] = {
    N.CONTRIBUTOR: "data",
    N.DATASET: "data",
    N.BATCH: "data",
    N.SAMPLE: "data",
    N.TRAINING_RUN: "training",
    N.MODEL: "model",
    N.DEPLOYMENT: "deployment",
    N.CONFIGURATION: "deployment",
    N.INPUT: "inference",
    N.INFERENCE: "inference",
    N.OUTPUT: "inference",
}

#: Legal (source type, target type) pairs per edge type. Direction is
#: upstream -> downstream. MODIFIED is special-cased (same type -> same type).
_ALLOWED_EDGES: dict[EdgeType, set[tuple[NodeType, NodeType]]] = {
    E.CONTRIBUTED: {(N.CONTRIBUTOR, t) for t in (N.DATASET, N.BATCH, N.SAMPLE)},
    E.CONTAINS: {(N.DATASET, N.BATCH), (N.DATASET, N.SAMPLE), (N.BATCH, N.SAMPLE)},
    E.TRAINED_ON: {(N.DATASET, N.TRAINING_RUN), (N.BATCH, N.TRAINING_RUN)},
    E.PRODUCED: {(N.TRAINING_RUN, N.MODEL)},
    E.DERIVED_FROM: {(N.MODEL, N.MODEL), (N.DATASET, N.DATASET),
                     (N.BATCH, N.BATCH), (N.SAMPLE, N.SAMPLE)},
    E.DEPLOYED_AS: {(N.MODEL, N.DEPLOYMENT)},
    E.CONFIGURES: {(N.CONFIGURATION, t)
                   for t in (N.TRAINING_RUN, N.DEPLOYMENT, N.INFERENCE)},
    E.INFERRED_ON: {(N.DEPLOYMENT, N.INFERENCE), (N.INPUT, N.INFERENCE)},
    E.GENERATED: {(N.INFERENCE, N.OUTPUT)},
    E.REPLAYED: {(N.INFERENCE, N.INFERENCE)},
}

#: node type -> upstream type(s) that must exist for its provenance to be known.
_REQUIRED_UPSTREAM: dict[NodeType, tuple[NodeType, ...]] = {
    N.BATCH: (N.CONTRIBUTOR,),
    N.TRAINING_RUN: (N.BATCH, N.DATASET),
    N.MODEL: (N.TRAINING_RUN,),
    N.DEPLOYMENT: (N.MODEL,),
    N.INFERENCE: (N.DEPLOYMENT,),
}

# Structural-proximity tuning (used by LineageRelation.proximity).
DIRECT_DECAY = 0.85           # per extra hop on a direct lifecycle path
COMMON_ORIGIN_DISCOUNT = 0.5  # sharing only a common ancestor is weaker


class CycleError(ValueError):
    """Adding the edge would break the DAG property."""


class SchemaViolationError(ValueError):
    """The edge type is not allowed between these node types."""


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    source_id: str
    evidence_type: str
    severity: float
    confidence: float
    detector: str
    stage: str
    payload: dict = field(default_factory=dict, compare=False)

    @property
    def weight(self) -> float:
        """Severity x confidence: how much this evidence should count."""
        return self.severity * self.confidence

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "source_id": self.source_id,
            "evidence_type": self.evidence_type,
            "severity": self.severity,
            "confidence": self.confidence,
            "detector": self.detector,
            "payload": dict(self.payload),
        }


class Relation(str, Enum):
    SAME_NODE = "SAME_NODE"
    UPSTREAM = "UPSTREAM"          # a is an ancestor of b
    DOWNSTREAM = "DOWNSTREAM"      # a is a descendant of b
    COMMON_ORIGIN = "COMMON_ORIGIN"  # neither, but they share an ancestor
    UNRELATED = "UNRELATED"


@dataclass(frozen=True)
class LineageRelation:
    kind: Relation
    distance: Optional[int] = None  # hops (for COMMON_ORIGIN: hops via `via`)
    via: Optional[str] = None       # shared ancestor for COMMON_ORIGIN

    @property
    def proximity(self) -> float:
        """0..1 structural closeness. 1.0 = same node, 0.0 = unrelated."""
        if self.kind is Relation.SAME_NODE:
            return 1.0
        if self.kind is Relation.UNRELATED or not self.distance:
            return 0.0
        base = DIRECT_DECAY ** (self.distance - 1)
        return base * COMMON_ORIGIN_DISCOUNT if self.kind is Relation.COMMON_ORIGIN else base


@dataclass(frozen=True)
class CorrelatedPair:
    """Two pieces of evidence whose source nodes sit on one lifecycle path."""
    evidence_a: str
    evidence_b: str
    relation: LineageRelation
    cross_stage: bool    # sources are in different lifecycle stages
    same_detector: bool  # same detector => less independent corroboration

    @property
    def proximity(self) -> float:
        return self.relation.proximity


@dataclass(frozen=True)
class OriginCandidate:
    node_id: str
    node_type: NodeType
    explained_fraction: float  # share of total evidence weight at/below this node
    evidence_ids: tuple[str, ...]
    stages: tuple[str, ...]    # lifecycle stages those evidence items span
    closure_size: int          # node + descendants (smaller = more specific)
    rank: int = 0


@dataclass(frozen=True)
class AttackPath:
    origin: str
    nodes: tuple[str, ...]            # lifecycle order
    edges: tuple[tuple[str, str], ...]
    evidence_nodes: tuple[str, ...]   # nodes on the path that carry evidence


# --------------------------------------------------------------------------- #
# Evidence normalisation
# --------------------------------------------------------------------------- #
_ID_KEYS = ("evidence_id", "id")
_SOURCE_KEYS = ("source_id", "source_node_id", "source", "node_id")
_TYPE_KEYS = ("evidence_type", "type")
_CORE_KEYS = set(_ID_KEYS + _SOURCE_KEYS + _TYPE_KEYS
                 + ("severity", "confidence", "detector", "payload"))
_MISSING = object()


def _plain(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _as_mapping(obj: Any) -> dict:
    if isinstance(obj, Mapping):
        return dict(obj)
    for method in ("model_dump", "to_dict", "dict"):
        fn = getattr(obj, method, None)
        if callable(fn):
            try:
                return dict(fn())
            except TypeError:
                continue
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return dict(vars(obj))


def _pick(m: Mapping, keys: tuple[str, ...], default: Any = _MISSING) -> Any:
    for k in keys:
        if k in m and m[k] is not None:
            return _plain(m[k])
    if default is _MISSING:
        raise ValueError(f"evidence is missing a required field (one of {keys})")
    return default


def _unit(name: str, value: Any) -> float:
    v = float(value)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"evidence {name} must be in [0, 1], got {v}")
    return v


# --------------------------------------------------------------------------- #
# The graph
# --------------------------------------------------------------------------- #
class IntegrityGraph:
    """Lifecycle DAG + attached evidence.

    ``strict=True`` (default) enforces the edge-type schema above. Cycles are
    always rejected.
    """

    def __init__(self, strict: bool = True) -> None:
        self.g: nx.DiGraph = nx.DiGraph()
        self.strict = strict
        self._evidence: dict[str, EvidenceRecord] = {}

    # ---- construction ---------------------------------------------------- #
    def add_node(self, node_id: str, node_type: NodeType | str,
                 label: Optional[str] = None, **attrs: Any) -> str:
        """Add a node (idempotent). Extra kwargs become metadata, e.g.
        ``sha256=...``, ``timestamp=...``, ``contributor_id=...``."""
        node_type = NodeType(node_type)
        if node_id in self.g:
            data = self.g.nodes[node_id]
            if data["node_type"] != node_type:
                raise ValueError(
                    f"node {node_id!r} already exists as {data['node_type'].value}, "
                    f"cannot re-add as {node_type.value}")
            if label:
                data["label"] = label
            data["attrs"].update(attrs)
            return node_id
        self.g.add_node(node_id, node_type=node_type, label=label or node_id,
                        attrs=dict(attrs), inspected_by=set(), evidence_ids=[])
        return node_id

    def add_edge(self, src: str, dst: str, edge_type: EdgeType | str,
                 **attrs: Any) -> None:
        """Add ``src -> dst`` (upstream -> downstream). Idempotent."""
        et = EdgeType(edge_type)
        for n in (src, dst):
            if n not in self.g:
                raise KeyError(f"unknown node {n!r}; add_node it first")
        if src == dst:
            raise CycleError(f"self-loop on {src!r}")
        if self.strict and not self._edge_allowed(et, src, dst):
            raise SchemaViolationError(
                f"{et.value} is not allowed from {self.node_type(src).value} "
                f"to {self.node_type(dst).value} ({src!r} -> {dst!r})")
        if self.g.has_edge(src, dst):
            existing = self.g.edges[src, dst]
            if existing["edge_type"] != et:
                raise ValueError(
                    f"{src!r} -> {dst!r} already exists as {existing['edge_type'].value}")
            existing["attrs"].update(attrs)
            return
        if nx.has_path(self.g, dst, src):
            raise CycleError(f"{src!r} -> {dst!r} would create a cycle")
        self.g.add_edge(src, dst, edge_type=et, attrs=dict(attrs))

    def _edge_allowed(self, et: EdgeType, src: str, dst: str) -> bool:
        pair = (self.node_type(src), self.node_type(dst))
        if et is E.MODIFIED:
            return pair[0] == pair[1]
        return pair in _ALLOWED_EDGES[et]

    # ---- basic lookups --------------------------------------------------- #
    def node_type(self, node_id: str) -> NodeType:
        return self.g.nodes[node_id]["node_type"]

    def stage_of(self, node_id: str) -> str:
        return NODE_STAGE[self.node_type(node_id)]

    def nodes_of_type(self, *types: NodeType | str) -> list[str]:
        wanted = {NodeType(t) for t in types}
        return sorted(n for n, d in self.g.nodes(data=True) if d["node_type"] in wanted)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self.g

    def __len__(self) -> int:
        return self.g.number_of_nodes()

    # ---- evidence -------------------------------------------------------- #
    def attach_evidence(self, evidence: Any, source_id: Optional[str] = None) -> EvidenceRecord:
        """Attach one evidence item (dict or object) to its source node."""
        m = _as_mapping(evidence)
        src = source_id or _pick(m, _SOURCE_KEYS)
        if src not in self.g:
            raise KeyError(f"evidence source {src!r} is not a node in the graph")
        raw_id = _pick(m, _ID_KEYS, default=None)
        eid = str(raw_id) if raw_id else f"E-{len(self._evidence)+1:05d}"
        if eid in self._evidence:
            raise ValueError(f"duplicate evidence id {eid!r}")
        payload = dict(m.get("payload") or {})
        payload.update({k: v for k, v in m.items() if k not in _CORE_KEYS})
        record = EvidenceRecord(
            evidence_id=eid,
            source_id=src,
            evidence_type=str(_pick(m, _TYPE_KEYS)),
            severity=_unit("severity", _pick(m, ("severity",))),
            confidence=_unit("confidence", _pick(m, ("confidence",))),
            detector=str(_pick(m, ("detector",), default="unknown")),
            stage=self.stage_of(src),
            payload=payload,
        )
        self._evidence[eid] = record
        self.g.nodes[src]["evidence_ids"].append(eid)
        self.g.nodes[src]["inspected_by"].add(record.detector)
        return record

    def attach_many(self, items: Iterable[Any]) -> list[EvidenceRecord]:
        return [self.attach_evidence(i) for i in items]

    def get_evidence(self, evidence_id: str) -> EvidenceRecord:
        return self._evidence[evidence_id]

    def all_evidence(self) -> list[EvidenceRecord]:
        return sorted(self._evidence.values(), key=lambda r: r.evidence_id)

    def evidence_for(self, node_id: str, *, include_upstream: bool = False,
                     include_downstream: bool = False) -> list[EvidenceRecord]:
        """Evidence on a node, optionally also on its ancestors/descendants."""
        nodes = {node_id}
        if include_upstream:
            nodes |= nx.ancestors(self.g, node_id)
        if include_downstream:
            nodes |= nx.descendants(self.g, node_id)
        ids = [e for n in nodes for e in self.g.nodes[n]["evidence_ids"]]
        return sorted((self._evidence[i] for i in ids), key=lambda r: r.evidence_id)

    def evidence_nodes(self) -> list[str]:
        return sorted(n for n, d in self.g.nodes(data=True) if d["evidence_ids"])

    def record_inspection(self, node_id: str, detector: str) -> None:
        """Record that ``detector`` looked at ``node_id`` (even if it found
        nothing). 'Clean' results count towards coverage; absence of evidence
        is only meaningful for nodes that were actually inspected."""
        self.g.nodes[node_id]["inspected_by"].add(detector)

    # ---- lineage --------------------------------------------------------- #
    def upstream(self, node_id: str, node_types: Optional[Iterable[NodeType | str]] = None) -> list[str]:
        """Ancestors, closest first. This is provenance: model -> run -> batch ..."""
        dist = dict(nx.single_target_shortest_path_length(self.g, node_id))
        dist.pop(node_id, None)
        return self._ordered(dist, node_types)

    def downstream(self, node_id: str, node_types: Optional[Iterable[NodeType | str]] = None) -> list[str]:
        """Descendants, closest first. This is the blast radius of a node."""
        dist = dict(nx.single_source_shortest_path_length(self.g, node_id))
        dist.pop(node_id, None)
        return self._ordered(dist, node_types)

    def _ordered(self, dist: dict[str, int], node_types) -> list[str]:
        if node_types is not None:
            wanted = {NodeType(t) for t in node_types}
            dist = {n: d for n, d in dist.items() if self.node_type(n) in wanted}
        return sorted(dist, key=lambda n: (dist[n], n))

    def trace_back(self, node_id: str) -> list[str]:
        """Provenance of a node, closest first (e.g. V17 -> TR42 -> DS19 -> B17 -> C04)."""
        return self.upstream(node_id)

    def lineage_paths(self, src: str, dst: str, limit: int = 20) -> list[list[str]]:
        """Up to ``limit`` simple lifecycle paths from ``src`` to ``dst``."""
        if not nx.has_path(self.g, src, dst):
            return []
        return list(islice(nx.all_simple_paths(self.g, src, dst), limit))

    def relation(self, a: str, b: str) -> LineageRelation:
        """How two nodes are structurally related (see ``Relation``)."""
        if a == b:
            return LineageRelation(Relation.SAME_NODE, 0)
        if nx.has_path(self.g, a, b):
            return LineageRelation(Relation.UPSTREAM, nx.shortest_path_length(self.g, a, b))
        if nx.has_path(self.g, b, a):
            return LineageRelation(Relation.DOWNSTREAM, nx.shortest_path_length(self.g, b, a))
        da = dict(nx.single_target_shortest_path_length(self.g, a))
        db = dict(nx.single_target_shortest_path_length(self.g, b))
        common = set(da) & set(db)
        if not common:
            return LineageRelation(Relation.UNRELATED)
        via = min(common, key=lambda n: (da[n] + db[n], n))
        return LineageRelation(Relation.COMMON_ORIGIN, da[via] + db[via], via)

    # ---- the cross-stage analysis the rest of the system needs ----------- #
    def correlation_pairs(self, min_proximity: float = 0.0) -> list[CorrelatedPair]:
        """Every pair of evidence items whose sources are structurally related.

        This is the graph's contribution to the corroboration term C_ij in
        fusion: evidence on a *connected* lifecycle path corroborates each
        other; evidence on unrelated parts of the pipeline does not.
        """
        records = self.all_evidence()
        cache: dict[tuple[str, str], LineageRelation] = {}
        pairs: list[CorrelatedPair] = []
        for i, a in enumerate(records):
            for b in records[i + 1:]:
                key = (a.source_id, b.source_id)
                if key not in cache:
                    cache[key] = self.relation(*key)
                rel = cache[key]
                if rel.kind is Relation.UNRELATED or rel.proximity < min_proximity:
                    continue
                pairs.append(CorrelatedPair(
                    evidence_a=a.evidence_id,
                    evidence_b=b.evidence_id,
                    relation=rel,
                    cross_stage=a.stage != b.stage,
                    same_detector=a.detector == b.detector,
                ))
        return pairs

    def candidate_origins(self, node_types: Optional[Iterable[NodeType | str]] = None,
                          min_fraction: float = 0.0) -> list[OriginCandidate]:
        """Rank nodes as possible origins of the observed anomalies.

        A node 'explains' the evidence located on itself or anything
        downstream of it. Candidates are ranked by the fraction of total
        evidence weight (severity x confidence) they explain; ties go to the
        node with the smaller downstream closure, i.e. the most specific
        explanation. So Batch B17 outranks Contributor B (same evidence, one
        node fewer) and outranks Dataset DS01 (same evidence, much bigger
        blast radius), while a clean batch that merely shares a training run
        explains less and ranks lower.

        Pass ``node_types={NodeType.BATCH}`` to rank at a single level.
        """
        total = sum(r.weight for r in self._evidence.values())
        if total <= 0:
            return []
        wanted = None if node_types is None else {NodeType(t) for t in node_types}
        scored: list[OriginCandidate] = []
        for node, data in self.g.nodes(data=True):
            if wanted is not None and data["node_type"] not in wanted:
                continue
            closure = {node} | nx.descendants(self.g, node)
            ids = sorted(e for n in closure for e in self.g.nodes[n]["evidence_ids"])
            weight = sum(self._evidence[i].weight for i in ids)
            fraction = weight / total
            if not ids or fraction < min_fraction:
                continue
            stages = {self._evidence[i].stage for i in ids}
            scored.append(OriginCandidate(
                node_id=node,
                node_type=data["node_type"],
                explained_fraction=fraction,
                evidence_ids=tuple(ids),
                stages=tuple(s for s in STAGES if s in stages),
                closure_size=len(closure),
            ))
        scored.sort(key=lambda c: (-round(c.explained_fraction, 9), c.closure_size, c.node_id))
        return [dataclasses.replace(c, rank=i + 1) for i, c in enumerate(scored)]

    def attack_path(self, origin: str) -> AttackPath:
        """The lifecycle path(s) from ``origin`` to every downstream node that
        carries evidence. This is what the UI highlights as the attack path."""
        nodes = {origin}
        edges: set[tuple[str, str]] = set()
        carriers = []
        for target in nx.descendants(self.g, origin) | {origin}:
            if not self.g.nodes[target]["evidence_ids"]:
                continue
            carriers.append(target)
            path = nx.shortest_path(self.g, origin, target)
            nodes.update(path)
            edges.update(zip(path, path[1:]))
        order = {n: i for i, n in enumerate(nx.lexicographical_topological_sort(self.g))}
        return AttackPath(
            origin=origin,
            nodes=tuple(sorted(nodes, key=order.__getitem__)),
            edges=tuple(sorted(edges, key=lambda e: (order[e[0]], order[e[1]]))),
            evidence_nodes=tuple(sorted(carriers, key=order.__getitem__)),
        )

    # ---- coverage / provenance health ------------------------------------ #
    def provenance_gaps(self) -> list[tuple[str, str]]:
        """Nodes whose provenance chain is broken, as ``(node_id, issue)``.
        e.g. a MODEL with no TRAINING_RUN upstream. Feeds both the assurance
        coverage number and (as evidence) the hypothesis engine."""
        gaps = []
        for node, data in self.g.nodes(data=True):
            needed = _REQUIRED_UPSTREAM.get(data["node_type"])
            if not needed:
                continue
            have = {self.node_type(a) for a in nx.ancestors(self.g, node)}
            if not have & set(needed):
                gaps.append((node, f"{data['node_type'].value}_WITHOUT_"
                                   f"{'_OR_'.join(t.value for t in needed)}"))
        return sorted(gaps)

    def inspection_coverage(self, node_types: Optional[Iterable[NodeType | str]] = None) -> float:
        """Fraction of nodes (optionally of given types) that some detector has
        inspected or produced evidence for. 0.0 if there are no such nodes."""
        wanted = None if node_types is None else {NodeType(t) for t in node_types}
        nodes = [d for _, d in self.g.nodes(data=True)
                 if wanted is None or d["node_type"] in wanted]
        if not nodes:
            return 0.0
        return sum(1 for d in nodes if d["inspected_by"] or d["evidence_ids"]) / len(nodes)

    def stages_present(self) -> list[str]:
        present = {NODE_STAGE[d["node_type"]] for _, d in self.g.nodes(data=True)}
        return [s for s in STAGES if s in present]

    def evidence_by_stage(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {s: [] for s in STAGES}
        for r in self.all_evidence():
            out[r.stage].append(r.evidence_id)
        return out

    # ---- counterfactual / ablation support ------------------------------- #
    def copy(self) -> "IntegrityGraph":
        clone = IntegrityGraph(strict=self.strict)
        clone.g = nx.DiGraph()
        for n, d in self.g.nodes(data=True):
            clone.g.add_node(n, node_type=d["node_type"], label=d["label"],
                             attrs=dict(d["attrs"]), inspected_by=set(d["inspected_by"]),
                             evidence_ids=list(d["evidence_ids"]))
        for u, v, d in self.g.edges(data=True):
            clone.g.add_edge(u, v, edge_type=d["edge_type"], attrs=dict(d["attrs"]))
        clone._evidence = dict(self._evidence)
        return clone

    def without_nodes(self, node_ids: Iterable[str]) -> "IntegrityGraph":
        """Copy of the graph with ``node_ids`` removed, together with their
        edges and their own evidence. The original is untouched.

        Evidence on *downstream* nodes is deliberately kept: whether it would
        disappear (e.g. the model is retrained without B17 and the trigger
        sensitivity vanishes) is an experimental result for
        ``counterfactual/analysis.py`` to supply, not something the graph can
        assume. Use ``downstream()`` to know which evidence is affected."""
        clone = self.copy()
        for n in set(node_ids):
            if n not in clone.g:
                raise KeyError(f"unknown node {n!r}")
            for eid in clone.g.nodes[n]["evidence_ids"]:
                clone._evidence.pop(eid, None)
            clone.g.remove_node(n)
        return clone

    def without_evidence(self, evidence_ids: Iterable[str] = (), *,
                         predicate: Optional[Callable[[EvidenceRecord], bool]] = None) -> "IntegrityGraph":
        """Copy with the given evidence removed (nodes/edges kept). ``predicate``
        removes every record it returns True for, handy for ablations such as
        ``lambda r: r.stage == "data"``."""
        drop = set(evidence_ids)
        if predicate is not None:
            drop |= {r.evidence_id for r in self._evidence.values() if predicate(r)}
        unknown = drop - set(self._evidence)
        if unknown:
            raise KeyError(f"unknown evidence ids: {sorted(unknown)}")
        clone = self.copy()
        for eid in drop:
            rec = clone._evidence.pop(eid)
            clone.g.nodes[rec.source_id]["evidence_ids"].remove(eid)
        return clone

    # ---- validation ------------------------------------------------------ #
    def validate(self) -> list[str]:
        """Human-readable problems; empty list means the graph is sound."""
        problems = []
        if not nx.is_directed_acyclic_graph(self.g):
            problems.append("graph contains a cycle")
        for n in nx.isolates(self.g):
            problems.append(f"node {n!r} is not connected to anything")
        problems += [f"{n}: {issue}" for n, issue in self.provenance_gaps()]
        return problems

    # ---- serialisation --------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "nodes": [
                {"id": n, "type": d["node_type"].value, "label": d["label"],
                 "attrs": d["attrs"], "inspected_by": sorted(d["inspected_by"])}
                for n, d in sorted(self.g.nodes(data=True))
            ],
            "edges": [
                {"source": u, "target": v, "type": d["edge_type"].value, "attrs": d["attrs"]}
                for u, v, d in sorted(self.g.edges(data=True), key=lambda e: (e[0], e[1]))
            ],
            "evidence": [r.to_dict() for r in self.all_evidence()],
        }

    @classmethod
    def from_dict(cls, data: Mapping, strict: bool = True) -> "IntegrityGraph":
        graph = cls(strict=strict)
        for n in data["nodes"]:
            graph.add_node(n["id"], n["type"], n.get("label"), **n.get("attrs", {}))
            graph.g.nodes[n["id"]]["inspected_by"].update(n.get("inspected_by", []))
        for e in data["edges"]:
            graph.add_edge(e["source"], e["target"], e["type"], **e.get("attrs", {}))
        for ev in data.get("evidence", []):
            graph.attach_evidence(ev)
        return graph

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), default=str, **kwargs)

    @classmethod
    def from_json(cls, text: str, strict: bool = True) -> "IntegrityGraph":
        return cls.from_dict(json.loads(text), strict=strict)

    def to_node_link(self, highlight: Optional[Iterable[str]] = None) -> dict:
        """UI-ready export (React Flow / Cytoscape friendly). ``highlight`` is a
        set of node ids to flag, e.g. ``attack_path(...).nodes``."""
        hot = set(highlight or ())
        nodes = []
        for n, d in sorted(self.g.nodes(data=True)):
            recs = [self._evidence[i] for i in d["evidence_ids"]]
            nodes.append({
                "id": n,
                "type": d["node_type"].value,
                "label": d["label"],
                "stage": NODE_STAGE[d["node_type"]],
                "rank": STAGES.index(NODE_STAGE[d["node_type"]]),
                "evidence_ids": list(d["evidence_ids"]),
                "evidence_count": len(recs),
                "max_severity": max((r.severity for r in recs), default=0.0),
                "highlighted": n in hot,
            })
        edges = [
            {"source": u, "target": v, "type": d["edge_type"].value,
             "highlighted": u in hot and v in hot}
            for u, v, d in sorted(self.g.edges(data=True), key=lambda e: (e[0], e[1]))
        ]
        return {"nodes": nodes, "edges": edges}


# --------------------------------------------------------------------------- #
# Demo pipeline (the B17 story from the design doc)
# --------------------------------------------------------------------------- #
def build_demo_graph() -> IntegrityGraph:
    """A small compromised pipeline for developing fusion / investigator /
    counterfactual code and for the UI, without waiting for the detectors.

        Contributor B -> Batch B17 --\\
                                      >-> TR42 -> V4 -> D7 -> I982 -> O982
        Contributor A -> Batch B03 --/

    Evidence sits on B17 (data), V4 (model) and O982 (inference); B03 is clean.
    """
    g = IntegrityGraph()
    g.add_node("C_A", N.CONTRIBUTOR, "Contributor A")
    g.add_node("C_B", N.CONTRIBUTOR, "Contributor B")
    g.add_node("DS01", N.DATASET, "Dataset DS01")
    g.add_node("B03", N.BATCH, "Batch B03")
    g.add_node("B17", N.BATCH, "Batch B17")
    for s, b in (("S001", "B03"), ("S184", "B17"), ("S185", "B17")):
        g.add_node(s, N.SAMPLE, f"Sample {s}")
        g.add_edge(b, s, E.CONTAINS)
    g.add_node("TR42", N.TRAINING_RUN, "Training Run 42")
    g.add_node("V4", N.MODEL, "Model V4")
    g.add_node("D7", N.DEPLOYMENT, "Deployment D7")
    g.add_node("IN982", N.INPUT, "Input 982")
    g.add_node("I982", N.INFERENCE, "Inference I982")
    g.add_node("O982", N.OUTPUT, "Output O982")

    g.add_edge("C_A", "B03", E.CONTRIBUTED)
    g.add_edge("C_B", "B17", E.CONTRIBUTED)
    g.add_edge("DS01", "B03", E.CONTAINS)
    g.add_edge("DS01", "B17", E.CONTAINS)
    g.add_edge("B03", "TR42", E.TRAINED_ON)
    g.add_edge("B17", "TR42", E.TRAINED_ON)
    g.add_edge("TR42", "V4", E.PRODUCED)
    g.add_edge("V4", "D7", E.DEPLOYED_AS)
    g.add_edge("D7", "I982", E.INFERRED_ON)
    g.add_edge("IN982", "I982", E.INFERRED_ON)
    g.add_edge("I982", "O982", E.GENERATED)

    g.attach_many([
        dict(evidence_id="E001", source_id="B17", evidence_type="NEAR_DUPLICATE_CLUSTER",
             severity=0.81, confidence=0.94, detector="duplicate_detector", affected_samples=421),
        dict(evidence_id="E002", source_id="B17", evidence_type="LABEL_DISTRIBUTION_ANOMALY",
             severity=0.74, confidence=0.88, detector="label_detector"),
        dict(evidence_id="E003", source_id="S184", evidence_type="OOD_ANOMALY",
             severity=0.67, confidence=0.83, detector="ood_detector"),
        dict(evidence_id="E010", source_id="V4", evidence_type="BEHAVIORAL_DEVIATION",
             severity=0.85, confidence=0.89, detector="behavior_detector"),
        dict(evidence_id="E011", source_id="V4", evidence_type="TRIGGER_SENSITIVITY",
             severity=0.95, confidence=0.94, detector="trigger_detector"),
        dict(evidence_id="E021", source_id="O982", evidence_type="PROVENANCE_MISMATCH",
             severity=0.60, confidence=0.90, detector="passport_verifier"),
    ])
    # Clean results still count towards coverage.
    for node, det in (("B03", "duplicate_detector"), ("B03", "label_detector"),
                      ("D7", "passport_verifier")):
        g.record_inspection(node, det)
    return g


if __name__ == "__main__":  # quick look: python -m backend.evidence.graph
    demo = build_demo_graph()
    print("trace_back(V4):", demo.trace_back("V4"))
    for c in demo.candidate_origins(node_types={N.BATCH, N.CONTRIBUTOR}):
        print(f"#{c.rank} {c.node_id:5s} explains {c.explained_fraction:.0%} "
              f"across {list(c.stages)}")
    print("attack path:", " -> ".join(demo.attack_path("B17").nodes))
    print("problems:", demo.validate() or "none")
