"""Grounded report writer for TRUST-X.

The deterministic investigator remains the security authority. This module only
turns its already-computed report into readable prose. An optional local
OpenAI-compatible LLM may rewrite the supplied facts, but the result is used
only when a citation/grounding check passes; otherwise the deterministic
narrative is returned.
"""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class Fact:
    id: str
    text: str


CITE_RE = re.compile(r"\[F(\d+)\]")
ID_RE = re.compile(r"\b[A-Za-z]+(?:[-_][A-Za-z]+)*[-_]?\d+(?:[-_]\d+)*\b")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _tokens(text: str) -> tuple[set[str], set[str]]:
    text = CITE_RE.sub("", text)
    ids = set(ID_RE.findall(text))
    nums = set(NUM_RE.findall(ID_RE.sub(" ", text)))
    return ids, nums


def build_facts(report: dict[str, Any]) -> list[Fact]:
    facts: list[str] = []
    facts.append(
        f"Assurance state is {report['assurance_state']} with recommended action {report['recommended_action']}; "
        f"risk is {float(report['risk_score']):.0f} out of 100, confidence is {float(report['confidence']):.0f} percent, "
        f"and inspection coverage is {float(report['coverage']):.0f} percent."
    )
    facts.append(
        f"The top hypothesis is {report['top_hypothesis']} with support {float(report['top_hypothesis_support']):.2f}."
    )
    if report.get("implicated_source"):
        facts.append(
            f"The implicated source is {report['implicated_source']} of type {report.get('implicated_source_type')}."
        )
    if report.get("attack_path"):
        facts.append("The reported lifecycle path is " + " -> ".join(report["attack_path"]) + ".")
    evidence = list(report.get("supporting_evidence") or ())
    if evidence:
        facts.append("Supporting evidence identifiers are " + ", ".join(evidence) + ".")
    if report.get("reasons"):
        facts.append("Investigator reasons: " + " ".join(str(x) for x in report["reasons"]) + ".")
    if report.get("warnings"):
        facts.append("Investigator warnings: " + " ".join(str(x) for x in report["warnings"]) + ".")
    summary_parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", str(report["human_readable_summary"]).strip()) if part.strip()]
    for part in summary_parts:
        facts.append("Deterministic investigator summary: " + part.rstrip("."))
    return [Fact(f"F{i}", text) for i, text in enumerate(facts, 1)]


def template_narrative(facts: Iterable[Fact]) -> str:
    return " ".join(f"{fact.text.rstrip('.')} [{fact.id}]." for fact in facts)


def verify_grounding(text: str, facts: list[Fact]) -> list[str]:
    by_id = {f.id: f for f in facts}
    normalized = re.sub(
        r"([.!?])\s*((?:\[F\d+\]\s*)+)",
        lambda m: " " + m.group(2).strip() + m.group(1) + " ",
        text,
    )
    sentences = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", normalized.strip()) if s.strip()]
    if not sentences:
        return ["empty report"]

    violations: list[str] = []
    cited_all: set[str] = set()
    for sentence in sentences:
        cites = {f"F{n}" for n in CITE_RE.findall(sentence)}
        if not cites:
            violations.append(f"uncited sentence: {sentence[:80]!r}")
            continue
        unknown = cites - by_id.keys()
        if unknown:
            violations.append(f"unknown citation(s): {sorted(unknown)}")
            continue
        cited_all |= cites
        allowed_ids: set[str] = set()
        allowed_nums: set[str] = set()
        for cite in cites:
            ids, nums = _tokens(by_id[cite].text)
            allowed_ids |= ids
            allowed_nums |= nums
        ids, nums = _tokens(sentence)
        extra_ids = ids - allowed_ids
        extra_nums = nums - allowed_nums
        if extra_ids:
            violations.append(f"identifiers not in cited facts: {sorted(extra_ids)}")
        if extra_nums:
            violations.append(f"numbers not in cited facts: {sorted(extra_nums)}")

    if "F1" not in cited_all:
        violations.append("required fact F1 not cited")
    if facts and facts[-1].id not in cited_all:
        violations.append(f"required fact {facts[-1].id} not cited")
    return violations


SYSTEM_PROMPT = (
    "You are the report writer for TRUST-X. Use ONLY the numbered facts supplied by the application. "
    "Every sentence must end with one or more citations such as [F2]. Never invent numbers, identifiers, "
    "or security conclusions. Use cautious language such as 'evidence supports'. Write 4-8 plain sentences."
)


class LocalLLM:
    """Optional local-only OpenAI-compatible report writer."""

    def __init__(self, url: str = "http://127.0.0.1:11434/v1/chat/completions", model: str = "llama3.1:8b", timeout: int = 60):
        host = urlparse(url).hostname
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("TRUST-X runs offline by default: LLM endpoint must be loopback")
        self.url = url
        self.model = model
        self.timeout = timeout

    def complete(self, facts: list[Fact]) -> str:
        body = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "FACTS:\n" + "\n".join(f"[{f.id}] {f.text}" for f in facts)},
            ],
        }
        request = urllib.request.Request(
            self.url,
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)["choices"][0]["message"]["content"].strip()


def generate_grounded_report(report: dict[str, Any], llm: Optional[LocalLLM] = None) -> dict[str, Any]:
    facts = build_facts(report)
    violations: list[str] = []
    narrative = template_narrative(facts)
    generator = "template"

    if llm is not None:
        try:
            candidate = llm.complete(facts)
            violations = verify_grounding(candidate, facts)
            if not violations:
                narrative = candidate
                generator = "llm"
        except Exception as exc:
            violations = [f"llm unavailable: {exc.__class__.__name__}"]

    final_violations = verify_grounding(narrative, facts)
    return {
        "headline": {
            "GREEN": "No integrity concern found",
            "AMBER": "Investigation required",
            "RED": "Compromise strongly indicated",
        }.get(report.get("assurance_state"), "Investigation required"),
        "narrative": narrative,
        "generator": generator,
        "grounded": not final_violations,
        "llm_rejected_reasons": violations,
        "facts": {f.id: f.text for f in facts},
    }
