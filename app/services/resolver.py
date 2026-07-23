"""Entity resolution: messy natural-language input -> exactly one machine_id.

This is safety-critical. A wrong machine means the wrong manual, the wrong spare
part, and the wrong lockout procedure, so the resolver never silently guesses:
it either returns a single high-confidence machine or it BLOCKS and asks.

The chain runs in a fixed order and stops at the first *confident* hit:

    1. EXACT_ID    input == machine_id (case/punctuation-insensitive)   1.00
    2. ALIAS       input == one of aliases[]                            0.95
    3. ERROR_CODE  codes in the text -> machine_model -> machines       0.90
    4. CONTEXT     boosts applied to the surviving candidate pool
    5. FUZZY       rapidfuzz token_set_ratio over id/name/model/aliases

Stages 1-2 short-circuit because an exact identifier is unambiguous by
construction. Stage 3 short-circuits only when the code maps to exactly one
machine; when it maps to several, those candidates fall through to CONTEXT and
FUZZY so the tie can be broken on evidence rather than on order.

Deterministic by design — there is no LLM in this path. The same input and the
same fleet always produce the same answer, which is what makes the result
auditable after an incident.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

from rapidfuzz import fuzz

from app.schemas.machine import COLLECTIONS
from app.schemas.resolution import (
    MatchMethod,
    ResolutionCandidate,
    ResolutionContext,
    ResolutionResult,
    ResolutionStatus,
)

# ---------------------------------------------------------------------------
# Tunables — every threshold in the chain lives here, nowhere else.
# ---------------------------------------------------------------------------
CONFIDENCE_EXACT_ID = 1.0
CONFIDENCE_ALIAS = 0.95
CONFIDENCE_ERROR_CODE = 0.9

#: A candidate scoring below this is not worth showing a human at all.
MIN_CANDIDATE_CONFIDENCE = 0.55
#: The top candidate must reach this to be actionable without asking.
RESOLVE_MIN_CONFIDENCE = 0.85
#: ...and must beat the runner-up by at least this much.
RESOLVE_MIN_MARGIN = 0.15
#: Never put more than this many options in front of an operator.
MAX_CANDIDATES = 5

#: Slack for binary float error, so a score sitting exactly on a threshold is
#: treated as meeting it. Without this, 0.95 - 0.80 evaluates to 0.1499999...
#: and a candidate that clears the margin by the book gets blocked.
_EPSILON = 1e-9

# CONTEXT-stage boosts. Deliberately small: context is corroborating evidence,
# not identification. Their sum cannot promote a sub-threshold guess on its own.
BOOST_LAST_MACHINE = 0.10
BOOST_ASSIGNED_LINE = 0.08
BOOST_ACTIVE_ALARM = 0.06
BOOST_FAULTED_STATUS = 0.04

#: Statuses that make a machine a more likely subject of a maintenance question.
_ATTENTION_STATUSES = frozenset({"fault", "stopped"})

#: e.g. "E104", "E-104", "AL_22", "SRV-1234"
ERROR_CODE_PATTERN = re.compile(r"\b[A-Z]{1,3}[-_]?\d{2,4}\b")


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------
def normalize_key(value: str) -> str:
    """Collapse a string to a comparison key: upper-case alphanumerics only.

    ``" cv-201. "``, ``"CV 201"`` and ``"cv201"`` all become ``"CV201"``.
    """
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def extract_error_codes(text: str) -> list[str]:
    """Return fault codes found in ``text``, upper-cased and de-duplicated.

    Order of first appearance is preserved so the primary code in a sentence
    stays first.
    """
    seen: dict[str, None] = {}
    for match in ERROR_CODE_PATTERN.findall(text.upper()):
        seen.setdefault(match, None)
    return list(seen)


def _code_variants(code: str) -> set[str]:
    """Return the spellings a code may be stored under, e.g. E-104 and E104."""
    bare = code.replace("-", "").replace("_", "")
    return {code, bare, f"{bare[:1]}-{bare[1:]}" if len(bare) > 1 else bare}


# ---------------------------------------------------------------------------
# Data access — a narrow port so callers (and tests) can swap the backend
# ---------------------------------------------------------------------------
class MachineRepository(Protocol):
    """The only database surface the resolver needs.

    Every method takes ``tenant_id`` explicitly: there is no way to ask this
    repository a question that is not scoped to one tenant.
    """

    async def list_machines(self, tenant_id: str) -> list[Mapping[str, Any]]:
        """Return every machine document in one tenant's fleet."""
        ...

    async def find_error_codes(
        self, tenant_id: str, codes: Sequence[str]
    ) -> list[Mapping[str, Any]]:
        """Return one tenant's error-code documents matching any of ``codes``."""
        ...


class MongoMachineRepository:
    """:class:`MachineRepository` backed by the application's Motor database.

    ``list_machines`` reads the whole collection because FUZZY has to score
    every machine anyway; the machine registry is a fleet inventory (hundreds of
    rows, not millions), so this stays cheap.
    """

    def __init__(self, database: Any = None) -> None:
        self._db = database

    def _scope(self, tenant_id: str) -> Any:
        # Imported lazily so importing the resolver never opens a connection.
        from app.db import get_tenant_scope

        return get_tenant_scope(tenant_id, self._db)

    async def list_machines(self, tenant_id: str) -> list[Mapping[str, Any]]:
        cursor = self._scope(tenant_id)[COLLECTIONS.machines].find({})
        return [doc async for doc in cursor]

    async def find_error_codes(
        self, tenant_id: str, codes: Sequence[str]
    ) -> list[Mapping[str, Any]]:
        if not codes:
            return []
        variants: set[str] = set()
        for code in codes:
            variants |= _code_variants(code)
        cursor = self._scope(tenant_id)[COLLECTIONS.error_codes].find(
            {"code": {"$in": sorted(variants)}}
        )
        return [doc async for doc in cursor]


class InMemoryMachineRepository:
    """:class:`MachineRepository` over plain dicts — for tests and dry runs."""

    def __init__(
        self,
        machines: Iterable[Mapping[str, Any]],
        error_codes: Iterable[Mapping[str, Any]] = (),
    ) -> None:
        self._machines = [dict(m) for m in machines]
        self._error_codes = [dict(c) for c in error_codes]

    async def list_machines(self, tenant_id: str) -> list[Mapping[str, Any]]:
        return [m for m in self._machines if m.get("tenant_id") == tenant_id]

    async def find_error_codes(
        self, tenant_id: str, codes: Sequence[str]
    ) -> list[Mapping[str, Any]]:
        variants: set[str] = set()
        for code in codes:
            variants |= _code_variants(code)
        return [
            c
            for c in self._error_codes
            if c.get("code") in variants and c.get("tenant_id") == tenant_id
        ]


# ---------------------------------------------------------------------------
# Internal scoring record
# ---------------------------------------------------------------------------
@dataclass
class _ScoredMachine:
    """A machine under consideration, before it becomes a public candidate."""

    doc: Mapping[str, Any]
    base_score: float
    matched_by: MatchMethod
    matched_value: str
    error_code: Optional[str] = None
    boost: float = 0.0
    boost_reasons: list[str] = field(default_factory=list)

    @property
    def machine_id(self) -> str:
        return str(self.doc.get("machine_id", ""))

    @property
    def confidence(self) -> float:
        """Base score plus context boosts, clamped to the unit interval."""
        return round(min(1.0, max(0.0, self.base_score + self.boost)), 4)

    def to_candidate(self) -> ResolutionCandidate:
        # Context is reported as the match method only when it was context that
        # lifted a weak fuzzy guess — an error-code hit stays an error-code hit.
        matched_by = self.matched_by
        matched_value = self.matched_value
        if self.boost > 0.0 and self.matched_by is MatchMethod.fuzzy:
            matched_by = MatchMethod.context
            matched_value = f"{self.matched_value} + {', '.join(self.boost_reasons)}"

        return ResolutionCandidate(
            machine_id=self.machine_id,
            name=str(self.doc.get("name", "")),
            model=str(self.doc.get("model", "")),
            line_id=str(self.doc.get("line_id", "")),
            status=self.doc.get("status", "running"),
            confidence=self.confidence,
            matched_by=matched_by,
            matched_value=matched_value,
            error_code=self.error_code,
        )


# ---------------------------------------------------------------------------
# Chain stages
# ---------------------------------------------------------------------------
def _match_exact_id(
    machines: Sequence[Mapping[str, Any]], key: str
) -> Optional[_ScoredMachine]:
    """Stage 1 — the input *is* a canonical machine id."""
    if not key:
        return None
    for doc in machines:
        if normalize_key(str(doc.get("machine_id", ""))) == key:
            return _ScoredMachine(
                doc=doc,
                base_score=CONFIDENCE_EXACT_ID,
                matched_by=MatchMethod.exact_id,
                matched_value=str(doc.get("machine_id", "")),
            )
    return None


def _match_alias(
    machines: Sequence[Mapping[str, Any]], key: str
) -> Optional[_ScoredMachine]:
    """Stage 2 — the input is a floor / ERP / drawing name for one machine."""
    if not key:
        return None
    for doc in machines:
        for alias in doc.get("aliases") or []:
            if normalize_key(str(alias)) == key:
                return _ScoredMachine(
                    doc=doc,
                    base_score=CONFIDENCE_ALIAS,
                    matched_by=MatchMethod.alias,
                    matched_value=str(alias),
                )
    return None


def _match_error_codes(
    machines: Sequence[Mapping[str, Any]],
    error_code_docs: Sequence[Mapping[str, Any]],
) -> list[_ScoredMachine]:
    """Stage 3 — codes name a machine *model*, which may map to many machines."""
    scored: list[_ScoredMachine] = []
    claimed: set[str] = set()

    for code_doc in error_code_docs:
        model = str(code_doc.get("machine_model", ""))
        code = str(code_doc.get("code", ""))
        if not model:
            continue
        for doc in machines:
            machine_id = str(doc.get("machine_id", ""))
            if str(doc.get("model", "")) != model or machine_id in claimed:
                continue
            claimed.add(machine_id)
            scored.append(
                _ScoredMachine(
                    doc=doc,
                    base_score=CONFIDENCE_ERROR_CODE,
                    matched_by=MatchMethod.error_code,
                    matched_value=code,
                    error_code=code,
                )
            )
    return scored


def _match_fuzzy(
    machines: Sequence[Mapping[str, Any]], text: str
) -> list[_ScoredMachine]:
    """Stage 5 — approximate match over id, name, model and every alias."""
    query = text.strip()
    if not query:
        return []

    scored: list[_ScoredMachine] = []
    for doc in machines:
        haystack: list[str] = [
            str(doc.get("machine_id", "")),
            str(doc.get("name", "")),
            str(doc.get("model", "")),
        ]
        haystack.extend(str(a) for a in (doc.get("aliases") or []))

        best_value, best_score = "", 0.0
        for value in haystack:
            if not value:
                continue
            score = fuzz.token_set_ratio(query, value) / 100.0
            if score > best_score:
                best_value, best_score = value, score

        if best_score >= MIN_CANDIDATE_CONFIDENCE - _EPSILON:
            scored.append(
                _ScoredMachine(
                    doc=doc,
                    base_score=round(best_score, 4),
                    matched_by=MatchMethod.fuzzy,
                    matched_value=best_value,
                )
            )
    return scored


def _apply_context(
    scored: Sequence[_ScoredMachine], context: ResolutionContext
) -> None:
    """Stage 4 — corroborate candidates with what we know about the operator.

    Mutates in place: each candidate accumulates a boost and a human-readable
    reason for it, so the audit trail explains *why* one option won.
    """
    active_alarms = {normalize_key(m) for m in (context.active_alarm_machine_ids or [])}
    last_machine = normalize_key(context.last_machine_id or "")
    assigned_line = normalize_key(context.assigned_line_id or "")

    for item in scored:
        machine_key = normalize_key(item.machine_id)

        if last_machine and machine_key == last_machine:
            item.boost += BOOST_LAST_MACHINE
            item.boost_reasons.append("previous turn")

        if assigned_line and normalize_key(str(item.doc.get("line_id", ""))) == assigned_line:
            item.boost += BOOST_ASSIGNED_LINE
            item.boost_reasons.append("assigned line")

        if machine_key in active_alarms:
            item.boost += BOOST_ACTIVE_ALARM
            item.boost_reasons.append("active alarm")

        if str(item.doc.get("status", "")).lower() in _ATTENTION_STATUSES:
            item.boost += BOOST_FAULTED_STATUS
            item.boost_reasons.append("machine is down")


# ---------------------------------------------------------------------------
# Merging, ranking, deciding
# ---------------------------------------------------------------------------
def _merge(*groups: Sequence[_ScoredMachine]) -> list[_ScoredMachine]:
    """Collapse candidate groups by machine_id, keeping the strongest evidence.

    A machine surfaced by both ERROR_CODE and FUZZY should appear once, carrying
    the error-code provenance rather than a weaker fuzzy string.
    """
    best: dict[str, _ScoredMachine] = {}
    for group in groups:
        for item in group:
            existing = best.get(item.machine_id)
            if existing is None or item.base_score > existing.base_score:
                # Preserve an error code discovered by another stage.
                if existing is not None and item.error_code is None:
                    item.error_code = existing.error_code
                best[item.machine_id] = item
            elif existing.error_code is None and item.error_code is not None:
                existing.error_code = item.error_code
    return list(best.values())


def _rank(scored: Sequence[_ScoredMachine]) -> list[_ScoredMachine]:
    """Sort by confidence descending; machine_id breaks ties deterministically."""
    return sorted(scored, key=lambda s: (-s.confidence, s.machine_id))


def build_clarification_question(candidates: Sequence[ResolutionCandidate]) -> str:
    """Phrase a question that names the actual options, never a generic prompt."""
    if not candidates:
        return "Which machine do you mean? I could not match that to any machine."

    described = [f"{c.machine_id} ({c.name})" for c in candidates]
    if len(described) == 1:
        return f"Did you mean {described[0]}?"

    options = f"{', '.join(described[:-1])} or {described[-1]}"

    lines = {c.line_id for c in candidates if c.line_id}
    if len(lines) == 1:
        return f"{len(described)} possible matches on {lines.pop()} — {options}?"
    return f"{len(described)} possible matches — {options}?"


def decide(
    scored: Sequence[_ScoredMachine],
    raw_input: str,
    extracted_error_codes: Sequence[str],
) -> ResolutionResult:
    """Apply the decision rules to a scored pool and produce the final result.

    RESOLVED requires both a high top score *and* clear separation from the
    runner-up. Anything else blocks: a near-tie between two machines is exactly
    the case where guessing gets someone hurt.
    """
    ranked = [
        s for s in _rank(scored) if s.confidence >= MIN_CANDIDATE_CONFIDENCE - _EPSILON
    ]

    if not ranked:
        return ResolutionResult(
            status=ResolutionStatus.not_found,
            machine=None,
            candidates=[],
            clarification_question=None,
            extracted_error_codes=list(extracted_error_codes),
            raw_input=raw_input,
        )

    candidates = [s.to_candidate() for s in ranked[:MAX_CANDIDATES]]
    top = candidates[0]
    clear_margin = len(ranked) == 1 or (
        (top.confidence - candidates[1].confidence) >= RESOLVE_MIN_MARGIN - _EPSILON
    )

    if top.confidence >= RESOLVE_MIN_CONFIDENCE - _EPSILON and clear_margin:
        return ResolutionResult(
            status=ResolutionStatus.resolved,
            machine=top,
            candidates=[top],
            clarification_question=None,
            extracted_error_codes=list(extracted_error_codes),
            raw_input=raw_input,
        )

    return ResolutionResult(
        status=ResolutionStatus.ambiguous,
        machine=None,
        candidates=candidates,
        clarification_question=build_clarification_question(candidates),
        extracted_error_codes=list(extracted_error_codes),
        raw_input=raw_input,
    )


def _single(
    scored: _ScoredMachine, raw_input: str, codes: Sequence[str]
) -> ResolutionResult:
    """Wrap a short-circuit hit from stage 1/2 as a RESOLVED result."""
    candidate = scored.to_candidate()
    return ResolutionResult(
        status=ResolutionStatus.resolved,
        machine=candidate,
        candidates=[candidate],
        clarification_question=None,
        extracted_error_codes=list(codes),
        raw_input=raw_input,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
async def resolve_machine(
    text: str,
    tenant_id: str,
    context: Optional[ResolutionContext] = None,
    repository: Optional[MachineRepository] = None,
) -> ResolutionResult:
    """Resolve free text to exactly one machine within one tenant, or block.

    ``tenant_id`` is required and positional: resolving "the conveyor" against
    the wrong tenant's fleet would hand an operator another plant's lockout
    procedure, so there is no unscoped call to make by accident.

    Callable directly by other services — no HTTP required. Pass a
    ``repository`` to resolve against something other than the live database.
    """
    from app.db import normalize_tenant_id

    tenant_id = normalize_tenant_id(tenant_id)
    context = context or ResolutionContext()
    repo = repository or MongoMachineRepository()
    raw_input = text
    key = normalize_key(text)
    codes = extract_error_codes(text)

    machines = await repo.list_machines(tenant_id)
    if not machines:
        return ResolutionResult(
            status=ResolutionStatus.not_found,
            extracted_error_codes=codes,
            raw_input=raw_input,
        )

    # 1. EXACT_ID — unambiguous by construction, stop here.
    exact = _match_exact_id(machines, key)
    if exact is not None:
        return _single(exact, raw_input, codes)

    # 2. ALIAS — an exact alias is equally unambiguous.
    alias = _match_alias(machines, key)
    if alias is not None:
        return _single(alias, raw_input, codes)

    # 3. ERROR_CODE — one machine ends the chain; several fall through.
    by_code: list[_ScoredMachine] = []
    if codes:
        code_docs = await repo.find_error_codes(tenant_id, codes)
        by_code = _match_error_codes(machines, code_docs)
        if len(by_code) == 1:
            return _single(by_code[0], raw_input, codes)

    # 5. FUZZY — gather approximate candidates to sit alongside the code hits.
    by_fuzzy = _match_fuzzy(machines, text)

    pool = _merge(by_code, by_fuzzy)

    # 4. CONTEXT — applied last because it grades the pool rather than filling it.
    _apply_context(pool, context)

    return decide(pool, raw_input, codes)
