"""Tests for the entity-resolution chain.

The database is mocked via :class:`InMemoryMachineRepository`, so no live Mongo
is required. The fixture fleet deliberately contains two same-model conveyors on
one line — the ambiguity this service exists to catch.

Every fixture belongs to :data:`TENANT`; resolution across tenants is covered
in ``tests/test_tenancy.py``.
"""

from __future__ import annotations

import pytest

from app.schemas.resolution import (
    MatchMethod,
    ResolutionContext,
    ResolutionStatus,
)
from app.services.resolver import (
    BOOST_LAST_MACHINE,
    InMemoryMachineRepository,
    MatchMethod as _MatchMethod,  # re-exported for convenience in assertions
    _ScoredMachine,
    decide,
    extract_error_codes,
    normalize_key,
    resolve_machine,
)


#: The tenant every fixture in this module belongs to.
TENANT = "demo"


async def _resolve(text, context=None, repository=None):
    """Resolve within :data:`TENANT`.

    ``resolve_machine`` takes the tenant as a required positional argument, so
    this wrapper keeps that explicit at the one place it is supplied rather
    than repeating it in forty calls.
    """
    return await resolve_machine(
        text, TENANT, context=context, repository=repository
    )


# ---------------------------------------------------------------------------
# Fixture fleet
# ---------------------------------------------------------------------------
CV_201 = {
    "tenant_id": TENANT,
    "machine_id": "CV-201",
    "name": "Infeed Belt Conveyor",
    "model": "SpanTech SB-3000",
    "manufacturer": "SpanTech",
    "site_id": "SITE-DETROIT",
    "line_id": "LINE-A",
    "position_in_line": 1,
    "criticality": 3,
    "status": "running",
    "aliases": ["Infeed Conveyor", "Line A Belt 1", "ERP-CNV-0201"],
}

CV_204 = {
    "tenant_id": TENANT,
    "machine_id": "CV-204",
    "name": "Outfeed Belt Conveyor",
    "model": "SpanTech SB-3000",
    "manufacturer": "SpanTech",
    "site_id": "SITE-DETROIT",
    "line_id": "LINE-A",
    "position_in_line": 4,
    "criticality": 3,
    "status": "running",
    "aliases": ["Outfeed Conveyor", "Line A Belt 2"],
}

MC_110 = {
    "tenant_id": TENANT,
    "machine_id": "MC-110",
    "name": "3-Axis CNC Milling Center",
    "model": "Haas VF-4",
    "manufacturer": "Haas Automation",
    "site_id": "SITE-DETROIT",
    "line_id": "LINE-B",
    "position_in_line": 2,
    "criticality": 5,
    "status": "fault",
    "aliases": ["Big Haas", "Mill 1"],
}

ERROR_CODES = [
    {
        "tenant_id": TENANT,
        "code": "E104",
        "machine_model": "Haas VF-4",
        "description": "Spindle over-temperature",
        "fault_class": "mechanical",
    },
    {
        "tenant_id": TENANT,
        "code": "E-220",
        "machine_model": "SpanTech SB-3000",
        "description": "Belt slip detected",
        "fault_class": "mechanical",
    },
]


@pytest.fixture
def repo() -> InMemoryMachineRepository:
    """A three-machine fleet with two indistinguishable-by-model conveyors."""
    return InMemoryMachineRepository(
        machines=[CV_201, CV_204, MC_110], error_codes=ERROR_CODES
    )


# ---------------------------------------------------------------------------
# Stage 1 — EXACT_ID
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["CV-201", "cv-201", "  cv 201 ", "cv201", "CV-201."])
async def test_exact_id_resolves_regardless_of_case_and_punctuation(repo, text):
    result = await _resolve(text, repository=repo)

    assert result.status == ResolutionStatus.resolved
    assert result.machine is not None
    assert result.machine.machine_id == "CV-201"
    assert result.machine.confidence == 1.0
    assert result.machine.matched_by == MatchMethod.exact_id
    assert result.machine.matched_value == "CV-201"
    assert result.raw_input == text
    assert result.is_blocking is False


# ---------------------------------------------------------------------------
# Stage 2 — ALIAS
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_alias_resolves_with_alias_confidence(repo):
    result = await _resolve("ERP-CNV-0201", repository=repo)

    assert result.status == ResolutionStatus.resolved
    assert result.machine.machine_id == "CV-201"
    assert result.machine.matched_by == MatchMethod.alias
    assert result.machine.matched_value == "ERP-CNV-0201"
    assert result.machine.confidence == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_alias_match_is_case_insensitive(repo):
    result = await _resolve("big haas", repository=repo)

    assert result.status == ResolutionStatus.resolved
    assert result.machine.machine_id == "MC-110"
    assert result.machine.matched_by == MatchMethod.alias


# ---------------------------------------------------------------------------
# Stage 3 — ERROR_CODE
# ---------------------------------------------------------------------------
def test_extract_error_codes_from_a_sentence():
    codes = extract_error_codes("The mill threw error E104 last night, then E-220")

    assert codes == ["E104", "E-220"]


def test_extract_error_codes_ignores_bare_words_and_numbers():
    assert extract_error_codes("the conveyor is jammed again") == []
    assert extract_error_codes("it stopped at 14:30") == []


@pytest.mark.asyncio
async def test_error_code_in_a_sentence_resolves_to_the_only_machine_of_that_model(repo):
    result = await _resolve(
        "second shift says it threw error E104 overnight", repository=repo
    )

    assert result.extracted_error_codes == ["E104"]
    assert result.status == ResolutionStatus.resolved
    assert result.machine.machine_id == "MC-110"
    assert result.machine.matched_by == MatchMethod.error_code
    assert result.machine.error_code == "E104"
    assert result.machine.confidence == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_error_code_matching_two_machines_is_ambiguous(repo):
    """E-220 belongs to a model both conveyors share — that must block."""
    result = await _resolve("we're getting E-220 on the belt", repository=repo)

    assert result.status == ResolutionStatus.ambiguous
    assert result.machine is None
    assert len(result.candidates) >= 2

    ids = {c.machine_id for c in result.candidates}
    assert {"CV-201", "CV-204"} <= ids
    assert result.clarification_question is not None
    assert "CV-201" in result.clarification_question
    assert "CV-204" in result.clarification_question
    assert result.is_blocking is True


@pytest.mark.asyncio
async def test_error_code_lookup_tolerates_separator_spelling(repo):
    """'E220' in the text must still find the stored 'E-220'."""
    result = await _resolve("fault E220 again", repository=repo)

    assert result.extracted_error_codes == ["E220"]
    assert {c.machine_id for c in result.candidates} >= {"CV-201", "CV-204"}


# ---------------------------------------------------------------------------
# Ambiguity and clarification
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ambiguous_input_returns_ranked_candidates_and_a_specific_question(repo):
    result = await _resolve("belt conveyor", repository=repo)

    assert result.status == ResolutionStatus.ambiguous
    assert result.machine is None
    assert len(result.candidates) >= 2

    confidences = [c.confidence for c in result.candidates]
    assert confidences == sorted(confidences, reverse=True)
    assert len(result.candidates) <= 5

    question = result.clarification_question
    assert question is not None
    # The question must name the actual options, not ask a generic "which one?".
    assert "CV-201" in question and "CV-204" in question
    assert "LINE-A" in question


# ---------------------------------------------------------------------------
# NOT_FOUND
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_garbage_input_is_not_found(repo):
    result = await _resolve("qqzxjv wibble 9981 flurm", repository=repo)

    assert result.status == ResolutionStatus.not_found
    assert result.machine is None
    assert result.candidates == []
    assert result.clarification_question is None
    assert result.is_blocking is True


@pytest.mark.asyncio
async def test_empty_fleet_is_not_found():
    empty = InMemoryMachineRepository(machines=[], error_codes=[])
    result = await _resolve("CV-201", repository=empty)

    assert result.status == ResolutionStatus.not_found


# ---------------------------------------------------------------------------
# Stage 4 — CONTEXT
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_context_boost_changes_the_winner_between_equal_fuzzy_candidates(repo):
    """Two conveyors the text cannot separate; context decides which ranks first."""
    neutral = await _resolve("belt conveyor", repository=repo)
    top_two = [c.machine_id for c in neutral.candidates[:2]]
    assert set(top_two) == {"CV-201", "CV-204"}
    # "Infeed"/"Outfeed" differ by one character, so the fuzzy scores are near
    # enough to be indistinguishable evidence — well inside a single boost.
    spread = neutral.candidates[0].confidence - neutral.candidates[1].confidence
    assert spread < BOOST_LAST_MACHINE
    assert neutral.candidates[0].machine_id == "CV-201"

    boosted = await _resolve(
        "belt conveyor",
        context=ResolutionContext(
            last_machine_id="CV-204", active_alarm_machine_ids=["CV-204"]
        ),
        repository=repo,
    )

    assert boosted.candidates[0].machine_id == "CV-204"
    assert boosted.candidates[0].confidence > neutral.candidates[0].confidence
    assert boosted.candidates[0].matched_by == MatchMethod.context


@pytest.mark.asyncio
async def test_assigned_line_boost_ranks_the_operators_own_line_first(repo):
    result = await _resolve(
        "belt conveyor",
        context=ResolutionContext(assigned_line_id="LINE-A"),
        repository=repo,
    )

    assert result.candidates[0].line_id == "LINE-A"


@pytest.mark.asyncio
async def test_context_alone_cannot_promote_a_machine_that_text_never_matched(repo):
    """Context corroborates; it must never conjure a candidate out of nothing."""
    result = await _resolve(
        "qqzxjv wibble flurm",
        context=ResolutionContext(
            last_machine_id="CV-201",
            assigned_line_id="LINE-A",
            active_alarm_machine_ids=["CV-201"],
        ),
        repository=repo,
    )

    assert result.status == ResolutionStatus.not_found
    assert result.candidates == []


@pytest.mark.asyncio
async def test_context_does_not_override_an_exact_id(repo):
    """An exact id wins even when context points somewhere else entirely."""
    result = await _resolve(
        "CV-204",
        context=ResolutionContext(
            last_machine_id="CV-201", active_alarm_machine_ids=["CV-201"]
        ),
        repository=repo,
    )

    assert result.status == ResolutionStatus.resolved
    assert result.machine.machine_id == "CV-204"


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------
def _scored(machine: dict, score: float) -> _ScoredMachine:
    return _ScoredMachine(
        doc=machine,
        base_score=score,
        matched_by=_MatchMethod.fuzzy,
        matched_value=machine["name"],
    )


def test_margin_rule_blocks_two_close_high_scorers():
    """0.90 vs 0.88: both above threshold, but 0.02 apart — must ask."""
    result = decide([_scored(CV_201, 0.90), _scored(CV_204, 0.88)], "belt", [])

    assert result.status == ResolutionStatus.ambiguous
    assert result.machine is None
    assert [c.machine_id for c in result.candidates] == ["CV-201", "CV-204"]
    assert result.clarification_question is not None


def test_clear_margin_resolves():
    """0.95 vs 0.60: above threshold and well clear — safe to act."""
    result = decide([_scored(CV_201, 0.95), _scored(CV_204, 0.60)], "infeed", [])

    assert result.status == ResolutionStatus.resolved
    assert result.machine.machine_id == "CV-201"


def test_high_score_alone_is_not_enough_without_margin():
    """Exactly at the margin boundary (0.15) resolves; a hair under does not."""
    at_boundary = decide([_scored(CV_201, 0.95), _scored(CV_204, 0.80)], "x", [])
    assert at_boundary.status == ResolutionStatus.resolved

    under = decide([_scored(CV_201, 0.95), _scored(CV_204, 0.81)], "x", [])
    assert under.status == ResolutionStatus.ambiguous


def test_lone_candidate_below_confidence_threshold_is_ambiguous_not_resolved():
    """One weak match is a question, not an answer."""
    result = decide([_scored(CV_201, 0.60)], "conveyer thing", [])

    assert result.status == ResolutionStatus.ambiguous
    assert result.clarification_question == "Did you mean CV-201 (Infeed Belt Conveyor)?"


def test_candidates_below_the_floor_are_dropped_entirely():
    result = decide([_scored(CV_201, 0.54), _scored(CV_204, 0.10)], "noise", [])

    assert result.status == ResolutionStatus.not_found
    assert result.candidates == []


def test_candidate_list_is_capped_at_five():
    fleet = [dict(CV_201, machine_id=f"CV-{i:03d}") for i in range(10)]
    result = decide([_scored(m, 0.70) for m in fleet], "conveyor", [])

    assert result.status == ResolutionStatus.ambiguous
    assert len(result.candidates) == 5


def test_extracted_error_codes_are_reported_even_when_not_found():
    result = decide([], "E999 on something", ["E999"])

    assert result.status == ResolutionStatus.not_found
    assert result.extracted_error_codes == ["E999"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("CV-201", "CV201"),
        (" cv 201 ", "CV201"),
        ("cv_201.", "CV201"),
        ("", ""),
    ],
)
def test_normalize_key(raw, expected):
    assert normalize_key(raw) == expected
