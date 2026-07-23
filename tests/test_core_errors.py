"""Tests for `app.core.errors.describe_failure`.

This is the single choke point every module and agent failure passes through
to split a caught exception into a safe, short ``reason`` and a full
``error_detail`` — see app/orchestrator/executor.py and app/agents/base.py.
"""

from __future__ import annotations

from app.core.errors import MAX_REASON_LENGTH, describe_failure


def test_reason_never_contains_the_exception_message():
    exc = RuntimeError(
        "Collection 'factorypilot.chunks' does not exist. $clusterTime: "
        "{clusterTime: Timestamp(1, 2), signature: {hash: b'\\x00', keyId: 7}}"
    )
    reason, detail = describe_failure(exc, label="Knowledge base query")

    assert "clusterTime" not in reason
    assert "signature" not in reason
    assert "Knowledge base query" in reason
    assert "RuntimeError" in reason
    # The full text is preserved, just not in reason.
    assert "clusterTime" in detail
    assert detail == f"RuntimeError: {exc}"


def test_reason_is_truncated_to_120_characters():
    exc = RuntimeError("x" * 500)
    reason, _detail = describe_failure(exc, label="A" * 200)

    assert len(reason) <= MAX_REASON_LENGTH


def test_short_reason_is_left_untouched():
    exc = ValueError("bad input")
    reason, _detail = describe_failure(exc, label="Inventory check")

    assert reason == "Inventory check failed (ValueError)."
    assert len(reason) < MAX_REASON_LENGTH
