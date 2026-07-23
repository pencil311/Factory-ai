"""Turning a caught exception into what an operator sees vs. what gets logged.

An exception's ``str()`` can carry things that must never reach a narrative —
a MongoDB error document with cluster timestamps and signature bytes, a raw
connection string, a stack frame. Every place that turns a caught exception
into a module or agent failure goes through :func:`describe_failure` so the
split is made the same way everywhere: a short, generic ``reason`` safe to
compose into prose, and the exception's full text as ``error_detail``, kept
for logs and for whoever debugs the run but never rendered into an answer.
"""

from __future__ import annotations

#: Reasons are meant for a one-line status, not a paragraph.
MAX_REASON_LENGTH = 120


def describe_failure(exc: Exception, *, label: str) -> tuple[str, str]:
    """``(reason, error_detail)`` for ``label`` (e.g. "Knowledge base query") having raised ``exc``.

    ``reason`` never includes the exception's own message — that message is
    exactly where server or driver internals leak — only its type name, which
    is informative without being a data leak. ``error_detail`` keeps the full
    text for debugging and is never meant to reach a narrative.
    """
    reason = _truncate(f"{label} failed ({type(exc).__name__}).")
    error_detail = f"{type(exc).__name__}: {exc}"
    return reason, error_detail


def _truncate(text: str, limit: int = MAX_REASON_LENGTH) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
