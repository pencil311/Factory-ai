"""Retrieval-augmented knowledge engine.

This package RETRIEVES ONLY. It parses, chunks, embeds, indexes and returns
passages with provenance. It does not diagnose, recommend, summarise, or
reason about what it returns — downstream agents do that. Every function here
that could be tempted to interpret returns raw text plus citations instead.
"""
