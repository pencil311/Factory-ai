"""The AI Orchestrator.

The orchestrator does not answer questions. It decides which modules take part,
runs them in dependency order with as much parallelism as the graph allows,
aggregates their structured results, and composes natural language exactly once
at the end. Modules never call each other.
"""
