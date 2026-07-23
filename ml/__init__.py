"""FactoryPilot ML: dataset acquisition, feature extraction, training, evaluation.

This package is imported by the serving path (``app/services/pdm.py`` imports
``ml.preprocess``) — that shared import is what guarantees train/serve feature
parity, so keep this package importable without heavy side effects.
"""
