"""Embedding providers.

Provider-agnostic by design: the vector store and retriever depend only on
:class:`EmbeddingProvider`, so swapping local inference for a hosted API is a
config change, not a code change.

``dimension`` is a property, never a constant. The Atlas vector index is built
from whatever the active provider reports, so changing models cannot leave a
384-dimension index being fed 1536-dimension vectors — a mismatch that Atlas
reports as an empty result set rather than an error, which is the worst
possible failure mode.

Three implementations:

``LocalEmbeddings``
    sentence-transformers, default ``all-MiniLM-L6-v2``. No API key, runs
    offline once the model is cached. The default.
``APIEmbeddings``
    Wired for a hosted provider, selected by config. Raises at the network
    call rather than pretending to work.
``HashingEmbeddings``
    Deterministic, dependency-light, no model download. This is what the test
    suite uses so it can run with no network at all, and it is also the
    honest fallback when the model cannot be loaded.
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)

LOCAL = "local"
API = "api"
HASHING = "hashing"

DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"
DEFAULT_HASHING_DIMENSION = 384

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-_/.]*")


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced."""


class EmbeddingProvider(ABC):
    """The interface the rest of the RAG stack depends on."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier reported by /knowledge/status, e.g. ``local:all-MiniLM-L6-v2``."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector width. The index is built from this, never from a literal."""

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch. Returns ``(len(texts), dimension)``, L2-normalised.

        Normalising here means cosine similarity is a plain dot product
        everywhere downstream, and both backends agree on what a score means.
        """

    def encode_one(self, text: str) -> list[float]:
        """Embed a single string as a plain list, ready for Mongo."""
        return self.encode([text])[0].tolist()


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale rows to unit length, leaving all-zero rows alone."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)


class LocalEmbeddings(EmbeddingProvider):
    """sentence-transformers running in-process.

    The model is loaded lazily on first use: importing this module must stay
    cheap, and a process that never embeds should never pay to load it.
    """

    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL, batch_size: int = 32) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None
        self._dimension: Optional[int] = None

    @property
    def name(self) -> str:
        return f"{LOCAL}:{self._model_name}"

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - env-dependent
                raise EmbeddingError(
                    "sentence-transformers is not installed. Install it, or set "
                    "EMBEDDING_PROVIDER=hashing to run without it."
                ) from exc
            logger.info("Loading embedding model '%s'", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            self._dimension = int(self._model.get_sentence_embedding_dimension())
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load()
        assert self._dimension is not None
        return self._dimension

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        model = self._load()
        vectors = model.encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class APIEmbeddings(EmbeddingProvider):
    """Hosted embedding provider.

    Fully wired apart from the HTTP call itself: config, batching, dimension
    reporting and normalisation are all real, so enabling this is a localised
    change. It raises at the request rather than returning zero vectors, which
    would poison the index silently.
    """

    def __init__(
        self,
        model_name: str,
        dimension: int,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        batch_size: int = 64,
    ) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._api_key = api_key
        self._endpoint = endpoint
        self._batch_size = batch_size

    @property
    def name(self) -> str:
        return f"{API}:{self._model_name}"

    @property
    def dimension(self) -> int:
        # Declared by config: a hosted model's width is known ahead of the
        # first call, and the index must be creatable before any data exists.
        return self._dimension

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        if not self._api_key:
            raise EmbeddingError(
                "EMBEDDING_PROVIDER=api but EMBEDDING_API_KEY is unset."
            )
        raise NotImplementedError(
            "Hosted embedding I/O is not wired up. To enable: POST batches of "
            f"<= {self._batch_size} texts to '{self._endpoint}' with model "
            f"'{self._model_name}', collect the returned vectors in request "
            "order, and return _l2_normalize(np.asarray(vectors))."
        )


class HashingEmbeddings(EmbeddingProvider):
    """Deterministic bag-of-tokens hashing embeddings.

    No model, no download, no network — which is what lets the test suite and
    a cold CI box exercise the whole pipeline. Semantically far weaker than a
    trained encoder, so it is never selected implicitly: it is either asked for
    by config or used as an explicitly-logged fallback.

    Tokens are hashed into a fixed number of buckets with sub-linear term
    weighting, so repeated exact strings (error codes, part numbers) produce
    stable, comparable vectors.
    """

    def __init__(self, dimension: int = DEFAULT_HASHING_DIMENSION) -> None:
        self._dimension = dimension

    @property
    def name(self) -> str:
        return f"{HASHING}:d{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _TOKEN_RE.findall((text or "").lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "big") % self._dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                matrix[row, bucket] += sign
            # Sub-linear damping so a term repeated ten times does not
            # dominate a passage that mentions it twice.
            matrix[row] = np.sign(matrix[row]) * np.sqrt(np.abs(matrix[row]))
        return _l2_normalize(matrix)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
_provider: Optional[EmbeddingProvider] = None


def build_provider(kind: Optional[str] = None) -> EmbeddingProvider:
    """Construct a provider without touching the process-wide singleton."""
    settings = get_settings()
    name = (kind or settings.embedding_provider or LOCAL).strip().lower()

    if name == LOCAL:
        return LocalEmbeddings(
            model_name=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
        )
    if name == API:
        return APIEmbeddings(
            model_name=settings.embedding_model,
            dimension=settings.embedding_dimension,
            api_key=settings.embedding_api_key or None,
            endpoint=settings.embedding_api_endpoint or None,
            batch_size=settings.embedding_batch_size,
        )
    if name == HASHING:
        return HashingEmbeddings(dimension=settings.embedding_dimension)

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER '{name}'. Expected one of: "
        f"{LOCAL}, {API}, {HASHING}."
    )


def get_embedding_provider() -> EmbeddingProvider:
    """The process-wide provider, created on first use."""
    global _provider
    if _provider is None:
        _provider = build_provider()
        logger.info("Embedding provider: %s", _provider.name)
    return _provider


def set_embedding_provider(provider: Optional[EmbeddingProvider]) -> None:
    """Replace the active provider. For tests and explicit wiring."""
    global _provider
    _provider = provider
