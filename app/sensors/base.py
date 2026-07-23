"""The abstraction every sensor source implements.

Downstream code depends on this interface and nothing else, which is what makes
the simulator swappable for real hardware without touching ingestion, the
routers, or the dashboard.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from app.models.reading import ReadingSource, SensorReading


class SensorSourceError(RuntimeError):
    """Raised when a source cannot be used as asked (e.g. read while closed)."""


class SensorSource(ABC):
    """A source of :class:`SensorReading` values, bound to one tenant.

    Sources carry ``tenant_id`` because a reading is meaningless without an
    owner: the field is required on :class:`SensorReading`, so a source that
    did not know its tenant could not construct one.

    Lifecycle is explicit — ``connect()`` before use, ``disconnect()`` after —
    because real drivers hold sockets and subscriptions that must be released.
    Implementations must be safe to connect and disconnect repeatedly.
    """

    #: Provenance stamped onto every reading this source emits.
    source_type: ReadingSource

    @property
    @abstractmethod
    def tenant_id(self) -> str:
        """The tenant every reading from this source belongs to."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short human-readable identifier, e.g. ``"simulator"``."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """True when the source is ready to serve readings."""

    @abstractmethod
    async def connect(self) -> None:
        """Open the connection / start the driver. Idempotent."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Release resources. Idempotent, and safe to call when never connected."""

    # Typed as a plain method returning an async iterator: implementations are
    # async *generator* functions (`async def stream(self): ... yield ...`), so
    # callers write `async for reading in source.stream():`.
    @abstractmethod
    def stream(self) -> AsyncIterator[SensorReading]:
        """Yield readings continuously until cancelled."""

    @abstractmethod
    async def read_once(self, sensor_id: str) -> SensorReading:
        """Return the current value of one sensor without advancing the stream.

        Raises :class:`KeyError` if the sensor is unknown to this source.
        """

    def _require_connected(self) -> None:
        """Guard for methods that need a live connection."""
        if not self.is_connected:
            raise SensorSourceError(
                f"Sensor source '{self.name}' is not connected — call connect() first"
            )
