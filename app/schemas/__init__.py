"""API/response schemas and Mongo collection metadata for FactoryPilot AI."""

from app.schemas.machine import (
    COLLECTIONS,
    ComponentOut,
    ErrorCodeOut,
    MachineOut,
    ProductionLineOut,
    SensorOut,
    SiteOut,
    strip_mongo_id,
)

__all__ = [
    "COLLECTIONS",
    "ComponentOut",
    "ErrorCodeOut",
    "MachineOut",
    "ProductionLineOut",
    "SensorOut",
    "SiteOut",
    "strip_mongo_id",
]
