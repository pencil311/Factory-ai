"""Tenant model.

A tenant is the ownership boundary for every other document in the system.
Nothing exists outside a tenant: ``tenant_id`` is required — not optional —
on every model, so a document that has not been assigned an owner cannot be
constructed, let alone written.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class UnitSystem(str, Enum):
    """Measurement system a tenant's UI renders in."""

    metric = "METRIC"
    imperial = "IMPERIAL"


class TenantSettings(BaseModel):
    """Per-tenant presentation preferences."""

    model_config = ConfigDict(use_enum_values=True)

    timezone: str = Field(default="UTC", description="IANA timezone, e.g. 'America/Detroit'")
    units: UnitSystem = UnitSystem.metric


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Tenant(BaseModel):
    """An organisation using FactoryPilot. The root of every ownership chain."""

    model_config = ConfigDict(use_enum_values=True, populate_by_name=True)

    tenant_id: str = Field(..., min_length=1, description="Canonical tenant id, e.g. 'demo'")
    name: str = Field(..., min_length=1, description="Display name, e.g. 'Demo Manufacturing'")
    slug: str = Field(..., min_length=1, description="URL-safe short name, e.g. 'demo'")
    created_at: datetime = Field(default_factory=utcnow)
    is_active: bool = True
    settings: TenantSettings = Field(default_factory=TenantSettings)

    @field_validator("tenant_id", "slug")
    @classmethod
    def _no_surrounding_whitespace(cls, value: str) -> str:
        """Reject padded ids: ' demo' and 'demo' must never be two tenants."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        if stripped != value:
            raise ValueError("must not have leading or trailing whitespace")
        return stripped
