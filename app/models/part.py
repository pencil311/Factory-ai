"""Part / inventory model."""

from __future__ import annotations

from typing import Optional

from pydantic import ConfigDict, Field

from app.models.machine import TenantOwned


class Part(TenantOwned):
    """A spare part in the inventory system."""

    model_config = ConfigDict(use_enum_values=True)

    part_number: str = Field(..., min_length=1)
    description: str
    category: str = "general"
    compatible_components: list[str] = Field(default_factory=list)
    compatible_machine_models: list[str] = Field(default_factory=list)
    quantity_on_hand: int = Field(default=0, ge=0)
    reorder_level: int = Field(default=1, ge=0)
    warehouse_location: Optional[str] = None
    unit_cost: float = Field(default=0.0, ge=0)
    lead_time_days: int = Field(default=7, ge=0)
    supplier: str = ""
    alternative_part_numbers: list[str] = Field(default_factory=list)
