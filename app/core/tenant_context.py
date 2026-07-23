"""Resolving which tenant a request belongs to.

Today the tenant comes from the ``X-Tenant-Id`` header, falling back to
``DEFAULT_TENANT_ID`` from the environment. That is deliberately thin: it is
the seam real authentication plugs into later, when the tenant will come from
a verified token claim instead. Everything downstream — routers, services,
repositories — already takes the tenant as an explicit argument, so swapping
the source out is a change to this file alone.

Resolution is separate from *validation* on purpose:

* :func:`resolve_tenant_id` is pure and importable from anywhere.
* :func:`get_current_tenant` is the FastAPI dependency, which additionally
  confirms the tenant exists and is active. An unrecognised header value must
  not conjure an empty namespace that silently accepts writes.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import Depends, Header, HTTPException, status

from app.config import get_settings
from app.db import TenantScope, get_database, get_tenant_scope
from app.schemas.machine import COLLECTIONS

logger = logging.getLogger(__name__)

TENANT_HEADER = "X-Tenant-Id"

#: Tenants change rarely; caching keeps a DB round-trip off every request.
_TENANT_CACHE_TTL_SECONDS = 60.0
_tenant_cache: dict[str, tuple[float, bool]] = {}


def resolve_tenant_id(header_value: Optional[str] = None) -> str:
    """Return the tenant for this request, or raise 400.

    Order: explicit header, then the configured default. A blank header is
    treated as absent rather than as a tenant named ``""``.
    """
    candidate = (header_value or "").strip()
    if not candidate:
        candidate = (get_settings().default_tenant_id or "").strip()

    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"No tenant could be resolved for this request. Send a "
                f"'{TENANT_HEADER}' header, or set DEFAULT_TENANT_ID in the "
                f"environment."
            ),
        )
    return candidate


def clear_tenant_cache() -> None:
    """Drop the tenant existence cache. Used by tests and after seeding."""
    _tenant_cache.clear()


async def _tenant_is_active(tenant_id: str) -> bool:
    """True when the tenant exists and is active, memoised briefly."""
    now = time.monotonic()
    cached = _tenant_cache.get(tenant_id)
    if cached is not None and cached[0] > now:
        return cached[1]

    # The tenant registry is the one collection that is not tenant-scoped: it
    # is what defines the scopes, so it cannot be read through one.
    doc = await get_database()[COLLECTIONS.tenants].find_one({"tenant_id": tenant_id})
    active = bool(doc and doc.get("is_active", False))
    _tenant_cache[tenant_id] = (now + _TENANT_CACHE_TTL_SECONDS, active)
    return active


async def get_current_tenant(
    x_tenant_id: Optional[str] = Header(default=None, alias=TENANT_HEADER),
) -> str:
    """FastAPI dependency yielding the validated tenant id for this request."""
    tenant_id = resolve_tenant_id(x_tenant_id)

    try:
        active = await _tenant_is_active(tenant_id)
    except HTTPException:
        raise
    except Exception as exc:  # database unreachable
        logger.exception("Tenant validation failed for '%s'", tenant_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not verify tenant '{tenant_id}': {exc}",
        ) from exc

    if not active:
        # 404 rather than 403: to an unauthorised caller, a tenant they may not
        # touch and a tenant that does not exist should look identical.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Tenant '{tenant_id}' is not a known, active tenant. "
                f"Seed it with `python -m app.seed.seed_machines`."
            ),
        )
    return tenant_id


async def get_tenant_db(tenant_id: str = Depends(get_current_tenant)) -> TenantScope:
    """FastAPI dependency yielding a database handle already bound to the tenant."""
    return get_tenant_scope(tenant_id)
