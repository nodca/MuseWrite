from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Header, HTTPException, status

from app.core.config import settings


@dataclass(frozen=True)
class AuthPrincipal:
    user_id: str


def _parse_token_mapping(raw: str) -> dict[str, str]:
    token_to_user: dict[str, str] = {}
    for chunk in (raw or "").split(","):
        item = chunk.strip()
        if not item or ":" not in item:
            continue
        user_id, token = item.split(":", 1)
        user = user_id.strip()
        token_value = token.strip()
        if not user or not token_value:
            continue
        token_to_user[token_value] = user
    return token_to_user


def _resolve_token_mapping() -> dict[str, str]:
    mapped = _parse_token_mapping(settings.auth_tokens)
    if mapped:
        return mapped
    fallback_token = str(settings.auth_token or "").strip()
    fallback_user = str(settings.auth_user or "").strip()
    if fallback_token and fallback_user:
        return {fallback_token: fallback_user}
    return {}


def _parse_project_owner_mapping(raw: str) -> dict[int, set[str]]:
    mapping: dict[int, set[str]] = {}
    for chunk in (raw or "").split(","):
        item = chunk.strip()
        if not item or ":" not in item:
            continue
        project_part, owners_part = item.split(":", 1)
        try:
            project_id = int(project_part.strip())
        except Exception:
            continue
        if project_id <= 0:
            continue
        owners = {owner.strip() for owner in owners_part.split("|") if owner.strip()}
        if owners:
            mapping[project_id] = owners
    return mapping


def _resolve_project_owner_mapping() -> dict[int, set[str]]:
    return _parse_project_owner_mapping(settings.auth_project_owners)


def get_current_principal(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> AuthPrincipal:
    if not settings.auth_enabled:
        fallback_user = str(settings.auth_disabled_user or "local-user").strip() or "local-user"
        return AuthPrincipal(user_id=fallback_user)

    token_to_user = _resolve_token_mapping()
    if not token_to_user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AUTH_TOKENS/AUTH_TOKEN is not configured",
        )

    auth_header = str(authorization or "").strip()
    if not auth_header:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing Authorization header")

    scheme, _, credential = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not credential.strip():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid Authorization header")

    user_id = token_to_user.get(credential.strip())
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid access token")

    return AuthPrincipal(user_id=user_id)


def ensure_project_access(user_id: str, project_id: int) -> None:
    if project_id <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid project_id")

    if not settings.auth_enabled:
        return

    owner_mapping = _resolve_project_owner_mapping()
    if not owner_mapping:
        # Single-user local mode: when no explicit project ACL is configured,
        # any authenticated identity can access project-scoped resources.
        return

    owners = owner_mapping.get(project_id)
    if not owners:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="project access denied")
    if "*" in owners:
        return
    if str(user_id or "").strip() not in owners:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="project access denied")


def filter_accessible_project_ids(user_id: str, requested_ids: list[int] | None) -> list[int]:
    if not isinstance(requested_ids, list):
        return []
    normalized: list[int] = []
    for raw in requested_ids:
        try:
            project_id = int(raw)
        except Exception:
            continue
        if project_id <= 0 or project_id in normalized:
            continue
        try:
            ensure_project_access(user_id, project_id)
        except HTTPException:
            continue
        normalized.append(project_id)
    return normalized
