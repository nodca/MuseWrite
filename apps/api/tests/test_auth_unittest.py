import unittest

from fastapi import HTTPException

from app.core.auth import (
    ensure_project_access,
    filter_accessible_project_ids,
    get_current_principal,
)
from app.core.config import settings


class AuthBehaviorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._snapshot = {
            "auth_enabled": settings.auth_enabled,
            "auth_tokens": settings.auth_tokens,
            "auth_token": settings.auth_token,
            "auth_user": settings.auth_user,
            "auth_project_owners": settings.auth_project_owners,
            "auth_disabled_user": settings.auth_disabled_user,
        }

    def tearDown(self) -> None:
        for key, value in self._snapshot.items():
            setattr(settings, key, value)

    def test_auth_disabled_uses_fallback_user(self) -> None:
        settings.auth_enabled = False
        settings.auth_disabled_user = "solo-author"
        principal = get_current_principal(None)
        self.assertEqual(principal.user_id, "solo-author")

    def test_auth_enabled_accepts_bearer_token_mapping(self) -> None:
        settings.auth_enabled = True
        settings.auth_tokens = "alice:token-a,bob:token-b"
        settings.auth_token = ""
        settings.auth_user = ""

        principal = get_current_principal("Bearer token-b")
        self.assertEqual(principal.user_id, "bob")

    def test_auth_enabled_can_use_legacy_single_token_fallback(self) -> None:
        settings.auth_enabled = True
        settings.auth_tokens = ""
        settings.auth_token = "legacy-token"
        settings.auth_user = "legacy-user"

        principal = get_current_principal("Bearer legacy-token")
        self.assertEqual(principal.user_id, "legacy-user")

    def test_auth_enabled_rejects_missing_or_invalid_token(self) -> None:
        settings.auth_enabled = True
        settings.auth_tokens = "local-user:secret-token"
        settings.auth_token = ""
        settings.auth_user = ""

        with self.assertRaises(HTTPException) as missing_header:
            get_current_principal(None)
        self.assertEqual(missing_header.exception.status_code, 401)

        with self.assertRaises(HTTPException) as invalid_header:
            get_current_principal("Basic secret-token")
        self.assertEqual(invalid_header.exception.status_code, 401)

        with self.assertRaises(HTTPException) as invalid_token:
            get_current_principal("Bearer not-matching")
        self.assertEqual(invalid_token.exception.status_code, 401)

    def test_auth_enabled_without_token_config_fails_closed(self) -> None:
        settings.auth_enabled = True
        settings.auth_tokens = ""
        settings.auth_token = ""
        settings.auth_user = ""

        with self.assertRaises(HTTPException) as missing_config:
            get_current_principal("Bearer any")
        self.assertEqual(missing_config.exception.status_code, 500)

    def test_project_access_single_user_mode_when_acl_is_empty(self) -> None:
        settings.auth_enabled = True
        settings.auth_project_owners = ""
        ensure_project_access("any-user", 1)

    def test_project_access_acl_enforced_when_mapping_configured(self) -> None:
        settings.auth_enabled = True
        settings.auth_project_owners = "1:alice|bob,2:*"

        ensure_project_access("alice", 1)
        ensure_project_access("someone", 2)

        with self.assertRaises(HTTPException) as denied:
            ensure_project_access("mallory", 1)
        self.assertEqual(denied.exception.status_code, 403)

        with self.assertRaises(HTTPException) as missing_project_acl:
            ensure_project_access("alice", 3)
        self.assertEqual(missing_project_acl.exception.status_code, 403)

    def test_filter_accessible_project_ids_filters_invalid_and_denied(self) -> None:
        settings.auth_enabled = True
        settings.auth_project_owners = "1:alice|bob,2:*"
        filtered = filter_accessible_project_ids("alice", [1, 1, "2", 3, -1, "oops"])
        self.assertEqual(filtered, [1, 2])


if __name__ == "__main__":
    unittest.main()
