from __future__ import annotations

from typing import Any


class CoreSettings:
    """Proxy view for core/auth/provider settings on root Settings."""

    FIELD_NAMES: tuple[str, ...] = (
        "api_prefix",
        "database_url",
        "auth_enabled",
        "auth_tokens",
        "auth_token",
        "auth_user",
        "auth_project_owners",
        "auth_disabled_user",
        "llm_provider",
        "llm_model",
        "llm_base_url",
        "llm_api_key",
        "llm_timeout_seconds",
        "llm_max_output_tokens",
        "llm_temperature",
        "llm_temperature_chat",
        "llm_temperature_action",
        "llm_temperature_ghost",
        "llm_temperature_brainstorm",
        "context_cache_enabled",
        "context_cache_ttl_seconds",
        "context_cache_max_entries",
        "openai_prompt_cache_key_enabled",
        "openai_prompt_cache_key_salt",
        "anthropic_enabled",
        "anthropic_base_url",
        "anthropic_api_key",
        "anthropic_model",
        "anthropic_version",
        "anthropic_prompt_caching_beta",
        "gemini_enabled",
        "gemini_base_url",
        "gemini_api_key",
        "gemini_model",
        "gemini_cache_enabled",
        "gemini_cache_ttl_seconds",
        "deepseek_enabled",
        "deepseek_base_url",
        "deepseek_api_key",
        "deepseek_model",
        "langfuse_enabled",
        "langfuse_host",
        "langfuse_public_key",
        "langfuse_secret_key",
    )

    def __init__(self, root: Any) -> None:
        object.__setattr__(self, "_root", root)

    def __getattr__(self, name: str) -> Any:
        if name in self.FIELD_NAMES:
            return getattr(self._root, name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.FIELD_NAMES:
            setattr(self._root, name, value)
            return
        object.__setattr__(self, name, value)
