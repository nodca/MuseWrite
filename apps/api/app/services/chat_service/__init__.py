from app.services.chat_service._common import *  # noqa: F401,F403
from app.services.chat_service.sessions import *  # noqa: F401,F403
from app.services.chat_service.messages import *  # noqa: F401,F403
from app.services.chat_service.actions import *  # noqa: F401,F403
from app.services.chat_service.mutations import *  # noqa: F401,F403
from app.services.chat_service.entity_merge import *  # noqa: F401,F403
from app.services.chat_service.entity_graph import *  # noqa: F401,F403
from app.services.chat_service.model_profiles import *  # noqa: F401,F403
from app.services.chat_service.prompt_templates import *  # noqa: F401,F403
from app.services.chat_service.project_assets import *  # noqa: F401,F403
from app.services.chat_service.volumes import *  # noqa: F401,F403
from app.services.chat_service.chapters import *  # noqa: F401,F403
from app.services.chat_service.scene_beats import *  # noqa: F401,F403

# ---- Explicit re-exports of underscore-prefixed helpers ----
# ``from module import *`` skips names starting with ``_``, so tests and
# other modules that import these via the package root need them here.
from app.services.chat_service._common import (  # noqa: F401
    _utc_now,
    _setting_key_from_payload,
    _normalize_aliases_payload,
    _collect_entity_merge_aliases,
    _normalize_graph_entity_token,
    _extract_aliases_from_content,
)
from app.services.chat_service.mutations import (  # noqa: F401
    _bump_project_mutation_version,
    _project_id_for_action,
    _index_lifecycle_meta,
)
from app.services.chat_service.entity_graph import (  # noqa: F401
    _sync_graph_for_action,
    _rewrite_chunk_with_llm_coref,
)
from app.services.chat_service.volumes import (  # noqa: F401
    _call_volume_memory_consolidation_llm,
)
