from uuid import uuid4
import copy
import time

from sqlmodel import Session, select

from app.core.config import settings
from app.models.chat import ChatAction
from app.models.content import SettingEntry, StoryCard
from app.services.graph_mutation_registry import (
    upsert_pending_graph_mutation,
    mark_pending_graph_mutation_status,
    mark_pending_graph_mutation_canceled,
)
from app.services.graph_job_queue import enqueue_graph_sync_job, _QUEUE as _GRAPH_SYNC_QUEUE
from app.services.index_lifecycle_queue import enqueue_index_lifecycle_job, _QUEUE as _INDEX_LIFECYCLE_QUEUE
from app.services.index_lifecycle_service import process_index_lifecycle_rebuild
from app.services.retrieval_adapters import (
    promote_neo4j_candidate_facts,
    update_neo4j_graph_fact_state,
    delete_neo4j_graph_facts,
)
from app.services.chat_service import (
    _bump_project_mutation_version,
    _collect_entity_merge_aliases,
    _extract_aliases_from_content,
    _index_lifecycle_meta,
    _normalize_aliases_payload,
    _normalize_graph_entity_token,
    _project_id_for_action,
    _setting_key_from_payload,
    _utc_now,
    create_action_audit_log,
    is_entity_merge_action_type,
    process_graph_sync_for_action,
)


def apply_action_effects(db: Session, action: ChatAction) -> ChatAction:
    project_id = _project_id_for_action(db, action)
    payload_raw = action.payload or {}
    provenance_raw = payload_raw.get("_provenance") if isinstance(payload_raw.get("_provenance"), dict) else {}
    payload = {key: value for key, value in payload_raw.items() if key != "_provenance"}
    graph_current_chapter = 0
    try:
        graph_current_chapter = int(provenance_raw.get("current_chapter_index") or 0)
    except Exception:
        graph_current_chapter = 0
    if graph_current_chapter > 0:
        payload["_graph_current_chapter"] = graph_current_chapter
    atype = action.action_type

    if atype == "setting.upsert":
        key = _setting_key_from_payload(payload)
        graph_anchor = key.replace("设定", "").strip() or key
        payload["_graph_anchor"] = graph_anchor
        value = payload.get("value", payload.get("content"))
        aliases_in_payload = "aliases" in payload
        aliases_from_payload = _normalize_aliases_payload(payload.get("aliases"))
        if value is None:
            raise ValueError("setting.upsert requires value/content")
        if not isinstance(value, dict):
            raise ValueError("setting value/content must be object")

        stmt = select(SettingEntry).where(SettingEntry.project_id == project_id, SettingEntry.key == key)
        existing = db.exec(stmt).first()
        if existing:
            next_aliases = (
                aliases_from_payload
                if aliases_in_payload
                else _normalize_aliases_payload(getattr(existing, "aliases", []))
            )
            if not next_aliases:
                next_aliases = _normalize_aliases_payload(_extract_aliases_from_content(value))
            before = {
                "exists": True,
                "value": copy.deepcopy(existing.value),
                "aliases": copy.deepcopy(existing.aliases or []),
            }
            existing.value = value
            existing.aliases = next_aliases
            existing.updated_at = _utc_now()
            db.add(existing)
        else:
            next_aliases = aliases_from_payload or _normalize_aliases_payload(_extract_aliases_from_content(value))
            before = {"exists": False}
            db.add(SettingEntry(project_id=project_id, key=key, value=value, aliases=next_aliases))

        action.apply_result = {"project_id": project_id, "key": key, "value": value, "aliases": next_aliases}
        action.undo_payload = {
            "kind": "setting.upsert",
            "project_id": project_id,
            "key": key,
            "before": before,
        }

    elif atype == "setting.delete":
        key = _setting_key_from_payload(payload)
        stmt = select(SettingEntry).where(SettingEntry.project_id == project_id, SettingEntry.key == key)
        existing = db.exec(stmt).first()
        if existing:
            before = {
                "exists": True,
                "value": copy.deepcopy(existing.value),
                "aliases": copy.deepcopy(existing.aliases or []),
            }
            db.delete(existing)
            deleted = True
        else:
            before = {"exists": False}
            deleted = False

        action.apply_result = {"project_id": project_id, "key": key, "deleted": deleted}
        action.undo_payload = {
            "kind": "setting.delete",
            "project_id": project_id,
            "key": key,
            "before": before,
        }

    elif atype == "card.create":
        title = payload.get("title") or "未命名卡片"
        payload["_graph_anchor"] = str(title)
        content = payload.get("content") or {}
        aliases = _normalize_aliases_payload(payload.get("aliases"))
        if not isinstance(content, dict):
            raise ValueError("card.create content must be object")
        if not aliases:
            aliases = _normalize_aliases_payload(_extract_aliases_from_content(content))

        card = StoryCard(project_id=project_id, title=title, content=content, aliases=aliases)
        db.add(card)
        db.flush()
        action.apply_result = {
            "project_id": project_id,
            "card_id": card.id,
            "title": card.title,
            "aliases": aliases,
        }
        action.undo_payload = {"kind": "card.create", "project_id": project_id, "card_id": card.id}

    elif atype == "card.update":
        card_id = payload.get("card_id")
        if not isinstance(card_id, int):
            raise ValueError("card.update requires integer card_id")

        card = db.get(StoryCard, card_id)
        if not card or card.project_id != project_id:
            raise ValueError("card not found in project")

        before = {
            "title": card.title,
            "content": copy.deepcopy(card.content),
            "aliases": copy.deepcopy(card.aliases or []),
        }
        if "title" in payload and isinstance(payload["title"], str):
            card.title = payload["title"]
        if "content" in payload:
            if not isinstance(payload["content"], dict):
                raise ValueError("card.update content must be object")
            merge = bool(payload.get("merge", True))
            card.content = {**(card.content or {}), **payload["content"]} if merge else payload["content"]
        if "aliases" in payload:
            card.aliases = _normalize_aliases_payload(payload.get("aliases"))
        payload["_graph_anchor"] = str(card.title or "")
        payload["_graph_anchor_before"] = str(before.get("title") or "")
        card.updated_at = _utc_now()
        db.add(card)

        action.apply_result = {
            "project_id": project_id,
            "card_id": card.id,
            "title": card.title,
            "aliases": copy.deepcopy(card.aliases or []),
        }
        action.undo_payload = {"kind": "card.update", "project_id": project_id, "card_id": card.id, "before": before}

    elif is_entity_merge_action_type(atype):
        target_card_id = payload.get("target_card_id")
        if not isinstance(target_card_id, int):
            target_card_id = payload.get("canonical_card_id")
        if not isinstance(target_card_id, int):
            target_card_id = payload.get("card_id")
        if not isinstance(target_card_id, int):
            raise ValueError("entity.merge requires integer target_card_id")

        card = db.get(StoryCard, target_card_id)
        if not card or card.project_id != project_id:
            raise ValueError("target card not found in project")

        incoming_aliases = _collect_entity_merge_aliases(payload)
        canonical_token = _normalize_graph_entity_token(str(card.title or ""))
        filtered_aliases = [
            alias for alias in incoming_aliases if _normalize_graph_entity_token(alias) != canonical_token
        ]
        if not filtered_aliases:
            raise ValueError("entity.merge requires at least one alias candidate")

        before = {
            "title": card.title,
            "aliases": copy.deepcopy(card.aliases or []),
        }
        merged_aliases = _normalize_aliases_payload([*(card.aliases or []), *filtered_aliases])
        payload["_graph_anchor"] = str(card.title or "")
        payload["_graph_anchor_before"] = str(card.title or "")
        card.aliases = merged_aliases
        card.updated_at = _utc_now()
        db.add(card)

        action.apply_result = {
            "project_id": project_id,
            "card_id": card.id,
            "title": card.title,
            "aliases": copy.deepcopy(card.aliases or []),
            "merge_aliases_added": filtered_aliases,
            "merge_mode": "aliases_only_manual",
        }
        action.undo_payload = {
            "kind": "entity.merge.aliases",
            "project_id": project_id,
            "card_id": card.id,
            "before": before,
        }

    elif atype == "graph.confirm_candidates":
        source_ref = str(payload.get("source_ref") or "").strip()
        fact_keys_raw = payload.get("fact_keys")
        fact_keys = (
            [str(item).strip() for item in fact_keys_raw if str(item).strip()]
            if isinstance(fact_keys_raw, list)
            else []
        )
        if not source_ref and not fact_keys:
            raise ValueError("graph.confirm_candidates requires source_ref or fact_keys")

        min_confidence_raw = payload.get("min_confidence")
        min_confidence: float | None = None
        if min_confidence_raw is not None:
            try:
                min_confidence = float(min_confidence_raw)
            except Exception:
                raise ValueError("graph.confirm_candidates min_confidence must be number")
            if min_confidence < 0.0 or min_confidence > 1.0:
                raise ValueError("graph.confirm_candidates min_confidence must be between 0 and 1")

        limit_raw = payload.get("limit", 200)
        try:
            limit = max(min(int(limit_raw), 1000), 1)
        except Exception:
            raise ValueError("graph.confirm_candidates limit must be integer")

        promoted_fact_keys = promote_neo4j_candidate_facts(
            project_id,
            fact_keys=fact_keys,
            source_ref=source_ref,
            min_confidence=min_confidence,
            limit=limit,
            current_chapter=graph_current_chapter if graph_current_chapter > 0 else None,
        )
        action.apply_result = {
            "project_id": project_id,
            "source_ref": source_ref or None,
            "requested_fact_keys": fact_keys,
            "requested_min_confidence": min_confidence,
            "limit": limit,
            "promoted_count": len(promoted_fact_keys),
            "promoted_fact_keys": promoted_fact_keys,
        }
        action.undo_payload = {
            "kind": "graph.confirm_candidates",
            "project_id": project_id,
            "promoted_fact_keys": promoted_fact_keys,
        }

    else:
        raise ValueError(f"unsupported action_type: {atype}")

    mutation_version = _bump_project_mutation_version(db, project_id)
    mutation_id = f"m-{project_id}-{mutation_version}-{uuid4().hex[:8]}"
    graph_job_idempotency_key = f"graph-sync:{action.id}:{mutation_id}"
    lifecycle_job_idempotency_key = f"index-lifecycle:{action.id}:{mutation_id}"
    action_provenance = {
        "source_action_id": action.id,
        "operator_id": action.operator_id,
        "source": provenance_raw.get("source") or "unknown",
        "current_chapter_index": graph_current_chapter if graph_current_chapter > 0 else None,
        "resolver_order": provenance_raw.get("resolver_order"),
        "providers": provenance_raw.get("providers", {}),
        "rag_route": provenance_raw.get("rag_route", {}),
        "quality_gate": provenance_raw.get("quality_gate", {}),
        "evidence_summary": provenance_raw.get("evidence_summary", {}),
        "evidence_refs": provenance_raw.get("evidence_refs", {}),
        "mutation_id": mutation_id,
        "expected_version": mutation_version,
    }

    action.status = "applied"
    action.applied_at = _utc_now()
    action.apply_result = {
        **(action.apply_result or {}),
        "provenance": action_provenance,
    }
    action.undo_payload = {
        **(action.undo_payload or {}),
        "provenance": action_provenance,
    }
    if atype in {"setting.upsert", "card.create", "card.update"}:
        action.apply_result = {
            **(action.apply_result or {}),
            "graph_sync": {
                "status": "pending_queue",
                "mode": "pending",
                "mutation_id": mutation_id,
                "expected_version": mutation_version,
                "job_idempotency_key": graph_job_idempotency_key,
            },
        }
    if atype == "setting.delete" and settings.index_lifecycle_enabled:
        action.apply_result = {
            **(action.apply_result or {}),
            "index_lifecycle": {
                "status": "pending_queue",
                "mode": "pending",
                "reason": "setting_delete",
                "mutation_id": mutation_id,
                "expected_version": mutation_version,
                "job_idempotency_key": lifecycle_job_idempotency_key,
            },
        }
    db.add(action)
    db.flush()

    if atype in {"setting.upsert", "card.create", "card.update"} and action.id:
        upsert_pending_graph_mutation(
            db,
            project_id=project_id,
            action_id=int(action.id),
            mutation_id=mutation_id,
            expected_version=mutation_version,
            status="pending_queue",
        )

    if atype in {"setting.upsert", "card.create", "card.update"}:
        if settings.graph_sync_async_enabled:
            queued = enqueue_graph_sync_job(
                action.id,
                project_id=project_id,
                action_type=atype,
                payload=payload,
                operator_id=action.operator_id,
                mutation_id=mutation_id,
                expected_version=mutation_version,
                idempotency_key=graph_job_idempotency_key,
                attempt=0,
                db=db,
            )
            if queued:
                mark_pending_graph_mutation_status(
                    db,
                    mutation_id=mutation_id,
                    status="queued",
                )
                action.apply_result = {
                    **(action.apply_result or {}),
                    "graph_sync": {
                        "status": "queued",
                        "mode": "async",
                        "queue": _GRAPH_SYNC_QUEUE,
                        "mutation_id": mutation_id,
                        "expected_version": mutation_version,
                        "job_idempotency_key": graph_job_idempotency_key,
                    },
                }
                db.add(action)
                db.commit()
                db.refresh(action)
                create_action_audit_log(
                    db=db,
                    action_id=action.id,
                    event_type="graph_queued",
                    operator_id=action.operator_id,
                    event_payload={
                        "queue": _GRAPH_SYNC_QUEUE,
                        "mode": "async",
                        "mutation_id": mutation_id,
                        "expected_version": mutation_version,
                        "job_idempotency_key": graph_job_idempotency_key,
                    },
                )
            else:
                graph_sync, _ = process_graph_sync_for_action(
                    db=db,
                    action=action,
                    project_id=project_id,
                    action_type=atype,
                    payload=payload,
                    operator_id=action.operator_id,
                    mutation_id=mutation_id,
                    expected_version=mutation_version,
                    job_idempotency_key=graph_job_idempotency_key,
                    sync_mode="sync_fallback",
                )
                if graph_sync:
                    action.apply_result = {
                        **(action.apply_result or {}),
                        "graph_sync": {
                            **graph_sync,
                            "mode": "sync_fallback",
                            "mutation_id": mutation_id,
                            "expected_version": mutation_version,
                            "job_idempotency_key": graph_job_idempotency_key,
                        },
                    }
                    db.add(action)
                    db.commit()
                    db.refresh(action)
        else:
            process_graph_sync_for_action(
                db=db,
                action=action,
                project_id=project_id,
                action_type=atype,
                payload=payload,
                operator_id=action.operator_id,
                mutation_id=mutation_id,
                expected_version=mutation_version,
                job_idempotency_key=graph_job_idempotency_key,
                sync_mode="sync_inline",
            )

    if atype == "setting.delete" and settings.index_lifecycle_enabled:
        queued = enqueue_index_lifecycle_job(
            project_id=project_id,
            operator_id=action.operator_id,
            reason="setting_delete",
            action_id=action.id,
            mutation_id=mutation_id,
            expected_version=mutation_version,
            idempotency_key=lifecycle_job_idempotency_key,
            lifecycle_slot="default",
            attempt=0,
            db=db,
        )
        if queued:
            action.apply_result = {
                **(action.apply_result or {}),
                "index_lifecycle": {
                    "status": "queued",
                    "mode": "async",
                    "queue": _INDEX_LIFECYCLE_QUEUE,
                    "reason": "setting_delete",
                    "mutation_id": mutation_id,
                    "expected_version": mutation_version,
                    "job_idempotency_key": lifecycle_job_idempotency_key,
                },
            }
            db.add(action)
            db.commit()
            db.refresh(action)
            create_action_audit_log(
                db=db,
                action_id=action.id,
                event_type="index_lifecycle_queued",
                operator_id=action.operator_id,
                event_payload={
                    "queue": _INDEX_LIFECYCLE_QUEUE,
                    "mode": "async",
                    "reason": "setting_delete",
                    "mutation_id": mutation_id,
                    "expected_version": mutation_version,
                    "job_idempotency_key": lifecycle_job_idempotency_key,
                },
            )
        else:
            lifecycle_result = process_index_lifecycle_rebuild(
                db=db,
                project_id=project_id,
                reason="setting_delete_sync_fallback",
                lifecycle_id=mutation_id,
            )
            action.apply_result = {
                **(action.apply_result or {}),
                "index_lifecycle": {
                    "status": "completed",
                    "mode": "sync_fallback",
                    "reason": "setting_delete",
                    "mutation_id": mutation_id,
                    "expected_version": mutation_version,
                    "job_idempotency_key": lifecycle_job_idempotency_key,
                    "result": lifecycle_result,
                },
            }
            db.add(action)
            db.commit()
            db.refresh(action)
            create_action_audit_log(
                db=db,
                action_id=action.id,
                event_type="index_lifecycle_done",
                operator_id=action.operator_id,
                event_payload={
                    "mode": "sync_fallback",
                    "reason": "setting_delete",
                    "mutation_id": mutation_id,
                    "expected_version": mutation_version,
                    "job_idempotency_key": lifecycle_job_idempotency_key,
                    "result": lifecycle_result,
                },
            )
    db.add(action)
    db.commit()
    db.refresh(action)
    return action


def undo_action_effects(db: Session, action: ChatAction) -> ChatAction:
    undo_payload = action.undo_payload or {}
    kind = undo_payload.get("kind")
    project_id = _project_id_for_action(db, action)
    provenance_meta = undo_payload.get("provenance") if isinstance(undo_payload.get("provenance"), dict) else {}
    undo_chapter_index = 0
    try:
        undo_chapter_index = int(provenance_meta.get("current_chapter_index") or 0)
    except Exception:
        undo_chapter_index = 0
    compensation_version = _bump_project_mutation_version(db, project_id)
    compensation_mutation_id = f"undo-{project_id}-{compensation_version}-{uuid4().hex[:8]}"

    if kind == "setting.upsert":
        key = undo_payload.get("key")
        before = undo_payload.get("before", {})
        stmt = select(SettingEntry).where(SettingEntry.project_id == project_id, SettingEntry.key == key)
        existing = db.exec(stmt).first()
        if before.get("exists"):
            value = before.get("value", {})
            aliases = _normalize_aliases_payload(before.get("aliases"))
            if existing:
                existing.value = value
                existing.aliases = aliases
                existing.updated_at = _utc_now()
                db.add(existing)
            else:
                db.add(SettingEntry(project_id=project_id, key=key, value=value, aliases=aliases))
        else:
            if existing:
                db.delete(existing)

    elif kind == "setting.delete":
        key = undo_payload.get("key")
        before = undo_payload.get("before", {})
        stmt = select(SettingEntry).where(SettingEntry.project_id == project_id, SettingEntry.key == key)
        existing = db.exec(stmt).first()
        if before.get("exists"):
            value = before.get("value", {})
            aliases = _normalize_aliases_payload(before.get("aliases"))
            if existing:
                existing.value = value
                existing.aliases = aliases
                existing.updated_at = _utc_now()
                db.add(existing)
            else:
                db.add(SettingEntry(project_id=project_id, key=key, value=value, aliases=aliases))

    elif kind == "card.create":
        card_id = undo_payload.get("card_id")
        card = db.get(StoryCard, card_id)
        if card and card.project_id == project_id:
            db.delete(card)

    elif kind == "card.update":
        card_id = undo_payload.get("card_id")
        before = undo_payload.get("before", {})
        card = db.get(StoryCard, card_id)
        if card and card.project_id == project_id:
            card.title = before.get("title", card.title)
            card.content = before.get("content", card.content)
            card.aliases = _normalize_aliases_payload(before.get("aliases"))
            card.updated_at = _utc_now()
            db.add(card)

    elif kind == "entity.merge.aliases":
        card_id = undo_payload.get("card_id")
        before = undo_payload.get("before", {})
        card = db.get(StoryCard, card_id)
        if card and card.project_id == project_id:
            card.aliases = _normalize_aliases_payload(before.get("aliases"))
            card.updated_at = _utc_now()
            db.add(card)

    elif kind == "graph.confirm_candidates":
        promoted_fact_keys_raw = undo_payload.get("promoted_fact_keys")
        promoted_fact_keys = (
            [str(item).strip() for item in promoted_fact_keys_raw if str(item).strip()]
            if isinstance(promoted_fact_keys_raw, list)
            else []
        )
        reverted = (
            update_neo4j_graph_fact_state(
                project_id,
                promoted_fact_keys,
                to_state="candidate",
                from_state="confirmed",
                current_chapter=undo_chapter_index if undo_chapter_index > 0 else None,
            )
            if promoted_fact_keys
            else 0
        )
        action.apply_result = {
            **(action.apply_result or {}),
            "graph_confirm_undo": {
                "requested": len(promoted_fact_keys),
                "reverted": reverted,
                "compensation_mutation_id": compensation_mutation_id,
                "compensation_version": compensation_version,
            },
        }

    else:
        raise ValueError("undo payload invalid")

    graph_fact_keys_raw = undo_payload.get("graph_fact_keys")
    graph_fact_keys = (
        [str(item).strip() for item in graph_fact_keys_raw if str(item).strip()]
        if isinstance(graph_fact_keys_raw, list)
        else []
    )
    lifecycle_compensation_needed = kind == "setting.delete" and settings.index_lifecycle_enabled
    lifecycle_compensation_idempotency_key = f"index-lifecycle:undo:{action.id}:{compensation_mutation_id}"
    base_apply_result = action.apply_result if isinstance(action.apply_result, dict) else {}
    graph_sync_meta = base_apply_result.get("graph_sync")
    graph_sync_mutation_id = str(graph_sync_meta.get("mutation_id") or "") if isinstance(graph_sync_meta, dict) else ""
    if graph_sync_mutation_id:
        mark_pending_graph_mutation_canceled(
            db,
            mutation_id=graph_sync_mutation_id,
            cancel_reason="undo_requested",
            canceled_by_mutation_id=compensation_mutation_id,
        )
    graph_sync_final = (
        {
            **graph_sync_meta,
            "status": "canceled",
            "canceled_by": compensation_mutation_id,
            "compensation_version": compensation_version,
        }
        if isinstance(graph_sync_meta, dict)
        else None
    )
    lifecycle_meta = _index_lifecycle_meta(action)
    lifecycle_default_final = (
        {
            **lifecycle_meta,
            "status": "canceled",
            "canceled_by": compensation_mutation_id,
            "compensation_version": compensation_version,
        }
        if kind == "setting.delete" and isinstance(lifecycle_meta, dict)
        else None
    )
    lifecycle_compensation_pending = (
        {
            "status": "pending_queue",
            "mode": "pending",
            "reason": "undo_setting_delete",
            "mutation_id": compensation_mutation_id,
            "expected_version": compensation_version,
            "job_idempotency_key": lifecycle_compensation_idempotency_key,
        }
        if lifecycle_compensation_needed
        else None
    )
    if graph_fact_keys:
        deleted = delete_neo4j_graph_facts(
            project_id,
            graph_fact_keys,
            current_chapter=undo_chapter_index if undo_chapter_index > 0 else None,
        )
        next_apply_result = {
            **base_apply_result,
            "graph_undo": {
                "requested": len(graph_fact_keys),
                "deleted": deleted,
                "compensation_mutation_id": compensation_mutation_id,
                "compensation_version": compensation_version,
            },
        }
        if graph_sync_final:
            next_apply_result["graph_sync"] = graph_sync_final
        if lifecycle_default_final:
            next_apply_result["index_lifecycle"] = lifecycle_default_final
        if lifecycle_compensation_pending:
            next_apply_result["index_lifecycle_compensation"] = lifecycle_compensation_pending
        action.apply_result = next_apply_result
    else:
        next_apply_result = {
            **base_apply_result,
            "graph_undo": {
                "requested": 0,
                "deleted": 0,
                "compensation_mutation_id": compensation_mutation_id,
                "compensation_version": compensation_version,
            },
        }
        if graph_sync_final:
            next_apply_result["graph_sync"] = graph_sync_final
        if lifecycle_default_final:
            next_apply_result["index_lifecycle"] = lifecycle_default_final
        if lifecycle_compensation_pending:
            next_apply_result["index_lifecycle_compensation"] = lifecycle_compensation_pending
        action.apply_result = next_apply_result

    action.status = "undone"
    action.undone_at = _utc_now()
    db.add(action)
    db.flush()

    if lifecycle_compensation_needed:
        queued = enqueue_index_lifecycle_job(
            project_id=project_id,
            operator_id=action.operator_id,
            reason="undo_setting_delete",
            action_id=action.id,
            mutation_id=compensation_mutation_id,
            expected_version=compensation_version,
            idempotency_key=lifecycle_compensation_idempotency_key,
            lifecycle_slot="compensation",
            attempt=0,
            db=db,
        )
        if queued:
            action.apply_result = {
                **(action.apply_result or {}),
                "index_lifecycle_compensation": {
                    "status": "queued",
                    "mode": "async",
                    "queue": _INDEX_LIFECYCLE_QUEUE,
                    "reason": "undo_setting_delete",
                    "mutation_id": compensation_mutation_id,
                    "expected_version": compensation_version,
                    "job_idempotency_key": lifecycle_compensation_idempotency_key,
                },
            }
            db.add(action)
            db.commit()
            db.refresh(action)
            create_action_audit_log(
                db=db,
                action_id=action.id,
                event_type="index_lifecycle_queued",
                operator_id=action.operator_id,
                event_payload={
                    "queue": _INDEX_LIFECYCLE_QUEUE,
                    "mode": "async",
                    "reason": "undo_setting_delete",
                    "mutation_id": compensation_mutation_id,
                    "expected_version": compensation_version,
                    "job_idempotency_key": lifecycle_compensation_idempotency_key,
                },
            )
        else:
            lifecycle_result = process_index_lifecycle_rebuild(
                db=db,
                project_id=project_id,
                reason="undo_setting_delete_sync_fallback",
                lifecycle_id=compensation_mutation_id,
            )
            action.apply_result = {
                **(action.apply_result or {}),
                "index_lifecycle_compensation": {
                    "status": "completed",
                    "mode": "sync_fallback",
                    "reason": "undo_setting_delete",
                    "mutation_id": compensation_mutation_id,
                    "expected_version": compensation_version,
                    "job_idempotency_key": lifecycle_compensation_idempotency_key,
                    "result": lifecycle_result,
                },
            }
            db.add(action)
            db.commit()
            db.refresh(action)
            create_action_audit_log(
                db=db,
                action_id=action.id,
                event_type="index_lifecycle_done",
                operator_id=action.operator_id,
                event_payload={
                    "mode": "sync_fallback",
                    "reason": "undo_setting_delete",
                    "mutation_id": compensation_mutation_id,
                    "expected_version": compensation_version,
                    "job_idempotency_key": lifecycle_compensation_idempotency_key,
                    "result": lifecycle_result,
                },
            )
    db.add(action)
    db.commit()
    db.refresh(action)
    return action
