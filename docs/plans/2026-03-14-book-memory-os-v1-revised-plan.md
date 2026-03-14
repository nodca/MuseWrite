# Book Memory OS v1 — Revised Implementation Plan

**Date:** 2026-03-14
**Supersedes:** `2026-03-13-book-memory-os-v1-implementation-plan.md`

## Strategy Change

The original 9-phase waterfall plan is replaced with a **5-phase vertical-slice** approach.

Key architectural decisions:

1. **Graphiti** (`graphiti-core`) manages the L4 temporal graph layer instead of hand-built `temporal_graph_updater.py`
2. **Instructor** replaces manual `generate_structured_sync` calls for structured LLM extraction
3. **Feature-flag parallel pipeline** — new `memory_pipeline.py` coexists with old `context_compiler`, switched via `USE_MEMORY_PIPELINE`
4. **Dual timeline** — `narrative_order` (chapter publish order) + `story_timeline` (in-story time) on temporal edges

## Architecture Overview

```
                   ┌───────────────────────────────────┐
                   │         Planning / Rewrite         │
                   │         Endpoint Layer             │
                   └──────┬───────────────┬─────────────┘
                          │               │
              ┌───────────▼──┐    ┌───────▼──────────┐
              │ memory_      │    │ context_compiler/ │
              │ pipeline.py  │    │ pipeline.py       │
              │ (NEW)        │    │ (LEGACY)          │
              └───┬──┬──┬───┘    └──────────────────-─┘
                  │  │  │
        ┌─────────┘  │  └─────────┐
        ▼            ▼            ▼
   ┌─────────┐ ┌──────────┐ ┌──────────┐
   │ L1-L3   │ │ L4       │ │ L5       │
   │ Postgres│ │ Graphiti │ │ LightRAG │
   │ Memory  │ │ (Neo4j)  │ │ Cold     │
   └─────────┘ └──────────┘ └──────────┘
```

## What Already Exists (from Phase 1-2 of original plan)

- **6 Postgres tables**: WorldRule, CharacterProfile, StoryStateSnapshot, StoryEpisode, CharacterKnowledgeState, MemoryMaterialization
- **Extraction pipeline**: extraction_service.py → episode_extractor.py → story_state_compiler.py
- **Consolidation queue**: BaseQueue integration, idempotent enqueue/dequeue
- **Materializers**: 5 memory pack types (world, character, story_state, chapter, knowledge)
- **Tests**: 3 test suites (~600 lines)

## Phase A: Graphiti Integration + Temporal Graph (Target: 2-3 days)

### A.1 Add Dependencies

- Add `graphiti-core>=0.28.0` to `requirements-docker.txt`
- Add `instructor>=1.14.0` to `requirements-docker.txt`
- Verify compatibility with existing `neo4j>=5.26.0`, `openai`, `pydantic>=2.x`

### A.2 Create Graphiti Adapter Layer

**New files:**

- `app/services/book_memory/graphiti_adapter.py` — sync wrapper around async Graphiti
- `app/services/book_memory/entity_types.py` — fiction-specific entity/edge types

**Design:**

- Lazy singleton Graphiti instance per process
- `group_id = f"project-{project_id}"` for multi-project isolation
- `reference_time = epoch + chapter_index * 1 day` for temporal ordering
- Fiction entity types: `FictionCharacter`, `FictionLocation`, `FictionFaction`, `FictionItem`
- Chinese extraction instructions for fiction domain

### A.3 Integrate Instructor into Extraction

**Modified files:**

- `app/services/book_memory/extraction_service.py` — replace `generate_structured_sync` with Instructor

**Benefit:** Automatic retry, type-safe output, 50%+ less boilerplate

### A.4 Hook Graphiti into Consolidation Pipeline

**Modified files:**

- `app/services/book_memory/consolidation_service.py` — after Postgres persist, call `graphiti_adapter.ingest_chapter_episodes()`

**Flow:**
```
chapter change → consolidation job → extraction → Postgres persist → Graphiti ingest
```

### A.5 Tests

- Unit tests for graphiti_adapter (mocked Graphiti)
- Integration test: chapter consolidation produces temporal graph entries

## Phase B: Memory Query Layer (Target: 2-3 days)

### B.1 Temporal Query Service

**New files:**

- `app/services/book_memory/temporal_query.py`

**Functions:**

- `query_character_knowledge(project_id, character_name, at_chapter) -> list[TemporalFact]`
- `query_temporal_facts(project_id, query, at_chapter, limit) -> list[TemporalFact]`
- `query_entity_relations(project_id, entity_name, at_chapter) -> list[Relation]`

### B.2 Valkey/Redis Hot Cache (Optional)

- Add Valkey to docker-compose for memory pack caching
- Key pattern: `bm:{project_id}:{pack_type}:{scope_key}`
- Falls back to Postgres MemoryMaterialization if Redis unavailable

## Phase C: Planning Memory Pipeline (Target: 3-4 days)

### C.1 New Memory Pipeline

**New files:**

- `app/services/book_memory/memory_pipeline.py`

**Assemblers:**

- `build_planning_memory_context(project_id, chapter_id, scene_beat_id)`
- Reads: L1 StoryStateSnapshot → L3 WorldRules + CharacterProfiles → L4 Graphiti temporal facts → L2 recent episodes → L5 LightRAG (fallback only)

### C.2 Planning Assistant Integration

**Modified files:**

- `app/api/endpoints/chat.py` — feature flag to use memory pipeline
- `app/api/endpoints/chat_story_workspace.py` — planning context from memory

### C.3 Context X-Ray Memory Origin

- Add `memory_layer_origin` field to evidence events
- Distinguish: `story_state`, `canon`, `temporal_graph`, `cold_retrieval`

## Phase D: Rewrite Memory Pipeline (Target: 2-3 days)

### D.1 Rewrite Context Assembler

**New files:**

- `app/services/book_memory/rewrite_context.py`

**Modes:**

- `polish`: L0 selection + L1 state + L3 hard constraints only
- `expand`: all polish + L4 temporal facts + L2 recent episodes

### D.2 Dedicated Rewrite LLM Contract

- Separate `rewrite.generate` from chat output contract
- No `proposed_actions` in rewrite response
- Return: generated text + usage + diagnostics

## Phase E: Consistency Engine + Evaluation (Target: 3-4 days)

### E.1 Consistency Context

- World-rule violation detection via L4 temporal facts
- Character knowledge leak detection (omniscience check)
- Chronology violation detection

### E.2 Evaluation Fixtures

- `evals/data/book_memory_eval.jsonl` — 50+ eval questions
- `evals/run_book_memory_eval.py` — automated evaluation runner
- Metrics: knowledge correctness, chronology correctness, L5 fallback rate

### E.3 Frontend Cleanup

- Hide `rag_mode` from default author settings
- Add non-blocking consistency warnings in editor
- Memory-origin visibility in Context X-Ray

## Configuration

New environment variables:

```
GRAPHITI_ENABLED=true              # Feature gate
USE_MEMORY_PIPELINE=false          # Feature flag for new pipeline
GRAPHITI_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
GRAPHITI_EMBEDDING_BASE_URL=https://router.tumuer.me/v1
GRAPHITI_EMBEDDING_DIM=4096
```

Reuses existing:

```
NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE
LIGHTRAG_LLM_MODEL, LIGHTRAG_LLM_BASE_URL, LIGHTRAG_LLM_API_KEY
```

## First Milestone

> Planning assistant answers character-knowledge and chronology questions correctly without depending on LightRAG by default.

This proves the architecture shift from retrieval-first to memory-first.

## File Ownership (for parallel execution)

| Agent | Scope |
|-------|-------|
| A | graphiti_adapter, entity_types, temporal_query |
| B | memory_pipeline, planning/rewrite context |
| C | extraction_service (Instructor), consolidation_service |
| D | frontend settings, x-ray, consistency warnings |
| E | eval fixtures, eval runner |

## Execution Order

Phase A → B → C (critical path)
Phase D starts after C
Phase E starts after B

Phases B+C can partially overlap: query layer can be tested before pipeline is fully wired.
