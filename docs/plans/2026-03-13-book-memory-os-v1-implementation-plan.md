# Book Memory OS v1 Implementation Plan

**Goal:** Replace the current retrieval-first context assembly model with a memory-first architecture for fiction writing. The system must support four first-class capabilities:

- world memory
- character memory
- story state memory
- event chronology memory

**Primary Architecture:** `Temporal Graph Memory OS`

- `Postgres` stores materialized memory, chapter state, episodes, and governance records
- `Neo4j` stores temporal truth, knowledge boundaries, and dynamic relations
- `LightRAG` serves only as `L5 Cold Retrieval`
- task-specific memory assemblers replace the current universal compiler over time

**Design Reference:** `docs/superpowers/specs/2026-03-13-book-memory-os-v1-design.md`

**Tech Stack:** Python, FastAPI, SQLModel, Neo4j, Postgres, React, TypeScript, unittest, Playwright

---

## Phase 0: Reference Baseline and Repo Study

### Task 0.1: Clone and catalog external memory references

**Goal:** Pull local copies of the reference projects and capture the specific ideas we want to reuse.

**Reference Repos:**

- `getzep/graphiti`
- `letta-ai/letta`
- `langchain-ai/langmem`
- `mem0ai/mem0`

**Target Directory:**

- Create: `references/agent-memory/`

**Deliverables:**

- local cloned repositories
- one short internal note per repo:
  - what to borrow
  - what not to borrow
  - how it maps to Novel Platform

**Verification:**

```bash
ls -la references/agent-memory
```

---

## Phase 1: Introduce Core Book Memory Data Model

### Task 1.1: Add Postgres models for structured memory

**Files:**

- Create: `apps/api/app/models/book_memory.py`
- Modify: `apps/api/app/models/__init__.py`

**Models to Add:**

- `WorldRule`
- `CharacterProfile`
- `StoryStateSnapshot`
- `StoryEpisode`
- `CharacterKnowledgeState`
- `MemoryMaterialization`

**Model Rules:**

- each model must include `project_id`
- each model must include provenance or source references where relevant
- snapshots and materializations must be replaceable without deleting history

**Verification:**

```bash
python -m unittest discover apps/api/tests -p "test_*memory*unittest.py"
```

### Task 1.2: Define temporal graph schema conventions

**Files:**

- Create: `docs/plans/2026-03-13-book-memory-graph-schema.md`
- Modify: `apps/api/app/services/retrieval_adapters.py`

**Graph Entities:**

- `Character`
- `WorldRule`
- `Location`
- `Faction`
- `Event`
- `Knowledge`

**Key Edge Types:**

- `KNOWS`
- `TRUSTS`
- `BETRAYS`
- `LOCATED_IN`
- `PARTICIPATED_IN`
- `VIOLATES_RULE`
- `AFFECTS`

**Required Edge Properties:**

- `valid_from_chapter`
- `valid_to_chapter`
- `confidence`
- `evidence`
- `source_episode_id`

**Verification:**

- add unit tests for new graph mutation/query helpers
- confirm chapter-bounded queries return different results for different chapter indexes

### Task 1.3: Add Pydantic schemas and CRUD helpers

**Files:**

- Create: `apps/api/app/schemas/book_memory.py`
- Create: `apps/api/app/services/book_memory/`

**Initial Services:**

- `world_service.py`
- `character_service.py`
- `story_state_service.py`
- `episode_service.py`
- `materialization_service.py`

**Verification:**

```bash
python -m unittest discover apps/api/tests -p "test_book_memory_*unittest.py"
```

---

## Phase 2: Build Background Memory Consolidation

### Task 2.1: Extract episodes from chapters and accepted actions

**Files:**

- Create: `apps/api/app/services/book_memory/episode_extractor.py`
- Create: `apps/api/app/services/book_memory/consolidation_queue.py`
- Modify: `apps/api/app/worker.py`

**Behavior:**

- chapter changes and accepted writing actions enqueue consolidation jobs
- extractor produces structured `StoryEpisode` candidates
- each candidate carries source chapter and text provenance

**Verification:**

```bash
python -m unittest discover apps/api/tests -p "test_episode_*unittest.py"
```

### Task 2.2: Recompute story state snapshots

**Files:**

- Create: `apps/api/app/services/book_memory/story_state_compiler.py`

**Behavior:**

- recompute active chapter goal
- recompute active characters
- recompute current location
- recompute open threads in local scope

**Verification:**

- adding a new chapter or beat updates the project story state snapshot
- snapshot refresh stays idempotent

### Task 2.3: Propose graph and knowledge-state updates

**Files:**

- Create: `apps/api/app/services/book_memory/knowledge_updater.py`
- Create: `apps/api/app/services/book_memory/temporal_graph_updater.py`

**Behavior:**

- update character knowledge only when an episode creates or reveals information
- mutate graph facts with validity windows instead of overwrite-only behavior
- close prior truths with `valid_to_chapter` when superseded

**Verification:**

- chapter-indexed queries show different knowledge states before and after reveal events
- relationship transitions are time-bounded, not blindly replaced

---

## Phase 3: Materialize Fast Hot-Path Memory Packs

### Task 3.1: Build memory materialization layer

**Files:**

- Create: `apps/api/app/services/book_memory/materializers.py`

**Materializations to Build:**

- `world_constitution_pack`
- `character_memory_pack`
- `story_state_pack`
- `chapter_memory_pack`
- `character_knowledge_pack`

**Rules:**

- materializations are small and prompt-ready
- each pack records source version hashes
- packs can be invalidated independently

**Verification:**

```bash
python -m unittest discover apps/api/tests -p "test_materialization_*unittest.py"
```

### Task 3.2: Cache memory packs for request hot path

**Files:**

- Modify: `apps/api/app/services/context_compiler/context_pack.py`
- Create: `apps/api/app/services/book_memory/cache.py`

**Behavior:**

- preheat memory packs for active project and chapter
- use memory-pack cache before loading generic settings/cards
- preserve existing context pack until migration completes

**Verification:**

- repeated requests for the same chapter reuse cached memory packs
- invalidation occurs after chapter or graph updates

---

## Phase 4: Introduce the Memory Router

### Task 4.1: Define task-specific memory assembly interfaces

**Files:**

- Create: `apps/api/app/services/book_memory/router.py`
- Create: `apps/api/app/services/book_memory/types.py`

**Assemblers:**

- `build_planning_memory_context`
- `build_rewrite_memory_context`
- `build_consistency_memory_context`

**Rules:**

- no single universal assembly function should remain the long-term center
- each assembler has explicit read order and fallback policy

**Verification:**

- unit tests assert each assembler reads only allowed layers by default

### Task 4.2: Integrate `L5` fallback policy

**Files:**

- Create: `apps/api/app/services/book_memory/l5_fallback.py`
- Modify: `apps/api/app/services/context_compiler/pipeline.py`

**Fallback Conditions:**

- memory miss
- low confidence
- ambiguity in entity resolution
- explicit brainstorming or exploratory mode

**Rules:**

- rewrite-polish must not hit `L5` by default
- rewrite-expand only allows targeted retrieval
- planning may call `L5` when local and graph memory are insufficient

**Verification:**

- tests assert `LightRAG` is skipped for polish requests
- tests assert `LightRAG` is invoked only under explicit fallback conditions

---

## Phase 5: Migrate Planning Assistant to Book Memory OS

### Task 5.1: Swap planning assistant off the universal compiler

**Files:**

- Modify: `apps/api/app/api/endpoints/chat.py`
- Modify: `apps/api/app/api/endpoints/chat_stream_pipeline.py`
- Modify: `apps/api/app/services/context_compiler/pipeline.py`

**Behavior:**

- planning assistant should consume `build_planning_memory_context`
- legacy evidence payload remains available during migration
- context x-ray provenance should include memory-layer origin

**Verification:**

- stream endpoint still works
- assistant can cite current story state and chapter-bounded knowledge
- no regression in action proposal flow

### Task 5.2: Add chapter-bounded character knowledge reasoning

**Files:**

- Create: `apps/api/app/services/book_memory/character_knowledge_query.py`

**Behavior:**

- planning assistant can query:
  - what a character knows at chapter N
  - what a character does not know at chapter N
  - what changed recently

**Verification:**

- targeted unit tests with staged chapter reveals

---

## Phase 6: Migrate Rewrite Endpoints to Rewrite Memory Context

### Task 6.1: Introduce a dedicated rewrite context assembler

**Files:**

- Create: `apps/api/app/services/book_memory/rewrite_context.py`
- Modify: `apps/api/app/api/endpoints/chat_writing_assist.py`

**Rewrite Modes:**

- `polish`
- `expand`

**Polish Context:**

- selected text
- local before/after window
- POV/person/tense guard
- world-rule hard constraints
- optional style template

**Expand Context:**

- all polish context
- current story state
- current beat
- next beat
- active characters
- current location
- targeted entity fetch

**Verification:**

- polish prompt tokens shrink compared to current chat-style prompt
- expand preserves world rules and timeline more reliably

### Task 6.2: Stop using generic chat output contract for rewrite

**Files:**

- Modify: `apps/api/app/services/llm_provider.py`
- Modify: `apps/api/app/schemas/chat.py`

**Behavior:**

- add dedicated `rewrite.generate` path
- return only:
  - generated text
  - usage
  - minimal rewrite diagnostics

**Rules:**

- no `proposed_actions` contract in rewrite path
- no generic planning-oriented system prompt in rewrite path

**Verification:**

```bash
python -m unittest apps/api/tests/test_chat_rewrite_endpoint_unittest.py -v
```

---

## Phase 7: Add Consistency and Contradiction Engine

### Task 7.1: Build consistency context on temporal truth

**Files:**

- Create: `apps/api/app/services/book_memory/consistency_context.py`
- Modify: `apps/api/app/services/consistency_audit_service.py`

**Checks:**

- world-rule violations
- chronology violations
- character knowledge leaks
- impossible relationship states

**Verification:**

- unit tests with deliberate contradictions
- audit output references source episode or temporal fact

### Task 7.2: Surface lightweight warnings in the editor

**Files:**

- Modify: `apps/web/src/App.tsx`
- Modify: `apps/web/src/components/AssistantActionsPanel.tsx`

**Behavior:**

- show non-blocking warnings for:
  - character omniscience leaks
  - chronology drift
  - world-rule conflicts

**Verification:**

- warning UI appears only when contradictions exist
- writing flow remains uninterrupted

---

## Phase 8: Frontend Product Contract Cleanup

### Task 8.1: Remove most retrieval knobs from default author settings

**Files:**

- Modify: `apps/web/src/components/settings/ContextTab.tsx`
- Modify: `apps/web/src/hooks/useAssistantSessionFlow.ts`

**Behavior:**

- hide low-level `rag_mode` and `deterministic_first` from default author flow
- replace with higher-level choices later if needed:
  - `strict canon`
  - `balanced`
  - `exploratory`

**Verification:**

- default author UI no longer exposes retrieval internals
- advanced diagnostics still preserve observability

### Task 8.2: Add memory-origin visibility to advanced panels

**Files:**

- Modify: `apps/web/src/components/chat/ContextXRayPopover.tsx`
- Modify: `apps/web/src/types.ts`

**Behavior:**

- show which layer supplied the supporting context:
  - `story_state`
  - `character_memory`
  - `temporal_graph`
  - `cold_retrieval`

**Verification:**

- context x-ray can distinguish memory-layer origin from cold retrieval fallback

---

## Phase 9: Testing and Evaluation

### Task 9.1: Build evaluation fixtures for the four target capabilities

**Files:**

- Create: `apps/api/evals/data/book_memory_eval.jsonl`
- Create: `apps/api/evals/run_book_memory_eval.py`

**Capability Sets:**

- world memory
- character memory
- story state memory
- event chronology memory

**Example Eval Questions:**

- "At chapter 18, does character A know secret B?"
- "Is rule X still active after event Y?"
- "What is the current chapter trying to resolve?"
- "Did event M happen before or after reveal N?"

### Task 9.2: Define success thresholds

**Metrics:**

- character knowledge correctness
- chronology correctness
- planning acceptance rate
- rewrite factual drift rate
- `L5` fallback rate
- planning and rewrite prompt token budgets

**Verification:**

- evaluation runner produces a report with the above dimensions

---

## Suggested File Ownership During Implementation

To reduce collision risk during execution:

- Agent A: Postgres models and schemas
- Agent B: graph and temporal query layer
- Agent C: rewrite context and LLM contract split
- Agent D: frontend settings and x-ray layer
- Agent E: eval harness and fixtures

One file should only be actively modified by one agent at a time.

---

## Execution Order Recommendation

Recommended sequence:

1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Phase 4
6. Phase 5
7. Phase 6
8. Phase 7
9. Phase 8
10. Phase 9

Reason:

- data model must exist before consolidation
- consolidation must exist before hot-path memory packs are credible
- task-specific routers must exist before assistant migration
- rewrite should migrate after planning, because planning exercises more of the memory stack

---

## First Milestone Definition

The first meaningful milestone is:

`Planning assistant answers character knowledge and chronology questions correctly without depending on LightRAG by default.`

That milestone proves the architecture is no longer retrieval-first.

---

## Immediate Next Action

Start with:

- Phase 0 reference clone
- Phase 1 data model
- Phase 2 background consolidation skeleton

Do not begin with frontend polish or advanced settings cleanup.
