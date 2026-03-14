# Book Memory OS v1 Design

Date: 2026-03-13

## Summary

This design replaces the current "RAG-heavy context assembly" mindset with a memory-first architecture for long-form fiction writing.

The primary product goal is not generic question answering. The goal is that the writing agent can reliably understand:

- world rules
- character knowledge and intent
- current story state
- event chronology

In the new architecture:

- `L1-L4` are the primary memory system
- `LightRAG` is demoted to `L5 Cold Retrieval`
- `Neo4j` becomes the authoritative temporal knowledge layer
- `Postgres` stores memory materializations, summaries, and episodic story records

The target behavior is that the agent can answer questions such as:

- "At this chapter, what does this character know?"
- "Did this event happen before or after that reveal?"
- "What is the current chapter trying to accomplish?"
- "Does this draft expansion violate any established world rule?"

## Why Change

The current system already contains strong retrieval and context compilation primitives, including:

- `DSL > GRAPH > RAG` ordering
- memory decay
- context pack preheating
- compressed context
- prompt cache prefixes

However, the current architecture still centers on assembling context per request. That is good for retrieval quality, but not sufficient for making an agent feel like it "knows the book".

For fiction writing, persistent understanding matters more than broad semantic recall. The agent must track state transitions, not just retrieve similar text.

## Product Priorities

Book Memory OS v1 focuses on four capabilities only:

1. World memory
2. Character memory
3. Story state memory
4. Event chronology memory

These four capabilities define whether the agent understands the book.

Not in v1 core:

- author preference memory
- style memory
- personalized writing taste modeling

Those can be added in v1.5 after factual and temporal correctness are stable.

## Architecture Decision

Adopt a hybrid design:

- `Graphiti-style temporal graph` as the core reasoning substrate
- `Letta-style memory files` as the human-readable governance layer
- `LangMem-style background consolidation` as the runtime memory maintenance pattern
- `LightRAG` as `L5 Cold Retrieval`

This is not a full framework replacement. It is a project-native design that borrows proven ideas while preserving the existing stack.

## Memory Layers

### L0 Working Buffer

Per-request working context.

Contains:

- current user instruction
- current selection
- local before/after text window
- cursor or chapter location
- active rewrite or planning intent

Purpose:

- immediate generation control
- local editing correctness

### L1 Active Writing Memory

The current writing-state snapshot.

Contains:

- current volume
- current chapter
- current scene beat
- chapter goal
- active characters
- current location
- unresolved local conflicts
- active constraints

Purpose:

- keep the model anchored to what the author is writing right now

### L2 Episodic Story Memory

Structured records of what has happened in the story.

Contains:

- chapter summaries
- scene summaries
- event records
- reveals
- promises and unresolved setups
- relationship changes
- status transitions

Purpose:

- allow "what happened so far" reasoning
- preserve progression over time

### L3 Canon Memory

Stable book knowledge.

Contains:

- world rules
- role definitions
- organizations
- places
- items
- power systems
- terminology
- canonical character sheets

Purpose:

- serve as the authoritative source for durable facts

### L4 Temporal Graph Memory

The authoritative graph for dynamic truth.

Contains:

- entities
- relations
- event nodes
- knowledge edges
- validity windows
- source provenance
- contradiction markers

Purpose:

- answer "what is true at time T"
- answer "who knows what at time T"
- support consistency checks and conflict detection

### L5 Cold Retrieval

Semantic recall only.

Implementation:

- `LightRAG`
- optional lexical search
- future external search if needed

Purpose:

- recover unstructured old prose
- recall weakly-linked passages
- assist brainstorming when memory layers are insufficient

Non-purpose:

- not the final truth source
- not the primary state carrier

## Authority Order

The new authority order is:

`L1 current writing state > L4 temporal graph > L3 canon memory > L2 episodic memory > L5 cold retrieval`

Interpretation:

- Immediate writing state overrides broad historical hints.
- Temporal truth overrides plain summaries.
- Canon overrides fuzzy semantic recall.
- Cold retrieval can propose evidence, but not define truth by itself.

This supersedes the previous retrieval-centric interpretation while remaining compatible with the existing `DSL > GRAPH > RAG` principle.

## Core Data Model

### 1. WorldRule

Represents durable story laws.

Fields:

- `id`
- `project_id`
- `scope`
- `title`
- `statement`
- `priority`
- `tags`
- `status`
- `source_refs`
- `created_at`
- `updated_at`

Examples:

- cultivation rules
- magic constraints
- social taboos
- political structure

### 2. CharacterProfile

Represents durable character identity.

Fields:

- `id`
- `project_id`
- `canonical_name`
- `aliases`
- `public_traits`
- `private_traits`
- `core_goals`
- `fears`
- `taboos`
- `default_voice_notes`
- `status`
- `source_refs`

Examples:

- "Lin Che wants revenge but will not betray innocents"

### 3. StoryStateSnapshot

Represents the latest structured writing state.

Fields:

- `id`
- `project_id`
- `volume_id`
- `chapter_id`
- `scene_beat_id`
- `chapter_goal`
- `active_characters`
- `current_location`
- `active_conflicts`
- `open_questions`
- `updated_at`

Purpose:

- fast hot-path load for planning and expansion

### 4. StoryEpisode

Represents a structured event or scene outcome.

Fields:

- `id`
- `project_id`
- `chapter_id`
- `scene_beat_id`
- `episode_index`
- `title`
- `summary`
- `event_type`
- `participants`
- `location`
- `visibility`
- `importance`
- `source_text_ref`
- `created_at`

Examples:

- confrontation
- reveal
- betrayal
- promise
- discovery

### 5. TemporalFact

Represents a graph truth with validity boundaries.

Fields:

- `fact_id`
- `project_id`
- `subject_entity_id`
- `predicate`
- `object_entity_id` or `object_text`
- `valid_from_chapter`
- `valid_to_chapter`
- `confidence`
- `source_episode_id`
- `evidence_ref`
- `status`

Examples:

- A trusts B from chapter 8 to chapter 17
- C knows secret X starting chapter 21
- faction D controls city E after chapter 33

### 6. CharacterKnowledgeState

Represents what a character knows at a given point.

Fields:

- `id`
- `project_id`
- `character_id`
- `knowledge_key`
- `knowledge_value`
- `gained_at_chapter`
- `lost_at_chapter`
- `source_episode_id`
- `confidence`

Purpose:

- make character epistemic boundaries queryable

### 7. MemoryMaterialization

Cached memory views for fast prompt assembly.

Fields:

- `id`
- `project_id`
- `materialization_type`
- `scope_key`
- `payload`
- `source_versions`
- `updated_at`

Examples:

- chapter memory pack
- character knowledge pack
- current story state pack
- world constitution pack

## Human-Readable Governance Layer

Introduce a memory file layer stored in Git-tracked docs or structured assets.

Suggested file groups:

- `memory/world/constitution/*.md`
- `memory/characters/*.md`
- `memory/story/current-state.md`
- `memory/timelines/*.md`

Purpose:

- human review
- deterministic correction
- easier editorial trust

These files are not the runtime source of truth by themselves. They are governance artifacts that sync into structured memory and graph state.

## Background Memory Manager

Memory should not be assembled entirely in the request path.

Introduce a background manager that performs:

1. Draft-to-episode extraction
2. Episode-to-graph mutation proposal
3. Character knowledge update proposal
4. Story state recomputation
5. Materialized memory pack refresh

Execution pattern:

- user applies content or actions
- background jobs extract structured memory deltas
- reviewable mutations update graph and state
- memory packs refresh asynchronously

This reduces request latency while improving persistent understanding.

## Context Assembly Pipelines

### Planning Pipeline

Used by:

- writing assistant
- chapter planning
- next-step suggestion

Read order:

1. `L1 StoryStateSnapshot`
2. relevant `CharacterProfile`
3. relevant `CharacterKnowledgeState`
4. recent `StoryEpisode`
5. relevant `TemporalFact`
6. `L5 LightRAG` only on miss or ambiguity

Output shape:

- current goal
- active constraints
- active actors
- what each actor knows
- recent causal chain
- safe next directions

### Rewrite Pipeline

Used by:

- polish
- expand

Polish context:

- selected text
- local surrounding text window
- POV/person/tense guard
- hard factual constraints
- optional style template

Expand context:

- all polish inputs
- current chapter goal
- current beat
- next beat
- active characters
- current location
- targeted character/world fetch if named entities appear

Rules:

- polish should not hit `L5` by default
- expand should only hit targeted retrieval, not full cold retrieval

### Consistency Pipeline

Used by:

- audits
- contradiction detection
- editor warnings

Read order:

1. candidate draft or chapter
2. temporal facts in scope
3. world rules in scope
4. character knowledge boundaries
5. recent story episodes

Purpose:

- detect world-rule violations
- detect chronology conflicts
- detect character knowledge leaks

## LightRAG as L5

`LightRAG` remains useful, but only as cold recall.

It should be triggered only when:

1. no relevant structured memory exists
2. current memory confidence is low
3. user asks for exploratory brainstorming
4. a named entity appears but has weak local coverage
5. old prose fragments may contain supporting detail not yet materialized

It should not be used as:

- the canonical truth source
- the main mechanism for character knowledge reasoning
- the default path for rewrite requests

## Read/Write Flows

### Author Writes New Draft Text

1. draft saved
2. optional inline rewrite runs using `L0 + L1`
3. on acceptance, background extractor creates episode candidates
4. candidate memory updates are proposed
5. graph and state refresh materialized packs

### Assistant Plans Next Scene

1. planning router loads `L1`
2. pull active characters and knowledge boundaries
3. load recent episodes and temporal facts
4. if ambiguity remains, call `L5`
5. generate proposal with provenance

### Consistency Check

1. analyze current chapter or diff
2. compare against world rules
3. compare against temporal facts
4. compare against character knowledge state
5. emit conflict report

## Proposed Service Boundaries

### New or Expanded Services

- `book_memory/world_service.py`
- `book_memory/character_service.py`
- `book_memory/story_state_service.py`
- `book_memory/episode_service.py`
- `book_memory/temporal_graph_service.py`
- `book_memory/materialization_service.py`
- `book_memory/router.py`
- `book_memory/rewrite_context_service.py`
- `book_memory/planning_context_service.py`
- `book_memory/consistency_context_service.py`

### Existing Services to Retain

- `llm_provider`
- action audit pipeline
- project assets CRUD
- chapter and beat CRUD
- Neo4j runtime
- worker queue infrastructure

### Existing Services to Reduce in Responsibility

- current context compiler should stop being the single universal assembly engine
- retrieval logic should become fallback logic, not primary logic

## Storage Strategy

### Postgres

Use for:

- world rules
- character profiles
- story state snapshots
- story episodes
- memory materializations
- sync metadata

### Neo4j

Use for:

- entities
- temporal facts
- knowledge edges
- relationship transitions
- event participation graph

### LightRAG

Use for:

- unstructured prose recall
- semantic cold retrieval

## Migration Plan

### Phase 1: Introduce New Data Model

Add new persistence tables and graph conventions without removing current retrieval paths.

Deliverables:

- SQL tables for world, character, state, episode, materialization
- graph schema conventions for temporal facts and knowledge edges
- internal serializers and provenance references

### Phase 2: Build Background Consolidation

Convert accepted content and chapter text into episodic memory.

Deliverables:

- episode extraction job
- story state recompute job
- graph fact mutation proposal job
- character knowledge update job

### Phase 3: Introduce Memory Router

Split the universal compiler into task-specific assemblers.

Deliverables:

- planning context assembler
- rewrite context assembler
- consistency context assembler
- `L5` fallback policy

### Phase 4: Migrate Assistant Flows

Planning first, rewrite second, consistency third.

Deliverables:

- planning assistant uses Book Memory OS
- rewrite endpoints use local memory packs
- consistency checks consume temporal truth

### Phase 5: Retire Retrieval-First Assumptions

Reduce front-end exposure of low-level retrieval switches.

Deliverables:

- remove or hide most end-user `rag_mode` controls
- replace with intent-level toggles
- keep deep debug visibility in advanced tools only

## API Direction

New internal API shapes should become more explicit.

Examples:

- `build_planning_memory_context(project_id, chapter_id, scene_beat_id)`
- `build_rewrite_memory_context(project_id, chapter_id, selection, mode)`
- `build_consistency_memory_context(project_id, chapter_id, draft_text)`
- `query_character_knowledge(character_id, at_chapter)`
- `query_temporal_fact(subject, predicate, at_chapter)`

The old single "compile everything" entry point can remain during migration, but should not remain the architectural center.

## Success Criteria

Book Memory OS v1 is successful if the agent can reliably:

1. answer what a character knows at a given chapter
2. answer whether a fact is valid at a given chapter
3. identify the current chapter's role in the story arc
4. suggest next-step planning without violating canon
5. expand text without introducing timeline or knowledge leaks

## Metrics

### Product Metrics

- character knowledge correctness
- chronology correctness
- planning acceptance rate
- rewrite factual drift rate
- contradiction detection precision

### System Metrics

- planning prompt tokens
- rewrite prompt tokens
- `L5` fallback rate
- materialization freshness latency
- graph update lag

## Out of Scope

Not part of v1:

- style imitation memory
- author preference adaptation
- cross-book universe memory federation beyond reference-project basics
- autonomous multi-agent writer rooms

## Recommendation

Proceed with:

- `Temporal Graph Memory OS` as the primary architecture
- `LightRAG` fixed as `L5 Cold Retrieval`
- rewrite and planning split into separate context assemblers
- background memory consolidation as the default update mechanism

This is the most direct path toward an agent that understands the book as an evolving world, not just a searchable document collection.
