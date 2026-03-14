# Book Memory Graph Schema

Date: 2026-03-13

## Goal

Define the temporal graph conventions for `Book Memory OS v1`.

This graph is not a generic entity graph. It exists to answer time-bounded fiction questions such as:

- what is true at chapter N
- what changed after event X
- what a character knows at chapter N
- which relationship is active during a given interval

## Node Types

### Character

Represents a durable character identity.

Key properties:

- `project_id`
- `character_id`
- `canonical_name`
- `aliases`
- `status`

### Event

Represents a story event or structured episode.

Key properties:

- `project_id`
- `episode_id`
- `chapter_id`
- `chapter_index`
- `scene_beat_id`
- `event_type`
- `title`

### WorldRule

Represents a rule or law in the fictional world.

Key properties:

- `project_id`
- `rule_id`
- `scope`
- `title`
- `status`

### Location

Represents a named place in the story world.

Key properties:

- `project_id`
- `location_key`
- `display_name`

### Faction

Represents an organization or political force.

Key properties:

- `project_id`
- `faction_key`
- `display_name`

### Knowledge

Represents a unit of knowable information.

Key properties:

- `project_id`
- `knowledge_key`
- `summary`
- `visibility`

## Edge Types

### KNOWS

`(:Character)-[:KNOWS]->(:Knowledge)`

Used for chapter-bounded epistemic state.

Required edge properties:

- `project_id`
- `valid_from_chapter`
- `valid_to_chapter`
- `confidence`
- `source_episode_id`
- `evidence`

### TRUSTS

`(:Character)-[:TRUSTS]->(:Character)`

Used for active trust relationships over time.

Required edge properties:

- `project_id`
- `valid_from_chapter`
- `valid_to_chapter`
- `confidence`
- `source_episode_id`
- `evidence`

### BETRAYS

`(:Character)-[:BETRAYS]->(:Character)`

Used for betrayal events or post-betrayal state markers.

### PARTICIPATED_IN

`(:Character)-[:PARTICIPATED_IN]->(:Event)`

Used to link characters to story episodes.

### LOCATED_IN

`(:Character)-[:LOCATED_IN]->(:Location)`

Used for chapter-bounded location state.

### AFFECTS

`(:Event)-[:AFFECTS]->(:Character|:Faction|:Location|:Knowledge)`

Used for causal reach from an event to world state.

### VIOLATES_RULE

`(:Event)-[:VIOLATES_RULE]->(:WorldRule)`

Used for consistency auditing and world-rule conflict detection.

## Temporal Rules

All dynamic truth edges must carry:

- `valid_from_chapter`
- `valid_to_chapter`

Open-ended truth uses:

- `valid_to_chapter = null`

When a new truth supersedes an old one:

1. close the old edge by setting `valid_to_chapter`
2. create the new edge with a later `valid_from_chapter`

Do not overwrite dynamic truth in place without preserving interval history.

## Provenance Rules

Every dynamic edge should preserve where it came from.

Minimum provenance:

- `source_episode_id`
- `evidence`
- `confidence`

Preferred provenance:

- `chapter_id`
- `scene_beat_id`
- `mutation_id`

## Query Principles

All graph queries that answer story truth must be chapter-bounded.

Example policy:

- if `current_chapter` is known, only return edges active at that chapter
- if chapter is unknown, either return the latest active truth or mark the result as ambiguous

This graph should never answer "what is true" without a time context if time context is available.

## Initial Focus

The first graph features to implement should support:

1. character knowledge at chapter N
2. active trust or hostility at chapter N
3. location state at chapter N
4. event chronology and causal links

## Non-Goals

Not part of the initial graph schema:

- style memory
- author preference memory
- generic semantic text similarity edges

Those belong elsewhere in Book Memory OS.
