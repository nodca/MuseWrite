# Simulation Director + ToM Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current round-robin simulation flow with a Director-driven orchestration system that supports dynamic bidding, explicit `paused/failed` outcomes, and session-scoped second-order Theory of Mind.

**Architecture:** Split simulation into `director`, `mind_engine`, `bidding`, `referee`, `actor_engine`, and `repository` modules. Add dedicated runtime state, event, and decision-trace storage so that speaker selection, belief updates, and turn generation are independently testable and no longer hidden inside `simulation_service.py`.

**Tech Stack:** FastAPI, SQLModel, Pydantic v2, pytest/unittest, existing structured output path in `llm_provider`

---

### Task 1: Introduce simulation runtime models

**Files:**
- Modify: `apps/api/app/models/simulation.py`
- Create: `apps/api/tests/test_simulation_models_unittest.py`

**Step 1: Write the failing tests**

- Add tests covering:
  - `SimulationSession` no longer exposes `pending_events`
  - new `SimulationSessionState`, `SimulationEvent`, `SimulationDecisionTrace` models serialize required fields
  - `revision`, `runtime_status`, and trace payloads default correctly

**Step 2: Run the focused tests and verify failure**

Run: `pytest apps/api/tests/test_simulation_models_unittest.py -q`  
Expected: FAIL because new models/fields do not exist yet

**Step 3: Implement the minimal model changes**

- Add `SimulationSessionState`
- Add `SimulationEvent`
- Add `SimulationDecisionTrace`
- Remove `pending_events` from `SimulationSession`
- Keep `SimulationTurn` as the final output record

**Step 4: Re-run the focused tests**

Run: `pytest apps/api/tests/test_simulation_models_unittest.py -q`  
Expected: PASS

### Task 2: Add repository and typed runtime contracts

**Files:**
- Create: `apps/api/app/services/simulation/__init__.py`
- Create: `apps/api/app/services/simulation/types.py`
- Create: `apps/api/app/services/simulation/repository.py`
- Create: `apps/api/tests/test_simulation_repository_unittest.py`

**Step 1: Write the failing tests**

- Add tests for:
  - creating/loading session state
  - appending/consuming events
  - persisting turn + decision trace in one transaction
  - optimistic `revision` updates

**Step 2: Run the focused tests and verify failure**

Run: `pytest apps/api/tests/test_simulation_repository_unittest.py -q`  
Expected: FAIL because repository/types are missing

**Step 3: Implement minimal repository contracts**

- Define typed contracts:
  - `WorldState`
  - `CharacterMindState`
  - `Belief`
  - `SecondOrderBelief`
  - `TurnCandidate`
  - `RefereeDecision`
  - `TurnOutcome`
- Implement repository helpers for session state, events, turns, and traces

**Step 4: Re-run the focused tests**

Run: `pytest apps/api/tests/test_simulation_repository_unittest.py -q`  
Expected: PASS

### Task 3: Build the Mind Engine

**Files:**
- Create: `apps/api/app/services/simulation/mind_engine.py`
- Create: `apps/api/tests/test_simulation_mind_engine_unittest.py`

**Step 1: Write the failing tests**

- Cover:
  - belief creation from observed turns/events
  - second-order belief updates
  - confidence decay / expiry
  - cooldown and agitation updates

**Step 2: Run the focused tests and verify failure**

Run: `pytest apps/api/tests/test_simulation_mind_engine_unittest.py -q`  
Expected: FAIL because `mind_engine.py` does not exist

**Step 3: Implement minimal logic**

- Add `advance_minds(world_state, session_state) -> session_state`
- Keep ToM capped at second order
- Do not write to Neo4j

**Step 4: Re-run the focused tests**

Run: `pytest apps/api/tests/test_simulation_mind_engine_unittest.py -q`  
Expected: PASS

### Task 4: Build bidding and referee selection

**Files:**
- Create: `apps/api/app/services/simulation/policy.py`
- Create: `apps/api/app/services/simulation/bidding.py`
- Create: `apps/api/app/services/simulation/referee.py`
- Create: `apps/api/tests/test_simulation_bidding_unittest.py`
- Create: `apps/api/tests/test_simulation_referee_unittest.py`

**Step 1: Write the failing tests**

- Add bidding tests for:
  - agitation score increases after provocation
  - cooldown suppresses repeat speakers
  - irrelevant actors score lower
- Add referee tests for:
  - chosen winner must be in candidate set
  - low-signal rounds return `none`
  - no implicit round-robin behavior remains

**Step 2: Run the focused tests and verify failure**

Run: `pytest apps/api/tests/test_simulation_bidding_unittest.py apps/api/tests/test_simulation_referee_unittest.py -q`  
Expected: FAIL because modules do not exist

**Step 3: Implement minimal selection pipeline**

- Add policy weights and thresholds
- Produce full candidate set from state
- Referee returns structured `winner_id | none`
- No fallback path to legacy selection

**Step 4: Re-run the focused tests**

Run: `pytest apps/api/tests/test_simulation_bidding_unittest.py apps/api/tests/test_simulation_referee_unittest.py -q`  
Expected: PASS

### Task 5: Build the Actor Engine and Director

**Files:**
- Create: `apps/api/app/services/simulation/actor_engine.py`
- Create: `apps/api/app/services/simulation/director.py`
- Create: `apps/api/tests/test_simulation_actor_engine_unittest.py`
- Create: `apps/api/tests/test_simulation_director_unittest.py`

**Step 1: Write the failing tests**

- Add actor engine tests asserting strict structured output validation（必须 JSON、`proposed_actions` 必须为空、台词不能为空）
- Add director tests covering:
  - `spoken` outcome writes turn + trace
  - `paused` outcome writes trace without turn
  - invalid actor output returns `failed`

**Step 2: Run the focused tests and verify failure**

Run: `pytest apps/api/tests/test_simulation_actor_engine_unittest.py apps/api/tests/test_simulation_director_unittest.py -q`  
Expected: FAIL because `actor_engine.py` and `director.py` do not exist

**Step 3: Implement minimal orchestration**

- `actor_engine` 负责台词生成（结构化 JSON；不允许动作）
- `director` executes the full turn lifecycle
- Persist `DecisionTrace` on every outcome
- Explicitly return `spoken`, `paused`, or `failed`

**Step 4: Re-run the focused tests**

Run: `pytest apps/api/tests/test_simulation_actor_engine_unittest.py apps/api/tests/test_simulation_director_unittest.py -q`  
Expected: PASS

### Task 6: Migrate the API surface to the new engine

**Files:**
- Modify: `apps/api/app/api/endpoints/simulation.py`
- Modify: `apps/api/app/services/simulation_service.py`
- Create: `apps/api/tests/test_simulation_endpoint_unittest.py`

**Step 1: Write the failing tests**

- Cover:
  - session creation/read still works after model changes
  - event injection writes `SimulationEvent`
  - run endpoint consumes `TurnOutcome`
  - response payload no longer depends on `pending_events`

**Step 2: Run the focused tests and verify failure**

Run: `pytest apps/api/tests/test_simulation_endpoint_unittest.py -q`  
Expected: FAIL because endpoint/service contracts still assume legacy flow

**Step 3: Implement the migration**

- Rewire endpoint logic to call `director.run_turn()`
- Update response schemas for session/runtime changes
- Reduce `simulation_service.py` to façade helpers only, or delete it if all imports are migrated cleanly

**Step 4: Re-run the focused tests**

Run: `pytest apps/api/tests/test_simulation_endpoint_unittest.py -q`  
Expected: PASS

### Task 7: Delete legacy round-robin flow and verify behavior

**Files:**
- Modify: `apps/api/app/services/simulation_service.py` or Delete if unused
- Modify: `apps/api/tests/test_structured_outputs_unittest.py`
- Modify: `apps/api/tests/test_simulation_*`

**Step 1: Write the failing cleanup assertions**

- Add assertions that:
  - `decide_next_actor_id()` no longer exists
  - no code path references `pending_events`
  - no test depends on round-robin order

**Step 2: Run the focused tests and verify failure**

Run: `pytest apps/api/tests/test_structured_outputs_unittest.py apps/api/tests/test_simulation_* -q`  
Expected: FAIL until legacy references are removed

**Step 3: Remove legacy code**

- Delete round-robin selection
- Delete JSON queue event handling
- Delete dead helpers made obsolete by `Mind Engine`
- Update or remove outdated tests and comments

**Step 4: Run focused verification**

Run: `pytest apps/api/tests/test_structured_outputs_unittest.py apps/api/tests/test_simulation_* -q`  
Expected: PASS

**Step 5: Run broader backend verification**

Run: `pytest apps/api/tests -q`  
Expected: PASS, or only unrelated pre-existing failures

### Task 8: Document outcomes and review risk

**Files:**
- Modify: `docs/plans/2026-03-10-simulation-director-tom-design.md`
- Modify: `docs/plans/2026-03-10-simulation-director-tom-implementation-plan.md`

**Step 1: Record migration notes**

- Note any schema adjustments made during implementation
- Note whether `simulation_service.py` was deleted or retained as façade

**Step 2: Run a final design sanity pass**

Run: `pytest apps/api/tests/test_simulation_* -q`  
Expected: PASS

**Step 3: Prepare execution handoff**

- Summarize:
  - deleted legacy paths
  - new runtime entities
  - remaining risks around prompt tuning and ToM drift
