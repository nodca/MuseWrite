# Neo4j PPR Projection Prewarm Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add worker-side asynchronous prewarm of named PPR projection graphs after successful graph sync writes.

**Architecture:** Reuse existing `graph_sync` worker pipeline. After a successful graph sync write, call a new `prewarm_neo4j_ppr_projection` helper in `retrieval_adapters` to build/refresh the named projection for the current scope. Failures are logged and do not affect graph sync completion.

**Tech Stack:** Python, FastAPI services, Neo4j GDS, unittest

---

### Task 1: Add projection prewarm helper in retrieval_adapters

**Files:**
- Modify: `apps/api/app/services/retrieval_adapters.py`

**Step 1: Write the failing test**

```python
def test_process_graph_job_triggers_projection_prewarm_on_sync(self) -> None:
    ...
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest apps.api.tests.test_worker_unittest.WorkerTestCase.test_process_graph_job_triggers_projection_prewarm_on_sync -v`  
Expected: FAIL with "prewarm not called"

**Step 3: Write minimal implementation**

```python
def prewarm_neo4j_ppr_projection(project_id: int, *, current_chapter: int | None = None, reason: str = "graph_sync_worker") -> dict:
    # build named projection if missing/outdated
```

**Step 4: Run test to verify it passes**

Run: `python -m unittest apps.api.tests.test_worker_unittest.WorkerTestCase.test_process_graph_job_triggers_projection_prewarm_on_sync -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add apps/api/app/services/retrieval_adapters.py apps/api/tests/test_worker_unittest.py
git commit -m "feat: add neo4j ppr projection prewarm helper"
```

---

### Task 2: Wire prewarm into graph_sync worker

**Files:**
- Modify: `apps/api/app/worker.py`

**Step 1: Write the failing test**

```python
def test_process_graph_job_triggers_projection_prewarm_on_sync(self) -> None:
    ...
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest apps.api.tests.test_worker_unittest.WorkerTestCase.test_process_graph_job_triggers_projection_prewarm_on_sync -v`  
Expected: FAIL with "prewarm not called"

**Step 3: Write minimal implementation**

```python
graph_sync, fact_keys = process_graph_sync_for_action(...)
if fact_keys and graph_sync and graph_sync.get("status") == "synced":
    prewarm_neo4j_ppr_projection(project_id, current_chapter=...)
```

**Step 4: Run test to verify it passes**

Run: `python -m unittest apps.api.tests.test_worker_unittest.WorkerTestCase.test_process_graph_job_triggers_projection_prewarm_on_sync -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add apps/api/app/worker.py apps/api/tests/test_worker_unittest.py
git commit -m "feat: prewarm neo4j ppr projection after graph sync"
```

---

### Task 3: Regression tests

**Files:**
- Test: `apps/api/tests/test_worker_unittest.py`

**Step 1: Run focused test**

Run: `python -m unittest apps.api.tests.test_worker_unittest.WorkerTestCase.test_process_graph_job_triggers_projection_prewarm_on_sync -v`  
Expected: PASS

**Step 2: Run graph sync unit tests**

Run: `python -m unittest apps.api.tests.test_worker_unittest -v`  
Expected: PASS

**Step 3: Commit**

```bash
git add apps/api/tests/test_worker_unittest.py
git commit -m "test: cover projection prewarm in worker"
```
