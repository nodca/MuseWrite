# Product Simplification and Concept Convergence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove dual-mode/Ghost-centric product complexity and converge the writing experience around a pure editor home, a planning-first assistant drawer, and a hidden advanced layer.

**Architecture:** Execute this as a staged refactor. First replace product concepts and regression anchors, then unify the shell, then relocate capabilities, and only after that clean up obsolete Ghost/mode state and shrink public configuration exposure. Preserve existing backend protocol compatibility until the new UI contract is stable.

**Tech Stack:** React 18, TypeScript, Zustand, Vite, Playwright, FastAPI, Pydantic, unittest

---

### Task 1: Freeze the new product contract in docs

**Files:**
- Create: `docs/plans/2026-03-11-product-simplification-design.md`
- Modify: `README.md`
- Modify: `docs/author-manual.md`

**Step 1: Write the failing wording checks**

```bash
rg -n "写作模式|工作台模式|Ghost Text|灵感工作台" README.md docs/author-manual.md
```

**Step 2: Run the checks to verify old concepts still exist**

Run: `rg -n "写作模式|工作台模式|Ghost Text|灵感工作台" README.md docs/author-manual.md`  
Expected: matches found

**Step 3: Rewrite the public narrative**

- `README.md`
  - describe one writing workspace
  - describe planning-first assistant drawer
  - remove mode-based language
- `docs/author-manual.md`
  - describe pure editor home
  - describe right-side floating button
  - describe selection-based rewrite only
  - remove Ghost Text and workbench references

**Step 4: Re-run the wording checks**

Run: `rg -n "写作模式|工作台模式|Ghost Text|灵感工作台" README.md docs/author-manual.md`  
Expected: no matches

**Step 5: Commit**

```bash
git add README.md docs/author-manual.md docs/plans/2026-03-11-product-simplification-design.md
git commit -m "docs: define simplified writing-first product contract"
```

### Task 2: Replace dual-mode shell assertions with the new shell contract

**Files:**
- Modify: `apps/web/tests/e2e/smoke.spec.ts`
- Modify: `apps/web/tests/e2e/blast-radius-preview.spec.ts`

**Step 1: Write the failing shell regression**

```ts
test("editor home opens without any mode switch", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByTestId("ui-mode-toggle")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "写作助手" })).toBeVisible();
});
```

**Step 2: Run focused tests to verify they fail**

Run: `cd apps/web && npm run test:e2e -- tests/e2e/smoke.spec.ts --project=chromium --grep "editor home opens without any mode switch"`  
Expected: FAIL

**Step 3: Rewrite test intent**

- remove all assertions that require switching to pro/workbench mode
- rewrite blast-radius access to enter from assistant drawer or advanced section instead of `切到工作台模式`

**Step 4: Re-run focused tests**

Run: `cd apps/web && npm run test:e2e -- tests/e2e/smoke.spec.ts tests/e2e/blast-radius-preview.spec.ts --project=chromium`  
Expected: still FAIL on implementation, but no longer fail because of old wording assumptions

**Step 5: Commit**

```bash
git add apps/web/tests/e2e/smoke.spec.ts apps/web/tests/e2e/blast-radius-preview.spec.ts
git commit -m "test: replace dual-mode shell regressions with new workspace contract"
```

### Task 3: Remove Ghost-first expectations from tests before UI refactor

**Files:**
- Modify: `apps/web/tests/e2e/diff-ghost-interactions.spec.ts`
- Modify: `apps/web/tests/e2e/resilience-regression.spec.ts`
- Delete or Modify: `apps/api/tests/test_ghost_text_endpoint_unittest.py`

**Step 1: Split the existing ghost/diff test intent**

```ts
test("selection rewrite supports polish and expand", async ({ page }) => {
  await page.goto("/");
  // focus only on selection-based rewrite behavior
});
```

**Step 2: Run the old Ghost-focused tests**

Run: `cd apps/web && npm run test:e2e -- tests/e2e/diff-ghost-interactions.spec.ts tests/e2e/resilience-regression.spec.ts --project=chromium`  
Expected: FAIL or rely on Ghost-specific UI

**Step 3: Rewrite test coverage**

- keep diff accept/ignore coverage
- keep assistant reply insert/replace coverage
- remove Ghost auto-suggestion assertions
- replace Ghost resilience coverage with assistant drawer / selection rewrite resilience coverage
- if Ghost endpoint is being deprecated immediately, delete `test_ghost_text_endpoint_unittest.py`
- if backend compatibility remains temporarily, rewrite the unittest to cover only deprecated shim behavior explicitly

**Step 4: Re-run focused tests**

Run: `cd apps/web && npm run test:e2e -- tests/e2e/diff-ghost-interactions.spec.ts tests/e2e/resilience-regression.spec.ts --project=chromium`  
Expected: FAIL only because implementation is pending, not because Ghost is assumed as a product requirement

**Step 5: Commit**

```bash
git add apps/web/tests/e2e/diff-ghost-interactions.spec.ts apps/web/tests/e2e/resilience-regression.spec.ts apps/api/tests
git commit -m "test: remove ghost-first regression assumptions"
```

### Task 4: Replace `uiMode` with explicit shell state

**Files:**
- Modify: `apps/web/src/store/chatStore.ts`
- Delete: `apps/web/src/store/chatStore.uiMode.test.ts`
- Create: `apps/web/src/store/chatStore.workspaceShell.test.ts`
- Modify: `apps/web/src/App.tsx`

**Step 1: Write the failing store contract**

```ts
import { useChatStore } from "./chatStore";

function run() {
  const state = useChatStore.getState();
  if (state.assistantDrawerOpen !== false) throw new Error("assistant drawer should default closed");
  if (state.advancedPanelOpen !== false) throw new Error("advanced panel should default closed");
}

run();
console.log("workspace shell store contract passed");
```

**Step 2: Run the contract to verify it fails**

Run: `cd apps/web && npx tsx src/store/chatStore.workspaceShell.test.ts`  
Expected: FAIL

**Step 3: Implement the minimal state migration**

- remove `uiMode` and `setUiMode`
- add explicit shell state for:
  - assistant drawer visibility
  - advanced area visibility
  - active assistant section
- update `App.tsx` call sites

**Step 4: Re-run store verification**

Run: `cd apps/web && npx tsx src/store/chatStore.workspaceShell.test.ts`  
Expected: PASS

**Step 5: Commit**

```bash
git add apps/web/src/store/chatStore.ts apps/web/src/store/chatStore.workspaceShell.test.ts apps/web/src/App.tsx
git rm apps/web/src/store/chatStore.uiMode.test.ts
git commit -m "refactor: replace uiMode with explicit workspace shell state"
```

### Task 5: Extract a writing-first shell with a single floating assistant entry

**Files:**
- Modify: `apps/web/src/App.tsx`
- Modify: `apps/web/src/styles.css`
- Delete: `apps/web/src/components/ModeSwitch.tsx`
- Delete: `apps/web/src/components/ProWorkspaceMode.tsx`
- Delete: `apps/web/src/components/WritingWorkspaceMode.tsx`

**Step 1: Write the failing shell behavior test**

```ts
test("home keeps the editor primary and opens assistant from a floating button", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("button", { name: "写作助手" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "正文工作区" })).toBeVisible();
});
```

**Step 2: Run the shell test**

Run: `cd apps/web && npm run test:e2e -- tests/e2e/smoke.spec.ts --project=chromium --grep "home keeps the editor primary"`  
Expected: FAIL

**Step 3: Implement the new shell**

- remove the top-level mode switch
- keep editor as the permanent main surface
- add one floating assistant button on the right-middle
- open the existing assistant drawer from that button
- stop rendering separate writing/pro workspace wrappers

**Step 4: Re-run focused tests**

Run: `cd apps/web && npm run test:e2e -- tests/e2e/smoke.spec.ts --project=chromium`  
Expected: PASS or reduced failures to later tasks

**Step 5: Commit**

```bash
git add apps/web/src/App.tsx apps/web/src/styles.css
git rm apps/web/src/components/ModeSwitch.tsx apps/web/src/components/ProWorkspaceMode.tsx apps/web/src/components/WritingWorkspaceMode.tsx
git commit -m "refactor: replace dual-mode shell with writing-first workspace"
```

### Task 6: Move planning into the assistant drawer and remove rewrite from the drawer contract

**Files:**
- Modify: `apps/web/src/App.tsx`
- Modify: `apps/web/src/hooks/useAssistantSessionFlow.ts`
- Modify: `apps/web/src/components/StoryPlanningPanel.tsx`
- Modify: `apps/web/src/debugPanels.tsx`

**Step 1: Write the failing assistant behavior test**

```ts
test("assistant drawer defaults to planning instead of rewrite or debug", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "写作助手" }).click();
  await expect(page.getByRole("heading", { name: "结构化大纲与伏笔" })).toBeVisible();
});
```

**Step 2: Run the focused assistant test**

Run: `cd apps/web && npm run test:e2e -- tests/e2e/smoke.spec.ts --project=chromium --grep "assistant drawer defaults to planning"`  
Expected: FAIL

**Step 3: Implement the assistant contract**

- default drawer tab to planning
- keep assistant chat available as a secondary tab or section
- keep prompt/debug/evidence in advanced collapse, not in the default assistant landing
- remove any drawer-level rewrite affordance

**Step 4: Re-run the focused assistant test**

Run: `cd apps/web && npm run test:e2e -- tests/e2e/smoke.spec.ts --project=chromium --grep "assistant drawer defaults to planning"`  
Expected: PASS

**Step 5: Commit**

```bash
git add apps/web/src/App.tsx apps/web/src/hooks/useAssistantSessionFlow.ts apps/web/src/components/StoryPlanningPanel.tsx apps/web/src/debugPanels.tsx
git commit -m "refactor: make assistant drawer planning-first"
```

### Task 7: Keep rewrite only as an in-editor selection action

**Files:**
- Modify: `apps/web/src/components/DraftWorkspacePanel.tsx`
- Modify: `apps/web/src/App.tsx`
- Modify: `apps/web/tests/e2e/diff-ghost-interactions.spec.ts`

**Step 1: Write the failing rewrite contract**

```ts
test("rewrite actions are only available from text selection", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("重写")).toHaveCount(0);
  await expect(page.getByText("Ghost")).toHaveCount(0);
});
```

**Step 2: Run the focused test**

Run: `cd apps/web && npm run test:e2e -- tests/e2e/diff-ghost-interactions.spec.ts --project=chromium`  
Expected: FAIL

**Step 3: Implement the rewrite cleanup**

- keep selection context menu with:
  - `润色选中`
  - `扩写选中`
- remove `重写` affordances if any remain
- remove drawer/shell references that imply rewrite is a primary assistant capability

**Step 4: Re-run focused tests**

Run: `cd apps/web && npm run test:e2e -- tests/e2e/diff-ghost-interactions.spec.ts --project=chromium`  
Expected: PASS

**Step 5: Commit**

```bash
git add apps/web/src/components/DraftWorkspacePanel.tsx apps/web/src/App.tsx apps/web/tests/e2e/diff-ghost-interactions.spec.ts
git commit -m "refactor: keep rewrite as selection-only editor action"
```

### Task 8: Delete Ghost UI/state and retain only temporary backend compatibility if needed

**Files:**
- Modify: `apps/web/src/App.tsx`
- Modify: `apps/web/src/components/editor/extensions/GhostTextExtension.ts`
- Modify or Delete: `apps/api/app/api/endpoints/chat_ghost_text.py`
- Modify or Delete: `apps/api/tests/test_ghost_text_endpoint_unittest.py`
- Modify: `apps/api/app/schemas/chat.py`

**Step 1: Write the failing cleanup grep**

```bash
rg -n "Ghost|ghost_|temperature_profile=\\\"ghost\\\"|llm_temperature_ghost|ghost-text-widget|ghost-preview" apps/web apps/api
```

**Step 2: Run the grep to confirm remaining Ghost surface**

Run: `rg -n "Ghost|ghost_|temperature_profile=\\\"ghost\\\"|llm_temperature_ghost|ghost-text-widget|ghost-preview" apps/web apps/api`  
Expected: many matches

**Step 3: Remove Ghost as a product feature**

- delete UI widgets, socket flow, editor extension hookups, auto-trigger settings
- decide whether backend keeps a temporary deprecated rewrite shim
- if compatibility remains, rename/document it as internal/deprecated only
- remove Ghost terminology from user-facing schemas and UI labels

**Step 4: Re-run cleanup grep**

Run: `rg -n "Ghost|ghost-text-widget|ghost-preview" apps/web apps/api`  
Expected: no user-facing Ghost matches, only intentional deprecated compatibility markers if any

**Step 5: Commit**

```bash
git add apps/web apps/api
git commit -m "refactor: remove ghost as a user-facing feature"
```

### Task 9: Shrink author-facing settings and keep advanced fields behind server defaults

**Files:**
- Modify: `apps/web/src/App.tsx`
- Modify: `.env.example`
- Modify: `apps/api/README.md`
- Modify: `apps/api/app/schemas/chat.py`

**Step 1: Write the failing settings contract**

```ts
test("author settings only expose writing essentials by default", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "写作设置" }).click();
  await expect(page.getByText("基础写作")).toBeVisible();
  await expect(page.getByText("进阶设置")).toBeVisible();
});
```

**Step 2: Run the focused settings test**

Run: `cd apps/web && npm run test:e2e -- tests/e2e/smoke.spec.ts --project=chromium --grep "author settings only expose writing essentials by default"`  
Expected: FAIL or show too many advanced terms

**Step 3: Implement the settings shrink**

- hide `rag_mode`, `thinking_enabled`, `enable_tot`, `context_window_profile`, `budget_mode`, `reference_project_ids`, `model_profile_id` from default author settings
- keep them on the server and in advanced/internal flows if necessary
- rewrite `.env.example` and `apps/api/README.md` to distinguish author-level settings from deployment/runtime settings

**Step 4: Re-run focused verification**

Run: `cd apps/web && npm run test:e2e -- tests/e2e/smoke.spec.ts --project=chromium --grep "author settings only expose writing essentials by default"`  
Expected: PASS

**Step 5: Commit**

```bash
git add apps/web/src/App.tsx .env.example apps/api/README.md apps/api/app/schemas/chat.py
git commit -m "refactor: shrink author-facing settings and keep advanced defaults server-side"
```

### Task 10: Remove leftovers and run full verification

**Files:**
- Modify: `apps/web/src/App.tsx`
- Modify: `apps/web/src/styles.css`
- Modify: `apps/web/tests/e2e/*.spec.ts`
- Modify: `docs/author-manual.md`
- Modify: `README.md`

**Step 1: Run leftover scans**

```bash
rg -n "uiMode|ModeSwitch|ProWorkspaceMode|WritingWorkspaceMode|工作台模式|写作模式|Ghost Text|ghost-text-widget|灵感工作台" apps/web apps/api docs
```

**Step 2: Remove leftovers**

- delete obsolete imports, styles, labels, aria names, docs wording, and stale comments

**Step 3: Run final verification**

Run: `rg -n "uiMode|ModeSwitch|ProWorkspaceMode|WritingWorkspaceMode|工作台模式|写作模式|Ghost Text|ghost-text-widget|灵感工作台" apps/web apps/api docs`  
Expected: no matches except explicitly documented deprecated compatibility notes

Run: `cd apps/web && npm run build`  
Expected: PASS

Run: `cd apps/web && npm run test:e2e`  
Expected: PASS

Run: `python -m unittest discover -s apps/api/tests -p "test_*_unittest.py"`  
Expected: PASS

**Step 4: Commit**

```bash
git add apps/web apps/api docs README.md .env.example
git commit -m "refactor: complete product simplification and remove legacy workspace concepts"
```
