# Single Workspace Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current writing/pro mode split with one always-on creative workspace whose advanced tools expand progressively instead of switching the whole page shell.

**Architecture:** Keep the draft editor as the permanent center of the page, move planning/actions/prompt/debug capabilities into a collapsible inspector rail, and replace `uiMode` with explicit shell state such as inspector visibility and pinned sections. Preserve the existing chat, draft, and action flows; this refactor is about layout, state semantics, and public configuration language, not backend protocol changes.

**Tech Stack:** React 18, TypeScript, Zustand, Playwright, Node 24, Vite

---

### Task 1: Lock the new shell contract in Playwright

**Files:**
- Modify: `apps/web/tests/e2e/smoke.spec.ts`
- Modify: `apps/web/src/App.tsx`

**Step 1: Write the failing shell regression test**

```ts
test("single workspace shell opens inspector without mode switching", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("button", { name: "展开进阶面板" })).toBeVisible();
  await expect(page.getByTestId("ui-mode-toggle")).toHaveCount(0);
  await expect(page.getByTestId("workspace-inspector")).toBeHidden();

  await page.getByRole("button", { name: "展开进阶面板" }).click();
  await expect(page.getByTestId("workspace-inspector")).toBeVisible();
});
```

**Step 2: Run test to verify it fails**

Run: `cd apps/web && npm run test:e2e -- tests/e2e/smoke.spec.ts --project=chromium --grep "single workspace shell opens inspector without mode switching"`  
Expected: FAIL because the page still renders the mode toggle and has no unified inspector shell

**Step 3: Write the minimal shell chrome change**

```tsx
<button
  type="button"
  className="btn ghost"
  onClick={onToggleWorkspaceInspector}
  data-testid="workspace-inspector-toggle"
>
  {workspaceInspectorOpen ? "收起进阶面板" : "展开进阶面板"}
</button>
```

Also remove the `ModeSwitch` render from the top bar and mount an empty-but-real inspector container with `data-testid="workspace-inspector"`.

**Step 4: Run test to verify it passes**

Run: `cd apps/web && npm run test:e2e -- tests/e2e/smoke.spec.ts --project=chromium --grep "single workspace shell opens inspector without mode switching"`  
Expected: PASS

**Step 5: Commit**

```bash
git add apps/web/tests/e2e/smoke.spec.ts apps/web/src/App.tsx
git commit -m "feat: replace mode switch with single workspace shell chrome"
```

---

### Task 2: Replace `uiMode` with explicit workspace shell state

**Files:**
- Modify: `apps/web/src/store/chatStore.ts`
- Create: `apps/web/src/store/chatStore.workspaceShell.test.ts`
- Delete: `apps/web/src/store/chatStore.uiMode.test.ts`
- Modify: `apps/web/src/App.tsx`

**Step 1: Write the failing store contract test**

```ts
import { useChatStore } from "./chatStore";

function run() {
  const state = useChatStore.getState();
  if (state.workspaceInspectorOpen !== false) throw new Error("inspector should default closed");

  state.setWorkspaceInspectorOpen(true);
  if (useChatStore.getState().workspaceInspectorOpen !== true) throw new Error("inspector did not open");
}

run();
console.log("chatStore workspace shell test passed");
```

**Step 2: Run test to verify it fails**

Run: `cd apps/web && npx tsx src/store/chatStore.workspaceShell.test.ts`  
Expected: FAIL because `workspaceInspectorOpen` and its setter do not exist yet

**Step 3: Write the minimal state migration**

```ts
interface ChatStoreState {
  workspaceInspectorOpen: boolean;
  workspaceSections: {
    planning: boolean;
    actions: boolean;
    prompt: boolean;
    context: boolean;
  };
  setWorkspaceInspectorOpen: (open: boolean) => void;
  toggleWorkspaceSection: (key: keyof ChatStoreState["workspaceSections"]) => void;
}
```

Remove `uiMode` and `setUiMode`, then update `App.tsx` call sites to consume the new shell state instead of mode state.

**Step 4: Run focused verification**

Run: `cd apps/web && npx tsx src/store/chatStore.workspaceShell.test.ts`  
Expected: PASS

Run: `cd apps/web && npm run build`  
Expected: PASS

**Step 5: Commit**

```bash
git add apps/web/src/store/chatStore.ts apps/web/src/store/chatStore.workspaceShell.test.ts apps/web/src/App.tsx
git commit -m "refactor: replace uiMode with workspace shell state"
```

---

### Task 3: Extract a unified creative workspace shell

**Files:**
- Create: `apps/web/src/components/CreativeWorkspaceShell.tsx`
- Modify: `apps/web/src/components/DraftWorkspacePanel.tsx`
- Modify: `apps/web/src/App.tsx`
- Modify: `apps/web/src/styles.css`

**Step 1: Write the failing behavior regression**

```ts
test("draft editor stays primary when inspector is opened", async ({ page }) => {
  await page.goto("/");

  const draftHeading = page.getByRole("heading", { name: "从第一章开始" });
  await expect(draftHeading).toBeVisible();

  await page.getByRole("button", { name: "展开进阶面板" }).click();
  await expect(draftHeading).toBeVisible();
  await expect(page.getByTestId("workspace-inspector")).toBeVisible();
});
```

**Step 2: Run test to verify it fails**

Run: `cd apps/web && npm run test:e2e -- tests/e2e/smoke.spec.ts --project=chromium --grep "draft editor stays primary when inspector is opened"`  
Expected: FAIL because inspector opening still depends on the old mode branch

**Step 3: Write the minimal shell extraction**

```tsx
export function CreativeWorkspaceShell({
  draftPane,
  inspector,
  inspectorOpen,
}: CreativeWorkspaceShellProps) {
  return (
    <main className={`creative-workspace ${inspectorOpen ? "inspector-open" : ""}`}>
      <section className="creative-main">{draftPane}</section>
      <aside className="creative-inspector" data-testid="workspace-inspector">
        {inspector}
      </aside>
    </main>
  );
}
```

Update `DraftWorkspacePanel` so it no longer receives `uiMode`; it should only care about draft concerns (`zenMode`, focus mode, editor state, chapter state).

**Step 4: Run focused verification**

Run: `cd apps/web && npm run test:e2e -- tests/e2e/smoke.spec.ts --project=chromium --grep "draft editor stays primary when inspector is opened"`  
Expected: PASS

Run: `cd apps/web && npm run build`  
Expected: PASS

**Step 5: Commit**

```bash
git add apps/web/src/components/CreativeWorkspaceShell.tsx apps/web/src/components/DraftWorkspacePanel.tsx apps/web/src/App.tsx apps/web/src/styles.css
git commit -m "refactor: extract unified creative workspace shell"
```

---

### Task 4: Move workbench capabilities into inspector sections and delete old mode wrappers

**Files:**
- Modify: `apps/web/src/App.tsx`
- Modify: `apps/web/src/styles.css`
- Delete: `apps/web/src/components/ModeSwitch.tsx`
- Delete: `apps/web/src/components/ProWorkspaceMode.tsx`
- Delete: `apps/web/src/components/WritingWorkspaceMode.tsx`
- Modify: `apps/web/tests/e2e/smoke.spec.ts`

**Step 1: Write the failing inspector-section test**

```ts
test("inspector sections expand independently", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "展开进阶面板" }).click();

  await page.getByRole("button", { name: "结构化规划" }).click();
  await expect(page.getByRole("heading", { name: "结构化大纲与伏笔" })).toBeVisible();

  await page.getByRole("button", { name: "动作提议" }).click();
  await expect(page.getByText("一致性报告")).toBeVisible();
});
```

**Step 2: Run test to verify it fails**

Run: `cd apps/web && npm run test:e2e -- tests/e2e/smoke.spec.ts --project=chromium --grep "inspector sections expand independently"`  
Expected: FAIL because planning/actions panels still live behind the old wrapper components

**Step 3: Write the minimal inspector section stack**

```tsx
const INSPECTOR_SECTIONS = [
  { key: "planning", label: "结构化规划" },
  { key: "actions", label: "动作提议" },
  { key: "prompt", label: "Prompt 与知识" },
  { key: "context", label: "Context 与检视" },
];
```

Render each section inside the unified inspector rail, wired to the new `workspaceSections` state. Remove the lazy `ProWorkspaceMode` / `WritingWorkspaceMode` branches and delete the now-unused wrapper files.

**Step 4: Run focused verification**

Run: `cd apps/web && npm run test:e2e -- tests/e2e/smoke.spec.ts --project=chromium --grep "inspector sections expand independently"`  
Expected: PASS

Run: `cd apps/web && npm run build`  
Expected: PASS

**Step 5: Commit**

```bash
git add apps/web/src/App.tsx apps/web/src/styles.css apps/web/tests/e2e/smoke.spec.ts
git rm apps/web/src/components/ModeSwitch.tsx apps/web/src/components/ProWorkspaceMode.tsx apps/web/src/components/WritingWorkspaceMode.tsx
git commit -m "refactor: move workbench panels into unified inspector rail"
```

---

### Task 5: Simplify settings and public configuration language

**Files:**
- Modify: `apps/web/src/App.tsx`
- Modify: `README.md`
- Modify: `docs/author-manual.md`
- Modify: `.env.example`

**Step 1: Write the failing shell/settings regression**

```ts
test("settings favor author basics and hide deploy-level terms", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "写作设置" }).click();

  await expect(page.getByText("基础写作（推荐）")).toBeVisible();
  await expect(page.getByText("进阶设置")).toBeVisible();
  await expect(page.getByText("工作台模式")).toHaveCount(0);
});
```

**Step 2: Run test to verify it fails**

Run: `cd apps/web && npm run test:e2e -- tests/e2e/smoke.spec.ts --project=chromium --grep "settings favor author basics and hide deploy-level terms"`  
Expected: FAIL because the current UI/docs still describe workbench mode and expose the old three-layer wording

**Step 3: Write the minimal settings/doc rewrite**

```tsx
<details className="settings-section" open>
  <summary>
    <strong>基础写作</strong>
    <small>默认只看这一组即可开始写作</small>
  </summary>
</details>

<details className="settings-section">
  <summary>
    <strong>进阶设置</strong>
    <small>模型、上下文与实验性辅助能力</small>
  </summary>
</details>
```

Update `README.md`, `docs/author-manual.md`, and `.env.example` so public guidance describes:

- one creative workspace
- progressive inspector sections
- `CONFIG_PROFILE` as the primary deployment abstraction

Keep deployment-only knobs documented tersely and out of the author path.

**Step 4: Run focused verification**

Run: `cd apps/web && npm run test:e2e -- tests/e2e/smoke.spec.ts --project=chromium --grep "settings favor author basics and hide deploy-level terms"`  
Expected: PASS

Run: `rg -n "工作台模式|写作模式" README.md docs/author-manual.md .env.example`  
Expected: no matches

**Step 5: Commit**

```bash
git add apps/web/src/App.tsx README.md docs/author-manual.md .env.example
git commit -m "docs: simplify single-workspace settings and guidance"
```

---

### Task 6: Remove dead references and run end-to-end verification

**Files:**
- Modify: `apps/web/src/App.tsx`
- Modify: `apps/web/src/styles.css`
- Modify: `apps/web/tests/e2e/smoke.spec.ts`
- Modify: `docs/author-manual.md`

**Step 1: Write the failing cleanup checks**

```bash
rg -n "uiMode|ModeSwitch|ProWorkspaceMode|WritingWorkspaceMode|mode-writing|mode-pro" apps/web/src apps/web/tests/e2e docs/author-manual.md
```

Expected before cleanup: matches remain

**Step 2: Remove the leftover references**

Delete dead imports, CSS selectors, stale aria labels, and obsolete wording that still refer to the old dual-mode shell.

**Step 3: Run final verification**

Run: `rg -n "uiMode|ModeSwitch|ProWorkspaceMode|WritingWorkspaceMode|mode-writing|mode-pro" apps/web/src apps/web/tests/e2e docs/author-manual.md`  
Expected: no matches

Run: `cd apps/web && npm run build`  
Expected: PASS

Run: `cd apps/web && npm run test:e2e -- tests/e2e/smoke.spec.ts --project=chromium`  
Expected: PASS

**Step 4: Commit**

```bash
git add apps/web/src/App.tsx apps/web/src/styles.css apps/web/tests/e2e/smoke.spec.ts docs/author-manual.md
git commit -m "refactor: remove dual-mode workspace leftovers"
```

---

Plan complete and saved to `docs/plans/2026-03-11-single-workspace-unification-implementation-plan.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
