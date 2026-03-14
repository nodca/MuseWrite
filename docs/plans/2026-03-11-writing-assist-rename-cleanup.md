# Writing Assist Rename Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the remaining `ghost` compatibility layer and converge the project on a single writing-assist vocabulary.

**Architecture:** Keep the writing-first product shape unchanged, but simplify the internal contract so backend routes, frontend state, and tests all use `rewrite` and `inline suggestion` semantics only. Preserve unrelated visual class names where `ghost` means button style or graph preview tone rather than writing-assist behavior.

**Tech Stack:** FastAPI, TypeScript, React, Playwright, Python unittest

---

### Task 1: Clean backend writing-assist contract

**Files:**
- Modify: `apps/api/app/api/endpoints/chat_writing_assist.py`
- Modify: `apps/api/app/api/endpoints/chat_helpers.py`
- Modify: `apps/api/app/services/llm_provider.py`
- Modify: `apps/api/app/services/context_compiler.py`

**Step 1: Remove compatibility-only aliases**

- Delete fallback handling for `ghost` profile names and old helper aliases.

**Step 2: Normalize naming**

- Use `suggestion` / `inline suggestion` wording in prompts, comments, and variable names where it refers to the writing feature.

**Step 3: Keep runtime stable**

- Preserve existing behavior for default temperatures and context compilation, but expose only the new naming path.

### Task 2: Clean frontend wording and dead UI remnants

**Files:**
- Modify: `apps/web/src/App.tsx`
- Modify: `apps/web/src/components/editor/extensions/InlineSuggestionExtension.ts`
- Modify: `apps/web/src/styles.css`

**Step 1: Fix remaining writing-assist labels**

- Replace user-facing `Ghost` wording with clearer planning / suggestion wording.

**Step 2: Verify rename integrity**

- Check bulk-renamed state and command names in `App.tsx` for accidental bad replacements.

**Step 3: Remove dead styles**

- Delete the old removed-panel CSS that no longer has any live markup.

### Task 3: Update tests and verify

**Files:**
- Modify: `apps/api/tests/test_chat_rewrite_endpoint_unittest.py`
- Modify: `apps/api/tests/test_writing_flow_unittest.py`
- Modify: `apps/web/tests/e2e/smoke.spec.ts`
- Modify: `apps/web/tests/e2e/diff-ghost-interactions.spec.ts`

**Step 1: Update expectations**

- Align endpoint paths, temperature profile assertions, and wording checks.

**Step 2: Run targeted verification**

- Run focused API tests for rewrite and writing flow behavior.
- Run focused Playwright tests for smoke and selection rewrite interactions.
