import { expect, test, type Locator, type Page, type Route } from "@playwright/test";

async function fulfillJson(route: Route, payload: unknown, status = 200): Promise<void> {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(payload),
  });
}

async function installMockChatApi(page: Page): Promise<void> {
  const projectId = 1;
  const chapter = {
    id: 101,
    project_id: projectId,
    volume_id: null,
    chapter_index: 1,
    title: "Mock Chapter",
    content: "Server draft content",
    version: 3,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };

  await page.route("**/api/chat/**", async (route) => {
    const request = route.request();
    const method = request.method().toUpperCase();
    const url = new URL(request.url());
    const path = url.pathname;

    if (
      (path === "/api/chat/rewrite/polish" || path === "/api/chat/rewrite/expand") &&
      method === "POST"
    ) {
      const parsedBody = JSON.parse(request.postData() ?? "{}") as { text?: string };
      const mode = path.endsWith("/expand") ? "expand" : "polish";
      const base = String(parsedBody.text ?? "").slice(0, 20) || "片段";
      await fulfillJson(route, {
        suggestion:
          mode === "expand"
            ? `${base}，扩写补充了一层细节与动作。`
            : `${base}，语句更凝练，情绪更清晰。`,
        usage: { provider: "mock", ghost_mode: mode },
      });
      return;
    }

    if (path.match(/^\/api\/chat\/projects\/\d+\/chapters\/?$/) && method === "GET") {
      await fulfillJson(route, [chapter]);
      return;
    }
    if (path.match(/^\/api\/chat\/projects\/\d+\/chapters\/\d+\/?$/) && method === "GET") {
      await fulfillJson(route, chapter);
      return;
    }
    if (path.match(/^\/api\/chat\/projects\/\d+\/chapters\/\d+\/?$/) && method === "PUT") {
      const parsedBody = JSON.parse(request.postData() ?? "{}") as { title?: string; content?: string; volume_id?: number | null };
      chapter.title = String(parsedBody.title ?? chapter.title);
      chapter.content = String(parsedBody.content ?? chapter.content);
      chapter.volume_id = parsedBody.volume_id ?? chapter.volume_id ?? null;
      chapter.version = Number(chapter.version ?? 0) + 1;
      chapter.updated_at = new Date().toISOString();
      await fulfillJson(route, chapter);
      return;
    }
    if (path.match(/^\/api\/chat\/projects\/\d+\/chapters\/\d+\/scene-beats\/?$/) && method === "GET") {
      await fulfillJson(route, []);
      return;
    }
    if (path.match(/^\/api\/chat\/projects\/\d+\/chapters\/\d+\/revisions\/?$/) && method === "GET") {
      await fulfillJson(route, []);
      return;
    }
    if (path.match(/^\/api\/chat\/projects\/\d+\/chapters\/?$/) && method === "POST") {
      await fulfillJson(route, chapter);
      return;
    }
    if (path === `/api/chat/projects/${projectId}/settings` && method === "GET") {
      await fulfillJson(route, []);
      return;
    }
    if (path === `/api/chat/projects/${projectId}/model-profiles` && method === "GET") {
      await fulfillJson(route, []);
      return;
    }
    if (path === `/api/chat/projects/${projectId}/cards` && method === "GET") {
      await fulfillJson(route, []);
      return;
    }
    if (path === `/api/chat/projects/${projectId}/prompt-templates` && method === "GET") {
      await fulfillJson(route, []);
      return;
    }
    if (path === `/api/chat/projects/${projectId}/volumes` && method === "GET") {
      await fulfillJson(route, []);
      return;
    }
    if (path === `/api/chat/projects/${projectId}/foreshadowing-cards` && method === "GET") {
      await fulfillJson(route, []);
      return;
    }
    if (path === `/api/chat/projects/${projectId}/consistency-audits` && method === "GET") {
      await fulfillJson(route, []);
      return;
    }
    if (path === `/api/chat/projects/${projectId}/graph-timeline` && method === "GET") {
      await fulfillJson(route, {
        project_id: projectId,
        chapter_index: 0,
        nodes: [],
        edges: [],
        stats: {},
      });
      return;
    }
    if (path === `/api/chat/projects/${projectId}/context-pack/preheat` && method === "POST") {
      await fulfillJson(route, {
        project_id: projectId,
        settings_count: 0,
        cards_count: 0,
        ttl_seconds: 30,
      });
      return;
    }
    if (path.startsWith("/api/chat/")) {
      await fulfillJson(route, []);
      return;
    }

    await route.fallback();
  });
}

async function ensureEditor(page: Page): Promise<Locator> {
  await page.goto("/");

  const emptyState = page.locator(".draft-empty-state");
  const editorShell = page.locator(".draft-editor-shell");

  await page.waitForFunction(() => {
    return Boolean(document.querySelector(".draft-empty-state") || document.querySelector(".draft-editor-shell"));
  });

  if (await emptyState.isVisible()) {
    const createChapterButton = page.getByRole("button", { name: "点击新建章节开始写作" });
    await createChapterButton.click();
    await expect(emptyState).toHaveCount(0);
  }

  const chapterOutlineItem = page.locator(".chapter-outline-item").first();
  if ((await chapterOutlineItem.count()) > 0) {
    await chapterOutlineItem.click();
  }

  await expect(editorShell).toBeVisible();
  const editorSurface = page.locator(".draft-editor .ProseMirror");
  await expect(page.locator(".draft-editor")).not.toHaveClass(/disabled/, { timeout: 15_000 });
  await expect(editorSurface).toBeVisible();
  await expect(editorSurface).toHaveAttribute("contenteditable", "true", { timeout: 15_000 });
  await editorSurface.click();
  return editorSurface;
}

test.describe("diff regression", () => {
  test("selection rewrite renders diff widget and cherry-pick applies suggestion", async ({ page }) => {
    await installMockChatApi(page);
    const editorSurface = await ensureEditor(page);

    await editorSurface.click();
    await page.keyboard.type("选中测试内容");
    await page.keyboard.press("Control+a");
    await expect(page.locator(".draft-hint")).toContainText("已选", { timeout: 10_000 });
    await expect(page.locator(".draft-hint")).not.toContainText("已选 0 字");
    await page.locator(".draft-editor-context").dispatchEvent("contextmenu", {
      bubbles: true,
      cancelable: true,
      clientX: 200,
      clientY: 200,
      button: 2,
    });

    const menu = page.locator(".selection-context-menu");
    await expect(menu).toBeVisible();
    await menu.locator("button").first().click({ force: true });

    const widget = page.locator(".diff-suggestion-widget");
    await expect(widget).toContainText("语句更凝练", { timeout: 15_000 });
    await widget.getByRole("button", { name: "接受" }).click();

    await expect(widget).toHaveCount(0);
    await expect(editorSurface).toContainText("语句更凝练");
  });
});
