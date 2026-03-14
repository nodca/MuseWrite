import { expect, test, type Page, type Route } from "@playwright/test";

type MockSessionMessage = {
  id: number;
  role: "user" | "assistant" | "system";
  content: string;
  created_at: string;
};

type MockSession = {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
  messages: MockSessionMessage[];
};

type MockChapter = {
  id: number;
  project_id: number;
  volume_id: number | null;
  chapter_index: number;
  title: string;
  content: string;
  version: number;
  created_at: string;
  updated_at: string;
};

type MockChatApiOptions = {
  assistantText?: string;
};

function nowIso(): string {
  return new Date().toISOString();
}

function toSse(event: Record<string, unknown>): string {
  return `data: ${JSON.stringify(event)}\n\n`;
}

async function fulfillJson(route: Route, payload: unknown, status = 200): Promise<void> {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(payload),
  });
}

async function installMockChatApi(page: Page, options: MockChatApiOptions = {}): Promise<void> {
  const projectId = 1;
  const sessions: MockSession[] = [];
  let nextSessionId = 1;
  let nextMessageId = 1;

  const chapters: MockChapter[] = [];
  let nextChapterId = 101;

  const summarizeSessions = () =>
    [...sessions]
      .sort((a, b) => Date.parse(b.updated_at) - Date.parse(a.updated_at))
      .map((session) => ({
        id: session.id,
        project_id: projectId,
        title: session.title,
        created_at: session.created_at,
        updated_at: session.updated_at,
      }));

  const findChapter = (chapterId: number) => chapters.find((item) => item.id === chapterId) ?? null;

  await page.route("**/api/chat/**", async (route) => {
    const request = route.request();
    const method = request.method().toUpperCase();
    const url = new URL(request.url());
    const path = url.pathname;

    if (path === "/api/chat/stream" && method === "POST") {
      const rawBody = request.postData() ?? "{}";
      const parsedBody = JSON.parse(rawBody) as { content?: string; session_id?: number | null };
      const content = String(parsedBody.content ?? "").trim();
      const requestedSessionId = Number(parsedBody.session_id ?? 0);

      let session =
        requestedSessionId > 0 ? sessions.find((item) => item.id === requestedSessionId) ?? null : null;
      if (!session) {
        const createdAt = nowIso();
        session = {
          id: nextSessionId++,
          title: `会话 #${nextSessionId - 1}`,
          created_at: createdAt,
          updated_at: createdAt,
          messages: [],
        };
        sessions.push(session);
      }

      const userMessage: MockSessionMessage = {
        id: nextMessageId++,
        role: "user",
        content,
        created_at: nowIso(),
      };
      const assistantText = options.assistantText ?? `mock-reply: ${content || "empty"}`;
      const assistantMessage: MockSessionMessage = {
        id: nextMessageId++,
        role: "assistant",
        content: assistantText,
        created_at: nowIso(),
      };
      session.messages.push(userMessage, assistantMessage);
      session.updated_at = nowIso();

      await route.fulfill({
        status: 200,
        contentType: "text/event-stream",
        headers: {
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
        body:
          toSse({
            type: "meta",
            session_id: session.id,
            assistant_message_id: assistantMessage.id,
            proposed_action_ids: [],
          }) +
          toSse({ type: "delta", text: assistantMessage.content }) +
          toSse({
            type: "done",
            assistant_message_id: assistantMessage.id,
            usage: { total_tokens: Math.max(1, assistantMessage.content.length) },
          }),
      });
      return;
    }

    if (path === `/api/chat/projects/${projectId}/sessions` && method === "GET") {
      await fulfillJson(route, summarizeSessions());
      return;
    }

    const sessionMessagesMatch = path.match(/^\/api\/chat\/sessions\/(\d+)\/messages$/);
    if (sessionMessagesMatch && method === "GET") {
      const sessionId = Number(sessionMessagesMatch[1]);
      const session = sessions.find((item) => item.id === sessionId);
      const messages =
        session?.messages.map((item) => ({
          id: item.id,
          session_id: sessionId,
          role: item.role,
          content: item.content,
          model: null,
          created_at: item.created_at,
        })) ?? [];
      await fulfillJson(route, messages);
      return;
    }

    if (path.match(/^\/api\/chat\/sessions\/\d+\/actions$/) && method === "GET") {
      await fulfillJson(route, []);
      return;
    }

    if (path === `/api/chat/projects/${projectId}/settings` && method === "GET") {
      await fulfillJson(route, []);
      return;
    }
    if (path === `/api/chat/projects/${projectId}/consistency-audits` && method === "GET") {
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

    if (path === `/api/chat/projects/${projectId}/context-pack/preheat` && method === "POST") {
      await fulfillJson(route, {
        project_id: projectId,
        settings_count: 0,
        cards_count: 0,
        ttl_seconds: 30,
      });
      return;
    }

    if (
      (path === "/api/chat/rewrite/polish" || path === "/api/chat/rewrite/expand") &&
      method === "POST"
    ) {
      const rawBody = request.postData() ?? "{}";
      const parsedBody = JSON.parse(rawBody) as {
        text?: string;
      };
      const source = String(parsedBody.text ?? "").trim() || "片段";
      const suggestion =
        path.endsWith("/expand")
          ? `${source}，扩写后补进了更多动作与氛围。`
          : `${source}，润色后语气更凝练。`;
      await fulfillJson(route, {
        suggestion,
        usage: { provider: "mock", mode: path.endsWith("/expand") ? "expand" : "polish" },
      });
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

    if (path === `/api/chat/projects/${projectId}/chapters` && method === "GET") {
      await fulfillJson(route, [...chapters]);
      return;
    }
    if (path === `/api/chat/projects/${projectId}/chapters` && method === "POST") {
      const createdAt = nowIso();
      const chapter: MockChapter = {
        id: nextChapterId++,
        project_id: projectId,
        volume_id: null,
        chapter_index: chapters.length + 1,
        title: `第${chapters.length + 1}章`,
        content: "",
        version: 1,
        created_at: createdAt,
        updated_at: createdAt,
      };
      chapters.push(chapter);
      await fulfillJson(route, chapter);
      return;
    }

    const chapterMatch = path.match(/^\/api\/chat\/projects\/(\d+)\/chapters\/(\d+)$/);
    if (chapterMatch && method === "GET") {
      const chapterId = Number(chapterMatch[2]);
      const chapter = findChapter(chapterId);
      await fulfillJson(route, chapter ?? { detail: "chapter_not_found" }, chapter ? 200 : 404);
      return;
    }
    if (chapterMatch && method === "PUT") {
      const chapterId = Number(chapterMatch[2]);
      const chapter = findChapter(chapterId);
      if (!chapter) {
        await fulfillJson(route, { detail: "chapter_not_found" }, 404);
        return;
      }
      const rawBody = request.postData() ?? "{}";
      const parsedBody = JSON.parse(rawBody) as { title?: string; content?: string };
      chapter.title = String(parsedBody.title ?? chapter.title);
      chapter.content = String(parsedBody.content ?? chapter.content);
      chapter.version = Math.max(1, chapter.version + 1);
      chapter.updated_at = nowIso();
      await fulfillJson(route, chapter);
      return;
    }

    const revisionsMatch = path.match(/^\/api\/chat\/projects\/\d+\/chapters\/\d+\/revisions$/);
    if (revisionsMatch && method === "GET") {
      await fulfillJson(route, []);
      return;
    }

    const beatsMatch = path.match(/^\/api\/chat\/projects\/\d+\/chapters\/\d+\/scene-beats$/);
    if (beatsMatch && method === "GET") {
      await fulfillJson(route, []);
      return;
    }

    if (path.startsWith("/api/chat/")) {
      await fulfillJson(route, {});
      return;
    }

    await route.fallback();
  });
}

async function ensureChapterReady(page: Page): Promise<void> {
  await expect(page.locator(".page-shell")).toBeVisible({ timeout: 30_000 });
  const createChapterButton = page.getByRole("button", { name: "点击新建章节开始写作" });
  if (await createChapterButton.isVisible()) {
    await createChapterButton.click();
  }
  const editorSurface = page.locator(".draft-editor .ProseMirror");
  await expect(editorSurface).toBeVisible({ timeout: 30_000 });
  await expect(editorSurface).toHaveAttribute("contenteditable", "true", { timeout: 30_000 });
}

test.describe("editor rewrite interactions", () => {
  test("diff suggestions support accept & ignore cherry-pick", async ({ page }) => {
    const assistantReply = "AI 建议替换：diff-accept";
    await installMockChatApi(page, { assistantText: assistantReply });
    await page.goto("/");

    await ensureChapterReady(page);

    const editorSurface = page.locator(".draft-editor .ProseMirror");
    await editorSurface.click();
    await page.keyboard.type("旧文本：alpha beta");

    await page.getByRole("button", { name: "写作助手" }).click();
    const drawer = page.locator("#assistant-drawer");
    await expect(drawer).toBeVisible();
    await drawer.getByRole("tab", { name: "对话" }).click();
    await expect(drawer.locator(".composer textarea")).toBeVisible();
    await drawer.locator(".composer textarea").fill("diff e2e");
    await drawer.locator(".composer button").click();
    await expect(drawer.locator(".chat-log article.msg.assistant pre")).toContainText(assistantReply);
    await drawer.getByRole("button", { name: "关闭写作助手" }).click();

    await editorSurface.click();
    await page.keyboard.press("Control+a");

    await page.getByRole("button", { name: "替换选中为助手回复" }).click();
    const diffWidget = page.locator(".diff-suggestion-widget");
    await expect(diffWidget).toBeVisible();

    await diffWidget.getByRole("button", { name: "接受" }).click();
    await expect(diffWidget).toHaveCount(0);
    await expect(editorSurface).toContainText("diff-accept");
    await expect(editorSurface).not.toContainText("旧文本：alpha beta");

    await editorSurface.click();
    await page.keyboard.press("Control+a");
    await page.keyboard.type("旧文本：reject-path");
    await page.keyboard.press("Control+a");

    await page.getByRole("button", { name: "替换选中为助手回复" }).click();
    const diffWidgetReject = page.locator(".diff-suggestion-widget");
    await expect(diffWidgetReject).toBeVisible();

    await diffWidgetReject.getByRole("button", { name: "忽略" }).click();
    await expect(diffWidgetReject).toHaveCount(0);
    await expect(editorSurface).toContainText("旧文本：reject-path");
  });

  test("selection rewrite opens diff feedback after polish action", async ({ page }) => {
    await installMockChatApi(page);
    await page.goto("/");

    await ensureChapterReady(page);

    const editorSurface = page.locator(".draft-editor .ProseMirror");
    await editorSurface.click();
    await page.keyboard.type("待润色片段");
    await page.keyboard.press("Control+a");
    await expect(page.locator(".draft-hint")).toContainText("已选", { timeout: 10_000 });

    await page.locator(".draft-editor-context").dispatchEvent("contextmenu", {
      bubbles: true,
      cancelable: true,
      clientX: 240,
      clientY: 240,
      button: 2,
    });
    const menu = page.locator(".selection-context-menu");
    await expect(menu).toBeVisible();
    await expect(menu.getByRole("menuitem", { name: /润色选中/ })).toBeVisible();
    await expect(menu.getByRole("menuitem", { name: /扩写选中/ })).toBeVisible();
    await menu.getByRole("menuitem", { name: /润色选中/ }).click();

    const diffWidget = page.locator(".diff-suggestion-widget");
    await expect(diffWidget).toBeVisible();
    await expect(diffWidget.getByRole("button", { name: "接受" })).toBeVisible();
    await expect(diffWidget.getByRole("button", { name: "忽略" })).toBeVisible();
  });
});

