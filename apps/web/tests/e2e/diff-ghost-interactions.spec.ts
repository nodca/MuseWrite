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
  await expect(page.getByRole("heading", { name: "正文工作区" })).toBeVisible({ timeout: 30_000 });
  const createChapterButton = page.getByRole("button", { name: "点击新建章节开始写作" });
  if (await createChapterButton.isVisible()) {
    await createChapterButton.click();
  }
  const editorSurface = page.locator(".draft-editor .ProseMirror");
  await expect(editorSurface).toBeVisible({ timeout: 30_000 });
  await expect(editorSurface).toHaveAttribute("contenteditable", "true", { timeout: 30_000 });
}

test.describe("diff/ghost interactions", () => {
  test("diff suggestions support accept & ignore cherry-pick", async ({ page }) => {
    const assistantReply = "AI 建议替换：diff-accept";
    await installMockChatApi(page, { assistantText: assistantReply });
    await page.goto("/");

    await ensureChapterReady(page);

    const editorSurface = page.locator(".draft-editor .ProseMirror");
    await editorSurface.click();
    await page.keyboard.type("旧文本：alpha beta");

    await page.getByRole("button", { name: "助手抽屉" }).click();
    const drawer = page.locator("#assistant-drawer");
    await expect(drawer).toHaveAttribute("aria-hidden", "false");
    await drawer.locator(".composer textarea").fill("diff e2e");
    await drawer.locator(".composer button").click();
    await expect(drawer.locator(".chat-log article.msg.assistant pre")).toContainText(assistantReply);
    await drawer.getByRole("button", { name: "关闭助手抽屉" }).click();

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

  test("ghost text accepts full suggestion via Tab and one word via Ctrl+ArrowRight", async ({ page }) => {
    await page.addInitScript(() => {
      const listenersSymbol = Symbol("ws_listeners");
      let sendCount = 0;

      function emit(target: any, type: string, event: any) {
        const map = target[listenersSymbol] as Map<string, Set<(event: any) => void>> | undefined;
        if (!map) return;
        const handlers = map.get(type);
        if (!handlers) return;
        handlers.forEach((cb) => {
          try {
            cb(event);
          } catch {
            // ignore handler failures in test shim
          }
        });
      }

      class MockWebSocket {
        static CONNECTING = 0;
        static OPEN = 1;
        static CLOSING = 2;
        static CLOSED = 3;

        url: string;
        readyState = MockWebSocket.CONNECTING;
        [listenersSymbol] = new Map<string, Set<(event: any) => void>>();

        constructor(url: string) {
          this.url = url;
          queueMicrotask(() => {
            this.readyState = MockWebSocket.OPEN;
            emit(this, "open", {});
          });
        }

        addEventListener(type: string, cb: (event: any) => void) {
          const map = this[listenersSymbol];
          const handlers = map.get(type) ?? new Set();
          handlers.add(cb);
          map.set(type, handlers);
        }

        removeEventListener(type: string, cb: (event: any) => void) {
          const map = this[listenersSymbol];
          map.get(type)?.delete(cb);
        }

        close() {
          if (this.readyState === MockWebSocket.CLOSED) return;
          this.readyState = MockWebSocket.CLOSED;
          emit(this, "close", {});
        }

        send(raw: string) {
          if (this.readyState !== MockWebSocket.OPEN) return;
          try {
            JSON.parse(String(raw ?? "{}"));
          } catch {
            // ignore invalid payloads
          }
          sendCount += 1;
          const suggestion = sendCount === 1 ? "spectral quill" : "gamma delta";

          const emitMessage = (payload: any) => {
            emit(this, "message", { data: JSON.stringify(payload) });
          };

          queueMicrotask(() => emitMessage({ type: "start", text: "" }));
          queueMicrotask(() => emitMessage({ type: "delta", text: suggestion.split(" ")[0] + " " }));
          queueMicrotask(() => emitMessage({ type: "delta", text: suggestion.split(" ").slice(1).join(" ") }));
          queueMicrotask(() => emitMessage({ type: "done", text: suggestion }));
        }
      }

      (window as any).WebSocket = MockWebSocket;
    });

    await installMockChatApi(page);
    await page.goto("/");

    await page.getByRole("button", { name: "写作设置" }).click();
    const settingsDialog = page.locator("#settings-dialog");
    await expect(settingsDialog).toBeVisible();
    await settingsDialog.locator("details.settings-section", { hasText: "高级调优" }).locator("summary").click();
    await settingsDialog.getByLabel("Ghost 自动建议").selectOption("on");
    await settingsDialog.getByRole("button", { name: "关闭写作设置" }).click();
    await expect(settingsDialog).toBeHidden();

    await ensureChapterReady(page);

    const editorSurface = page.locator(".draft-editor .ProseMirror");
    await editorSurface.click();
    await page.keyboard.type("prefix ");

    const ghostWidget = page.locator(".ghost-text-widget");
    await expect(ghostWidget).toContainText("spectral quill", { timeout: 15_000 });

    await editorSurface.focus();
    await page.keyboard.press("Tab");
    await expect(ghostWidget).toHaveCount(0);
    await expect(editorSurface).toContainText("spectral quill");

    await expect(page.locator(".ghost-text-widget")).toContainText("gamma delta", { timeout: 15_000 });
    const saveDraftButton = page.locator(".draft-actions button.btn.primary.tiny").first();
    await expect(saveDraftButton).toHaveText("保存正文", { timeout: 15_000 });
    await editorSurface.focus();
    await page.keyboard.down("Control");
    await page.keyboard.press("ArrowRight");
    await page.keyboard.up("Control");
    await expect(page.locator(".ghost-text-widget")).toHaveText("delta");
    await expect(editorSurface).toContainText("gamma");
  });
});
