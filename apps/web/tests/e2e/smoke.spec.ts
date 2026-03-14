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

async function installMockChatApi(page: Page, assistantText: string): Promise<void> {
  const projectId = 1;
  const chapterId = 101;
  const sessions: MockSession[] = [];
  let nextSessionId = 1;
  let nextMessageId = 100;

  const summarizeSessions = () =>
    [...sessions]
      .sort((left, right) => Date.parse(right.updated_at) - Date.parse(left.updated_at))
      .map((session) => ({
        id: session.id,
        project_id: projectId,
        title: session.title,
        created_at: session.created_at,
        updated_at: session.updated_at,
      }));

  await page.route("**/api/chat/**", async (route) => {
    const request = route.request();
    const method = request.method().toUpperCase();
    const url = new URL(request.url());
    const path = url.pathname;

    if (path === "/api/chat/stream" && method === "POST") {
      const parsedBody = JSON.parse(request.postData() ?? "{}") as {
        content?: string;
        session_id?: number | null;
      };
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

    const messagesMatch = path.match(/^\/api\/chat\/sessions\/(\d+)\/messages$/);
    if (messagesMatch && method === "GET") {
      const sessionId = Number(messagesMatch[1]);
      const session = sessions.find((item) => item.id === sessionId);
      await fulfillJson(
        route,
        session?.messages.map((item) => ({
          id: item.id,
          session_id: sessionId,
          role: item.role,
          content: item.content,
          model: null,
          created_at: item.created_at,
        })) ?? []
      );
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
    if (path === `/api/chat/projects/${projectId}/cards` && method === "GET") {
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
        chapter_index: 1,
        nodes: [],
        edges: [],
        stats: {},
      });
      return;
    }
    if (path === `/api/chat/projects/${projectId}/chapters` && method === "GET") {
      await fulfillJson(route, [
        {
          id: chapterId,
          project_id: projectId,
          volume_id: null,
          chapter_index: 1,
          title: "第一章",
          content: "Smoke draft content",
          version: 1,
          created_at: nowIso(),
          updated_at: nowIso(),
        },
      ]);
      return;
    }
    if (path === `/api/chat/projects/${projectId}/chapters/${chapterId}` && method === "GET") {
      await fulfillJson(route, {
        id: chapterId,
        project_id: projectId,
        volume_id: null,
        chapter_index: 1,
        title: "第一章",
        content: "Smoke draft content",
        version: 1,
        created_at: nowIso(),
        updated_at: nowIso(),
      });
      return;
    }
    if (path === `/api/chat/projects/${projectId}/chapters/${chapterId}/revisions` && method === "GET") {
      await fulfillJson(route, []);
      return;
    }
    if (path === `/api/chat/projects/${projectId}/chapters/${chapterId}/scene-beats` && method === "GET") {
      await fulfillJson(route, []);
      return;
    }

    if (path.startsWith("/api/chat/")) {
      await fulfillJson(route, []);
      return;
    }

    await route.fallback();
  });
}

test.describe("smoke regression", () => {
  test("editor home opens in the writing-first shell", async ({ page }) => {
    await installMockChatApi(page, "smoke-mode-toggle");
    await page.goto("/");

    await expect(page.getByTestId("ui-mode-toggle")).toHaveCount(0);
    await expect(page.locator(".assistant-fab")).toHaveText("写作助手");
    await expect(page.getByRole("button", { name: "写作助手" })).toBeVisible();
    await expect(page.locator(".page-shell")).toBeVisible();
    await expect(page.getByRole("button", { name: "展开进阶面板" })).toBeVisible();
    await expect(page.getByRole("banner").getByRole("button", { name: "专注模式" })).toBeVisible();
  });

  test("shell interactions remain healthy", async ({ page }) => {
    await installMockChatApi(page, "smoke-shell");
    await page.goto("/");
    await expect(page.locator(".page-shell")).toBeVisible();
    await expect(page.locator(".advanced-panel-stack")).toHaveCount(0);

    await page.getByRole("button", { name: "写作设置" }).click();
    const settingsDialog = page.locator("#settings-dialog");
    await expect(settingsDialog).toBeVisible();
    await settingsDialog.getByRole("button", { name: "关闭写作设置" }).click();
    await expect(settingsDialog).toBeHidden();

    await page.getByRole("button", { name: "展开进阶面板" }).click();
    await expect(page.locator(".advanced-panel-stack")).toBeVisible();

    await page.getByRole("button", { name: "写作助手" }).click();
    await expect(page.locator("#assistant-drawer")).toBeVisible();
  });

  test("settings keep beginner options primary and hide deprecated wording", async ({ page }) => {
    await installMockChatApi(page, "smoke-settings");
    await page.goto("/");

    await page.getByRole("button", { name: "写作设置" }).click();
    const settingsDialog = page.locator("#settings-dialog");
    await expect(settingsDialog).toBeVisible();

    await expect(settingsDialog.getByRole("tab", { name: "写作" })).toBeVisible();
    await expect(settingsDialog.getByRole("tab", { name: "AI 模型" })).toBeVisible();
    await expect(settingsDialog.getByRole("tab", { name: "上下文" })).toBeVisible();
    await expect(settingsDialog.getByRole("tab", { name: "行为" })).toBeVisible();

    await expect(settingsDialog.getByText("基础写作（推荐）")).toHaveCount(0);
    await expect(settingsDialog.getByText("AI 高级（谨慎调整）")).toHaveCount(0);
    await expect(settingsDialog.getByText("Ghost 自动建议")).toHaveCount(0);
  });

  test("assistant drawer send flow is not regressed", async ({ page }) => {
    const assistantReply = `smoke-reply-${Date.now()}`;
    await installMockChatApi(page, assistantReply);
    await page.goto("/");
    await page.getByRole("button", { name: "写作助手" }).click();

    const drawer = page.locator("#assistant-drawer");
    await expect(drawer).toBeVisible();
    await drawer.getByRole("tab", { name: "对话" }).click();

    const composer = drawer.locator(".composer textarea");
    await expect(composer).toBeVisible();

    const prompt = `e2e smoke ${Date.now()}`;
    const messageRows = drawer.locator(".chat-log article.msg");
    const assistantRows = drawer.locator(".chat-log article.msg.assistant");
    const beforeCount = await messageRows.count();
    const beforeAssistantCount = await assistantRows.count();

    await composer.fill(prompt);

    const sendButton = drawer.locator(".composer button");
    await expect(sendButton).toHaveText("发送");
    await sendButton.click();
    await expect(sendButton).toHaveText("发送", { timeout: 45_000 });

    await expect
      .poll(async () => messageRows.count(), { timeout: 45_000 })
      .toBeGreaterThanOrEqual(beforeCount + 2);

    const userMessage = drawer.locator(".chat-log article.msg.user pre", { hasText: prompt });
    await expect(userMessage).toBeVisible();

    await expect
      .poll(async () => assistantRows.count(), { timeout: 45_000 })
      .toBeGreaterThanOrEqual(beforeAssistantCount + 1);

    const latestAssistantMessage = assistantRows.last().locator("pre");
    await expect(latestAssistantMessage).toContainText(assistantReply, { timeout: 45_000 });
  });

  test("assistant drawer defaults to planning instead of chat-first", async ({ page }) => {
    await installMockChatApi(page, "smoke-planning-default");
    await page.goto("/");
    await page.getByRole("button", { name: "写作助手" }).click();

    const drawer = page.locator("#assistant-drawer");
    await expect(drawer).toBeVisible();
    await expect(drawer.getByRole("heading", { name: "结构化大纲与伏笔" })).toBeVisible();
  });
});
