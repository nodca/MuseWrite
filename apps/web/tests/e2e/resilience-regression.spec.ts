import { expect, test, type Locator, type Page, type Route } from "@playwright/test";

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

type MockChatApiOptions = {
  failFirstStream?: boolean;
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

async function installMockChatApi(page: Page, options: MockChatApiOptions = {}): Promise<{ chapterId: number }> {
  const projectId = 1;
  const chapter = {
    id: 101,
    project_id: projectId,
    volume_id: null,
    chapter_index: 1,
    title: "Mock Chapter",
    content: "Server draft content",
    version: 3,
    created_at: nowIso(),
    updated_at: nowIso(),
  };
  const sessions: MockSession[] = [];
  let nextSessionId = 1;
  let nextMessageId = 1;
  let failFirstStream = Boolean(options.failFirstStream);

  await page.route("**/api/chat/**", async (route) => {
    const request = route.request();
    const method = request.method().toUpperCase();
    const url = new URL(request.url());
    const path = url.pathname;

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

    if (path === "/api/chat/stream" && method === "POST") {
      if (failFirstStream) {
        failFirstStream = false;
        await fulfillJson(route, { detail: "e2e-forced-stream-failure" }, 500);
        return;
      }

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
      const assistantMessage: MockSessionMessage = {
        id: nextMessageId++,
        role: "assistant",
        content: `mock-reply: ${content || "empty"}`,
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

    if (path.match(new RegExp(`^/api/chat/projects/${projectId}/chapters/?$`)) && method === "GET") {
      await fulfillJson(route, [chapter]);
      return;
    }
    if (path.match(new RegExp(`^/api/chat/projects/${projectId}/chapters/\\d+/?$`)) && method === "GET") {
      await fulfillJson(route, chapter);
      return;
    }
    if (path.match(new RegExp(`^/api/chat/projects/${projectId}/chapters/\\d+/revisions/?$`)) && method === "GET") {
      await fulfillJson(route, []);
      return;
    }
    if (path.match(new RegExp(`^/api/chat/projects/${projectId}/chapters/\\d+/scene-beats/?$`)) && method === "GET") {
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

    if (path.startsWith("/api/chat/")) {
      await fulfillJson(route, {});
      return;
    }

    await route.fallback();
  });

  return { chapterId: chapter.id };
}

async function openAssistantDrawer(page: Page): Promise<Locator> {
  await page.goto("/");
  await page.getByRole("button", { name: "助手抽屉" }).click();
  const drawer = page.locator("#assistant-drawer");
  await expect(drawer).toHaveAttribute("aria-hidden", "false");
  return drawer;
}

async function sendPrompt(drawer: Locator, prompt: string): Promise<void> {
  const composer = drawer.locator(".composer textarea");
  const sendButton = drawer.locator(".composer button");
  await composer.fill(prompt);
  await expect(sendButton).toHaveText("发送");
  await sendButton.click();
  await expect(sendButton).toHaveText("发送", { timeout: 45_000 });

  await expect
    .poll(
      async () =>
        await drawer
          .locator(".chat-log article.msg.user pre", {
            hasText: prompt,
          })
          .count(),
      { timeout: 45_000 }
    )
    .toBeGreaterThan(0);

  await expect(drawer.locator(".chat-log article.msg.user pre", { hasText: prompt })).toBeVisible();
}

test.describe("resilience regression", () => {
  test("session switching keeps per-session history isolated", async ({ page }) => {
    await installMockChatApi(page);
    const drawer = await openAssistantDrawer(page);
    const sessionSelect = drawer.getByRole("combobox", { name: "会话" });

    const firstPrompt = `e2e session-a ${Date.now()}`;
    await sendPrompt(drawer, firstPrompt);
    await expect
      .poll(async () => (await sessionSelect.inputValue()).trim(), { timeout: 45_000 })
      .not.toBe("");
    const firstSessionId = (await sessionSelect.inputValue()).trim();

    await drawer.getByRole("button", { name: "新会话" }).click();
    await expect(sessionSelect).toHaveValue("");

    const secondPrompt = `e2e session-b ${Date.now() + 1}`;
    await sendPrompt(drawer, secondPrompt);
    await expect
      .poll(async () => (await sessionSelect.inputValue()).trim(), { timeout: 45_000 })
      .not.toBe("");
    const secondSessionId = (await sessionSelect.inputValue()).trim();
    expect(secondSessionId).not.toBe(firstSessionId);

    await sessionSelect.selectOption(firstSessionId);
    await expect(sessionSelect).toHaveValue(firstSessionId);
    await expect(drawer.locator(".chat-log article.msg.user pre", { hasText: firstPrompt })).toBeVisible();
    await expect(drawer.locator(".chat-log article.msg.user pre", { hasText: secondPrompt })).toBeHidden();

    await sessionSelect.selectOption(secondSessionId);
    await expect(sessionSelect).toHaveValue(secondSessionId);
    await expect(drawer.locator(".chat-log article.msg.user pre", { hasText: secondPrompt })).toBeVisible();
  });

  test("send failure surfaces error and next retry recovers", async ({ page }) => {
    await installMockChatApi(page, { failFirstStream: true });
    const drawer = await openAssistantDrawer(page);

    const failedPrompt = `e2e fail-once ${Date.now()}`;
    await sendPrompt(drawer, failedPrompt);
    await expect(page.locator(".error-banner")).toContainText("错误：e2e-forced-stream-failure");
    await expect(drawer.locator(".chat-log article.msg.assistant pre")).toContainText(
      "请求失败：e2e-forced-stream-failure"
    );

    const recoveryPrompt = `e2e retry-success ${Date.now() + 1}`;
    await sendPrompt(drawer, recoveryPrompt);
    await expect(page.locator(".error-banner")).toBeHidden();
    await expect(drawer.locator(".chat-log article.msg.user pre", { hasText: recoveryPrompt })).toBeVisible();

    const assistantRows = drawer.locator(".chat-log article.msg.assistant pre");
    await expect
      .poll(async () => ((await assistantRows.last().textContent()) ?? "").trim().length, { timeout: 45_000 })
      .toBeGreaterThan(0);
  });

  test("refresh keeps assistant send flow healthy", async ({ page }) => {
    await installMockChatApi(page);

    const firstPrompt = `e2e before-refresh ${Date.now()}`;
    const drawer = await openAssistantDrawer(page);
    await sendPrompt(drawer, firstPrompt);

    await page.reload();
    await page.getByRole("button", { name: "助手抽屉" }).click();
    const reloadedDrawer = page.locator("#assistant-drawer");
    await expect(reloadedDrawer).toHaveAttribute("aria-hidden", "false");
    const secondPrompt = `e2e after-refresh ${Date.now() + 1}`;
    await sendPrompt(reloadedDrawer, secondPrompt);
    await expect(page.locator(".error-banner")).toBeHidden();
    await expect(reloadedDrawer.locator(".chat-log article.msg.user pre", { hasText: secondPrompt })).toBeVisible();
  });
});
