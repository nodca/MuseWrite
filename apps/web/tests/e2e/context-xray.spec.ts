import { expect, test, type Locator, type Page, type Route } from "@playwright/test";

type MockSessionMessage = {
  id: number;
  role: "user" | "assistant" | "system";
  content: string;
  created_at: string;
  context_xray?: {
    version: number;
    evidence: Record<string, unknown> | null;
  } | null;
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
    created_at: nowIso(),
    updated_at: nowIso(),
  };

  let nextSessionId = 7;
  let nextMessageId = 100;
  let delayNextMessagesFetch = false;
  const streamEvidence = {
    type: "evidence",
    policy: {
      mode: "character",
      anchor: "神剑",
      notes: [],
      resolver_order: "DSL > GRAPH > RAG",
    },
    summary: {
      dsl: 1,
      graph: 0,
      rag: 0,
    },
    sources: {
      dsl: [
        {
          kind: "setting",
          id: 9001,
          title: "神剑",
          snippet: "神剑：只认李承渊一人为主。",
        },
      ],
      graph: [],
      rag: [],
    },
  };

  const sessions: MockSession[] = [];

  await page.route("**/api/chat/**", async (route) => {
    const request = route.request();
    const method = request.method().toUpperCase();
    const url = new URL(request.url());
    const path = url.pathname;

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

    if (path === "/api/chat/stream" && method === "POST") {
      const parsedBody = JSON.parse(request.postData() ?? "{}") as {
        content?: string;
        session_id?: number | null;
      };
      const content = String(parsedBody.content ?? "").trim();
      const requestedSessionId = Number(parsedBody.session_id ?? 0);
      let session = requestedSessionId > 0 ? sessions.find((item) => item.id === requestedSessionId) ?? null : null;
      if (!session) {
        const createdAt = nowIso();
        session = {
          id: nextSessionId++,
          title: "Context X-Ray 会话",
          created_at: createdAt,
          updated_at: createdAt,
          messages: [],
        };
        sessions.push(session);
      }

      const usesEvidence = content.includes("神剑");
      const assistantReply = usesEvidence ? "神剑裂空而出，剑鸣直指李承渊。" : "古剑在黑夜中低鸣，像是在等待旧主归来。";

      const userMessage: MockSessionMessage = {
        id: nextMessageId++,
        role: "user",
        content,
        created_at: nowIso(),
      };
      const assistantMessage: MockSessionMessage = {
        id: nextMessageId++,
        role: "assistant",
        content: assistantReply,
        created_at: nowIso(),
        context_xray: usesEvidence
          ? {
              version: 1,
              evidence: streamEvidence,
            }
          : null,
      };

      session.messages.push(userMessage, assistantMessage);
      session.updated_at = nowIso();
      delayNextMessagesFetch = usesEvidence;

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
          (usesEvidence ? toSse(streamEvidence) : "") +
          toSse({ type: "delta", text: assistantMessage.content }) +
          toSse({
            type: "done",
            assistant_message_id: assistantMessage.id,
            usage: { total_tokens: assistantMessage.content.length },
          }),
      });
      return;
    }

    if (path === `/api/chat/projects/${projectId}/sessions` && method === "GET") {
      await fulfillJson(route, summarizeSessions());
      return;
    }
    if (path.match(/^\/api\/chat\/projects\/\d+\/sessions$/) && method === "GET") {
      await fulfillJson(route, summarizeSessions());
      return;
    }

    const messagesMatch = path.match(/^\/api\/chat\/sessions\/(\d+)\/messages$/);
    if (messagesMatch && method === "GET") {
      if (delayNextMessagesFetch) {
        delayNextMessagesFetch = false;
        await page.waitForTimeout(1_500);
      }
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
          context_xray: item.context_xray ?? null,
        })) ?? [];
      await fulfillJson(route, messages);
      return;
    }

    if (path.match(/^\/api\/chat\/sessions\/\d+\/actions$/) && method === "GET") {
      await fulfillJson(route, []);
      return;
    }

    if (path === `/api/chat/projects/${projectId}/settings` && method === "GET") {
      await fulfillJson(route, [
        {
          id: 1,
          project_id: projectId,
          key: "神剑",
          value: {
            aliases: ["古剑", "天命神剑"],
            description: "神剑：沉睡百年，唯认命定之主。",
          },
          created_at: nowIso(),
          updated_at: nowIso(),
        },
      ]);
      return;
    }

    if (path === `/api/chat/projects/${projectId}/cards` && method === "GET") {
      await fulfillJson(route, [
        {
          id: 11,
          project_id: projectId,
          title: "李承渊",
          content: {
            aliases: ["李少主"],
            description: "剑宗少主，神剑宿主。",
          },
          created_at: nowIso(),
          updated_at: nowIso(),
        },
      ]);
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
        settings_count: 1,
        cards_count: 1,
        ttl_seconds: 30,
      });
      return;
    }

    if (path.match(/^\/api\/chat\/projects\/\d+\/chapters\/?$/) && method === "GET") {
      await fulfillJson(route, [chapter]);
      return;
    }
    if (path.match(/^\/api\/chat\/projects\/\d+\/chapters\/?$/) && method === "POST") {
      await fulfillJson(route, chapter);
      return;
    }
    if (path.match(/^\/api\/chat\/projects\/\d+\/chapters\/\d+\/?$/) && method === "GET") {
      await fulfillJson(route, chapter);
      return;
    }
    if (path.match(/^\/api\/chat\/projects\/\d+\/chapters\/\d+\/revisions\/?$/) && method === "GET") {
      await fulfillJson(route, []);
      return;
    }
    if (path.match(/^\/api\/chat\/projects\/\d+\/chapters\/\d+\/scene-beats\/?$/) && method === "GET") {
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

async function openAssistantDrawer(page: Page): Promise<Locator> {
  await page.goto("/");
  await page.getByRole("button", { name: "写作助手" }).click();
  const drawer = page.locator("#assistant-drawer");
  await expect(drawer).toBeVisible();
  return drawer;
}

test.describe("context x-ray regression", () => {
  test("assistant messages render evidence-first bindings and settings fallback with popovers", async ({ page }) => {
    await installMockChatApi(page);
    const settingsResponse = page.waitForResponse((response) =>
      response.request().method() === "GET" && response.url().includes("/api/chat/projects/1/settings")
    );
    const cardsResponse = page.waitForResponse((response) =>
      response.request().method() === "GET" && response.url().includes("/api/chat/projects/1/cards")
    );
    const drawer = await openAssistantDrawer(page);
    await Promise.all([settingsResponse, cardsResponse]);
    await page.waitForTimeout(150);

    await drawer.getByRole("tab", { name: "对话" }).click();
    const composer = drawer.locator(".composer textarea");
    const sendButton = drawer.locator(".composer button");

    await composer.fill("让古剑鸣响");
    await sendButton.click();
    await expect(sendButton).toHaveText("发送", { timeout: 10_000 });

    const fallbackToken = drawer
      .locator("article.msg.assistant")
      .last()
      .locator('[data-context-xray-source="fallback"]', { hasText: "古剑" });
    await expect(fallbackToken).toBeVisible();
    await fallbackToken.hover();
    await expect(page.getByRole("tooltip")).toContainText("设定回退");
    await expect(page.getByRole("tooltip")).toContainText("沉睡百年");
    await page.mouse.move(0, 0);
    await fallbackToken.focus();
    await expect(page.getByRole("tooltip")).toContainText("设定回退");
    await expect(page.getByRole("tooltip")).toContainText("神剑");

    await page.mouse.move(0, 0);

    await composer.fill("让神剑现身");
    await sendButton.click();
    await expect(sendButton).toHaveText("发送中...");

    const streamingEvidenceToken = drawer
      .locator("article.msg.assistant")
      .last()
      .locator('[data-context-xray-source="evidence"]', { hasText: "神剑" });
    await expect(streamingEvidenceToken).toBeVisible({ timeout: 1_200 });

    await expect(sendButton).toHaveText("发送", { timeout: 10_000 });
    await streamingEvidenceToken.hover();
    await expect(page.getByRole("tooltip")).toContainText("本轮引用");
    await expect(page.getByRole("tooltip")).toContainText("只认李承渊一人为主");
  });
});
