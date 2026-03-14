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
  const actionId = 701;
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
  let nextSessionId = 41;
  let nextMessageId = 300;

  const proposedAction = {
    id: actionId,
    session_id: nextSessionId,
    action_type: "card.update",
    status: "proposed",
    payload: {
      title: "神剑",
      content: {
        描述: "神剑表面浮现宿命烙印，祖祠封印正在松动。",
      },
      _graph_blast_radius: {
        source: "action_projection",
        action_type: "card.update",
        chapter_index: 1,
        nodes: [
          {
            id: "神剑",
            label: "神剑",
            change: "update",
            role: "anchor",
            in_current_graph: true,
          },
          {
            id: "祖祠",
            label: "祖祠",
            change: "touch",
            role: "related",
            in_current_graph: true,
          },
          {
            id: "宿命烙印",
            label: "宿命烙印",
            change: "create",
            role: "related",
            in_current_graph: false,
          },
          {
            id: "旧剑盟约",
            label: "旧剑盟约",
            change: "delete",
            role: "related",
            in_current_graph: true,
          },
        ],
        edges: [
          {
            key: "神剑|GUARDS|祖祠",
            source: "神剑",
            relation: "GUARDS",
            target: "祖祠",
            change: "update",
            in_current_graph: true,
          },
          {
            key: "神剑|MARKS|宿命烙印",
            source: "神剑",
            relation: "MARKS",
            target: "宿命烙印",
            change: "add",
            in_current_graph: false,
          },
          {
            key: "旧剑盟约|BINDS|神剑",
            source: "旧剑盟约",
            relation: "BINDS",
            target: "神剑",
            change: "delete",
            in_current_graph: true,
          },
        ],
        summary: {
          nodes: {
            create: 1,
            update: 1,
            touch: 1,
            delete: 1,
          },
          edges: {
            add: 1,
            update: 1,
            delete: 1,
          },
        },
        notes: ["Hover 后应立即展示动作爆炸半径。"],
      },
    },
    apply_result: {},
    undo_payload: {
      before: {
        title: "神剑",
        content: {
          描述: "旧设定",
        },
      },
    },
    idempotency_key: "blast-radius-701",
    operator_id: "assistant",
    created_at: nowIso(),
    applied_at: null,
    undone_at: null,
  };

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
      const rawBody = request.postData() ?? "{}";
      const parsedBody = JSON.parse(rawBody) as { content?: string; session_id?: number | null };
      const content = String(parsedBody.content ?? "").trim();
      const requestedSessionId = Number(parsedBody.session_id ?? 0);

      let session =
        requestedSessionId > 0 ? sessions.find((item) => item.id === requestedSessionId) ?? null : null;
      if (!session) {
        const createdAt = nowIso();
        session = {
          id: nextSessionId,
          title: "Blast Radius 会话",
          created_at: createdAt,
          updated_at: createdAt,
          messages: [],
        };
        sessions.push(session);
        nextSessionId += 1;
      }

      const replyText = "已生成带图谱投影的动作提议，请在工作台查看爆炸半径。";
      const userMessage: MockSessionMessage = {
        id: nextMessageId++,
        role: "user",
        content,
        created_at: nowIso(),
      };
      const assistantMessage: MockSessionMessage = {
        id: nextMessageId++,
        role: "assistant",
        content: replyText,
        created_at: nowIso(),
      };
      session.messages.push(userMessage, assistantMessage);
      session.updated_at = nowIso();

      proposedAction.session_id = session.id;

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
            proposed_action_ids: [actionId],
          }) +
          toSse({ type: "delta", text: replyText }) +
          toSse({
            type: "done",
            assistant_message_id: assistantMessage.id,
            usage: { total_tokens: Math.max(1, replyText.length) },
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
          context_xray: null,
        })) ?? [];
      await fulfillJson(route, messages);
      return;
    }

    const actionsMatch = path.match(/^\/api\/chat\/sessions\/(\d+)\/actions$/);
    if (actionsMatch && method === "GET") {
      const sessionId = Number(actionsMatch[1]);
      await fulfillJson(route, sessionId === proposedAction.session_id ? [proposedAction] : []);
      return;
    }

    if (path.match(/^\/api\/chat\/actions\/\d+\/logs$/) && method === "GET") {
      await fulfillJson(route, []);
      return;
    }

    if (path === `/api/chat/projects/${projectId}/graph-timeline` && method === "GET") {
      await fulfillJson(route, {
        project_id: projectId,
        chapter_index: 1,
        nodes: [
          { id: "神剑", label: "神剑", kind: "entity", degree: 2 },
          { id: "祖祠", label: "祖祠", kind: "location", degree: 1 },
          { id: "旧剑盟约", label: "旧剑盟约", kind: "concept", degree: 1 },
        ],
        edges: [
          {
            id: "神剑|GUARDS|祖祠",
            source: "神剑",
            relation: "GUARDS",
            target: "祖祠",
          },
          {
            id: "旧剑盟约|BINDS|神剑",
            source: "旧剑盟约",
            relation: "BINDS",
            target: "神剑",
          },
        ],
        stats: {
          nodes: 3,
          edges: 2,
        },
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

    if (path.match(/^\/api\/chat\/projects\/\d+\/chapters\/\d+\/revisions\/?$/) && method === "GET") {
      await fulfillJson(route, []);
      return;
    }

    if (path.match(/^\/api\/chat\/projects\/\d+\/chapters\/\d+\/scene-beats\/?$/) && method === "GET") {
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

    if (path.startsWith("/api/chat/")) {
      await fulfillJson(route, {});
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

async function sendPrompt(drawer: Locator, prompt: string): Promise<void> {
  await drawer.getByRole("tab", { name: "对话" }).click();
  const composer = drawer.locator(".composer textarea");
  const sendButton = drawer.locator(".composer button");
  await composer.fill(prompt);
  await sendButton.click();
  await expect(sendButton).toHaveText("发送", { timeout: 45_000 });
}

test.describe("blast radius regression", () => {
  test("sending a proposed action previews its graph blast radius in drawer", async ({ page }) => {
    await installMockChatApi(page);
    const drawer = await openAssistantDrawer(page);

    await sendPrompt(drawer, `blast-radius ${Date.now()}`);

    const actionCard = drawer.locator(".action-card", {
      hasText: "爆炸半径",
    });
    await expect(actionCard).toContainText("神剑", { timeout: 45_000 });

    await actionCard.hover();

    const previewBar = drawer.locator(".timeline-preview-bar");
    await expect(previewBar).toBeVisible();
    await expect(previewBar).toContainText("动作爆炸半径");
    await expect(previewBar).toContainText("+1 节点");
    await expect(previewBar).toContainText("新增 / 投影");
    await expect(previewBar).toContainText("宿命烙印");
    await expect(previewBar).toContainText("查看明细");

    const timelineGraph = drawer.locator(".timeline-graph");
    await expect(timelineGraph).toBeVisible();
    await expect(timelineGraph.locator(".timeline-edge.preview-add")).toHaveCount(1);
    await expect(timelineGraph.locator(".timeline-edge.preview-update")).toHaveCount(1);
    await expect(timelineGraph.locator(".timeline-edge.preview-delete")).toHaveCount(1);
    await expect(timelineGraph.locator(".timeline-node.preview-add")).toHaveCount(1);
    await expect(timelineGraph.locator(".timeline-node.preview-update")).toHaveCount(2);
    await expect(timelineGraph.locator(".timeline-node.preview-delete")).toHaveCount(1);

    const detail = previewBar.locator("details");
    await detail.locator("summary").click();
    await expect(detail).toHaveAttribute("open", "");

    const detailGrid = detail.locator(".grid");
    const nodeSection = detailGrid.locator("section").nth(0);
    const edgeSection = detailGrid.locator("section").nth(1);
    const detailSearch = detail.locator("input[type=\"search\"]");
    const detailFilter = detail.locator("select");

    await detailSearch.fill("宿命");
    await expect(nodeSection.locator("button")).toHaveCount(1);
    await expect(nodeSection).toContainText("宿命烙印");
    await expect(edgeSection.locator("button")).toHaveCount(1);
    await expect(edgeSection).toContainText("MARKS");

    await detailSearch.fill("");
    await detailFilter.selectOption("delete");
    await expect(nodeSection.locator("button")).toHaveCount(1);
    await expect(nodeSection).toContainText("旧剑盟约");
    await expect(edgeSection.locator("button")).toHaveCount(1);
    await expect(edgeSection).toContainText("BINDS");
  });
});
