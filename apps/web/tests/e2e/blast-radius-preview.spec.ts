import { expect, test, type Locator, type Page, type Route } from "@playwright/test";

type MockSessionMessage = {
  id: number;
  role: "user" | "assistant" | "system";
  content: string;
  created_at: string;
};

type MockChatAction = {
  id: number;
  session_id: number;
  action_type: string;
  status: string;
  payload: Record<string, unknown>;
  apply_result: Record<string, unknown>;
  undo_payload: Record<string, unknown>;
  idempotency_key: string;
  operator_id: string;
  created_at: string;
  applied_at: string | null;
  undone_at: string | null;
};

type BlastRadiusFixtureOptions = {
  sessionId?: number;
  actionId?: number;
  actionType?: string;
  assistantText?: string;
  actionPayload?: Record<string, unknown>;
  blastRadius?: Record<string, unknown>;
  graphTimeline?: Record<string, unknown>;
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

async function installMockChatApiWithBlastRadius(
  page: Page,
  options: BlastRadiusFixtureOptions = {}
): Promise<{
  sessionId: number;
  actionId: number;
  getApplyCount: () => number;
}> {
  const projectId = 1;
  const chapterId = 101;
  const sessionId = options.sessionId ?? 77;
  const actionId = options.actionId ?? 501;
  const actionType = options.actionType ?? "card.create";
  const createdAt = "2026-03-11T08:00:00.000Z";
  const assistantText = options.assistantText ?? "我建议补完沈夜与白璃的同盟关系。";
  const blastRadius =
    options.blastRadius ?? {
      source: "rule_preview",
      action_type: "card.create",
      chapter_index: 3,
      nodes: [
        { id: "沈夜", label: "沈夜", change: "update", role: "anchor", in_current_graph: true },
        { id: "白璃", label: "白璃", change: "create", role: "related", in_current_graph: false },
      ],
      edges: [
        {
          key: "沈夜|ALLY_OF|白璃",
          source: "沈夜",
          relation: "ALLY_OF",
          target: "白璃",
          change: "add",
          in_current_graph: false,
        },
      ],
      summary: {
        nodes: { create: 1, update: 1, delete: 0, touch: 0 },
        edges: { add: 1, update: 0, delete: 0 },
      },
      notes: ["将为沈夜补全同盟关系。"],
    };
  const actionPayload =
    options.actionPayload ?? {
      title: "沈夜",
      content: { ally: "白璃" },
    };
  const graphTimeline =
    options.graphTimeline ?? {
      project_id: projectId,
      chapter_index: 3,
      nodes: [
        { id: "沈夜", label: "沈夜", kind: "entity", degree: 1 },
        { id: "长夜城", label: "长夜城", kind: "entity", degree: 1 },
      ],
      edges: [
        {
          id: "edge:沈夜|LOCATED_IN|长夜城",
          source: "沈夜",
          relation: "LOCATED_IN",
          target: "长夜城",
        },
      ],
      stats: { nodes: 2, edges: 1 },
    };

  let nextMessageId = 1000;
  let sessionActive = false;
  let sessionUpdatedAt = createdAt;
  let applyCount = 0;
  let actionStatus: "proposed" | "applied" = "proposed";
  let actionAppliedAt: string | null = null;
  const messages: MockSessionMessage[] = [];

  const buildAction = (): MockChatAction => ({
    id: actionId,
    session_id: sessionId,
    action_type: actionType,
    status: actionStatus,
    payload: {
      ...actionPayload,
      _graph_blast_radius: blastRadius,
    },
    apply_result: actionStatus === "applied" ? { ok: true, applied_action_type: actionType } : {},
    undo_payload: {},
    idempotency_key: `action-${actionId}`,
    operator_id: "human-user",
    created_at: createdAt,
    applied_at: actionAppliedAt,
    undone_at: null,
  });

  const summarizeSessions = () =>
    sessionActive
      ? [
          {
            id: sessionId,
            project_id: projectId,
            title: "Blast Radius 会话",
            created_at: createdAt,
            updated_at: sessionUpdatedAt,
          },
        ]
      : [];

  await page.route("**/api/chat/**", async (route) => {
    const request = route.request();
    const method = request.method().toUpperCase();
    const url = new URL(request.url());
    const path = url.pathname;

    if (path === "/api/chat/stream" && method === "POST") {
      const parsedBody = JSON.parse(request.postData() ?? "{}") as { content?: string };
      const content = String(parsedBody.content ?? "").trim();
      sessionActive = true;
      sessionUpdatedAt = nowIso();
      messages.push(
        { id: nextMessageId++, role: "user", content, created_at: nowIso() },
        { id: nextMessageId++, role: "assistant", content: assistantText, created_at: nowIso() }
      );
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
            session_id: sessionId,
            assistant_message_id: nextMessageId,
            proposed_action_ids: [actionId],
          }) +
          toSse({ type: "delta", text: assistantText }) +
          toSse({
            type: "done",
            assistant_message_id: nextMessageId,
            usage: { total_tokens: assistantText.length },
          }),
      });
      return;
    }

    if (path === `/api/chat/projects/${projectId}/sessions` && method === "GET") {
      await fulfillJson(route, summarizeSessions());
      return;
    }
    if (path === `/api/chat/sessions/${sessionId}/messages` && method === "GET") {
      await fulfillJson(
        route,
        messages.map((item) => ({
          id: item.id,
          session_id: sessionId,
          role: item.role,
          content: item.content,
          model: null,
          created_at: item.created_at,
          context_xray: null,
        }))
      );
      return;
    }
    if (path === `/api/chat/sessions/${sessionId}/actions` && method === "GET") {
      await fulfillJson(route, [buildAction()]);
      return;
    }
    if (path === `/api/chat/actions/${actionId}/logs` && method === "GET") {
      await fulfillJson(route, []);
      return;
    }
    if (path === `/api/chat/actions/${actionId}/apply` && method === "POST") {
      applyCount += 1;
      actionStatus = "applied";
      actionAppliedAt = nowIso();
      await fulfillJson(route, buildAction());
      return;
    }
    if (path === `/api/chat/actions/${actionId}/reject` && method === "POST") {
      await fulfillJson(route, { ...buildAction(), status: "rejected" });
      return;
    }
    if (path === `/api/chat/actions/${actionId}/undo` && method === "POST") {
      await fulfillJson(route, { ...buildAction(), status: "undone", undone_at: nowIso() });
      return;
    }

    if (path === `/api/chat/projects/${projectId}/graph-timeline` && method === "GET") {
      await fulfillJson(route, graphTimeline);
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
      path === `/api/chat/projects/${projectId}/settings` ||
      path === `/api/chat/projects/${projectId}/cards` ||
      path === `/api/chat/projects/${projectId}/consistency-audits` ||
      path === `/api/chat/projects/${projectId}/model-profiles` ||
      path === `/api/chat/projects/${projectId}/prompt-templates` ||
      path === `/api/chat/projects/${projectId}/volumes` ||
      path === `/api/chat/projects/${projectId}/foreshadowing-cards`
    ) {
      await fulfillJson(route, []);
      return;
    }
    if (path === `/api/chat/projects/${projectId}/chapters` && method === "GET") {
      await fulfillJson(route, [
        {
          id: chapterId,
          project_id: projectId,
          volume_id: null,
          chapter_index: 3,
          title: "第三章",
          content: "沈夜独行于长夜城。",
          version: 1,
          created_at: createdAt,
          updated_at: createdAt,
        },
      ]);
      return;
    }
    if (path === `/api/chat/projects/${projectId}/chapters/${chapterId}` && method === "GET") {
      await fulfillJson(route, {
        id: chapterId,
        project_id: projectId,
        volume_id: null,
        chapter_index: 3,
        title: "第三章",
        content: "沈夜独行于长夜城。",
        version: 1,
        created_at: createdAt,
        updated_at: createdAt,
      });
      return;
    }
    if (
      path === `/api/chat/projects/${projectId}/chapters/${chapterId}/revisions` ||
      path === `/api/chat/projects/${projectId}/chapters/${chapterId}/scene-beats`
    ) {
      await fulfillJson(route, []);
      return;
    }
    if (path.startsWith("/api/chat/")) {
      await fulfillJson(route, []);
      return;
    }

    await route.fallback();
  });

  return {
    sessionId,
    actionId,
    getApplyCount: () => applyCount,
  };
}

async function openAssistantDrawer(page: Page): Promise<Locator> {
  await page.goto("/");
  const switchToProMode = page.getByRole("button", { name: "切到工作台模式" });
  if (await switchToProMode.isVisible()) {
    await switchToProMode.click();
  }
  await page.getByRole("button", { name: "助手抽屉" }).click();
  const drawer = page.locator("#assistant-drawer");
  await expect(drawer).toHaveAttribute("aria-hidden", "false");
  return drawer;
}

async function sendPromptAndCloseDrawer(drawer: Locator, prompt: string): Promise<void> {
  const composer = drawer.locator(".composer textarea");
  const sendButton = drawer.locator(".composer button");
  await composer.fill(prompt);
  await sendButton.click();
  await expect(sendButton).toHaveText("发送", { timeout: 10_000 });
  await drawer.getByRole("button", { name: "关闭助手抽屉" }).click();
  await expect(drawer).toHaveAttribute("aria-hidden", "true");
}

test.describe("blast radius preview regression", () => {
  test("hover shows blast radius and apply is two-step gated", async ({ page }) => {
    const { actionId, getApplyCount } = await installMockChatApiWithBlastRadius(page);
    const drawer = await openAssistantDrawer(page);

    await sendPromptAndCloseDrawer(drawer, "补完沈夜和白璃的关系");

    const actionCard = page.locator(".action-card").filter({ hasText: `#${actionId}` }).first();
    await expect(actionCard).toBeVisible({ timeout: 10_000 });
    await expect(actionCard).toContainText("爆炸半径：+1 节点 / ~1 节点 / +1 边");

    await actionCard.hover();

    const blastRadiusPanel = page.locator(".blast-radius-panel");
    await expect(blastRadiusPanel).toBeVisible();
    await expect(blastRadiusPanel).toContainText("动作爆炸半径");
    await expect(blastRadiusPanel).toContainText("沈夜");
    await expect(blastRadiusPanel).toContainText("白璃");
    await expect(blastRadiusPanel).toContainText("将为沈夜补全同盟关系。");
    await expect.poll(async () => page.locator(".timeline-node.preview-update").count()).toBeGreaterThan(0);
    await expect.poll(async () => page.locator(".timeline-node.preview-add.is-preview-ghost").count()).toBeGreaterThan(0);
    await expect.poll(async () => page.locator(".timeline-edge.preview-add.is-preview-ghost").count()).toBeGreaterThan(0);
    await expect.poll(async () => page.locator(".timeline-edge.is-dim").count()).toBeGreaterThan(0);

    await actionCard.getByRole("button", { name: "应用到项目" }).click();
    await expect(actionCard).toContainText("应用前总览");
    expect(getApplyCount()).toBe(0);

    await actionCard.getByRole("button", { name: "确认应用" }).click();
    await expect.poll(getApplyCount).toBe(1);
    await expect(actionCard).toContainText("已应用");
    await expect(actionCard.getByRole("button", { name: "撤销应用" })).toBeVisible();
  });

  test("keyboard focus previews delete blast radius and blur clears the panel", async ({ page }) => {
    const { actionId } = await installMockChatApiWithBlastRadius(page, {
      actionId: 502,
      actionType: "card.update",
      assistantText: "我建议移除沈夜对白璃的旧关系，并引入新的敌对线。",
      actionPayload: {
        card_id: 12,
        title: "沈夜",
        content: { rival: "玄冥" },
      },
      blastRadius: {
        source: "rule_preview",
        action_type: "card.update",
        chapter_index: 3,
        nodes: [
          { id: "沈夜", label: "沈夜", change: "update", role: "anchor", in_current_graph: true },
          { id: "白璃", label: "白璃", change: "delete", role: "related", in_current_graph: true },
          { id: "玄冥", label: "玄冥", change: "create", role: "related", in_current_graph: false },
        ],
        edges: [
          {
            key: "沈夜|ALLY_OF|白璃",
            source: "沈夜",
            relation: "ALLY_OF",
            target: "白璃",
            change: "delete",
            in_current_graph: true,
          },
          {
            key: "沈夜|RIVAL_OF|玄冥",
            source: "沈夜",
            relation: "RIVAL_OF",
            target: "玄冥",
            change: "add",
            in_current_graph: false,
          },
        ],
        summary: {
          nodes: { create: 1, update: 1, delete: 1, touch: 0 },
          edges: { add: 1, update: 0, delete: 1 },
        },
        notes: ["将移除旧同盟，并引入新的敌对关系。"],
      },
      graphTimeline: {
        project_id: 1,
        chapter_index: 3,
        nodes: [
          { id: "沈夜", label: "沈夜", kind: "entity", degree: 2 },
          { id: "白璃", label: "白璃", kind: "entity", degree: 1 },
          { id: "长夜城", label: "长夜城", kind: "entity", degree: 1 },
        ],
        edges: [
          {
            id: "edge:沈夜|ALLY_OF|白璃",
            source: "沈夜",
            relation: "ALLY_OF",
            target: "白璃",
          },
          {
            id: "edge:沈夜|LOCATED_IN|长夜城",
            source: "沈夜",
            relation: "LOCATED_IN",
            target: "长夜城",
          },
        ],
        stats: { nodes: 3, edges: 2 },
      },
    });
    const drawer = await openAssistantDrawer(page);

    await sendPromptAndCloseDrawer(drawer, "替换沈夜与白璃的旧关系");

    const actionCard = page.locator(".action-card").filter({ hasText: `#${actionId}` }).first();
    await expect(actionCard).toBeVisible({ timeout: 10_000 });
    const applyButton = actionCard.getByRole("button", { name: "应用到项目" });
    await applyButton.scrollIntoViewIfNeeded();
    await applyButton.focus();
    await expect(applyButton).toBeFocused();

    const blastRadiusPanel = page.locator(".blast-radius-panel");
    await expect(blastRadiusPanel).toBeVisible();
    await expect(blastRadiusPanel).toContainText("-1 节点");
    await expect(blastRadiusPanel).toContainText("-1 边");
    await expect(blastRadiusPanel).toContainText("将移除旧同盟，并引入新的敌对关系。");
    await expect.poll(async () => page.locator(".timeline-node.preview-delete").count()).toBeGreaterThan(0);
    await expect.poll(async () => page.locator(".timeline-edge.preview-delete").count()).toBeGreaterThan(0);

    await page.getByRole("button", { name: "写作设置" }).focus();
    await expect(blastRadiusPanel).toBeHidden();
  });

  test("note-only blast radius keeps graph stable and still gates apply", async ({ page }) => {
    const { actionId, getApplyCount } = await installMockChatApiWithBlastRadius(page, {
      actionId: 503,
      actionType: "setting.delete",
      assistantText: "我建议删除已经废弃的旧盟约设定。",
      actionPayload: {
        key: "旧盟约",
      },
      blastRadius: {
        source: "none",
        action_type: "setting.delete",
        chapter_index: 3,
        nodes: [],
        edges: [],
        summary: {
          nodes: { create: 0, update: 0, delete: 0, touch: 0 },
          edges: { add: 0, update: 0, delete: 0 },
        },
        notes: ["此动作不会直接改写图谱关系，仅影响设定与索引生命周期。"],
      },
      graphTimeline: {
        project_id: 1,
        chapter_index: 3,
        nodes: [
          { id: "沈夜", label: "沈夜", kind: "entity", degree: 1 },
          { id: "白璃", label: "白璃", kind: "entity", degree: 1 },
        ],
        edges: [
          {
            id: "edge:沈夜|ALLY_OF|白璃",
            source: "沈夜",
            relation: "ALLY_OF",
            target: "白璃",
          },
        ],
        stats: { nodes: 2, edges: 1 },
      },
    });
    const drawer = await openAssistantDrawer(page);

    await sendPromptAndCloseDrawer(drawer, "删除过期设定旧盟约");

    const actionCard = page.locator(".action-card").filter({ hasText: `#${actionId}` }).first();
    await expect(actionCard).toBeVisible({ timeout: 10_000 });
    await expect(actionCard).toContainText("爆炸半径：此动作不会直接改写图谱关系，仅影响设定与索引生命周期。");

    await actionCard.hover();

    const blastRadiusPanel = page.locator(".blast-radius-panel");
    await expect(blastRadiusPanel).toBeVisible();
    await expect(blastRadiusPanel).toContainText("此动作不会直接改写图谱关系，仅影响设定与索引生命周期。");
    await expect(blastRadiusPanel).toContainText("仅锚点波及");
    await expect(page.locator(".timeline-node.preview-add, .timeline-node.preview-update, .timeline-node.preview-delete")).toHaveCount(0);
    await expect(page.locator(".timeline-edge.preview-add, .timeline-edge.preview-update, .timeline-edge.preview-delete")).toHaveCount(0);

    await actionCard.getByRole("button", { name: "应用到项目" }).click();
    await expect(actionCard).toContainText("应用前总览");
    await expect(actionCard).toContainText("仅锚点波及");
    expect(getApplyCount()).toBe(0);

    await actionCard.getByRole("button", { name: "确认应用" }).click();
    await expect.poll(getApplyCount).toBe(1);
    await expect(actionCard).toContainText("已应用");
  });
});
