import { expect, test, type Dialog, type Locator, type Page, type TestInfo } from "@playwright/test";

function handleNextDialog(page: Page, handler: (dialog: Dialog) => Promise<void> | void): Promise<void> {
  return new Promise((resolve, reject) => {
    page.once("dialog", (dialog) => {
      Promise.resolve(handler(dialog))
        .then(() => resolve())
        .catch(reject);
    });
  });
}

async function openAssistantDrawer(page: Page): Promise<Locator> {
  await page.goto("/");
  await page.getByRole("button", { name: "助手抽屉" }).click();

  const drawer = page.locator("#assistant-drawer");
  await expect(drawer).toHaveAttribute("aria-hidden", "false");
  return drawer;
}

function parseSseEvents(rawBody: string): Array<Record<string, unknown>> {
  const events: Array<Record<string, unknown>> = [];
  for (const block of rawBody.split("\n\n")) {
    const line = block.trim();
    if (!line.startsWith("data:")) continue;
    const payload = line.slice(5).trim();
    if (!payload) continue;
    const parsed = JSON.parse(payload);
    if (parsed && typeof parsed === "object") {
      events.push(parsed as Record<string, unknown>);
    }
  }
  return events;
}

function resolveApiToken(): string | null {
  const token = (process.env.E2E_API_TOKEN ?? process.env.VITE_API_TOKEN ?? "").trim();
  return token.length > 0 ? token : null;
}

async function bootstrapSessionViaApi(page: Page, prompt: string, token: string): Promise<number> {
  const response = await page.request.post("/api/chat/stream", {
    headers: {
      Authorization: `Bearer ${token}`,
    },
    data: {
      project_id: 1,
      content: prompt,
    },
  });

  if (!response.ok()) {
    const raw = await response.text();
    throw new Error(`bootstrap session failed: ${response.status()} ${raw.slice(0, 300)}`);
  }

  const raw = await response.text();
  const events = parseSseEvents(raw);
  const meta = events.find((item) => item.type === "meta");
  const sessionId = Number(meta?.session_id ?? 0);
  if (!Number.isFinite(sessionId) || sessionId <= 0) {
    throw new Error("bootstrap session failed: missing meta.session_id");
  }
  return sessionId;
}

test.describe("session management regression", () => {
  test("rename and delete session flow remains healthy (fixture-seeded)", async ({ page }, testInfo: TestInfo) => {
    testInfo.annotations.push({
      type: "flaky",
      description: "Known flaky when backend session index update is delayed in CI.",
    });

    const token = resolveApiToken();
    test.skip(!token, "Skipping fixture-seeded session test: E2E_API_TOKEN/VITE_API_TOKEN not set.");

    test.setTimeout(120_000);

    await page.goto("/");
    const sessionId = await bootstrapSessionViaApi(page, `e2e session seed ${Date.now()}`, token!);
    await page.reload();

    const drawer = await openAssistantDrawer(page);
    const sessionSelect = drawer.getByRole("combobox", { name: "会话" });

    const option = sessionSelect.locator(`option[value="${sessionId}"]`);
    await expect(option).toBeVisible({ timeout: 45_000 });
    await sessionSelect.selectOption(String(sessionId));
    await expect(sessionSelect).toHaveValue(String(sessionId));
    await expect(drawer.getByRole("button", { name: "重命名" })).toBeVisible();

    const renamedTitle = `e2e renamed ${Date.now()}`;
    const renameButton = drawer.getByRole("button", { name: "重命名" });
    await expect(renameButton).toBeVisible();
    const renameRequestPromise = page.waitForRequest((request) => {
      return request.method() === "PUT" && request.url().endsWith(`/api/chat/projects/1/sessions/${sessionId}`);
    });

    const renameDialogHandled = handleNextDialog(page, async (dialog) => {
      expect(dialog.type()).toBe("prompt");
      await dialog.accept(renamedTitle);
    });
    await renameButton.click();
    await renameDialogHandled;
    const renameRequest = await renameRequestPromise;
    expect(renameRequest.postDataJSON()).toEqual({ title: renamedTitle });

    const deleteButton = drawer.getByRole("button", { name: "删除会话" });
    await expect(deleteButton).toBeVisible();
    const deleteRequestPromise = page.waitForRequest((request) => {
      return request.method() === "DELETE" && request.url().endsWith(`/api/chat/projects/1/sessions/${sessionId}`);
    });
    const deleteDialogHandled = handleNextDialog(page, async (dialog) => {
      expect(dialog.type()).toBe("confirm");
      await dialog.accept();
    });
    await deleteButton.click();
    await deleteDialogHandled;
    await deleteRequestPromise;

    await expect(sessionSelect).toHaveValue("");
  });

  test("assistant tools tabs toggle without regression", async ({ page }) => {
    const drawer = await openAssistantDrawer(page);

    const actionsTab = drawer.getByRole("button", { name: "动作提议" });
    const candidatesTab = drawer.getByRole("button", { name: "候选审核" });

    await expect(actionsTab).toHaveClass(/\bactive\b/);
    await expect(candidatesTab).not.toHaveClass(/\bactive\b/);

    await candidatesTab.click();
    await expect(candidatesTab).toHaveClass(/\bactive\b/);
    await expect(actionsTab).not.toHaveClass(/\bactive\b/);

    await actionsTab.click();
    await expect(actionsTab).toHaveClass(/\bactive\b/);
    await expect(candidatesTab).not.toHaveClass(/\bactive\b/);
  });
});
