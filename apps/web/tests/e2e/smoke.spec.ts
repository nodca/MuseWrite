import { expect, test } from "@playwright/test";

test.describe("smoke regression", () => {
  test("defaults to writing mode and toggles to pro mode", async ({ page }) => {
    await page.goto("/");

    await expect(page.locator(".page-shell")).toHaveAttribute("data-ui-mode", "writing");
    await expect(page.getByTestId("ui-mode-toggle")).toHaveText("切到工作台模式");
    await expect(page.getByRole("heading", { name: "工作台面板" })).toBeHidden();

    await page.getByTestId("ui-mode-toggle").click();

    await expect(page.locator(".page-shell")).toHaveAttribute("data-ui-mode", "pro");
    await expect(page.getByTestId("ui-mode-toggle")).toHaveText("切到写作模式");
    await expect(page.getByRole("heading", { name: "工作台面板" })).toBeVisible();
  });

  test("shell interactions remain healthy", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { name: "AI 辅助写作工作台" })).toBeVisible();

    await page.getByRole("button", { name: "写作设置" }).click();
    const settingsDialog = page.locator("#settings-dialog");
    await expect(settingsDialog).toBeVisible();
    await settingsDialog.getByRole("button", { name: "关闭写作设置" }).click();
    await expect(settingsDialog).toBeHidden();

    const switchToProMode = page.getByRole("button", { name: "切到工作台模式" });
    if (await switchToProMode.isVisible()) {
      await switchToProMode.click();
    }
    await expect(page.getByRole("heading", { name: "工作台面板" })).toBeVisible();
  });

  test("assistant drawer send flow is not regressed", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: "助手抽屉" }).click();

    const drawer = page.locator("#assistant-drawer");
    await expect(drawer).toHaveAttribute("aria-hidden", "false");

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
    await expect
      .poll(async () => ((await latestAssistantMessage.textContent()) ?? "").trim().length, { timeout: 45_000 })
      .toBeGreaterThan(0);
  });
});
