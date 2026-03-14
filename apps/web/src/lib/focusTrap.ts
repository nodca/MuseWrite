export const FOCUSABLE_SELECTOR = [
  "a[href]",
  "area[href]",
  "button:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  '[tabindex]:not([tabindex="-1"])',
].join(",");

export function getFocusableElements(container: HTMLElement): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter((element) => {
    if (element.getAttribute("aria-hidden") === "true") return false;
    if (element.hasAttribute("disabled")) return false;
    if (element.tabIndex < 0) return false;
    return element.getClientRects().length > 0;
  });
}

export function focusFirstDialogElement(container: HTMLElement): void {
  const preferred = container.querySelector<HTMLElement>("[data-autofocus]");
  if (preferred && !preferred.hasAttribute("disabled") && preferred.getClientRects().length > 0) {
    preferred.focus();
    return;
  }
  const firstField = container.querySelector<HTMLElement>(
    "input:not([disabled]), select:not([disabled]), textarea:not([disabled])"
  );
  if (firstField) {
    firstField.focus();
    return;
  }
  const focusables = getFocusableElements(container);
  if (focusables.length > 0) {
    focusables[0].focus();
    return;
  }
  container.focus();
}
