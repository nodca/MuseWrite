import { createPortal } from "react-dom";

type ContextXRayPopoverProps = {
  anchorRect: DOMRect | null;
  canonical: string;
  badge: string;
  excerpt: string;
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

export function ContextXRayPopover({
  anchorRect,
  canonical,
  badge,
  excerpt,
}: ContextXRayPopoverProps) {
  if (!anchorRect || typeof document === "undefined") return null;
  const width = 320;
  const gutter = 12;
  const viewportWidth = typeof window !== "undefined" ? window.innerWidth : width + gutter * 2;
  const popoverLeft = clamp(anchorRect.left + anchorRect.width / 2 - width / 2, gutter, viewportWidth - width - gutter);
  const showAbove = anchorRect.top > 140;
  const top = showAbove ? anchorRect.top - 12 : anchorRect.bottom + 12;

  return createPortal(
    <div
      className={`context-xray-popover ${showAbove ? "above" : "below"}`}
      role="tooltip"
      style={{
        position: "fixed",
        top,
        left: popoverLeft,
        width,
      }}
    >
      <div className="context-xray-popover-badge">{badge}</div>
      <strong>{canonical}</strong>
      <p>{excerpt}</p>
    </div>,
    document.body
  );
}
