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
      className="rounded-lg bg-surface-elevated border border-border-default shadow-lg p-3 text-sm z-50"
      role="tooltip"
      style={{
        position: "fixed",
        top,
        left: popoverLeft,
        width,
      }}
    >
      <div className="inline-block rounded-full bg-accent-secondary text-accent-primary text-xs font-medium px-2 py-0.5 mb-1">{badge}</div>
      <strong>{canonical}</strong>
      <p>{excerpt}</p>
    </div>,
    document.body
  );
}
