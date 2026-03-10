import { memo, useMemo, useState, type MouseEvent, type FocusEvent } from "react";
import type { SettingEntry, StoryCard, UiMessage } from "../../types";
import { buildContextXRaySegments } from "./contextXRay";
import { ContextXRayPopover } from "./ContextXRayPopover";

type ActivePopoverState = {
  anchorRect: DOMRect;
  canonical: string;
  badge: string;
  excerpt: string;
} | null;

type ContextXRayMessageProps = {
  message: UiMessage;
  settings: SettingEntry[];
  cards: StoryCard[];
};

export const ContextXRayMessage = memo(function ContextXRayMessage({
  message,
  settings,
  cards,
}: ContextXRayMessageProps) {
  const [activePopover, setActivePopover] = useState<ActivePopoverState>(null);
  const segments = useMemo(() => buildContextXRaySegments(message, settings, cards), [cards, message, settings]);

  const openPopover = (
    event: MouseEvent<HTMLSpanElement> | FocusEvent<HTMLSpanElement>,
    canonical: string,
    badge: string,
    excerpt: string
  ) => {
    setActivePopover({
      anchorRect: event.currentTarget.getBoundingClientRect(),
      canonical,
      badge,
      excerpt,
    });
  };

  return (
    <>
      <pre className="context-xray-message">
        {segments.map((segment, index) => {
          if (segment.kind === "text") {
            return <span key={`text-${index}`}>{segment.text}</span>;
          }
          return (
            <span
              key={`token-${index}-${segment.text}`}
              className="context-xray-token"
              data-context-xray-source={segment.binding.source}
              tabIndex={0}
              onMouseEnter={(event) =>
                openPopover(event, segment.binding.canonical, segment.binding.sourceLabel, segment.binding.excerpt)
              }
              onFocus={(event) =>
                openPopover(event, segment.binding.canonical, segment.binding.sourceLabel, segment.binding.excerpt)
              }
              onMouseLeave={() => setActivePopover(null)}
              onBlur={() => setActivePopover(null)}
            >
              {segment.text}
            </span>
          );
        })}
      </pre>
      <ContextXRayPopover
        anchorRect={activePopover?.anchorRect ?? null}
        canonical={activePopover?.canonical ?? ""}
        badge={activePopover?.badge ?? ""}
        excerpt={activePopover?.excerpt ?? ""}
      />
    </>
  );
});
