import { memo } from "react";
import clsx from "clsx";
import { formatJson, formatRole } from "../utils/formatting";
import { ContextXRayMessage } from "./chat/ContextXRayMessage";
import type { SettingEntry, StoryCard, UiMessage } from "../types";

export type AssistantChatPanelProps = {
  usage: Record<string, unknown> | null;
  messages: UiMessage[];
  settings: SettingEntry[];
  cards: StoryCard[];
  input: string;
  streaming: boolean;
  composerInputRef: { current: HTMLTextAreaElement | null };
  setInput: (value: string) => void;
  onSend: () => Promise<void>;
};

export const AssistantChatPanel = memo(function AssistantChatPanel({
  usage,
  messages,
  settings,
  cards,
  input,
  streaming,
  composerInputRef,
  setInput,
  onSend,
}: AssistantChatPanelProps) {
  return (
    <section className="flex flex-col min-h-0">
      {/* Panel header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border-default shrink-0">
        <h2 className="text-sm font-semibold text-text-primary">聊天流</h2>
        {usage ? (
          <small className="text-xs text-text-tertiary truncate max-w-[200px]">{formatJson(usage)}</small>
        ) : (
          <small className="text-xs text-text-tertiary">尚未返回 usage</small>
        )}
      </div>

      {/* Chat log — keep .chat-log and article.msg.{role} classes for E2E */}
      <div className="chat-log flex-1 overflow-y-auto p-4 space-y-3">
        {messages.length === 0 ? (
          <p className="text-sm text-text-tertiary text-center py-8">
            先发一条消息，验证 SSE + 动作提议。
          </p>
        ) : null}
        {messages.map((message) => (
          <article
            key={message.id}
            className={clsx(
              "msg rounded-lg p-3 text-sm",
              message.role,
              message.role === "user"
                ? "bg-accent-secondary/50 ml-8"
                : "bg-surface-elevated mr-8",
            )}
          >
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-medium text-text-secondary">{formatRole(message.role)}</span>
              {message.streaming ? (
                <em className="text-xs text-accent-primary animate-pulse">streaming...</em>
              ) : null}
            </div>
            {message.role === "assistant" ? (
              <ContextXRayMessage message={message} settings={settings} cards={cards} />
            ) : (
              <pre className="whitespace-pre-wrap break-words text-text-primary text-sm font-sans">
                {message.content}
              </pre>
            )}
          </article>
        ))}
      </div>

      {/* Composer — keep .composer class for E2E */}
      <div className="composer border-t border-border-default p-4 shrink-0">
        <textarea
          aria-label="给助手的输入"
          ref={composerInputRef}
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="例：请补充设定，主角的第一动机是复仇但不能越过底线。"
          rows={4}
          disabled={streaming}
          className="w-full rounded-lg border border-border-default bg-surface-base px-3 py-2 text-sm text-text-primary placeholder:text-text-tertiary resize-none focus:outline-none focus:ring-2 focus:ring-accent-primary/30"
        />
        <button
          type="button"
          className="mt-2 w-full rounded-lg bg-accent-primary px-4 py-2 text-sm font-medium text-white hover:bg-accent-primary-hover disabled:opacity-40 transition-colors"
          onClick={() => void onSend()}
          disabled={streaming || !input.trim()}
        >
          {streaming ? "发送中..." : "发送"}
        </button>
      </div>
    </section>
  );
});
