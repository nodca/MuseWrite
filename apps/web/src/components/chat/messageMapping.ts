import type { ChatMessageDto, EvidencePayload, UiMessage } from "../../types";

export function toUiMessage(message: ChatMessageDto): UiMessage {
  return {
    id: `msg-${message.id}`,
    role: message.role,
    content: message.content,
    contextXRay: message.context_xray ?? null,
  };
}

export function attachEvidenceToUiMessage(message: UiMessage, evidence: EvidencePayload): UiMessage {
  return {
    ...message,
    contextXRay: {
      version: 1,
      evidence,
    },
  };
}
