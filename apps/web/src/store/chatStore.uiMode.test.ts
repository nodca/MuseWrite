import { useChatStore } from "./chatStore";

function expectEqual<T>(label: string, actual: T, expected: T) {
  if (actual !== expected) {
    throw new Error(`[chatStore uiMode] ${label}: expected ${String(expected)}, got ${String(actual)}`);
  }
}

function resetStore() {
  useChatStore.setState({
    uiMode: "writing",
    projectId: 1,
    model: "",
    povMode: "global",
    povAnchor: "",
    ragMode: "mix",
    deterministicFirst: false,
    thinkingEnabled: false,
    sessionId: null,
    streaming: false,
    error: null,
    usage: null,
    messages: [],
    actions: [],
    pendingActionIds: [],
    settings: [],
    cards: [],
    selectedActionId: null,
    actionLogs: [],
    evidence: null,
  });
}

function run() {
  resetStore();

  const initial = useChatStore.getState();
  expectEqual("default uiMode", initial.uiMode, "writing");

  initial.setUiMode("pro");
  expectEqual("switch to pro", useChatStore.getState().uiMode, "pro");

  initial.setUiMode("writing");
  expectEqual("switch back to writing", useChatStore.getState().uiMode, "writing");

  console.log("chatStore uiMode test passed");
}

run();
