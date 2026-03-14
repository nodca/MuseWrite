import { useChatStore } from "./chatStore";

function expectEqual<T>(label: string, actual: T, expected: T) {
  if (actual !== expected) {
    throw new Error(`[chatStore workspaceShell] ${label}: expected ${String(expected)}, got ${String(actual)}`);
  }
}

function resetStore() {
  useChatStore.setState({
    assistantDrawerOpen: false,
    advancedPanelOpen: false,
    assistantSection: "planning",
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
  expectEqual("default assistant drawer", initial.assistantDrawerOpen, false);
  expectEqual("default advanced panel", initial.advancedPanelOpen, false);
  expectEqual("default assistant section", initial.assistantSection, "planning");

  initial.setAssistantDrawerOpen(true);
  expectEqual("open assistant drawer", useChatStore.getState().assistantDrawerOpen, true);

  initial.setAdvancedPanelOpen(true);
  expectEqual("open advanced panel", useChatStore.getState().advancedPanelOpen, true);

  initial.setAssistantSection("chat");
  expectEqual("switch assistant section", useChatStore.getState().assistantSection, "chat");

  console.log("chatStore workspace shell test passed");
}

run();
