# Context X-Ray Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add message-level evidence provenance and hover popovers so authors can inspect which DSL/GRAPH context an assistant reply term came from.

**Architecture:** Persist a lightweight `context_xray` payload on assistant chat messages at stream time, expose it through the session messages API, and render assistant message text through a dedicated Context X-Ray component that resolves entity bindings from message evidence first and project DSL/GRAPH fallback second.

**Tech Stack:** Python, FastAPI, SQLModel, Pydantic v2, React, TypeScript, Zustand, Vitest, unittest

---

### Task 1: Lock the backend message provenance contract with tests

**Files:**
- Modify: `apps/api/tests/test_chat_stream_endpoint_unittest.py`
- Modify: `apps/api/tests/test_chat_endpoint_edges_unittest.py`

**Step 1: Write the failing stream persistence test**

```python
def test_chat_stream_persists_context_xray_provenance_on_assistant_message(self) -> None:
    response = self.client.post("/api/chat/stream", json=payload)
    ...
    with Session(self.engine) as db:
        rows = list_messages(db, session_id)
        assistant = rows[-1]
        self.assertIn("context_xray", assistant.provenance)
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest apps.api.tests.test_chat_stream_endpoint_unittest.ChatStreamEndpointTestCase.test_chat_stream_persists_context_xray_provenance_on_assistant_message -v`  
Expected: FAIL with missing `provenance` field or missing `context_xray`

**Step 3: Write the failing session messages API test**

```python
def test_session_messages_returns_context_xray_payload(self) -> None:
    response = self.client.get(f"/api/chat/sessions/{session_id}/messages", headers=self.auth_headers)
    body = response.json()
    assert body[-1]["context_xray"]["evidence"]["summary"]["dsl"] == 1
```

**Step 4: Run test to verify it fails**

Run: `python -m unittest apps.api.tests.test_chat_endpoint_edges_unittest.ChatEndpointEdgesTestCase.test_session_messages_returns_context_xray_payload -v`  
Expected: FAIL because response model does not include `context_xray`

**Step 5: Commit**

```bash
git add apps/api/tests/test_chat_stream_endpoint_unittest.py apps/api/tests/test_chat_endpoint_edges_unittest.py
git commit -m "test: lock chat context xray provenance contract"
```

---

### Task 2: Add persistent provenance support to chat messages

**Files:**
- Modify: `apps/api/app/models/chat.py`
- Modify: `apps/api/app/schemas/chat.py`
- Modify: `apps/api/app/services/chat_service.py`

**Step 1: Write minimal model and schema support**

```python
class ChatMessage(SQLModel, table=True):
    ...
    provenance: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class ChatMessageRead(BaseModel):
    ...
    context_xray: dict | None = None
```

**Step 2: Extend message service helpers**

```python
def append_message(..., provenance: dict[str, Any] | None = None) -> ChatMessage:
    msg = ChatMessage(..., provenance=provenance or {})


def update_message_provenance(message_id: int, provenance: dict[str, Any], *, db: Session | None = None) -> None:
    ...
```

**Step 3: Map stored provenance into API shape**

```python
def _serialize_context_xray(msg: ChatMessage) -> dict[str, Any] | None:
    raw = msg.provenance.get("context_xray")
    return raw if isinstance(raw, dict) else None
```

**Step 4: Run focused backend tests**

Run: `python -m unittest apps.api.tests.test_chat_stream_endpoint_unittest.ChatStreamEndpointTestCase.test_chat_stream_persists_context_xray_provenance_on_assistant_message -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add apps/api/app/models/chat.py apps/api/app/schemas/chat.py apps/api/app/services/chat_service.py
git commit -m "feat: persist context xray provenance on chat messages"
```

---

### Task 3: Store stream evidence on assistant messages

**Files:**
- Modify: `apps/api/app/api/endpoints/chat_stream_pipeline.py`
- Modify: `apps/api/tests/test_chat_stream_endpoint_unittest.py`

**Step 1: Write the failing stream update test**

```python
def test_chat_stream_attaches_compiled_evidence_to_assistant_message(self) -> None:
    events = list(read_sse_events(response))
    ...
    self.assertEqual(assistant.provenance["context_xray"]["evidence"]["summary"]["graph"], 1)
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest apps.api.tests.test_chat_stream_endpoint_unittest.ChatStreamEndpointTestCase.test_chat_stream_attaches_compiled_evidence_to_assistant_message -v`  
Expected: FAIL because stream pipeline never writes evidence into message provenance

**Step 3: Implement the minimal pipeline change**

```python
assistant_msg = append_message(...)
update_message_provenance(
    assistant_msg.id,
    {
        "context_xray": {
            "version": 1,
            "evidence": compiled_bundle.evidence_event,
        }
    },
    db=db,
)
```

**Step 4: Run the focused stream test**

Run: `python -m unittest apps.api.tests.test_chat_stream_endpoint_unittest.ChatStreamEndpointTestCase.test_chat_stream_attaches_compiled_evidence_to_assistant_message -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add apps/api/app/api/endpoints/chat_stream_pipeline.py apps/api/tests/test_chat_stream_endpoint_unittest.py
git commit -m "feat: attach context xray evidence during chat stream"
```

---

### Task 4: Expose provenance through the web API and frontend types

**Files:**
- Modify: `apps/web/src/types.ts`
- Modify: `apps/web/src/api/chatApi.ts`
- Modify: `apps/web/src/App.tsx`

**Step 1: Write the failing frontend type/test coverage**

```ts
it("preserves contextXRay when mapping ChatMessageDto to UiMessage", () => {
  const dto = {
    id: 7,
    role: "assistant",
    content: "神剑已觉醒",
    context_xray: { version: 1, evidence: sampleEvidence },
  };
  expect(toUiMessage(dto).contextXRay?.evidence?.summary.dsl).toBe(1);
});
```

**Step 2: Run test to verify it fails**

Run: `npm test -- chatStore.uiMode.test.ts --runInBand`  
Expected: FAIL because DTO / UI message types do not include `context_xray`

**Step 3: Add DTO and UI types**

```ts
export interface ContextXRayPayload {
  version: number;
  evidence: EvidencePayload | null;
}

export interface UiMessage {
  ...
  contextXRay?: ContextXRayPayload | null;
}
```

**Step 4: Map API messages into UI messages**

```ts
function toUiMessage(message: ChatMessageDto): UiMessage {
  return {
    ...,
    contextXRay: message.context_xray ?? null,
  };
}
```

**Step 5: Commit**

```bash
git add apps/web/src/types.ts apps/web/src/api/chatApi.ts apps/web/src/App.tsx
git commit -m "feat: expose context xray payload in web message types"
```

---

### Task 5: Preserve message-level evidence during streaming

**Files:**
- Modify: `apps/web/src/store/chatStore.ts`
- Modify: `apps/web/src/hooks/useAssistantSessionFlow.ts`
- Test: `apps/web/src/store/chatStore.uiMode.test.ts`

**Step 1: Write the failing stream message update test**

```ts
it("attaches evidence to the current assistant message when an evidence event arrives", () => {
  ...
  expect(message.contextXRay?.evidence?.summary.graph).toBe(2);
});
```

**Step 2: Run test to verify it fails**

Run: `npm test -- chatStore.uiMode.test.ts --runInBand`  
Expected: FAIL because evidence is only stored globally

**Step 3: Add the minimal message updater**

```ts
if (event.type === "evidence") {
  setEvidence(event);
  updateMessage(assistantLocalId, (message) => ({
    ...message,
    contextXRay: { version: 1, evidence: event },
  }));
}
```

**Step 4: Run the focused frontend test**

Run: `npm test -- chatStore.uiMode.test.ts --runInBand`  
Expected: PASS

**Step 5: Commit**

```bash
git add apps/web/src/store/chatStore.ts apps/web/src/hooks/useAssistantSessionFlow.ts apps/web/src/store/chatStore.uiMode.test.ts
git commit -m "feat: retain context xray evidence on streaming assistant messages"
```

---

### Task 6: Build the Context X-Ray message renderer

**Files:**
- Create: `apps/web/src/components/chat/contextXRay.ts`
- Create: `apps/web/src/components/chat/ContextXRayMessage.tsx`
- Create: `apps/web/src/components/chat/ContextXRayPopover.tsx`
- Modify: `apps/web/src/App.tsx`
- Modify: `apps/web/src/styles.css`

**Step 1: Write the failing renderer test**

```tsx
it("renders matched assistant entities with a context xray trigger", () => {
  render(<ContextXRayMessage message={message} settings={settings} cards={cards} />);
  expect(screen.getByText("神剑")).toHaveAttribute("data-context-xray-source", "evidence");
});
```

**Step 2: Run test to verify it fails**

Run: `npm test -- ContextXRayMessage.test.tsx --runInBand`  
Expected: FAIL because the component does not exist

**Step 3: Implement entity resolution helpers**

```ts
export function buildContextXRayBindings(message: UiMessage, settings: SettingEntry[], cards: StoryCard[]) {
  // message evidence first, project fallback second
}
```

**Step 4: Implement the renderer and popover**

```tsx
<span
  className="context-xray-token"
  data-context-xray-source={binding.source}
  tabIndex={0}
>
  {binding.text}
</span>
```

**Step 5: Replace raw `<pre>{message.content}</pre>` in the assistant drawer**

```tsx
{message.role === "assistant" ? (
  <ContextXRayMessage message={message} settings={settings} cards={cards} />
) : (
  <pre>{message.content}</pre>
)}
```

**Step 6: Run the focused renderer test**

Run: `npm test -- ContextXRayMessage.test.tsx --runInBand`  
Expected: PASS

**Step 7: Commit**

```bash
git add apps/web/src/components/chat/contextXRay.ts apps/web/src/components/chat/ContextXRayMessage.tsx apps/web/src/components/chat/ContextXRayPopover.tsx apps/web/src/App.tsx apps/web/src/styles.css
git commit -m "feat: add context xray hover rendering for assistant messages"
```

---

### Task 7: Verify fallback behavior and regressions

**Files:**
- Test: `apps/api/tests/test_chat_stream_endpoint_unittest.py`
- Test: `apps/api/tests/test_chat_endpoint_edges_unittest.py`
- Test: `apps/web/src/store/chatStore.uiMode.test.ts`
- Test: `apps/web/src/components/chat/ContextXRayMessage.test.tsx`

**Step 1: Run focused backend tests**

Run: `python -m unittest apps.api.tests.test_chat_stream_endpoint_unittest apps.api.tests.test_chat_endpoint_edges_unittest -v`  
Expected: PASS

**Step 2: Run focused frontend tests**

Run: `npm test -- chatStore.uiMode.test.ts ContextXRayMessage.test.tsx --runInBand`  
Expected: PASS

**Step 3: Run the web build**

Run: `npm run build --prefix apps/web`  
Expected: PASS

**Step 4: Summarize fallback semantics**

```text
"本轮引用" only when the matched token is backed by this message's evidence payload.
"设定回退" only when message evidence misses and the project DSL/GRAPH index provides support.
```

**Step 5: Commit**

```bash
git add apps/api/tests/test_chat_stream_endpoint_unittest.py apps/api/tests/test_chat_endpoint_edges_unittest.py apps/web/src/store/chatStore.uiMode.test.ts apps/web/src/components/chat/ContextXRayMessage.test.tsx
git commit -m "test: verify context xray fallback behavior"
```
