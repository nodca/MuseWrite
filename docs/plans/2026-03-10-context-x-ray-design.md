# Context X-Ray 设计稿

日期：2026-03-10

## 背景

当前聊天侧边栏虽然已经有统一的 `evidence` 数据流，但作者若想知道 AI 回复里的某个名词到底引用了哪条设定，仍需切到工作台手动翻看。这个链路割裂了“生成结果”和“引用依据”，信任建立成本过高。

`Context X-Ray` 的目标，是让作者直接在聊天消息里 hover 某个实体词，立即看到该词在本轮上下文中的设定依据原文，并明确区分“本轮真实引用”与“项目设定回退”。

## 目标

- 让 `assistant` 回复中的关键实体支持 hover/focus 显影。
- 优先显示该条消息当轮命中的 `evidence` 原文。
- 若该条消息的 `evidence` 中没有命中，再回退到项目 DSL / GRAPH 设定。
- 历史消息、刷新页面、切换会话后仍能复现原消息的依据。
- 不引入泛化 NER，不做无法解释的“猜测式高亮”。

## 非目标

- 不解释模型完整推理链，不生成“为什么这样想”的自然语言说明。
- 不在首版支持多依据对比面板、人工修正绑定、证据排序配置。
- 不对 `user` / `system` 消息做相同级别的显影。
- 不为每个词在后端预计算并持久化最终绑定结果。

## 方案对比

### 方案 A：纯前端即时显影

前端直接拿当前全局 `evidence + settings/cards`，对聊天消息做 hover 绑定。

优点：

- 开发最快。
- 几乎不改后端。

缺点：

- 历史消息在刷新后无法知道“当时到底引用了什么”。
- 当前全局 `evidence` 会污染旧消息，产生错绑。
- 透明化只是表象，不可审计。

### 方案 B：前端按消息缓存 evidence

流式回复期间，将 `assistantLocalId -> evidence snapshot` 暂存在前端 store，再按消息渲染。

优点：

- 本轮流式体验较好。
- 比全局 `evidence` 更准确。

缺点：

- 会话刷新后失忆。
- 重新拉取历史消息时没有 provenance。
- 无法支撑“可信回放”。

### 方案 C：消息级 provenance + 前端按消息显影

后端把每条 `assistant` 消息对应的 `evidence snapshot` 保存并通过消息列表接口返回；前端基于消息级 provenance 执行实体识别与 Popover 显示。

优点：

- 历史消息与当前消息都能稳定复现依据。
- 避免全局 `evidence` 错绑。
- 最符合“思考透明化”的产品目标。

缺点：

- 需要联动前后端与消息 schema。
- 首版实现成本高于前两种方案。

## 定案

采用方案 C。

核心原则：

1. 透明化建立在“消息级 provenance”之上，而不是“当前页面状态”之上。
2. 绑定优先级固定为“本条消息 evidence 优先，缺失时回退项目 DSL / GRAPH”。
3. 仅显影显式实体，不高亮泛词、代词、普通名词。

## 数据模型

### 后端消息模型

现有 `ChatMessage` 只保存：

- `id`
- `session_id`
- `role`
- `content`
- `model`
- `created_at`

需要扩展一个 JSON 字段，例如：

```python
provenance: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
```

该字段首版只承载消息级上下文依据，不承载最终词级绑定结果。

推荐结构：

```json
{
  "context_xray": {
    "version": 1,
    "evidence": {
      "policy": {},
      "summary": {},
      "sources": {
        "dsl": [],
        "graph": [],
        "rag": []
      }
    }
  }
}
```

### API schema

`ChatMessageRead` 与前端 `ChatMessageDto` / `UiMessage` 需要新增可选字段：

```ts
context_xray?: {
  version: number;
  evidence: EvidencePayload | null;
};
```

说明：

- `ChatMessageDto` 保持对后端返回结构一一映射。
- `UiMessage` 在前端渲染层持有同名字段或归一后的 `contextXRay` 字段。
- `user` / `system` 消息默认该字段为空。

## 流式写入链路

### 当前状态

当前流式过程里：

1. 后端创建 assistant message。
2. SSE 先推送 `meta`。
3. 再推送 `evidence`。
4. 最后持续推送 `delta`。

前端只把 `evidence` 存进全局 store，消息本身没有绑定它自己的依据。

### 目标链路

改造后链路：

1. 后端在创建 assistant message 后，将本轮 `evidence_event` 摘要写入该消息的 `provenance.context_xray.evidence`。
2. SSE 仍继续发送 `evidence` 事件，供工作台与当前流式体验使用。
3. 前端在流式阶段收到 `evidence` 后，也同步写入当前 `assistantLocalId` 对应的 `UiMessage.contextXRay`，确保消息在流式中就能显影。
4. 页面刷新或切会话后，`getSessionMessages()` 返回的历史消息仍带上同一份 provenance。

这样可以同时兼顾：

- 当前轮流式体验。
- 历史消息稳定回放。

## 前端渲染设计

### 组件拆分

当前消息内容在 `App.tsx` 中以：

```tsx
<pre>{message.content}</pre>
```

直接渲染。

改造后应拆为独立组件，例如：

- `ChatMessageContent`
- `ContextXRayMessage`
- `ContextXRayPopover`

职责建议：

- `ChatMessageContent`：按消息角色选择普通文本或显影文本。
- `ContextXRayMessage`：根据消息内容与消息级 provenance 生成可 hover 的实体 span。
- `ContextXRayPopover`：负责浮窗定位、边界翻转、可访问性和展示文案。

### 识别策略

首版不引入通用 NER，而是采用“知识库反向匹配”：

1. 从本条消息的 `evidence.sources.dsl` 中提取 `title / snippet / alias-like fields`。
2. 从本条消息的 `evidence.sources.graph` 中提取 `title / fact` 中的实体名。
3. 若本条消息 evidence 中找不到，则回退到项目 `settings/cards` 中可归一的实体名与 aliases。
4. 生成 token lookup 表后，对消息文本执行“长词优先”的字符串匹配。

过滤规则：

- 过滤长度过短的 token。
- 过滤常见泛词、纯数字、纯标点。
- 过滤代词与高歧义常用词。
- 同一位置若多词竞争，选择长度更长、优先级更高、来源更可信的候选。

### 绑定优先级

每个命中的实体 token 绑定来源按以下优先级确定：

1. 本条消息 `evidence` 的 DSL 命中
2. 本条消息 `evidence` 的 GRAPH 命中
3. 本条消息 `evidence` 的 RAG 命中
4. 项目 DSL / GRAPH 回退命中

若多个来源都命中，同词只展示一个主绑定，避免首版交互过载。

## Popover 交互规则

### 触发方式

- 鼠标 hover 时弹出。
- 键盘 focus 时同样弹出。
- 失焦或 pointer 离开后关闭。

### 展示内容

Popover 首版固定三层：

1. 实体 canonical 名
2. 来源标签
3. 摘录原文

来源标签文案：

- `本轮引用`：命中本条消息 evidence
- `设定回退`：未在本条消息 evidence 命中，但在项目 DSL / GRAPH 中找到支持

### 多依据策略

若同一词可命中多条记录：

- 首版只展示 1 条主依据
- 若存在额外命中，显示“另有 N 条依据”
- 不在首版展开二级列表

### 可访问性

- 可 hover 的实体 span 需要支持 `tabIndex=0`
- Popover 需要 `role="tooltip"`
- 使用 `aria-describedby` 或等效属性进行关联

## 样式与性能

### 样式原则

正文编辑器中已有 `entity-inline-hint` 样式，但聊天抽屉存在滚动容器，直接复用伪元素 tooltip 容易被裁切。

因此首版建议：

- 消息文本中的实体使用轻量高亮样式
- Popover 使用受控浮层组件，挂在聊天消息 DOM 层
- 支持边界翻转与顶部/底部定位

实体视觉建议：

- 使用较轻的 underline + 微弱底色
- hover 时加深底色
- 避免整段文本大面积高亮

### 性能边界

需要按以下维度 memo：

- `message.content`
- `message.contextXRay.evidence`
- `settings/cards` 提取出的 fallback 索引版本

避免每次输入、每次消息追加时重扫整个会话历史。

## 后端改动点

建议涉及文件：

- `apps/api/app/models/chat.py`
- `apps/api/app/schemas/chat.py`
- `apps/api/app/services/chat_service.py`
- `apps/api/app/api/endpoints/chat_actions.py`
- `apps/api/app/api/endpoints/chat_stream_pipeline.py`

改动目标：

- `ChatMessage` 增加 provenance JSON 字段。
- `append_message()` 支持写入 provenance。
- 增加更新消息 provenance 的服务方法，供流式链路补齐。
- `ChatMessageRead` 输出 provenance/context_xray。
- `session_messages()` 返回扩展后的消息结构。
- 在流式聊天管道中，将 `compiled_bundle.evidence_event` 归档到 assistant message provenance。

## 前端改动点

建议涉及文件：

- `apps/web/src/types.ts`
- `apps/web/src/api/chatApi.ts`
- `apps/web/src/store/chatStore.ts`
- `apps/web/src/hooks/useAssistantSessionFlow.ts`
- `apps/web/src/App.tsx`
- `apps/web/src/styles.css`

可选新增：

- `apps/web/src/components/chat/ContextXRayMessage.tsx`
- `apps/web/src/components/chat/ContextXRayPopover.tsx`
- `apps/web/src/components/chat/contextXRay.ts`

改动目标：

- 扩展消息 DTO / UI 类型。
- 流式阶段把 `evidence` 写入当前 assistant message。
- 刷新会话时从后端恢复历史消息 provenance。
- 将消息文本渲染拆成可复用显影组件。
- 增加聊天区 Popover 样式和边界处理。

## 测试与验收

### 后端

建议新增或扩展测试：

- `apps/api/tests/test_chat_stream_endpoint_unittest.py`
- `apps/api/tests/test_chat_endpoint_edges_unittest.py`

覆盖点：

- 流式聊天后 assistant message 保存了 `context_xray.evidence`
- `/sessions/{session_id}/messages` 返回 provenance
- `user` 消息不意外携带 assistant-only provenance

### 前端

建议新增或扩展测试：

- `apps/web/src/store/chatStore.uiMode.test.ts`
- 新增 `apps/web/src/components/chat/*.test.tsx`

覆盖点：

- 流式过程中收到 `evidence` 后当前 assistant message 获得 `contextXRay`
- 历史消息渲染时优先使用消息级 evidence
- fallback 命中时标签为“设定回退”
- 无匹配词不高亮

### 手工验收

1. 发送一条包含明确实体的请求。
2. 等待 assistant 回复完成。
3. hover 回复中的实体词，看到来源标签与原文摘录。
4. 刷新页面后再次 hover，结果保持一致。
5. 切到其他会话再切回，结果保持一致。
6. 对普通词、代词、泛词 hover 时无显影。

## 风险

### 风险 1：历史消息错绑

若仍依赖全局 `evidence`，旧消息会被当前状态污染。

规避：

- 必须保存消息级 provenance。

### 风险 2：高亮泛滥

若 token 过滤不严，聊天消息会满屏高亮。

规避：

- 长词优先
- 过滤短词和歧义词
- 首版仅处理强实体

### 风险 3：浮窗被滚动容器裁切

规避：

- 使用受控 Popover，而不是纯 CSS 伪元素 tooltip。

### 风险 4：消息 provenance 过重

若直接保存整份过大的 evidence，会抬高消息表体积。

规避：

- 首版先保存必要字段
- 后续可引入裁剪或外部引用方案

## 分阶段落地建议

### Phase 1：后端持久化与前端类型打通

- 消息模型增加 provenance
- 流式链路写入消息级 evidence
- 历史消息接口返回 provenance
- 前端类型与 store 能消费该字段

### Phase 2：聊天消息显影组件

- 拆出消息显影组件
- 完成 evidence 优先与 fallback 绑定
- 接入 Popover

### Phase 3：过滤、样式与测试补强

- 优化实体过滤策略
- 修复滚动与定位边界
- 补全单测与构建验证

## 结论

`Context X-Ray` 的真正价值不在 tooltip，而在“让某条 AI 回复能够长期、稳定、可回放地证明自己引用了什么”。因此最优雅且最稳的方案，是为 assistant message 建立消息级 provenance，再在前端按消息做实体显影和 Popover 展示。
