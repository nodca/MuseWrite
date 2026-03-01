# Novel 平台 v1 执行基线（冻结版）

日期：2026-02-27  
状态：`frozen-v1`（本文件是执行唯一基线）

## 1. 目标（一句话）

做一个 **AI辅助写作平台**：人写正文，AI 提建议、提动作、提上下文，**AI 不直接接管写作**。

## 2. 必做能力（MUST）

1. 卡片/Schema 体系（类型、实例、字段校验）。
2. 编辑器写作闭环（建议 -> 采纳/拒绝 -> 保存）。
3. AI 聊天控制中枢（聊天+动作提议）。
4. 动作安全流：`proposed -> apply -> undo`（`apply_requested` 与 `applied` 分别记录审计事件）。
5. 三合一上下文：`DSL + GRAPH + LightRAG`。
6. POV 认知沙箱（含 POV 切换桥接记忆重置）。
7. Prompt 工坊 + 知识库注入。
8. 助手工具链（读卡/建卡/改卡/删卡/移动卡，危险操作需确认）。
9. 灵感工作台（自由卡片区 + 回流正式项目）。
10. 工作流系统（触发器、执行、暂停/恢复/取消、运行记录）。
11. Workflow Agent（自然语言改工作流，预览/校验/应用）。
12. Docker 部署（web/api/worker/postgres/neo4j/redis）。

## 3. 明确不做（DROP）

1. 自动整章 pipeline 生成。
2. 诊断中心（Consistency/Diagnostic Center）。
3. 桌面本地版本（Electron）。
4. AI 绕过确认直接写库。

## 4. 固定技术栈（不再摇摆）

- 前端：React + Vite + TypeScript + Tiptap + Zustand
- 后端：FastAPI + SQLModel + Alembic
- 数据：PostgreSQL + Neo4j + LightRAG
- 异步：Redis + Worker
- 部署：Docker Compose + Nginx/Caddy

### 4.1 当前实现说明（2026-02-27）

1. Web 正文编辑器已接入 `Tiptap`，替代早期 `contenteditable` 过渡实现。
2. 当前保留“纯正文优先”的交互策略（章节正文、选区润色/扩写、助手写回），并通过模板注入面板补齐 Prompt + 知识注入最小闭环。
3. 后续编辑器能力扩展（Ghost Text、差异层、复杂富文本节点）应继续围绕 Tiptap Extension 机制演进，避免退回 DOM 直改模式。

## 5. 质量红线

1. 冲突裁决基线固定：`DSL > GRAPH > RAG`；同层必须再按 `freshness + confidence + relevance` 排序，旧事实不得硬压新事实。
2. 写入操作必须带 `operator_id` 和审计日志。
3. 异步图谱写入必须执行一致性协议：`mutation_id + graph_job_idempotency_key + expected_version + compensation`。
4. 所有 AI 动作必须可撤销（至少一步），且撤销后不得残留晚到图谱写入副作用。
5. RAG 默认走 `mix`，并允许按请求临时切换 `local/global/hybrid/mix`，route reason 必须可观测。
5.1 精确字段查询支持显式“事实短路”（`deterministic_first=true`）：优先 DSL/GRAPH，命中达阈值可跳过 RAG。
6. Reranker 与 citation 结果必须可观测并纳入 DoD；默认观测优先，强约束由开关控制。
7. 文档删除必须触发索引/图谱重建或补偿删除，禁止只删文本不删索引。
8. 每次改检索策略都要跑同一评测集（RAGAS 离线回归 gate）。
9. 图谱权威事实源固定为 Neo4j；LightRAG 作为召回与候选图来源，不保留并行最终真相。
10. Langfuse 追踪（trace/span/cost/latency/retrieval params）必须在 Phase A/B 接入，不后置到运营期。

## 6. 分阶段验收门槛

### Phase A（基础可用）

- 写作闭环可用。
- 聊天动作闭环可用。
- 审计/撤销可用。

### Phase B（生产力）

- 灵感工作台可用。
- 助手会话历史/跨项目引用/模型覆盖可用。
- 右键润色扩写 + Ghost Text 可用。

### Phase C（一致性）

- 三合一上下文上线。
- POV 沙箱全链路生效。
- 图谱抽取/入图/更新可用。
- 查询路由策略（`local/global/hybrid/mix`）可观测、可切换、可回归验证。
- reranker/citation 开启策略可验证，命中质量纳入验收指标。
- 删除文档后的索引与图谱重建链路可用。

### Phase D（自动化）

- 工作流编辑/执行/恢复可用。
- Workflow Agent 预览校验应用闭环可用。

### Phase E（运营）

- 模型管理、成本统计、性能观测可用。
- 单机多项目隔离和最小 RBAC 可用。

## 7. 执行规则

1. 功能范围看 `fusion-matrix`，开发顺序看 `execution-board`。
2. 本文件只改“方向与边界”，不写实现细节。
3. 任何新增需求必须先标记到矩阵再进看板。
4. 优先复用 LightRAG 原生能力（PostgreSQL 存储、WebUI、删除重建、RAGAS、Langfuse），避免重复造轮子。

## 8. 配套文档

- [三项目融合矩阵](./2026-02-27-fusion-matrix.md)
- [v1 执行看板](./2026-02-27-v1-execution-board.md)
