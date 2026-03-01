# v1 执行看板（可落地）

日期：2026-02-27  
状态：`active`

## Phase A：基础闭环（P0）

1. 聊天流式接口：`POST /api/chat/stream`。
- DoD：前端可实时接收 delta，消息可持久化。

2. 动作接口：`apply/reject/undo`。
- DoD：动作状态流转完整，幂等键生效。

3. 动作审计表。
- DoD：每次动作都记录 `operator_id`、时间、前后值摘要。

4. 编辑器建议闭环。
- DoD：建议可采纳/拒绝，正文保存后可追踪来源。

## Phase B：生产力体验（P0/P1）

1. 灵感工作台（自由卡片）。
- DoD：可创建自由卡并回流正式项目。

2. 助手会话能力。
- DoD：历史会话、跨项目引用、模型覆盖、Thinking 可用。

3. 编辑器增强。
- DoD：右键润色/扩写、Ghost Text 接受/拒绝/重生成功。

4. Prompt + 知识库面板。
- DoD：可创建模板、引用知识库并在请求中生效。

5. Langfuse 追踪接入（前置）。
- DoD：trace/span、token/cost/latency、检索参数（route/provider/top_k）可追踪。

## Phase C：一致性与记忆（P0）

1. 三合一上下文编译器。
- DoD：`DSL + GRAPH + LightRAG` 输出统一 evidence。

2. 冲突裁决。
- DoD：`DSL > GRAPH > RAG` 可观测且可解释，同层包含 freshness/confidence 评分。

3. POV 沙箱。
- DoD：POV 切换触发桥接清空，越权事实不过审。

4. 图谱流水线。
- DoD：抽取 -> 预览 -> 入图 -> 更新完整可用。

5. 查询路由策略。
- DoD：默认固定 `mix`，并记录 route source/reason；支持请求级临时切换。

5.1 事实短路策略（显式开关）。
- DoD：`deterministic_first=true` 时，DSL/GRAPH 命中达阈值可跳过 RAG，并记录 short-circuit reason。

6. 引用与重排质量。
- DoD：reranker 可开关、citation 必须返回来源且可追溯到 chunk/file。

7. 文档生命周期一致性。
- DoD：删除文档可触发索引与图谱重建/补偿删除，无悬挂引用。

8. RAGAS 离线回归（前置）。
- DoD：固定小说评测集、固定评分脚本、每次检索策略改动必须跑回归并出报告。

## 进展更新（2026-02-27）

1. 已落地：三合一上下文编译器（`DSL + Neo4j + LightRAG`）及 evidence 可观测。
2. 已落地：POV 沙箱请求参数与检索侧过滤。
3. 已落地：图谱门控写入第一版（LightRAG 抽取优先 + 规则兜底，`apply` 后入 Neo4j，`undo` 回滚图谱边）。
4. 已落地：图谱异步入库 worker（Redis 队列 + 重试 + 同步兜底）。
5. 已落地：LightRAG v1.4 契约适配（`/query/data` + `hl_keywords/ll_keywords` 旁路关键词抽取）。
6. 已落地：异步图谱一致性协议第一版（`mutation_id + expected_version + job_idempotency_key + compensation`）。
7. 已落地：查询路由策略支持 `local/global/hybrid/mix`（默认 `mix` + 请求临时覆盖 + route source/reason 可观测）。
8. 已落地：citation/reranker 运行时质量门槛（degraded 信号 + usage 标记，默认观测优先不强制改写回答）。
9. 已落地：异步图谱一致性协议第二版（写前/写后/提交前三次 stale 校验 + `graph_skipped_stale`/`graph_compensated` 指标）。
10. 已落地：事实短路开关（DSL/GRAPH 命中阈值后跳过 RAG，减少 mix 噪声）。
11. 已落地：Langfuse 追踪接入第一版（可选开关 + retrieval policy metadata 上报）。
12. 已落地：文档生命周期一致性链路第一版（`setting.delete/undo` -> `index_lifecycle` 队列 -> worker 重建 -> dead-letter 监控）。
13. 已落地：RAGAS 离线回归基线脚本（固定数据集 + 阈值 gate + 报告输出）。
14. 已落地：LightRAG documents 原生接口接入（文本插入/分页/删除/管线状态），平台不再重复实现文档索引删除流程。
15. 已落地：dead-letter 重放运维接口（按项目批量重放 + `index_lifecycle_replayed` 审计事件）。
16. 已落地：RAGAS CI gate（策略相关改动触发，自动准备 fixture 文档后执行回归）。
17. 已落地：评测集冻结清单（manifest + fixture docs + question set 版本化）。
18. 已落地：RAGAS 趋势对比（相对 baseline 输出 `gate_score/averages/subset` 分差）。
19. 已落地：多 POV 子集分数（`global/character`）与样本级 POV 元信息写入报告。
20. 已落地：正文工作区服务端持久化（章节保存 + 版本历史 + 一键回滚）。
21. 已落地：author 风格编辑增强（选中润色/扩写提示注入 + 助手回复写回正文）。
22. 已落地：章节化写作闭环（章节列表/切换/新建 + 章节保存 + 章节历史回滚）。
23. 已落地：章节管理操作（章节上移/下移重排 + 删除章节 + 删除后自动选中可编辑章节）。
24. 已落地：章节大纲侧栏与拖拽重排（前端拖拽 + 后端 `chapters/reorder` 一次性重排）。
25. 已落地：author 风格编辑体验第二版（`Tiptap` 正文编辑内核、专注模式、选区驱动的润色/扩写、助手回复插入/替换保持可用）。
26. 已落地：Prompt + 知识库面板最小闭环（模板创建/更新/删除、设定与卡片注入选择、请求携带 `prompt_template_id`、evidence 可观测注入元信息）。
27. 已落地：Ghost Text 闭环（独立 `ghost-text` 接口、前端手动触发默认 + 可选自动触发、接受/拒绝/重生、本地短缓存与去抖）。
28. 已落地：会话能力补齐（`thinking_enabled` 与 `reference_project_ids` 请求参数、跨项目上下文引用、evidence/runtime 可观测）。
29. 已落地：Prompt 面板增强（模板复制、模板版本历史与回滚、注入预览与冲突提示）。
30. 已落地：正文零感保存（章节自动保存与状态反馈）+ 后端迁移脚本与回归测试补齐。
31. 已落地：写作/调试双模式与助手抽屉（写作界面聚焦正文，调试能力按需展开）。
32. 已落地：选区润色/扩写升级为真实写作动作（自动打开助手抽屉并聚焦输入框）。
33. 已落地：本地崩溃恢复基线（未落库正文写入 localStorage 快照，刷新后自动恢复）。
34. 已落地：Ghost Text 轻上下文策略（默认仅 prefix + 模板风格约束，不走 compile_context_bundle）。
35. 已落地：动作 provenance 增强（proposed/confirmed/applied/undone 审计链路可追溯 source/evidence/mutation_version）。
36. 已落地：异步队列升级为 PostgreSQL `AsyncJob`（`SKIP LOCKED` claim + processing 超时回收 + 显式 ack/retry/fail），替代 `LPUSH/BRPOP` 丢任务风险。
37. 已落地：worker 增加 project 级 advisory lock + `expected_version` 对账（版本落后 job 自动 skip），降低 rebuild/增量竞态覆盖。
38. 已落地：图谱更新补“投影语义”基线（同源实体更新前先删旧边，再写新边）。
39. 已落地：动作接口语义对齐（`confirm` -> `apply`，前端与按钮语义统一为“应用并记录”）。
40. 已落地：上下文编译链路性能优化（最近消息改 SQL 级“取最后 12 条并反转”，settings/cards 改 SQL 级 LIMIT，保持原排序语义）。
41. 已落地：`update_message_content` 合并为单次 commit + `AsyncJob` done/failed 历史定时清理。
42. 已落地：P1 多温策略（`action/chat/ghost/brainstorm`）+ 请求级温度覆盖（`temperature_profile/temperature_override`）。
43. 已落地：P1 动态上下文滑窗（`context_window_profile` 策略管道：`balanced/chapter_focus/world_focus/minimal`，并在 evidence 透出）。
44. 已落地：P2 写作沉浸与认知透出（Zen 模式、内联实体高亮 + Tooltip、AI 当前认知标签）+ 图谱入库前别名对齐（alias -> canonical，可观测对齐计数）。
45. 已落地：DSL 前置别名驱动（`StoryCard/SettingEntry.aliases` 原生字段 + 抽取前 Aho-Corasick 命中扫描注入 alias -> canonical 映射），并保留入库前二次归一化兜底。
46. 调整：方案二轻量版已停用（暂停代词预处理与滑窗继承，当前线上链路仅保留方案一的别名归一化）。
47. 已落地：实体合并强制人工确认（禁用自动合并；`entity.merge*` 仅允许人工 `apply`，并要求 `manual_confirmed=true`）。
48. 已落地：方案三/四第一版（`entity_merge_scan_jobs` 后台巡检 + 手动扫描接口），仅生成 `entity.merge.proposal`，应用阶段仅 aliases 软合并并可 `undo`。

## Phase D：工作流体系（P1）

1. 工作流工作室 + 触发器。
- DoD：支持创建/执行/暂停/恢复/取消/查看运行记录。

2. Workflow Agent。
- DoD：支持“预览补丁 -> 校验 -> 应用”，失败可重试。

## Phase E：运营与扩展（P1）

1. 模型管理与成本统计。
- DoD：连通性测试、模型发现、用量重置可用。

2. 观测与评测。
- DoD：输出命中率、POV违规率、延迟、成本。

3. 多用户预留。
- DoD：`project_id` 隔离 + 最小 RBAC（owner/editor/viewer）。

## 当前只做的下一步（本周）

1. 回到写作主线：补“助手会话能力”剩余项（跨项目引用、模型覆盖、Thinking 开关联动前端）。
2. 回到写作主线：补“编辑器增强”剩余项（右键润色/扩写 + Ghost Text 接受/拒绝/重生）。
3. 回到写作主线：补“Prompt + 知识库面板”剩余项（模板版本管理、模板复制、注入策略预览与冲突提示）。
