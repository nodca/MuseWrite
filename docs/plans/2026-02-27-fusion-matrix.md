# 三项目融合矩阵（冻结版）

日期：2026-02-27  
状态：`frozen-v1`

| 能力 | 来源 | 结论 | 说明 |
|---|---|---|---|
| 卡片/Schema | NovelForge | MUST | 核心数据模型，必须保留 |
| `@DSL` 精确引用 | NovelForge | MUST | 三合一检索中的精确层 |
| 图谱记忆（Neo4j） | NovelForge | MUST | 关系事实层 |
| 灵感助手工具调用 | NovelForge | MUST | 可执行协作，不是纯聊天 |
| Prompt 工坊 + 知识库 | NovelForge | MUST | 提示词可控性核心 |
| 灵感工作台（自由卡） | NovelForge | MUST | 脑暴与正式写作解耦 |
| 工作流系统 | NovelForge | MUST | 自动化资料流程核心 |
| Workflow Agent | NovelForge | MUST | 用自然语言维护工作流 |
| 伏笔管理 | NovelForge | SHOULD | Phase C 纳入 |
| 编辑器 Ghost Text 体验 | author | MUST | 主要交互体验来源 |
| AI侧栏聊天 | author | MUST | 低打扰写作协作 |
| 聊天改设定动作 | author | MUST | 聊天 -> 动作 -> 应用 |
| 设定树实时更新 | author | MUST | 动作确认后即时反映 |
| 快照/时光机 | author | SHOULD | 结合后端审计做服务端版本化 |
| 本地 IndexedDB 存储 | author | DROP | 与 Web+Docker+PostgreSQL 冲突 |
| POV 沙箱 | AI-Novel V2 | MUST | 避免越权叙述 |
| POV 切换桥接重置 | AI-Novel V2 | MUST | 防跨视角污染 |
| Token 分层预算 | AI-Novel V2 | MUST | must/important/nice 裁剪策略 |
| LightRAG 实践 | AI-Novel V2 | MUST | 语义召回层 |
| 自动整章 pipeline | AI-Novel V2 | DROP | 与“AI辅助而非接管”冲突 |
| 一致性诊断流水线 | AI-Novel V2 | DROP | 当前范围明确不做 |
| 桌面壳（Electron） | AI-Novel V2 | DROP | 当前仅 Web + Docker |

## 融合规则

1. 体验优先采用 `author`，结构能力优先采用 `NovelForge`。
2. 记忆与约束采用 `NovelForge + AI-Novel V2` 组合。
3. 冲突时按产品原则裁决：先“AI辅助写作”，再“可控与可回滚”，最后“自动化”。
