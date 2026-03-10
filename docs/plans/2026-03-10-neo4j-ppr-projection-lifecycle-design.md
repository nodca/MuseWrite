# Neo4j PPR 命名 Projection 生命周期设计

日期：2026-03-10

## 背景

图检索从“每次查询临时投影 + 立即 drop”升级为“命名 projection graph + 懒重建”，以降低多次查询的重复投影成本，并保证改稿后的检索一致性。

## 目标

- 保持质量优先：GDS 仍是硬依赖，不允许静默降级。
- 可复用：同一项目与同一 scope 下复用已有 projection。
- 可失效：写路径只标记 dirty，不在写请求内全量重建。
- 可重建：读取时按版本懒重建，并清理旧版本。
- 可运维：通过命名规则与元数据节点快速判断状态。

## 核心设计

### 命名规则

```
{NEO4J_GDS_GRAPH_NAME_PREFIX}_{project_id}_{scope}_v{version}
```

- `prefix`: 由 `NEO4J_GDS_GRAPH_NAME_PREFIX` 指定，默认 `novel_ppr`
- `scope`: 由时间过滤条件决定
  - `all`: 未启用时间过滤
  - `chapter_{n}`: 指定章节过滤
  - `latest`: 未传章节，但要求只取“仍有效”的事实
  - `all_temporal`: 启用时间过滤但未限制章节
- `version`: projection 版本号，每次图事实变更都会自增

### 元数据模型

使用 Neo4j 节点保存 projection 版本与失效原因：

```
(:GdsProjectionState {
  project_id,
  kind: "ppr",
  projection_version,
  created_at,
  updated_at,
  invalidated_at,
  invalidated_reason
})
```

### 读路径（检索）

1. 计算 `scope` 与 `graph_name`。
2. `gds.graph.exists(graph_name)`：
   - 已存在：直接复用。
   - 不存在：执行 `gds.graph.project(...)` 创建。
3. 若创建成功，清理同 `project + scope` 的旧版本图。
4. 在命名图上执行 `gds.pageRank.stream(...)`。

### 写路径（失效）

以下写操作会标记 dirty（projection 版本自增）：

- `upsert_neo4j_graph_facts`
- `promote_neo4j_candidate_facts`
- `update_neo4j_graph_fact_state`
- `delete_neo4j_graph_facts`
- `delete_neo4j_graph_facts_by_sources`
- `delete_all_neo4j_graph_facts`

写路径只 bump version，不做投影重建。

### 时间轴字段策略

为了避免 Neo4j missing-property warning，`valid_to_chapter` 使用显式 sentinel：

```
valid_to_chapter = 2147483647
```

读取时将 sentinel 映射为 `None`，表现为“开放区间”。

## 失败与边界

- 若 GDS 不可用：启动时直接失败，运行时遵循 strict graph mode（无本地 fallback）。
- projection 创建失败：读路径返回空结果或 error（由 strict 模式控制）。
- 旧版本投影清理失败：只记录为 best-effort，不影响当前查询结果。

## 测试覆盖

- 单测：`tests.test_graph_ppr_retrieval_unittest`  
  - 复用命名投影  
  - 未命中则创建投影
- 集成：`tests.test_writing_flow_unittest`  
  - 命名投影复用  
  - 变更后版本重建  
  - 长距线索回收
- 启动门禁：`tests.test_startup_gds_gate_unittest`

## 配置

新增变量：

```
NEO4J_GDS_GRAPH_NAME_PREFIX=novel_ppr
```

其它 GDS 相关变量保持不变。

## 运维要点

- 使用 `CALL gds.graph.list()` 可以查看当前命名投影。
- 通过 `GdsProjectionState` 节点追溯最近一次失效原因。
- 若需要手动清理，可按 prefix 批量 drop。

## 兼容性

- 仅新增配置项与元数据节点，不破坏既有 Neo4j 图事实结构。
- 对旧数据无迁移要求；首次查询自动初始化 projection 版本。

## 后续可选优化

- 加入投影缓存命中率指标。
- 将 projection rebuild 纳入 async worker（在高写频场景减少读路径冷启动）。
- 增加 per-scope 的最小 TTL 或背景重建策略。
