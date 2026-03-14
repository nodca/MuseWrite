# Neo4j PPR Projection 异步预热设计

日期：2026-03-10

## 背景

当前命名 projection graph 采用懒重建：写入只 bump 版本，第一次读会承担投影创建成本。为了避免首次查询冷启动延迟，引入 worker 异步预热。

## 目标

- 预热仅在 worker 端执行，避免阻塞请求路径。
- 不增加开关和指标，保持系统最小配置面。
- 不改变严格质量门禁（GDS 仍为硬依赖）。

## 触发方案

选用 **方案 A：接入 graph_sync worker**。

触发点：`_process_graph_job` 在 `process_graph_sync_for_action` 完成后执行预热。

理由：

- graph_sync 本身就是图写入的异步管线，适合承载预热。
- 写入后的第一次查询最容易遇到冷投影，预热在这里最有效。
- 不需要新增队列或引入额外配置。

## 行为说明

### 何时预热

- 当 graph_sync 成功写入事实（`status="synced"` 且写入有事实）时触发。
- 若无事实写入，则跳过。

### 预热做什么

- 计算 scope（`all` / `chapter_{n}` / `latest` / `all_temporal`）。
- 基于最新 projection version 创建命名图（若尚不存在）。
- 若已存在且版本一致则复用，不重复构建。

### 失败策略

- 预热失败不会回滚已完成的 graph_sync。
- 失败会被 log 记录，但不影响 worker 主流程。

## 影响范围

- 读路径性能提升：首次查询更易命中已有 projection。
- 写路径成本几乎不变：额外成本转移到 worker。
- 兼容现有命名 projection 生命周期与版本机制。

## 测试计划

- 单元测试：worker 在 graph_sync 成功时调用 `prewarm_neo4j_ppr_projection`。
- 集成测试不必扩展（已有命名 projection 复用/重建验证）。

## 兼容性

- 不新增开关、不新增指标。
- 与现有 `graph_sync`/`index_lifecycle` 并行共存。
