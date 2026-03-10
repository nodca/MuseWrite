# Neo4j GDS 硬门禁实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标：** 为质量优先的图检索补齐 Neo4j Graph Data Science 运行依赖，默认在 Docker 中启用 GDS，并在 API 启动阶段执行硬门禁校验，缺失时直接失败。

**架构：** 保持现有 `DSL + GRAPH + LightRAG` 检索编排不变，但把 Neo4j GDS 视为强依赖基础设施。Docker 负责默认装载插件，运行时配置显式表达策略，FastAPI 启动时执行一次真实能力探测，避免悄悄降级到低质量图检索。

> 2026-03-09 现场补记：官方 `neo4j:5.26.21-community` 通过 `NEO4J_PLUGINS=["graph-data-science"]` 自动解析插件时，会因 `versions.json` 缺少 `5.26.21` 精确条目而启动失败到“无 GDS”状态。现已改为仓库内自定义 Neo4j 镜像，直接 baked-in `GDS 2.13.7`，并用 `gds.version()` healthcheck 锁死运行时质量。

**技术栈：** Docker Compose、Neo4j 5 community、FastAPI lifespan、Python `neo4j` driver、`unittest`

---

### 任务 1：补齐 GDS 运行时配置契约

**文件：**
- 修改：`apps/api/app/core/config.py`
- 修改：`apps/api/app/core/settings/runtime.py`
- 测试：`apps/api/tests/test_settings_contract_unittest.py`

**步骤 1：先写失败测试**

补断言，确保新增的 Neo4j GDS 设置字段进入根配置与 runtime 代理映射。

**步骤 2：运行测试，确认先失败**

运行：`python -m unittest apps.api.tests.test_settings_contract_unittest -v`

预期：因新字段尚未加入配置契约而失败。

**步骤 3：写最小实现**

新增以下配置：
- `neo4j_gds_required`
- `neo4j_gds_min_version`

并把它们暴露到 `RuntimeSettings.FIELD_NAMES`。

**步骤 4：再次运行测试，确认转绿**

运行：`python -m unittest apps.api.tests.test_settings_contract_unittest -v`

预期：通过。

### 任务 2：在 Docker 中默认启用 GDS，并补文档

**文件：**
- 修改：`docker-compose.yml`
- 修改：`README.md`
- 修改：`apps/api/README.md`

**步骤 1：先约束验证方式**

不单独新增 Compose 解析测试，统一通过最终 `docker compose config` 验证语法与渲染结果。

**步骤 2：写最小实现**

更新 Neo4j 服务，默认预装 GDS plugin，并在文档中声明“质量优先图检索需要 GDS”。

**步骤 3：手工验证**

运行：`docker compose config`

预期：YAML 正常展开，Neo4j 环境变量中包含 GDS plugin 相关配置。

### 任务 3：在 API 启动阶段执行 GDS 硬门禁

**文件：**
- 修改：`apps/api/app/main.py`
- 修改：`apps/api/app/services/retrieval_adapters.py`
- 测试：`apps/api/tests/test_startup_gds_gate_unittest.py`

**步骤 1：先写失败测试**

补启动测试，覆盖以下场景：
- `neo4j_gds_required=false` 时启动可通过
- `neo4j_enabled=false` 时启动可通过
- `neo4j_gds_required=true` 且 GDS 校验失败时启动必须抛错

**步骤 2：运行测试，确认先失败**

运行：`python -m unittest apps.api.tests.test_startup_gds_gate_unittest -v`

预期：因尚未存在 GDS 启动门禁而失败。

**步骤 3：写最小实现**

增加一个轻量 Neo4j 能力探针，执行 `RETURN gds.version()`；若插件缺失或版本低于配置要求，抛出清晰的 runtime error。

**步骤 4：再次运行测试，确认转绿**

运行：`python -m unittest apps.api.tests.test_startup_gds_gate_unittest -v`

预期：通过。

### 任务 4：最终验尸

**文件：**
- 无新增文件

**步骤 1：运行目标测试**

运行：`python -m unittest apps.api.tests.test_settings_contract_unittest apps.api.tests.test_startup_gds_gate_unittest -v`

预期：全部通过。

**步骤 2：运行 Compose 校验**

运行：`docker compose config`

预期：通过，且 Neo4j plugin 环境变量正确渲染。
