# Simulation Director + ToM Design

**日期：** 2026-03-10  
**主题：** 用 `Director / Mind Engine / Actor Engine` 重构场景模拟，替代 `Round-Robin`

---

## 1. 判词

当前 `apps/api/app/services/simulation_service.py` 把发言调度、角色上下文、LLM 生成、持久化揉在一起，核心发言权仍由 `decide_next_actor_id()` 轮询驱动。该设计无法支撑“动机驱动抢话”和“二阶 ToM”。

本次设计直接放弃补丁式扩展，改为系统级重构：

- 用 `Simulation Director` 统筹单轮运行
- 用 `Mind Engine` 承载角色主观状态与二阶 ToM
- 用 `Actor Engine` 专职生成发言
- 用显式失败与 `paused` 替代业务级 fallback

目标不是“更复杂”，而是“更优雅、更可解释、更接近真人互动”。

## 2. 已确认约束

- 产品取向：**自然度优先**
- 行为取向：**允许一定程度失控**
- ToM 范围：**仅会话内、仅二阶**
- 架构取向：**允许删除旧设计，不保留长期 fallback**
- 图谱边界：**Neo4j 继续只承载客观事实，不承载主观 belief**

## 3. 总体架构

新链路：

`SimulationEvent -> Director -> Mind Engine -> Bidding -> Referee -> Actor Engine -> Repository`

职责拆分：

- `Director`：单轮 orchestration，只负责流程控制
- `Mind Engine`：维护角色私有知识、belief、二阶 ToM、情绪与压力
- `Bidding`：为所有角色产出 `TurnCandidate`
- `Referee`：从候选中裁决本轮 speaker，允许返回 `none`
- `Actor Engine`：为被选中角色生成结构化 `TurnAction`
- `Repository`：统一管理 session、state、event、turn、decision_trace 持久化
- `Policy`：调度参数与约束，不参与存储

旧 `simulation_service.py` 最终降格为 façade，或被完全删除。

## 4. 领域模型

### 4.1 客观层

- `SimulationSession`：静态配置与生命周期
- `SimulationEvent`：外部事件、系统事件、导演事件
- `SimulationTurn`：最终剧情输出

### 4.2 主观层

- `SimulationSessionState`
  - `revision`
  - `runtime_status`
  - `last_speaker_id`
  - `active_pressures`
  - `mind_states`
- `CharacterMindState`
  - `known_fact_keys`
  - `beliefs`
  - `beliefs_about_others`
  - `goals`
  - `agitation`
  - `cooldown`
  - `focus_target`

### 4.3 裁决层

- `Belief`
  - `fact_key`
  - `stance`
  - `confidence`
  - `source_turn_id`
  - `expires_at`
- `SecondOrderBelief`
  - `holder_id`
  - `subject_id`
  - `fact_key`
  - `believed_knowledge_state`
  - `confidence`
  - `source_turn_id`
- `TurnCandidate`
  - `actor_id`
  - `agitation_score`
  - `motive`
  - `target_id`
  - `interrupt_kind`
  - `confidence`
- `RefereeDecision`
  - `winner_id | none`
  - `reason`
  - `applied_rules`
  - `pause_suggested`
- `SimulationDecisionTrace`
  - `session_id`
  - `turn_index`
  - `candidate_snapshot`
  - `decision_payload`
  - `failure_code`

## 5. 单轮运行协议

`Director.run_turn()` 固定执行：

1. 读取 `SimulationSession` 与最新 `SimulationSessionState`
2. 拉取未消费 `SimulationEvent`
3. 组装 `WorldState`
4. 调用 `Mind Engine.advance()`
5. 调用 `Bidding.produce_candidates()`
6. 调用 `Referee.pick()`
7. 若返回 `none`，提交 `paused` 与 `DecisionTrace`
8. 若选中 speaker，调用 `Actor Engine.generate_turn()`
9. 持久化 `SimulationTurn`、更新 `SimulationSessionState`
10. 同事务提交 `DecisionTrace`

结果只能是：

- `spoken`
- `paused`
- `failed`

不允许：

- 回退到轮询
- 伪造默认台词
- 结构化失败后静默修补

## 6. 调度算法

### 6.1 agitation 评分

加分因子：

- 刚被冒犯或点名
- 秘密暴露风险上升
- 目标对象刚发言
- 外部事件命中角色目标
- 怀疑链被触发

减分因子：

- 刚刚发言
- 连续霸场
- 与当前局势低相关
- belief 置信度过低
- 冷却中

系统目标不是公平轮流，而是“谁此刻最忍不住，谁先开口”。

### 6.2 paused 语义

若候选全部低相关、低置信，或 Director 判断应等待新事件，则返回 `paused`。  
`paused` 是合法剧情结果，不是降级或失败。

## 7. 二阶 ToM

ToM 仅做到二阶：

- `A knows X`
- `A believes B knows X`

不做三阶及以上递归猜心。

更新来源仅允许：

- 直接观察到的发言/行动
- 明确注入事件
- 角色自身推断

每条主观状态都必须带：

- `confidence`
- `source_turn_id`
- 可选 `expires_at`

错误 belief 不进入 Neo4j，不写入客观知识层，只在会话态中生灭。

## 8. LLM Prompt 契约

说明：

- `Bidding` 与 `Referee` 当前采用 heuristic（可控、可解释），不走 LLM。
- LLM 仅用于：`Actor` 台词生成 + `ToM Updater` 的二阶 belief 增量。

### 8.1 ToM Updater（自动二阶 belief 增量）

输入：holder 的 mind state + 最近对白（recent turns）  
输出：严格结构化增量（写入会话态，不写 Neo4j）：

- 仅更新 `beliefs_about_others`（二阶）
- 每轮 cap（例如 ≤8）
- 必须带 `confidence` 与可追溯的 `source_turn_id`（可为空）

禁止：

- 基于规则/关键词匹配做机械判断
- 三阶及以上递归猜心
- 写入 Neo4j

### 8.2 Actor（台词生成）

输入：被选中角色 mind state + 当前局势 + 最近对白  
输出：严格结构化 JSON：`assistant_text` 为一句第一人称台词，`proposed_actions` 必须为空数组

禁止：

- 输出旁白/解释/多句对白
- 产出任何动作（proposed_actions 非空即视为越权）

## 9. 存储重构

建议新增：

- `SimulationSessionState`
- `SimulationEvent`
- `SimulationDecisionTrace`

建议收缩：

- `SimulationSession.pending_events` 删除
- `SimulationSession.status` 语义简化为 session 生命周期

继续保留：

- `SimulationTurn`

数据库策略：

- `SimulationSessionState.revision` 做乐观锁
- `Turn` 与 `DecisionTrace` 同事务提交
- `Event` 消费后写 `consumed_at`

## 10. 失败语义

显式失败类型示例：

- `NO_VALID_CANDIDATE`
- `REFEREE_INVALID_DECISION`
- `ACTOR_INVALID_OUTPUT`
- `STATE_REVISION_CONFLICT`

原则：

- 失败必须可观测
- 失败必须可定位
- 失败不靠 fallback 掩盖

## 11. 测试策略

- `mind_engine`：belief / 二阶 belief 更新、衰减、覆盖
- `bidding`：评分、候选排序、霸场抑制
- `referee`：winner 合法性、`paused` 判定
- `actor_engine`：结构化 `TurnAction` 合法性
- `director`：一次 run turn 的状态推进、事务一致性
- 端到端：验证抢话、停顿、试探、误判是否出现

关键指标：

- `speaker_entropy`
- `interrupt_rate`
- `suspicion_chain_depth`
- `pause_rate`
- `invalid_schema_rate`
- `latency_p95`

## 12. 迁移路径

1. 新建 simulation 子模块与新状态模型
2. 先接通 `Director` 新链路
3. API endpoint 切到新链路
4. 删除 `pending_events`
5. 删除 `decide_next_actor_id()`
6. 删除旧单体调度逻辑与相关旧测试

迁移期间允许旧代码仍在仓内，但主链只走新实现；稳定后立即删旧骨，不保留长期双轨。

## 13. 非目标

- 本轮不把主观 ToM 持久化到 Neo4j
- 本轮不做三阶 ToM
- 本轮不做“完全无静默”的强制发言
- 本轮不保留业务级 fallback

## 14. 验收口径

满足以下条件，视为设计达标：

- 发言权不再轮询
- 角色可基于动机、冲突、误判主动抢话
- 系统可合法进入 `paused`
- 所有模型调用均为严格 structured output
- `DecisionTrace` 可解释每轮选人原因
- 旧轮询逻辑与 `pending_events` 被删除
