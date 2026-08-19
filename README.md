# robot_brain

> 展厅轮式移动操作机器人的**机器人大脑**早期原型 —— 由 LLM 在已注册的技能中选择并制定计划，通过 **Plan-Execute 循环**驱动机器人完成多步任务。

`robot_brain` 是一个演示性的任务编排系统：用户用自然语言下达任务（如 *"去展柜旁拿一瓶水，然后回来递给用户"*），LLM 把任务拆成可执行的分步计划，逐条调用机器人技能执行，再根据执行反馈动态修正计划，直到任务完成并给出最终回复。

这是一个"机器人大脑"的前额叶原型：**LLM 负责思考（规划与重规划），确定性的技能工具负责行动（执行）**。

```
┌─────┐      ┌──────────┐      ┌────────┐      ┌──────────┐      ┌──────────┐
│用户任务│ ──▶ │ Planner   │ ──▶ │Executor│ ──▶ │Replanner │ ──▶ │ 完成 ✅   │
└─────┘      │制定分步计划│      │(ReAct) │      │更新剩余计划│      │ 最终回复  │
             └──────────┘      │逐步调用技能│    └────┬─────┘      └──────────┘
                               └────┬─────┘         │计划未收敛时基于反馈重新计划
                                    │               │
                                    └───────────────┘
```

## 系统怎么运转

核心是一个 **Plan-and-Execute** 循环（基于 LangGraph 实现）：

1. **Planner（规划器）** 把用户任务拆成一份按顺序执行的分步计划；
2. **Executor（执行器，ReAct）** 每次取出计划中的**第一条**步骤，以 ReAct（推理-行动-观察）方式调用对应技能并客观汇报执行结果；
3. **Replanner（复盘器）** 根据执行反馈**剪枝已完成步骤**、保留/修正剩余步骤（失败时追加备选方案），产出更新后的计划；
4. 循环往复，直到剩余计划为空、任务收敛，最终输出对用户的回复。

**两个关键设计点：**

- **结构化输出区分"计划"与"回复"**：planner / replanner 通过 `with_structured_output` 强制返回类型化结构（`Plan`：`type=plan` + 步骤列表；`Response`：`type=response` + 回复文本），保证模型"下一步还该干嘛"不会被当成"对用户的最终回复"。
- **本仓库对执行循环做了本地修改**：`execute_step` 会在执行前消费 `plan[0]`（上游版本不消费、完全依赖 replanner 剪枝，会导致死循环）。

## 当前技能

四个 mock 机器人技能（真实技能 API 的无 ROS 替代品），共享一个 `RobotState`（`location` / `holding`）模拟世界状态。每个技能的**前置条件 / 成功条件**写在工具描述里，供 LLM 在规划时自行判断"现在调用是否安全 / 该不该调用"：

| 技能 | 作用 | 关键前提 / 成功条件 |
|------|------|---------------------|
| `navigate_to` | 导航到 `display_area` / `reception` / `door_main` | 成功：已到达目标位置 |
| `pick` | 拾取物体（`water_1` / `water_2` / `brochure_1`） | 前提：底盘已停、物体可及、**未持有物体**；成功：持物 |
| `handover` | 把手中物体递给用户（`person_1` / `person_2`） | 前提：**持有物体**且停在接收者旁；成功：脱手 |
| `orient_to` | 朝向某个展示物 / 人 | 无前置条件；只返回朝向文本，不改变状态 |

> 演示任务：`"去展柜旁拿一瓶水，然后回来递给用户"` —— 机器人需要规划出 `navigate_to → pick → navigate_to → handover` 这样的多步流程，并在每一步后接受复盘修正。

## 模型轨迹日志

`model_trace.py` 为每次 LLM 调用提供**完整的控制台调试日志**：原始输入/输出、耗时、token 用量（`ModelTraceCallbackHandler` 挂在模型上，自动捕获 planner / replanner / executor 的所有调用）。日志左下角会按提示词特征自动标注角色：

- **planner** —— 含"针对给定的目标"
- **replanner** —— 含"当前计划按顺序的步骤依次为"
- **executor** —— 其余

在排查"机器人为什么绕了远路 / 计划为什么没收敛"这类多步任务问题时非常好用。

## 快速开始

需要 `uv`（Python ≥ 3.10）。

```bash
# 1. 安装依赖
uv sync

# 2. 配置 API Key —— 本演示实时调用 Qwen（DashScope OpenAI 兼容端点）
cp .env.example .env
#    在 .env 中填入 QWEN_API_KEY=...（DashScope key）

# 3. 运行演示
uv run python main.py
```

> ⚠️ 运行会**实时调用 Qwen 大模型 API**，需要联网，并产生 API 费用。

## 目录结构

```
robot_brain/
├── main.py            # 演示入口：构建模型、创建 Plan-Execute agent、执行取水任务
├── skills.py          # 四个 mock 机器人技能 + 共享 RobotState
├── model_trace.py     # 本地模型轨迹日志（调试 planner/replanner/executor）
├── config.toml        # 模型供应商、模型名、tracing 配置
├── rai/               # 随仓库引入的 Robotec.AI 机器人代理库（Apache-2.0）
├── learning_notes/    # 设计文档（目标架构的权威规范 / 演进方案 / 启发）
├── uv.lock
└── .env.example       # 环境变量模板
```

## 技术栈

- **编排**：`LangChain` + `LangGraph`（plan-and-execute 图：`planner → agent → replan →(should_end) agent|END`）
- **模型**：`qwen3.7-plus`（阿里云 DashScope **OpenAI 兼容端点**，ChatOpenAI）
- **机器人代理库**：`rai`（[Robotec.AI](https://github.com/RobotecAI/rai) 的机器人代理库，Apache-2.0，本仓库直接导入、未作为 pip 依赖安装）
- **语言 / 工具**：Python 3.10–3.12，`uv` 包管理；提示词为**中文**

## 当前局限

- 这是**演示原型**：技能是 mock 的（无 ROS、无真实导航/夹爪），状态是简化模拟，没有 `observe`/查找步骤为后续 `pick` 返回物体坐标。
- **多步任务收敛依赖模型质量**：qwen3.7-plus 对多步任务收敛性一般，replanner 偶尔会重复规划已完成步骤、出现不必要的反复导航，曾经触发过 `recursion_limit`；本仓库对执行循环的本地修改已缓解死循环问题，但更彻底的解法见下。

## 设计意图与路线

`learning_notes/` 中的《展厅机器人大脑需求文档》描述了目标架构：**LLM 只负责产出受约束的计划**，由**确定性执行器**校验并消费步骤、维护任务状态，配合**安全监督器**与**能力运行时** —— 而不是像当前 demo 这样让模型自我修正。多步任务收敛的彻底解决，需按此方向演进。

> 详细设计见 [`learning_notes/展厅机器人大脑需求文档.md`](learning_notes/展厅机器人大脑需求文档.md)。

## 许可

`rai/` 目录为 Apache-2.0，其余部分未单独声明。