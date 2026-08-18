# CLAUDE.md

本文件为 Claude Code（claude.ai/code）在此仓库中处理代码时提供指导。

## 项目概述

展厅轮式移动操作机器人的早期原型（"机器人大脑"）。由 LLM 通过在已注册的技能中选择来驱动机器人；设计目标（见 `learning_notes/`）是一个灵活的任务编排循环：LLM 提出计划，确定性层对其进行审查/校验，然后才发送给技能接口执行。目前代码是一个针对 mock 技能运行 Plan-Execute 循环的单一演示脚本。

是 git 仓库（2026-08-18 初始化，尚无提交）。

## 运行

- 包管理器：`uv`（见 `uv.lock`）。无构建步骤；纯 Python。
- 安装 / 同步：`uv sync`（创建 `.venv`）
- 运行演示：`uv run python main.py`
  - 需要 `robot_brain/.env` 中包含 `QWEN_API_KEY=...`（DashScope key；`main.py` 也兼容 `DASHSCOPE_API_KEY` / `DEEPSEEK_API_KEY`），并在构建模型前将其复制为 `OPENAI_API_KEY`。
  - 会实时调用 Qwen（DashScope OpenAI 兼容端点）API —— 需要联网且会产生费用。
- 仓库中没有测试，也未配置 pytest；仓库中没有 lint/format 配置。

## 架构

- `main.py` —— 演示入口。加载 `.env` 和 `config.toml`，通过 `rai.initialization.get_llm_model("complex_model", vendor="openai", config_path=...)` 构建 Qwen `ChatOpenAI`（DashScope OpenAI 兼容端点），用 `rai.agents.langchain.core.plan_agent.create_plan_execute_agent` 创建 Plan-Execute agent（planner/replanner 用 `with_structured_output`，executor 是 ReAct），用取水示例任务调用它，并打印计划、已执行步骤和最终回复。
- `skills.py` —— mock 机器人技能（`navigate_to`、`pick`、`handover`、`orient_to`），是真实技能 API 的无 ROS 替代品。每个都是带 pydantic `args_schema` 的 LangChain `BaseTool` 子类；工具 `description` 声明前置条件 / 成功条件，以便 LLM 推理什么调用是安全的以及如何验证。`make_skills()` 返回交给 agent 的技能列表。
- `rai/` —— **随仓库引入的第三方代码**：Robotec.AI 的 "Rai" 机器人代理库（Apache-2.0）。直接导入，不作为依赖安装；建议不要修改。本仓库使用的关键接口：
  - `rai.initialization.get_llm_model` / `get_embeddings_model` / `get_tracing_callbacks` —— 由 `config.toml` 驱动的模型构建；支持 `openai`（包括 DeepSeek 风格的 `base_url`）、`aws`、`ollama`、`google` 等供应商。
  - `rai.agents.langchain.core.react_agent.create_react_runnable` —— ReAct 工具调用图；`create_plan_execute_agent` 内部的 executor 使用它。
  - `rai.agents.langchain.core.plan_agent.create_plan_execute_agent` —— Plan-and-Execute（计划并执行）代理，`main.py` 当前使用。planner/replanner 用 `with_structured_output`（默认 `method="json_schema"`，Qwen 兼容端点支持；注意 Qwen thinking 模式拒绝 `method="function_calling"` 的强制 `tool_choice`）。**本地修改**：`execute_step` 执行后消费 `plan[0]` —— 上游版本不消费、完全依赖 replanner 剪枝，会导致死循环。
  - `rai.messages.*MultimodalMessage` —— 可携带图像/音频工件（artifact）的消息类型。

## 模型配置

- `config.toml` 中的 `[vendor]` 选择由哪个供应商服务 `simple_model` / `complex_model` / `embeddings_model`；`[openai]` 提供模型名称和 `base_url`。目前 `openai` → Qwen（`qwen3.7-plus`，`https://llm-rjwzwzgyuh2m7sfv.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`）。
- `[tracing]` 控制 Langfuse / LangSmith；两者默认关闭。

## 设计意图（来源：`learning_notes/`）

`learning_notes/展厅机器人大脑需求文档.md` 是目标架构的权威规范（大部分尚未实现）：

- 四个具有单写者所有权（single-writer ownership）的状态存储 —— `WorldState`、`DialogueState`、`TaskState`、`SystemState` —— 以及一个 `ContextBuilder`，用于为 LLM 调用组装只读的、特定用途的 `AgentContext` 快照。
- 一个 LLM **规划器（planner）**，只发出基于当前可用已注册技能构建的 `TaskPlanProposal`；一个确定性的**任务执行器（task executor）**，负责校验并接受计划、运行步骤、记录效果、检查提交点、处理重试/失败，并且是 `TaskState` 的唯一写者；一个具有独立否决/停止权的**安全监督器（safety supervisor）**。
- 一个能力运行时（capability runtime），持续计算带版本号的只读 `CapabilitySnapshot`（机器人*当前*能做什么，而非静态技能列表）。
- 硬约束：LLM 绝不能直接调用技能、写入状态或控制硬件 —— 所有模型输出都必须经过确定性执行器和安全层。

当前演示与目标之间的已知差距：mock 技能加入了简易共享状态（`skills.py` 的 `RobotState`，模拟 `WorldState`/`TaskState`），但仍是无 ROS 的简化模拟，没有 `observe`/查找步骤为后续的 `pick` 返回物体坐标。**已知局限（2026-08-18）**：qwen3.7-plus 在 Rai 的 naive plan_agent 上对多步任务收敛性差 —— replanner 会重复规划已完成步骤、反复导航，循环最终触发 `recursion_limit`，`uv run python main.py` 可能跑不完就抛 `GraphRecursionError`。要做到稳健，需要按需求文档走向确定性执行器（LLM 只出计划、确定性层消费步骤并维护状态）。

## 规则约束

### 重大改动需向用户申请

凡属**重大改动**，必须先向用户说明并征得同意，**不得直接实施**。重大改动包括但不限于：

- 架构级改动（新增/移除/重构模块、改变模块间依赖关系）
- 修改核心流程（任务编排循环、状态存储、执行器/安全层等）
- 跨多个文件的较大重构或重命名
- 删除或重命名现有重要文件/功能
- 新增或变更依赖（pyproject.toml、依赖库）

申请时必须说明：

- **前因后果**：为什么需要这个改动，当前的问题或背景是什么
- **解决方案**：打算具体怎么做（涉及哪些文件、如何实施）
- **涉及的风险**：可能引入什么问题、影响范围、如何缓解
- **涉及的收益**：改动完成后的好处

用户批准后才可实施。用户明确授权过的范围除外。
