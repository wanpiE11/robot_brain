"""Runtime state and prompt context builders for the robot demo."""

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class WorldState:
    """Deterministic world state maintained by skill code."""

    location: str | None = None
    holding: str | None = None
    facing: str | None = None


@dataclass
class StepRecord:
    """One executed plan step and its objective result."""

    step: str
    result: str
    success: bool = True
    error: str | None = None


@dataclass
class TaskRuntimeState:
    """Runtime facts for one plan-execute task."""

    world: WorldState = field(default_factory=WorldState)
    past_steps: list[StepRecord] = field(default_factory=list)
    plan_version: int = 1


def format_world_state(world: WorldState) -> str:
    return "\n".join(
        [
            f"- 位置: {world.location or '未知'}",
            f"- 持有物: {world.holding or '无'}",
            f"- 朝向: {world.facing or '未知'}",
        ]
    )


def format_step_records(past_steps: Sequence[StepRecord]) -> str:
    if not past_steps:
        return "- none"
    lines: list[str] = []
    for index, record in enumerate(past_steps, start=1):
        status = "成功" if record.success else "失败"
        lines.append(f"{index}. [{status}] {record.step} -> {record.result}")
        if record.error:
            lines.append(f"   错误: {record.error}")
    return "\n".join(lines)


def format_plan_steps(steps: Sequence[str]) -> str:
    if not steps:
        return "- none"
    return "\n".join(f"{index}. {step}" for index, step in enumerate(steps, start=1))


def build_state_hint(world: WorldState, current_step: str) -> str:
    """Return deterministic guidance derived from current state."""

    hints = [
        "请把当前世界状态当作事实，不要自行编造或改写。",
        "只有已注册的工具可以改变世界状态。",
    ]

    step_lower = current_step.lower()
    if world.holding:
        hints.append(
            f"机器人当前已经持有 {world.holding}；除非当前步骤明确要求更换物体，否则不要再次调用 pick。"
        )
    else:
        hints.append("机器人当前没有持有物体；在 pick 成功之前，handover 一定会失败。")

    if "递" in current_step or "交" in current_step or "handover" in step_lower:
        if world.holding:
            hints.append(
                "对于当前递交步骤，请直接调用 handover，并传入目标人物。不要先调用 pick，因为物体已经在手中。"
            )
        else:
            hints.append(
                "对于当前递交步骤，请先补回缺失的持有物体，再调用 handover。"
            )

    return "\n".join(f"- {hint}" for hint in hints)


def build_executor_context(
    *,
    original_task: str,
    current_step: str,
    remaining_plan: Sequence[str],
    past_steps: Sequence[StepRecord],
    world: WorldState,
) -> str:
    """Build the per-step prompt for the ReAct executor."""

    return f"""你正在执行机器人计划中的一个步骤。

原始任务：
{original_task}

当前要执行的步骤：
{current_step}

执行完当前步骤后的剩余计划：
{format_plan_steps(remaining_plan)}

已完成步骤记录：
{format_step_records(past_steps)}

当前世界状态：
{format_world_state(world)}

状态提示：
{build_state_hint(world, current_step)}

执行规则：
- 只执行当前步骤。
- 只调用完成当前步骤所必需的最少注册工具。
- 执行后，只用一句话客观汇报当前步骤的结果。
- 不要与用户寒暄，不要提及无关的后续步骤。"""


def build_replanner_context(
    *,
    original_task: str,
    current_plan: Sequence[str],
    past_steps: Sequence[StepRecord],
    world: WorldState,
    plan_version: int,
) -> str:
    """Build the prompt for the replanner."""

    return f"""# 角色
你是展厅机器人的复盘模块，只负责更新剩余计划。

# 规则
1. 只要当前计划里还有未执行步骤，就必须返回 Plan 动作，只保留仍需执行的步骤。
2. 只有原始任务已经真正完成、且没有待执行步骤时，才允许返回 Response 动作。
3. 不要把“剩余计划 / 更新后的计划”之类的文本写进 response 字段。
4. 不要把已经成功完成的步骤放进新计划。
5. 如果上一步失败，就保留或修复该步骤，方便重试。
6. 当前世界状态是事实。你可以读取它，但不能直接修改它。

原始任务：
{original_task}

当前有效计划（版本 {plan_version}）：
{format_plan_steps(current_plan)}

已完成步骤记录：
{format_step_records(past_steps)}

当前世界状态：
{format_world_state(world)}

请仅根据剩余需要执行的步骤重新生成计划。"""
