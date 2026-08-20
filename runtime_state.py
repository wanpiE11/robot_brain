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
            f"- location: {world.location or 'unknown'}",
            f"- holding: {world.holding or 'none'}",
            f"- facing: {world.facing or 'unknown'}",
        ]
    )


def format_step_records(past_steps: Sequence[StepRecord]) -> str:
    if not past_steps:
        return "- none"
    lines: list[str] = []
    for index, record in enumerate(past_steps, start=1):
        status = "success" if record.success else "failed"
        lines.append(f"{index}. [{status}] {record.step} -> {record.result}")
        if record.error:
            lines.append(f"   error: {record.error}")
    return "\n".join(lines)


def format_plan_steps(steps: Sequence[str]) -> str:
    if not steps:
        return "- none"
    return "\n".join(f"{index}. {step}" for index, step in enumerate(steps, start=1))


def build_state_hint(world: WorldState, current_step: str) -> str:
    """Return deterministic guidance derived from current state."""

    hints = [
        "Use the current world state as fact. Do not invent or rewrite it.",
        "Only the registered tools can change world state.",
    ]

    step_lower = current_step.lower()
    if world.holding:
        hints.append(
            f"The robot is already holding {world.holding}; do not call pick again "
            "unless the current step explicitly requires replacing that object."
        )
    else:
        hints.append("The robot is not holding an object; handover will fail until pick succeeds.")

    if "递" in current_step or "交" in current_step or "handover" in step_lower:
        if world.holding:
            hints.append(
                "For this delivery/handover step, call handover with the target person. "
                "Do not call pick first because the object is already held."
            )
        else:
            hints.append(
                "For this delivery/handover step, first recover the missing held object "
                "before calling handover."
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

    return f"""You are executing one step of a robot plan.

Original user task:
{original_task}

Current step to execute now:
{current_step}

Remaining effective plan after this step:
{format_plan_steps(remaining_plan)}

Completed step records:
{format_step_records(past_steps)}

Current world state:
{format_world_state(world)}

State guidance:
{build_state_hint(world, current_step)}

Execution rules:
- Execute only the current step.
- Choose the minimal registered tool call(s) needed for this step.
- After executing, report this step's objective result in one concise sentence.
- Do not chat with the user and do not mention unrelated future steps."""


def build_replanner_context(
    *,
    original_task: str,
    current_plan: Sequence[str],
    past_steps: Sequence[StepRecord],
    world: WorldState,
    plan_version: int,
) -> str:
    """Build the prompt for the replanner."""

    return f"""# Role
You are the replanner for an exhibition robot task. You update only the remaining plan.

# Rules
1. If any steps still need execution, return a Plan action with only those remaining steps.
2. Return a Response action only when the original task is actually complete.
3. Never put remaining-plan text into the response field.
4. Never include successfully completed steps in the new plan.
5. If the last step failed, keep or repair that step so it can be retried.
6. Treat the world state below as factual. You may read it, but you cannot modify it directly.

Original user task:
{original_task}

Current effective plan, version {plan_version}:
{format_plan_steps(current_plan)}

Completed step records:
{format_step_records(past_steps)}

Current world state:
{format_world_state(world)}

Replan now using only the remaining required steps."""
