# Copyright (C) 2025 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from rai.agents.langchain.core import ReActAgentState
from rai.agents.langchain.core.react_agent import create_react_runnable
from rai.initialization import get_llm_model
from rai.messages import HumanMultimodalMessage


class Plan(BaseModel):
    """用于帮助解决用户请求的计划。"""

    steps: List[str] = Field(
        description="需要依次执行的步骤，应按顺序排列"
    )


class Response(BaseModel):
    """回复给用户的内容。"""

    response: str


class Act(BaseModel):
    """要执行的动作。"""

    action: Union[Response, Plan] = Field(
        description="要执行的动作。如果想直接回复用户，请用 Response；如果还需要执行步骤，请用 Plan。"
    )


class PlanExecuteState(ReActAgentState):
    """State for the plan and execute agent."""

    # NOTE (jmatejcz) should original_task be replaced with
    # passing first message? The message can contain images etc.
    original_task: str
    plan: List[str]
    past_steps: List[Tuple[str, str]]
    response: str


def should_end(state: PlanExecuteState) -> str:
    """Check if we should end or continue planning."""
    if state["response"]:
        return END
    else:
        return "agent"


def create_plan_execute_agent(
    tools: List[BaseTool],
    planner_llm: Optional[BaseChatModel] = None,
    executor_llm: Optional[BaseChatModel] = None,
    replanner_llm: Optional[BaseChatModel] = None,
    system_prompt: Optional[str] = None,
) -> CompiledStateGraph:
    """Create a plan and execute agent that can break down complex tasks into steps.

    Parameters
    ----------
    tools : List[BaseTool]
        List of tools the agent can use during execution
    llm : Optional[BaseChatModel], default=None
        Language model to use. If None, will use complex_model from config
    system_prompt : Optional[str | SystemMultimodalMessage], default=None
        System prompt to use (currently not used in this implementation)

    Returns
    -------
    CompiledStateGraph
        Compiled state graph for the plan and execute agent

    Raises
    ------
    ValueError
        If tools are not provided or invalid
    """
    if planner_llm is None:
        planner_llm = get_llm_model("complex_model", streaming=True)
    if executor_llm is None:
        executor_llm = get_llm_model("complex_model", streaming=True)
    if replanner_llm is None:
        replanner_llm = get_llm_model("complex_model", streaming=True)

    if not tools:
        raise ValueError("Tools must be provided for plan and execute agent")
    if system_prompt is None:
        system_prompt = ""

    planner_prompt = """针对给定的目标，制定一个简单、分步骤的计划。

制定计划时请注意：
- 让每一步尽量使用上面工具列表中合适的工具
- 具体说明每一步要获取哪些信息或执行什么动作
- 把步骤写成可以用现有工具执行的清晰指令
- 不要自己实际调用或使用任何工具，只负责制定计划
- 每一步都应当可执行、且与工具匹配

这个计划应当由一系列子任务组成，正确执行即可得到正确答案。
不要添加多余的步骤。最后一步的结果就应当是最终答案。
确保每一步都包含所需的信息，不要遗漏任何步骤。"""

    agent_executor = create_react_runnable(
        llm=executor_llm, system_prompt=system_prompt, tools=tools
    )
    # the prompt will be filled with values when passed to invoke
    planner_llm_with_tools = planner_llm.bind_tools(tools)
    planner = planner_llm_with_tools.with_structured_output(Plan)  # type: ignore
    replanner = replanner_llm.with_structured_output(Act)  # type: ignore

    def execute_step(state: PlanExecuteState):
        """Execute the current step of the plan."""

        plan = state["plan"]
        if not plan:
            return {}
        task = plan[0]
        task_formatted = f"""你要执行的任务是：{task}。"""

        agent_response = agent_executor.invoke(
            {"messages": [HumanMultimodalMessage(content=task_formatted)]},
            config={"recursion_limit": 50},
        )
        # Local fix vs upstream: consume the executed step so the plan shrinks
        # each iteration. Upstream leaves plan untouched and relies entirely on
        # the replanner to prune done steps, which causes loops.
        return {
            # "plan": plan[1:],
            "past_steps": [(task, agent_response["messages"][-1].content)],
        }

    def plan_step(state: PlanExecuteState):
        """Initial planning step."""
        messages = [
            SystemMessage(content=system_prompt + "\n" + planner_prompt),
            HumanMultimodalMessage(content=state["original_task"]),
        ]
        plan = planner.invoke(messages)
        return {"plan": plan.steps}

    def replan_step(state: PlanExecuteState):
        """Replan based on execution results."""
        # Format past steps for the prompt
        past_steps_str = "\n".join(
            [
                f"{step}: {result}"
                for i, (step, result) in enumerate(state["past_steps"])
            ]
        )

        # Format remaining plan
        plan_str = "\n".join([step for i, step in enumerate(state["plan"])])

        replanner_prompt = f"""你不是规划器，不要从零制定计划。你负责复盘任务的执行进度，并决定下一步。

请检查已完成步骤与剩余计划：
更新计划，只保留仍然需要执行的步骤，不需要返回回复（Response），已经完成的步骤不要再次出现，也不要添加多余步骤。

原始目标如下：
{state["original_task"]}

当前计划是：
{plan_str}

目前已完成的步骤：
{past_steps_str}

请据此更新计划。若不需要更多步骤、可以直接回复用户，请直接回复。"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMultimodalMessage(content=replanner_prompt),
        ]
        output = replanner.invoke(messages)

        if isinstance(output.action, Response):
            return {"response": output.action.response}
        else:
            return {"plan": output.action.steps}

    workflow = StateGraph(PlanExecuteState)

    workflow.add_node("planner", plan_step)
    workflow.add_node("agent", execute_step)
    workflow.add_node("replan", replan_step)

    workflow.add_edge(START, "planner")
    # From plan we go to agent
    workflow.add_edge("planner", "agent")
    # From agent, we replan
    workflow.add_edge("agent", "replan")

    workflow.add_conditional_edges(
        "replan",
        should_end,
        ["agent", END],
    )

    return workflow.compile()


def create_initial_plan_execute_state(
    original_task: str,
    messages: Optional[List[BaseMessage]] = None,
) -> PlanExecuteState:
    """Create initial state for the plan and execute agent.

    Parameters
    ----------
    input_text : str
        The user's input/objective to accomplish
    messages : Optional[List[BaseMessage]], default=None
        Initial messages for the conversation

    Returns
    -------
    PlanExecuteState
        Initial state for the agent
    """
    if messages is None:
        messages = []

    return PlanExecuteState(
        messages=messages,
        original_task=original_task,
        plan=[],
        past_steps=[],
        response="",
    )


def run_plan_execute_agent(
    agent: CompiledStateGraph,
    original_task: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the plan and execute agent on a given input.

    Parameters
    ----------
    agent : CompiledStateGraph
        The compiled plan and execute agent
    input_text : str
        The user's input/objective
    config : Optional[Dict[str, Any]], default=None
        Configuration for the agent execution

    Returns
    -------
    Dict[str, Any]
        Final state after execution
    """
    initial_state = create_initial_plan_execute_state(original_task)

    # Execute the agent
    result = agent.invoke(initial_state, config=config)

    return result
