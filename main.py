"""Run a real LLM-driven Plan-Execute loop over the mock skills.

Chain: config.toml -> get_llm_model (vendor=openai, base_url = DashScope
OpenAI-compatible endpoint) -> create_plan_execute_agent -> the Qwen model
plans with with_structured_output (json_schema) and executes each step by
calling registered mock skills via native function calling.
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_dotenv() -> None:
    dotenv_path = HERE / ".env"
    if not dotenv_path.is_file():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    _load_dotenv()
    for var in ("QWEN_API_KEY", "DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY"):
        if os.environ.get(var):
            # ChatOpenAI reads OPENAI_API_KEY by default; copy the vendor key.
            os.environ["OPENAI_API_KEY"] = os.environ[var]
            break

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "No API key. Set QWEN_API_KEY / DASHSCOPE_API_KEY / DEEPSEEK_API_KEY "
            "(or OPENAI_API_KEY), or create robot_brain/.env."
        )

    from rai.agents.langchain.core.plan_agent import (
        create_initial_plan_execute_state,
        create_plan_execute_agent,
    )
    from rai.initialization import get_llm_model

    from skills import make_skills

    llm = get_llm_model(
        "complex_model", vendor="openai", config_path=str(HERE / "config.toml")
    )
    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", "?")
    base_url = getattr(llm, "openai_api_base", None) or getattr(llm, "base_url", "?")
    print(f"LLM: {llm.__class__.__name__} | base_url={base_url} | model={model_name}")

    agent = create_plan_execute_agent(
        tools=make_skills(),
        planner_llm=llm,
        executor_llm=llm,
        replanner_llm=llm,
        system_prompt=(
            "You drive an exhibition hall robot. Use only the registered skills. "
            "Do not invent actions outside the provided tools."
        ),
    )

    task = "去展柜旁拿一瓶水，然后回来递给用户"
    print(f"\n=== task ===\n{task}\n")

    result = agent.invoke(
        create_initial_plan_execute_state(task),
        config={"recursion_limit": 30},
    )

    print("\n=== plan ===")
    for step in result.get("plan", []):
        print(f"- {step}")

    print("\n=== past steps ===")
    for step, out in result.get("past_steps", []):
        print(f"- {step}\n  -> {out}")

    print("\n=== final response ===")
    print(result.get("response"))


if __name__ == "__main__":
    main()
