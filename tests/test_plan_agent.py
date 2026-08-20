import unittest

from langgraph.graph import END

from rai.agents.langchain.core.plan_agent import (
    Act,
    Plan,
    Response,
    _merge_replanner_output,
    should_end,
)


class PlanAgentReplannerTests(unittest.TestCase):
    def test_empty_replanned_steps_fall_back_to_current_plan(self) -> None:
        output = Act(action=Plan(type="plan", steps=[]))

        updated_plan, response = _merge_replanner_output(output, ["抓取一瓶水"])

        self.assertEqual(updated_plan, ["抓取一瓶水"])
        self.assertIsNone(response)

    def test_blank_response_falls_back_to_current_plan(self) -> None:
        output = Act(action=Response(type="response", response="   "))

        updated_plan, response = _merge_replanner_output(output, ["抓取一瓶水"])

        self.assertEqual(updated_plan, ["抓取一瓶水"])
        self.assertIsNone(response)

    def test_should_end_when_plan_and_response_are_empty(self) -> None:
        self.assertEqual(should_end({"plan": [], "response": ""}), END)

    def test_should_end_when_response_exists(self) -> None:
        self.assertEqual(should_end({"plan": ["抓取一瓶水"], "response": "完成"}), END)


if __name__ == "__main__":
    unittest.main()
