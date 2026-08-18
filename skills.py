"""Mock skill tools for the exhibition robot brain.

Skills follow the rai_core skill-definition pattern (LangChain BaseTool +
pydantic args_schema, cf. ``rai.tools.ros2.base.BaseROS2Tool``) but are
ROS-free. They are stand-ins for the real robot skill APIs declared in the
requirements doc (``navigate_to``, ``pick``, ``handover``, ``orient_to``).

Each tool declares preconditions / success conditions in its description so
the LLM planner can reason about what is safe to call and how to verify it.
"""

from typing import List

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class RobotState:
    """Shared mutable mock robot state (single writer per skill, simplified).

    Stands in for the WorldState/TaskState stores from the requirements doc.
    """

    def __init__(self) -> None:
        self.location: str | None = None
        self.holding: str | None = None


class NavigateToArgs(BaseModel):
    location_id: str = Field(
        description="取值之一：display_area（展柜/展区）、reception（接待台）、door_main（大门）"
    )


class NavigateToTool(BaseTool):
    """Wheeled-base navigation skill.

    Preconditions: system.localization == healthy, path not blocked.
    Success condition: robot arrived at location_id.
    """

    name: str = "navigate_to"
    description: str = (
        "将轮式底盘导航到语义位置。"
        "合法位置：display_area（展柜/展区）、reception（接待台）、door_main（大门）。"
        "前置条件：system.localization 正常、路径畅通。"
        "成功条件：机器人已到达 location_id。"
    )
    args_schema: type[BaseModel] = NavigateToArgs
    known_locations: List[str] = Field(
        default_factory=lambda: ["display_area", "reception", "door_main"]
    )
    state: RobotState

    def _run(self, location_id: str) -> str:
        if self.known_locations and location_id not in self.known_locations:
            raise ValueError(f"unknown location: {location_id}")
        self.state.location = location_id
        return f"arrived at {location_id}"


class PickArgs(BaseModel):
    object_id: str = Field(
        description="取值之一：water_1（瓶装水）、water_2（瓶装水）、brochure_1（宣传册）"
    )


class PickTool(BaseTool):
    """Manipulator pick skill.

    Preconditions: robot stopped, target object fresh and reachable.
    Success condition: robot.holding_object_id == object_id.
    """

    name: str = "pick"
    description: str = (
        "用机械臂抓取物体。"
        "合法物体：water_1（瓶装水）、water_2（瓶装水）、brochure_1（宣传册）。"
        "前置条件：机器人底盘已停止、目标物体在位且可触及。"
        "若机器人已持物则失败。"
        "成功条件：robot.holding_object_id == object_id。"
    )
    args_schema: type[BaseModel] = PickArgs
    grabbable_objects: List[str] = Field(
        default_factory=lambda: ["water_1", "water_2", "brochure_1"]
    )
    state: RobotState

    def _run(self, object_id: str) -> str:
        if self.grabbable_objects and object_id not in self.grabbable_objects:
            raise ValueError(f"object not grabbable: {object_id}")
        if self.state.holding is not None:
            raise ValueError(
                f"already holding {self.state.holding}; hand it over before picking again"
            )
        self.state.holding = object_id
        return f"picked {object_id} (tactile confirmed)"


class HandoverArgs(BaseModel):
    person_id: str = Field(description="取值之一：person_1、person_2")


class HandoverTool(BaseTool):
    """Object handover skill.

    Preconditions: robot holding an object, robot stopped at recipient.
    Success condition: object delivered to person_id, robot no longer holding it.
    """

    name: str = "handover"
    description: str = (
        "把持有的物体交给某人。"
        "合法对象：person_1、person_2。"
        "前置条件：机器人持有物体且已停在接收者旁。"
        "若机器人未持有任何物体则失败。"
        "成功条件：物体已交付给 person_id，机器人不再持有。"
    )
    args_schema: type[BaseModel] = HandoverArgs
    known_persons: List[str] = Field(default_factory=lambda: ["person_1", "person_2"])
    state: RobotState

    def _run(self, person_id: str) -> str:
        if self.known_persons and person_id not in self.known_persons:
            raise ValueError(f"unknown person: {person_id}")
        if self.state.holding is None:
            raise ValueError("not holding any object to hand over")
        delivered = self.state.holding
        self.state.holding = None
        return f"delivered held object ({delivered}) to {person_id}"


class OrientToArgs(BaseModel):
    person_or_entity_id: str = Field(
        description="需要朝向的某人或某实体的 id"
    )


class OrientToTool(BaseTool):
    """Sensor-head orient skill.

    Preconditions: none (low risk). Success condition: robot facing person/entity.
    """

    name: str = "orient_to"
    description: str = (
        "让机器人朝向某人或某实体。"
        "前置条件：无，低风险。"
        "成功条件：机器人传感器朝向 person_or_entity_id。"
    )
    args_schema: type[BaseModel] = OrientToArgs

    def _run(self, person_or_entity_id: str) -> str:
        return f"oriented toward {person_or_entity_id}"


def make_skills() -> List[BaseTool]:
    state = RobotState()
    return [
        NavigateToTool(state=state),
        PickTool(state=state),
        HandoverTool(state=state),
        OrientToTool(),
    ]
