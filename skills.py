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
        description="One of: display_area (展柜/展区), reception (接待台), door_main (大门)"
    )


class NavigateToTool(BaseTool):
    """Wheeled-base navigation skill.

    Preconditions: system.localization == healthy, path not blocked.
    Success condition: robot arrived at location_id.
    """

    name: str = "navigate_to"
    description: str = (
        "Navigate the wheeled base to a semantic location. "
        "Valid locations: display_area (展柜/展区), reception (接待台), door_main (大门). "
        "Preconditions: system.localization == healthy, path not blocked. "
        "Success condition: robot arrived at location_id."
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
        description="One of: water_1 (瓶装水), water_2 (瓶装水), brochure_1 (宣传册)"
    )


class PickTool(BaseTool):
    """Manipulator pick skill.

    Preconditions: robot stopped, target object fresh and reachable.
    Success condition: robot.holding_object_id == object_id.
    """

    name: str = "pick"
    description: str = (
        "Pick up an object with the manipulator. "
        "Valid objects: water_1 (瓶装水), water_2 (瓶装水), brochure_1 (宣传册). "
        "Preconditions: robot base stopped, target object fresh and reachable. "
        "Fails if the robot is already holding an object. "
        "Success condition: robot.holding_object_id == object_id."
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
    person_id: str = Field(description="One of: person_1, person_2")


class HandoverTool(BaseTool):
    """Object handover skill.

    Preconditions: robot holding an object, robot stopped at recipient.
    Success condition: object delivered to person_id, robot no longer holding it.
    """

    name: str = "handover"
    description: str = (
        "Hand over the held object to a person. "
        "Valid persons: person_1, person_2. "
        "Preconditions: robot holding an object and stopped at recipient. "
        "Fails if the robot is not holding any object. "
        "Success condition: object delivered to person_id, robot no longer holding it."
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
        description="Person or entity id to orient the sensor head toward"
    )


class OrientToTool(BaseTool):
    """Sensor-head orient skill.

    Preconditions: none (low risk). Success condition: robot facing person/entity.
    """

    name: str = "orient_to"
    description: str = (
        "Orient the robot toward a person or entity. "
        "Preconditions: none, low risk. "
        "Success condition: robot sensor head facing person_or_entity_id."
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
