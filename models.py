from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class RowData(BaseModel):
    name: str | None = None
    age: int | str | None = None
    extra: str | None = None


class Action(BaseModel):
    action_type: Literal[
        "remove_duplicates",
        "fill_nulls",
        "normalize_name",
        "cast_age_to_int",
        "remove_invalid_rows",
    ] = Field(alias="type")
    column: Literal["name", "age", "extra"] | None = None
    fill_value: int | str | None = None

    model_config = {"populate_by_name": True}


class Reward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    score: float = Field(ge=0.0, le=1.0)
    delta_score: float = Field(ge=-1.0, le=1.0)
    action_success: bool
    message: str


class Observation(BaseModel):
    task_id: str
    difficulty: str
    instruction: str
    data: list[RowData]
    available_actions: list[str]
    metrics: dict[str, float | int]
    step_count: int
    max_steps: int


class TaskSpec(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    instruction: str
    initial_data: list[RowData]
    expected_data: list[RowData]
    max_steps: int = Field(default=6, ge=1, le=20)

    @field_validator("initial_data", "expected_data")
    @classmethod
    def non_empty_rows(cls, rows: list[RowData]) -> list[RowData]:
        if not rows:
            raise ValueError("Task datasets must not be empty.")
        return rows
