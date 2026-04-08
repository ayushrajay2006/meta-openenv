from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel

from models import (
    Action,
    Observation,
    Reward,
    RowData,
    TaskSpec,
)


TASKS_DIR = Path(__file__).resolve().parent / "tasks"
VALID_ACTIONS = {
    "remove_duplicates",
    "fill_nulls",
    "normalize_name",
    "cast_age_to_int",
    "remove_invalid_rows",
}


class EnvironmentState(BaseModel):
    task_id: str | None = None
    difficulty: str | None = None
    instruction: str | None = None
    rows: list[RowData] = []
    expected_rows: list[RowData] = []
    step_count: int = 0
    max_steps: int = 6
    done: bool = False
    last_score: float = 0.0
    action_history: list[dict[str, Any]] = []


class CleanEnv:
    def __init__(self) -> None:
        self._state = EnvironmentState()

    def reset(self, task: str = "easy") -> Observation:
        task_spec = self._load_task(task)
        self._state = EnvironmentState(
            task_id=task_spec.task_id,
            difficulty=task_spec.difficulty,
            instruction=task_spec.instruction,
            rows=[row.model_copy(deep=True) for row in task_spec.initial_data],
            expected_rows=[row.model_copy(deep=True) for row in task_spec.expected_data],
            max_steps=task_spec.max_steps,
            last_score=0.0,
        )
        initial_score = self._compute_score(self._state.rows, self._state.expected_rows)
        self._state.last_score = initial_score
        return self._build_observation()

    def step(self, action: Action | dict[str, Any]) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        if self._state.task_id is None:
            raise ValueError("Environment has not been reset. Call reset(task=...) first.")

        action_model = action if isinstance(action, Action) else Action.model_validate(action)
        if self._state.done:
            reward = Reward(
                value=0.0,
                score=self._state.last_score,
                delta_score=0.0,
                action_success=False,
                message="Episode already completed.",
            )
            return self._build_observation(), reward, True, self._build_info()

        self._state.step_count += 1
        previous_rows = [row.model_copy(deep=True) for row in self._state.rows]
        previous_score = self._state.last_score
        changed_count = self._apply_action(action_model)
        current_score = self._compute_score(self._state.rows, self._state.expected_rows)
        self._state.last_score = current_score

        delta_score = current_score - previous_score
        invalid_or_noop = changed_count == 0
        loop_penalty = 0.08 if invalid_or_noop else 0.0
        step_pressure_penalty = 0.02
        shaped = (0.65 * current_score) + (0.35 * max(delta_score, 0.0)) - loop_penalty - step_pressure_penalty
        reward_value = round(min(1.0, max(0.0, shaped)), 4)

        is_exact = self._canonical_counter(self._state.rows) == self._canonical_counter(self._state.expected_rows)
        exhausted = self._state.step_count >= self._state.max_steps
        self._state.done = is_exact or exhausted

        self._state.action_history.append(
            {
                "step": self._state.step_count,
                "action": action_model.model_dump(),
                "changed_rows": changed_count,
                "score_after_step": current_score,
            }
        )

        reward = Reward(
            value=reward_value,
            score=round(current_score, 4),
            delta_score=round(delta_score, 4),
            action_success=not invalid_or_noop,
            message=self._build_reward_message(action_model, changed_count, previous_rows),
        )
        return self._build_observation(), reward, self._state.done, self._build_info()

    def state(self) -> dict[str, Any]:
        if self._state.task_id is None:
            return {"status": "idle"}
        return {
            "task_id": self._state.task_id,
            "difficulty": self._state.difficulty,
            "instruction": self._state.instruction,
            "rows": [row.model_dump() for row in self._state.rows],
            "expected_rows": [row.model_dump() for row in self._state.expected_rows],
            "step_count": self._state.step_count,
            "max_steps": self._state.max_steps,
            "done": self._state.done,
            "last_score": round(self._state.last_score, 4),
            "action_history": self._state.action_history,
        }

    def available_tasks(self) -> list[str]:
        return sorted(path.stem for path in TASKS_DIR.glob("*.json"))

    def _load_task(self, task: str) -> TaskSpec:
        path = TASKS_DIR / f"{task}.json"
        if not path.exists():
            raise FileNotFoundError(f"Task '{task}' was not found in {TASKS_DIR}.")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return TaskSpec.model_validate(payload)

    def _build_observation(self) -> Observation:
        metrics = {
            "rows": len(self._state.rows),
            "duplicates": self._count_duplicates(self._state.rows),
            "null_cells": self._count_null_cells(self._state.rows),
            "non_title_names": self._count_non_title_names(self._state.rows),
            "string_ages": self._count_string_ages(self._state.rows),
            "invalid_rows": self._count_invalid_rows(self._state.rows),
            "score": round(self._state.last_score, 4),
        }
        return Observation(
            task_id=self._state.task_id or "",
            difficulty=self._state.difficulty or "",
            instruction=self._state.instruction or "",
            data=[row.model_copy(deep=True) for row in self._state.rows],
            available_actions=sorted(VALID_ACTIONS),
            metrics=metrics,
            step_count=self._state.step_count,
            max_steps=self._state.max_steps,
        )

    def _build_info(self) -> dict[str, Any]:
        return {
            "grader_score": round(self._state.last_score, 4),
            "task_id": self._state.task_id,
            "difficulty": self._state.difficulty,
            "steps_remaining": max(0, self._state.max_steps - self._state.step_count),
            "completed_exact_match": self._canonical_counter(self._state.rows)
            == self._canonical_counter(self._state.expected_rows),
        }

    def _build_reward_message(
        self, action: Action, changed_count: int, previous_rows: list[RowData]
    ) -> str:
        if changed_count == 0:
            return f"Action '{action.action_type}' made no useful change."
        if action.action_type == "remove_duplicates":
            removed = len(previous_rows) - len(self._state.rows)
            return f"Removed {removed} duplicate rows."
        if action.action_type == "fill_nulls":
            return f"Filled {changed_count} null values in '{action.column}'."
        if action.action_type == "normalize_name":
            return f"Normalized {changed_count} names."
        if action.action_type == "cast_age_to_int":
            return f"Casted {changed_count} age values to integers."
        if action.action_type == "remove_invalid_rows":
            return f"Removed {changed_count} invalid rows."
        return f"Applied '{action.action_type}'."

    def _apply_action(self, action: Action) -> int:
        if action.action_type == "remove_duplicates":
            return self._remove_duplicates()
        if action.action_type == "fill_nulls":
            return self._fill_nulls(action.column, action.fill_value)
        if action.action_type == "normalize_name":
            return self._normalize_name()
        if action.action_type == "cast_age_to_int":
            return self._cast_age_to_int()
        if action.action_type == "remove_invalid_rows":
            return self._remove_invalid_rows()
        return 0

    def _remove_duplicates(self) -> int:
        seen: set[tuple[tuple[str, Any], ...]] = set()
        deduped: list[RowData] = []
        removed = 0
        for row in self._state.rows:
            key = self._row_key(row)
            if key in seen:
                removed += 1
                continue
            seen.add(key)
            deduped.append(row)
        self._state.rows = deduped
        return removed

    def _fill_nulls(self, column: str | None, fill_value: int | str | None) -> int:
        if not column:
            return 0
        replacement = 0 if fill_value is None else fill_value
        changed = 0
        for row in self._state.rows:
            value = getattr(row, column, None)
            if value is None:
                setattr(row, column, replacement)
                changed += 1
        return changed

    def _normalize_name(self) -> int:
        changed = 0
        for row in self._state.rows:
            if row.name is None:
                continue
            normalized = " ".join(part.capitalize() for part in row.name.strip().split())
            if row.name != normalized:
                row.name = normalized
                changed += 1
        return changed

    def _cast_age_to_int(self) -> int:
        changed = 0
        for row in self._state.rows:
            if isinstance(row.age, str):
                stripped = row.age.strip()
                if stripped.isdigit():
                    row.age = int(stripped)
                    changed += 1
        return changed

    def _remove_invalid_rows(self) -> int:
        kept: list[RowData] = []
        removed = 0
        for row in self._state.rows:
            if row.name is None and row.age is None:
                removed += 1
                continue
            kept.append(row)
        self._state.rows = kept
        return removed

    def _compute_score(self, rows: list[RowData], expected_rows: list[RowData]) -> float:
        if not expected_rows:
            return 1.0

        matched_current: set[int] = set()
        total = 0.0
        for expected in expected_rows:
            best_index = None
            best_score = 0.0
            for idx, current in enumerate(rows):
                if idx in matched_current:
                    continue
                candidate_score = self._row_similarity(current, expected)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_index = idx
            if best_index is not None:
                matched_current.add(best_index)
            total += best_score

        structure_penalty = 0.05 * abs(len(rows) - len(expected_rows))
        normalized = (total / len(expected_rows)) - structure_penalty
        return round(max(0.0, min(1.0, normalized)), 4)

    def _row_similarity(self, current: RowData, expected: RowData) -> float:
        checks = [
            1.0 if current.name == expected.name else 0.0,
            1.0 if current.age == expected.age else 0.0,
        ]
        if current.extra == expected.extra:
            checks.append(1.0)
        elif expected.extra is not None:
            checks.append(0.0)
        return sum(checks) / len(checks)

    def _canonical_counter(self, rows: Iterable[RowData]) -> Counter[tuple[tuple[str, Any], ...]]:
        return Counter(self._row_key(row) for row in rows)

    def _row_key(self, row: RowData) -> tuple[tuple[str, Any], ...]:
        data = row.model_dump(exclude_none=True)
        return tuple(sorted(data.items()))

    def _count_duplicates(self, rows: list[RowData]) -> int:
        counts = self._canonical_counter(rows)
        return sum(count - 1 for count in counts.values() if count > 1)

    def _count_null_cells(self, rows: list[RowData]) -> int:
        return sum((row.name is None) + (row.age is None) for row in rows)

    def _count_non_title_names(self, rows: list[RowData]) -> int:
        return sum(
            1
            for row in rows
            if isinstance(row.name, str) and row.name != " ".join(part.capitalize() for part in row.name.strip().split())
        )

    def _count_string_ages(self, rows: list[RowData]) -> int:
        return sum(1 for row in rows if isinstance(row.age, str))

    def _count_invalid_rows(self, rows: list[RowData]) -> int:
        return sum(1 for row in rows if row.name is None and row.age is None)
