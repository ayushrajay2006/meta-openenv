from __future__ import annotations

import json
import os
from typing import Any

import requests
from openai import OpenAI


ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MAX_STEPS = 6
TASKS = ["easy", "medium", "hard"]


SYSTEM_PROMPT = """You are controlling a data-cleaning environment.
Return exactly one JSON object with keys: type, column, fill_value.
Valid action types are: remove_duplicates, fill_nulls, normalize_name, cast_age_to_int, remove_invalid_rows.
Choose the single best next action from the observation.
Use fill_nulls with column='age' and fill_value=0 when age values are missing.
Set column and fill_value to null when not needed.
Do not include markdown fences or extra text."""


def log(tag: str, payload: dict[str, Any]) -> None:
    print(f"[{tag}] {json.dumps(payload, sort_keys=True)}", flush=True)


def build_client() -> OpenAI | None:
    if not API_BASE_URL or not MODEL_NAME or not HF_TOKEN:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def reset_task(task: str) -> dict[str, Any]:
    response = requests.post(f"{ENV_BASE_URL}/reset", json={"task": task}, timeout=30)
    response.raise_for_status()
    return response.json()["observation"]


def step_task(action: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{ENV_BASE_URL}/step", json=action, timeout=30)
    response.raise_for_status()
    return response.json()


def parse_action(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    payload = json.loads(cleaned)
    action_type = payload.get("type") or payload.get("action_type")
    if not action_type:
        raise ValueError(f"Model response missing action type: {content}")
    return {
        "type": action_type,
        "column": payload.get("column"),
        "fill_value": payload.get("fill_value"),
    }


def choose_action_with_model(client: OpenAI, observation: dict[str, Any]) -> dict[str, Any]:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(observation, sort_keys=True)},
        ],
    )
    content = completion.choices[0].message.content or "{}"
    return parse_action(content)


def choose_action_heuristic(observation: dict[str, Any]) -> dict[str, Any]:
    metrics = observation["metrics"]
    if metrics["string_ages"] > 0:
        return {"type": "cast_age_to_int", "column": None, "fill_value": None}
    if metrics["non_title_names"] > 0:
        return {"type": "normalize_name", "column": None, "fill_value": None}
    if metrics["invalid_rows"] > 0:
        return {"type": "remove_invalid_rows", "column": None, "fill_value": None}
    if metrics["null_cells"] > 0:
        return {"type": "fill_nulls", "column": "age", "fill_value": 0}
    if metrics["duplicates"] > 0:
        return {"type": "remove_duplicates", "column": None, "fill_value": None}
    return {"type": "remove_duplicates", "column": None, "fill_value": None}


def run_task(task: str, client: OpenAI | None) -> float:
    observation = reset_task(task)
    log(
        "START",
        {
            "task": task,
            "difficulty": observation["difficulty"],
            "instruction": observation["instruction"],
            "planner": "model" if client else "heuristic",
        },
    )

    final_score = 0.0
    for step_index in range(1, MAX_STEPS + 1):
        action = choose_action_with_model(client, observation) if client else choose_action_heuristic(observation)
        result = step_task(action)
        observation = result["observation"]
        reward = result["reward"]
        info = result["info"]
        final_score = info["grader_score"]
        log(
            "STEP",
            {
                "task": task,
                "step": step_index,
                "action": action,
                "reward": reward["value"],
                "score": final_score,
                "done": result["done"],
            },
        )
        if result["done"]:
            break

    log("END", {"task": task, "final_score": final_score})
    return final_score


def main() -> None:
    client = build_client()
    scores = {task: run_task(task, client) for task in TASKS}
    mean_score = round(sum(scores.values()) / len(scores), 4)
    print(json.dumps({"scores": scores, "mean_score": mean_score}, sort_keys=True))


if __name__ == "__main__":
    main()
