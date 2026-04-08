from __future__ import annotations

import json
import os
from typing import Any

import requests
from openai import OpenAI


ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_STEPS = 6
TASKS = ["easy", "medium", "hard"]
BENCHMARK = "openenv-data-cleaning"


SYSTEM_PROMPT = """You are controlling a data-cleaning environment.
Return exactly one JSON object with keys: type, column, fill_value.
Valid action types are: remove_duplicates, fill_nulls, normalize_name, cast_age_to_int, remove_invalid_rows.
Choose the single best next action from the observation.
Use fill_nulls with column='age' and fill_value=0 when age values are missing.
Set column and fill_value to null when not needed.
Do not include markdown fences or extra text."""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def build_client() -> OpenAI | None:
    if not HF_TOKEN:
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
    observation: dict[str, Any] | None = None
    final_score = 0.0
    rewards: list[float] = []
    steps_taken = 0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = reset_task(task)
        for step_index in range(1, MAX_STEPS + 1):
            action = (
                choose_action_with_model(client, observation)
                if client
                else choose_action_heuristic(observation)
            )
            result = step_task(action)
            observation = result["observation"]
            reward_value = float(result["reward"]["value"])
            final_score = float(result["info"]["grader_score"])
            done = bool(result["done"])
            error_value = result["info"].get("last_action_error")
            rewards.append(reward_value)
            steps_taken = step_index
            action_str = json.dumps(action, sort_keys=True, separators=(",", ":"))
            log_step(
                step=step_index,
                action=action_str,
                reward=reward_value,
                done=done,
                error=error_value,
            )
            if done:
                break
        success = final_score >= 1.0
    finally:
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

    return final_score


def main() -> None:
    client = build_client()
    for task in TASKS:
        run_task(task, client)


if __name__ == "__main__":
    main()
