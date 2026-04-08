from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from environment import CleanEnv
from models import Action, Observation, Reward


class ResetRequest(BaseModel):
    task: str = "easy"


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict


app = FastAPI(
    title="OpenEnv Data Cleaning Environment",
    version="1.0.0",
    description="A real-world data cleaning environment for benchmarking agentic dataset remediation.",
)
env = CleanEnv()


@app.get("/")
def root() -> dict:
    return {
        "name": "openenv-data-cleaning",
        "tasks": env.available_tasks(),
        "endpoints": ["/reset", "/step", "/state"],
    }


@app.post("/reset", response_model=dict)
def reset(req: ResetRequest) -> dict:
    try:
        observation = env.reset(req.task)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"observation": observation.model_dump()}


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
    try:
        observation, reward, done, info = env.step(action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StepResponse(
        observation=observation,
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state", response_model=dict)
def state() -> dict:
    return env.state()
