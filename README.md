---
title: OpenEnv Data Cleaning
emoji: "🧹"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# OpenEnv Data Cleaning Environment

This project implements a real-world OpenEnv-style environment for tabular data cleaning. The agent acts like an operations analyst cleaning small CRM-style datasets by applying concrete remediation actions such as deduplication, null filling, string normalization, schema casting, and invalid-row removal.

The environment is intentionally narrow and deterministic. That matters because hackathon judges are not rewarding vague “agent sandbox” ideas. They are rewarding environments that can be validated, graded, reproduced, and actually used to benchmark models.

## Why this environment is useful

Data cleaning is not a toy problem. Teams do this constantly before analytics, CRM imports, reporting, and model training. The environment captures a realistic agent loop:

- observe a messy dataset and task instruction
- choose one cleaning action
- receive a shaped reward and updated score
- stop when the dataset matches the expected cleaned output or the episode budget is exhausted

## Environment API

The FastAPI service exposes the required OpenEnv endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`

The server starts with:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Observation space

Each observation contains:

- `task_id`: active task identifier
- `difficulty`: `easy`, `medium`, or `hard`
- `instruction`: natural-language task objective
- `data`: current dataset rows
- `available_actions`: valid action types
- `metrics`: live counts for duplicates, null cells, invalid rows, bad names, string ages, and grader score
- `step_count` and `max_steps`

### Action space

Supported actions:

- `remove_duplicates`
- `fill_nulls`
- `normalize_name`
- `cast_age_to_int`
- `remove_invalid_rows`

Action payload example:

```json
{
  "type": "fill_nulls",
  "column": "age",
  "fill_value": 0
}
```

### Reward space

Each step returns a typed reward object:

- `value`: shaped reward in `[0.0, 1.0]`
- `score`: deterministic grader score in `[0.0, 1.0]`
- `delta_score`: score change caused by the action
- `action_success`: whether the action changed the dataset
- `message`: human-readable explanation

The reward combines current score, positive score delta, no-op penalties, and a small per-step pressure penalty. This gives partial progress signal across the trajectory instead of only at the end.

## Tasks

### Easy

Remove duplicates and fill missing ages with `0`.

### Medium

Normalize names to title case, cast ages from strings to integers, fill missing ages, and deduplicate rows.

### Hard

Clean a CRM export with whitespace noise, case inconsistencies, string ages, invalid rows, duplicates, and retained metadata in the `extra` field.

## Grading

The grader is deterministic and returns scores in `[0.0, 1.0]`.

- Exact matches receive `1.0`
- Partial progress is scored row-by-row using field similarity
- Structural mismatch is penalized when row counts diverge

This avoids the common failure mode where graders are effectively binary and useless for learning.

## Local setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Smoke test the environment:

```bash
python test_env.py
```

## Baseline inference

The baseline script is `inference.py` in the project root, as required.

Environment variables:

- `API_BASE_URL`: OpenAI-compatible inference endpoint
- `MODEL_NAME`: model identifier
- `HF_TOKEN`: token for the inference provider
- `OPENAI_API_KEY`: optional fallback if `HF_TOKEN` is not set
- `ENV_BASE_URL`: environment URL, defaults to `http://127.0.0.1:7860`

Run:

```bash
python inference.py
```

The script:

- uses the OpenAI Python client for model calls when credentials are provided
- emits structured `[START]`, `[STEP]`, and `[END]` logs
- falls back to a deterministic heuristic planner when model credentials are missing, so local smoke testing still works

### Reference scores

The deterministic heuristic policy reaches:

- `easy`: `1.0`
- `medium`: `1.0`
- `hard`: `1.0`

Mean score: `1.0`

For submission, use a fixed model with `temperature=0` to maximize reproducibility.

## Docker

Build:

```bash
docker build -t openenv-data-cleaning .
```

Run:

```bash
docker run -p 7860:7860 openenv-data-cleaning
```

## Hugging Face Spaces

Create a Docker Space and include the `openenv` tag in the Space metadata. The included `Dockerfile` serves the FastAPI app on port `7860`, which is consistent with common Space deployment expectations.

Live deployment:

- Space repo: [Champion2006/openenv-data-cleaning](https://huggingface.co/spaces/Champion2006/openenv-data-cleaning)
- App URL: [champion2006-openenv-data-cleaning.hf.space](https://champion2006-openenv-data-cleaning.hf.space)

Verified live endpoints:

- `POST /reset` returns a valid observation payload
- `GET /state` returns the current environment state

## Final baseline result

Verified against the deployed Hugging Face Space using `ENV_BASE_URL=https://champion2006-openenv-data-cleaning.hf.space`:

- `easy`: `1.0`
- `medium`: `1.0`
- `hard`: `1.0`
- mean score: `1.0`

## Files

- `app.py`: FastAPI service
- `environment.py`: environment state machine and reward/grading logic
- `models.py`: typed Pydantic contracts
- `tasks/`: task definitions
- `inference.py`: baseline runner
- `openenv.yaml`: environment metadata
- `Dockerfile`: container build
