from environment import CleanEnv


POLICY = {
    "easy": [
        {"type": "remove_duplicates"},
        {"type": "fill_nulls", "column": "age", "fill_value": 0},
    ],
    "medium": [
        {"type": "cast_age_to_int"},
        {"type": "normalize_name"},
        {"type": "fill_nulls", "column": "age", "fill_value": 0},
        {"type": "remove_duplicates"},
    ],
    "hard": [
        {"type": "cast_age_to_int"},
        {"type": "normalize_name"},
        {"type": "remove_invalid_rows"},
        {"type": "fill_nulls", "column": "age", "fill_value": 0},
        {"type": "remove_duplicates"},
    ],
}


def run_task(task_name: str) -> None:
    env = CleanEnv()
    observation = env.reset(task_name)
    print(f"\nTask={task_name} initial_score={observation.metrics['score']}")
    done = False
    for action in POLICY[task_name]:
        observation, reward, done, info = env.step(action)
        print(action, reward.model_dump(), info)
        if done:
            break


if __name__ == "__main__":
    for task in ("easy", "medium", "hard"):
        run_task(task)
