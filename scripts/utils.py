import json
import os
import datetime


def get_best_model_dir(output_path: str) -> str:
    checkpoints = []
    for name in os.listdir(output_path):
        if not name.startswith("checkpoint-"):
            continue

        step = name.removeprefix("checkpoint-")
        if step.isdigit():
            checkpoints.append((int(step), os.path.join(output_path, name)))

    if not checkpoints:
        raise ValueError(f"No checkpoint directories in {output_path} !")

    latest_checkpoint = max(checkpoints, key=lambda item: item[0])[1]
    trainer_state_path = os.path.join(latest_checkpoint, "trainer_state.json")
    with open(trainer_state_path, "r", encoding="utf-8") as f:
        trainer_state = json.load(f)

    return trainer_state.get("best_model_checkpoint")


def get_utc_time_str(format="%Y-%m-%d_%H-%M-%S", timedelta_hours: int = 8):
    utc = datetime.timezone(datetime.timedelta(hours=timedelta_hours))
    current_time = datetime.datetime.now(utc)
    time_str = current_time.strftime(format)
    return time_str
