# sahi_predict_wandb.py

import os
from pathlib import Path

import wandb
from dotenv import load_dotenv


def download_model_from_wandb(wb_project, wb_artifact, run_name) -> Path:
    load_dotenv()

    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    else:
        wandb.login()

    run = wandb.init(
        name=run_name,
        project=wb_project,
        job_type="sahi_inference",
    )

    artifact = run.use_artifact(wb_artifact, type="model")
    artifact_dir = Path(artifact.download())

    candidates = list(artifact_dir.rglob("best.pt"))

    if not candidates:
        candidates = list(artifact_dir.rglob("*.pt"))

    if not candidates:
        raise FileNotFoundError(
            f"No .pt model file found in downloaded artifact directory: {artifact_dir}"
        )

    model_path = candidates[0]
    print(f"Using model: {model_path}")

    return model_path, run


