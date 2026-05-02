import os
from pprint import pformat

import wandb
from dotenv import load_dotenv
from ultralytics import YOLO, settings
from read_config import read_config
import albumentations as A


def main():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'yolo11s.pt')
    DATA_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'yolo_config.yaml')

    RUN_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "run_config_basic_augmentations.yaml")
    run_config = read_config(RUN_CONFIG_PATH)

    RUN_NAME = f"baseline_{run_config["model_name"]}_{run_config["epochs"]}epochs_{run_config["image_size"]}_eric"
    USER_DESCRIPTION = f"test"
    run_description = pformat(run_config, indent=2, sort_dicts=False)

    description = f"{USER_DESCRIPTION} \n\n {run_description}"
    print(description)
    settings.update({"wandb": True})

    load_dotenv()
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    wandb.login(key=WANDB_API_KEY)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(DATA_CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found: {DATA_CONFIG_PATH}")

    train_params = {
        "data": DATA_CONFIG_PATH,
        "epochs": run_config["epochs"],
        "imgsz": run_config["image_size"],
        "batch": run_config["batch_size"],
        "name": RUN_NAME,
        "seed": run_config["seed"],
        "lr0": run_config["lr0"],
        "lr1": run_config["lr1"],
        "auto_augment": run_config["auto_augment"],

        # Color augmentations
        "hsv_h": run_config["hsv_h"],
        "hsv_s": run_config["hsv_s"],
        "hsv_v": run_config["hsv_v"],

        # Geometric augmentations
        "translate": run_config["translate"],
        "scale": run_config["scale"],
        "fliplr": run_config["fliplr"]
    }

    custom_augmentations = []
    if run_config.get("blur", False):
        custom_augmentations.append(A.Blur(blur_limit=run_config["blur_limit"], p=run_config["blur_p"]))
    if run_config.get("motion_blur", False):
        custom_augmentations.append(A.MotionBlur(blur_limit=run_config["motion_blur_limit"], p=run_config["motion_blur_p"]))

    if custom_augmentations:
        train_params.update({"augmentations": custom_augmentations})

    with wandb.init(
            project="Rail-Bench-Object-Detection-Eric-Toudik",
            name=RUN_NAME,
            notes=description,
            tags=[f"{run_config["model_name"]}"]
    ) as run:

        model = YOLO(MODEL_PATH)
        model.train(**train_params)

if __name__ == '__main__':
    main()