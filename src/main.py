import os
import wandb
from dotenv import load_dotenv
from torch.utils.data._utils import pin_memory
from ultralytics import YOLO, settings


def main():
    # Resolve paths relative to project root (works regardless of current working directory)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'yolo11n.pt')
    CONFIG_PATH = os.path.join(PROJECT_ROOT, 'data', 'yolo_config.yaml')

    settings.update({"wandb": True})

    load_dotenv()
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    wandb.login(key=WANDB_API_KEY)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    model = YOLO(MODEL_PATH)
    model.train(data=CONFIG_PATH, epochs=1, imgsz=1024, batch=16,  project="Rail-Bench-Object-Detection-Eric-Toudik",
                name="test1"
)

if __name__ == '__main__':
    main()