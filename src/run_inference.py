import json
import os
import sys
from pathlib import Path

import numpy as np
import wandb
import torch
import yaml
from dotenv import load_dotenv
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torchvision.transforms as T

from src.calculate_railbench_benchmark import BenchmarkCalculator

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------
# Paths / Config
# ---------------------------------------------------------------------
TEST_PIPELINE = False
USE_SAHI = False  # Set  False for direct/"normal" inference instead of sahi
MODEL_BACKEND = "ecdet"  # "yolo" or "ecdet"
ECDET_MODEL_NAME = "large"  # "small", "medium", "large", or "xlarge"
ECDET_CHECKPOINT_SOURCE = "wandb"  # "wandb" or "local"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ECDET_LOCAL_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "EdgeCrafter" / "ecdetseg" / "outputs" / "ecdet_m" / "best.pth"

if TEST_PIPELINE:
    DATA_CONFIG_PATH = PROJECT_ROOT / "config" / "yolo_config_pipeline_test.yaml"
    DEFAULT_GT_COCO_PATH = PROJECT_ROOT / "data" / "pipeline_test" / "annotations_pipeline_test.json"
    YOLO_WANDB_ARTIFACT =  "disable eric-walsh-fau/Rail-Bench-Object-Detection-Eric-Toudik/run_tvxbvrao_model:v0"
    ECDET_WANDB_ARTIFACT = "disable eric-walsh-fau/Rail-Bench-Object-Detection-EdgeCrafter/sz437qhg_model:v0"

else:
    DATA_CONFIG_PATH = PROJECT_ROOT / "config" / "yolo_config.yaml"
    DEFAULT_GT_COCO_PATH = PROJECT_ROOT / "data" / "coco_annotations" / "objects" / "annotations_val.json"
    YOLO_WANDB_ARTIFACT = "eric-walsh-fau/Rail-Bench-Object-Detection-Eric-Toudik/run_keylpne9_model:v0"
    ECDET_WANDB_ARTIFACT = "eric-walsh-fau/Rail-Bench-Object-Detection-EdgeCrafter/3qpxzl2b_model:v0"

WANDB_ARTIFACT = ECDET_WANDB_ARTIFACT if MODEL_BACKEND == "ecdet" else YOLO_WANDB_ARTIFACT

if MODEL_BACKEND == "yolo":
    WANDB_PROJECT = "Rail-Bench-Object-Detection-Eric-Toudik"
else:
    WANDB_PROJECT = "Rail-Bench-Object-Detection-EdgeCrafter"

# ---------------------------------------------------------------------
# SAHI / Evaluation Parameters
# ---------------------------------------------------------------------

SPLIT = "val"

SLICE_HEIGHT = 2048
SLICE_WIDTH = 768
OVERLAP_HEIGHT_RATIO = 0.2
OVERLAP_WIDTH_RATIO = 0.3

# Für mAP-Evaluation bewusst niedrig wählen.
# Für reine Visualisierung wäre 0.25 sinnvoller.
CONF_THRESHOLD = 0.001

POSTPROCESS_MATCH_THRESHOLD = 0.5
DEVICE = "cuda:0"  # oder "cpu"

if USE_SAHI:
    RUN_NAME = (
        f"{MODEL_BACKEND}_sahi_{SPLIT}_"
        f"{SLICE_WIDTH}x{SLICE_HEIGHT}_"
        f"conf{str(CONF_THRESHOLD).replace('.', '')}_"
        f"overlap{str(OVERLAP_WIDTH_RATIO).replace('.', '')}"
    )
else:
    RUN_NAME = (
        f"{MODEL_BACKEND}_direct_{SPLIT}_"
        f"conf{str(CONF_THRESHOLD).replace('.', '')}"
    )


def init_wandb_run() -> None:
    if wandb.run is not None:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    load_dotenv()

    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    else:
        wandb.login()

    wandb.init(
        project=WANDB_PROJECT,
        name=RUN_NAME,
        dir=str(OUTPUT_DIR),
        job_type="sahi_inference_eval",
        config={
            "split": SPLIT,
            "slice_height": SLICE_HEIGHT,
            "slice_width": SLICE_WIDTH,
            "overlap_height_ratio": OVERLAP_HEIGHT_RATIO,
            "overlap_width_ratio": OVERLAP_WIDTH_RATIO,
            "confidence_threshold": CONF_THRESHOLD,
            "postprocess_match_threshold": POSTPROCESS_MATCH_THRESHOLD,
            "device": DEVICE,
            "wandb_artifact": WANDB_ARTIFACT,
            "model_backend": MODEL_BACKEND,
            "ecdet_model": ECDET_MODEL_NAME if MODEL_BACKEND == "ecdet" else None,
        },
    )


def download_model_from_wandb() -> Path:
    init_wandb_run()

    artifact = wandb.use_artifact(WANDB_ARTIFACT, type="model")
    artifact_name = WANDB_ARTIFACT.rsplit("/", 1)[-1].replace(":", "-")
    artifact_dir = Path(artifact.download(root=OUTPUT_DIR / "artifacts" / artifact_name))

    expected_model_file = "best.pth" if MODEL_BACKEND == "ecdet" else "best.pt"
    model_path = artifact_dir / expected_model_file

    if not model_path.exists():
        raise FileNotFoundError(
            f"Expected model file '{expected_model_file}' not found in downloaded artifact directory: {artifact_dir}"
        )

    print(f"Using model: {model_path}")

    return model_path


def resolve_ecdet_checkpoint_path() -> Path:
    """Resolve the ECDet checkpoint from W&B or from a local file."""
    if ECDET_CHECKPOINT_SOURCE == "wandb":
        return download_model_from_wandb()

    if ECDET_CHECKPOINT_SOURCE == "local":
        if not ECDET_LOCAL_CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"Local ECDet checkpoint not found: {ECDET_LOCAL_CHECKPOINT_PATH}"
            )

        print(f"Using local ECDet checkpoint: {ECDET_LOCAL_CHECKPOINT_PATH}")
        return ECDET_LOCAL_CHECKPOINT_PATH

    raise ValueError(
        "Unsupported ECDET_CHECKPOINT_SOURCE value: "
        f"{ECDET_CHECKPOINT_SOURCE!r}. Expected 'wandb' or 'local'."
    )


def print_yolo_model_information(model_path: Path) -> None:
    """Print available YOLO checkpoint metadata before running inference."""
    from ultralytics import YOLO

    print("\nYOLO model information:")
    print(f"W&B artifact: {WANDB_ARTIFACT}")
    print(f"Checkpoint path: {model_path}")

    model = YOLO(str(model_path))
    model_module = getattr(model, "model", None)
    overrides = getattr(model, "overrides", {}) or {}
    yaml_metadata = getattr(model_module, "yaml", None)

    try:
        print(f"Task: {getattr(model, 'task', None) or 'unknown'}")

        model_name = (
            getattr(model_module, "name", None)
            or overrides.get("model")
            or getattr(model_module, "yaml_file", None)
        )
        if model_name:
            print(f"Model name: {model_name}")

        if isinstance(yaml_metadata, dict):
            yaml_file = yaml_metadata.get("yaml_file") or yaml_metadata.get("model")
            scale = yaml_metadata.get("scale")
            if yaml_file:
                print(f"YAML metadata: {yaml_file}")
            if scale:
                print(f"Model scale: {scale}")

        names = getattr(model, "names", None)
        if isinstance(names, dict):
            print(f"Classes: {len(names)}")
            print(f"Class names: {names}")
        elif isinstance(names, list):
            print(f"Classes: {len(names)}")
            print(f"Class names: {dict(enumerate(names))}")
    except Exception as exc:
        print(f"Warning: Could not extract all YOLO model metadata: {exc}")

    try:
        model.info(verbose=True)
    except Exception as exc:
        print(f"Warning: Could not print Ultralytics model summary: {exc}")


# ---------------------------------------------------------------------
# Data Config Helpers
# ---------------------------------------------------------------------

def load_yolo_data_config(data_config_path: Path) -> dict:
    with open(data_config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_split_image_dir(data_config_path: Path, split: str) -> Path:
    data_cfg = load_yolo_data_config(data_config_path)

    if split not in data_cfg:
        raise KeyError(f"Split '{split}' not found in {data_config_path}")

    image_dir = Path(data_cfg[split])

    if not image_dir.is_absolute():
        image_dir = (data_config_path.parent / image_dir).resolve()

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    return image_dir


def infer_label_dir_from_image_dir(image_dir: Path, split: str) -> Path:
    """
    Expected common YOLO structure:
      data/images/val
      data/labels/val
    """
    image_dir_str = str(image_dir)

    old = str(Path("images") / split)
    new = str(Path("labels") / split)

    label_dir = Path(image_dir_str.replace(old, new))

    if not label_dir.exists():
        raise FileNotFoundError(
            f"Label directory not found: {label_dir}\n"
            f"Expected labels next to images, e.g. data/labels/{split}."
        )

    return label_dir


def get_default_gt_coco_path() -> Path:
    return DEFAULT_GT_COCO_PATH


# ---------------------------------------------------------------------
# YOLO Labels -> COCO Ground Truth
# ---------------------------------------------------------------------

def yolo_labels_to_coco_gt(
    data_config_path: Path,
    split: str,
    output_json_path: Path,
    category_id_offset: int = 0,
) -> Path:
    """
    Converts YOLO detection labels to COCO ground-truth JSON.

    YOLO format per line:
      class_id x_center y_center width height

    Normalized coordinates in [0, 1].

    category_id_offset:
      0 -> categories are 0..n-1
      1 -> categories are 1..n

    Start with 0. If COCOeval gives category mismatch problems,
    check SAHI result.json category_ids and adjust if needed.
    """
    data_cfg = load_yolo_data_config(data_config_path)
    names = data_cfg["names"]

    image_dir = resolve_split_image_dir(data_config_path, split)
    label_dir = infer_label_dir_from_image_dir(image_dir, split)

    categories = [
        {
            "id": class_id + category_id_offset,
            "name": class_name,
            "supercategory": "object",
        }
        for class_id, class_name in enumerate(names)
    ]

    images = []
    annotations = []

    annotation_id = 1
    image_id = 1

    image_paths = sorted(
        list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.jpeg"))
        + list(image_dir.glob("*.png"))
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    for image_path in image_paths:
        with Image.open(image_path) as img:
            width, height = img.size

        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        label_path = label_dir / f"{image_path.stem}.txt"

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()

                if not parts:
                    continue

                if len(parts) < 5:
                    raise ValueError(
                        f"Invalid YOLO label line in {label_path}: {line}"
                    )

                class_id = int(float(parts[0]))
                x_center = float(parts[1])
                y_center = float(parts[2])
                bbox_width = float(parts[3])
                bbox_height = float(parts[4])

                x_min = (x_center - bbox_width / 2.0) * width
                y_min = (y_center - bbox_height / 2.0) * height
                box_width = bbox_width * width
                box_height = bbox_height * height

                # Optional: clamp very small numerical issues
                x_min = max(0.0, x_min)
                y_min = max(0.0, y_min)
                box_width = min(box_width, width - x_min)
                box_height = min(box_height, height - y_min)

                if box_width <= 0 or box_height <= 0:
                    continue

                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id + category_id_offset,
                        "bbox": [
                            float(x_min),
                            float(y_min),
                            float(box_width),
                            float(box_height),
                        ],
                        "area": float(box_width * box_height),
                        "iscrowd": 0,
                    }
                )

                annotation_id += 1

        image_id += 1

    coco_gt = {
        "info": {
            "description": f"Temporary COCO GT generated from YOLO labels for split '{split}'"
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(coco_gt, f, indent=2)

    print(f"Saved COCO GT to: {output_json_path}")
    print(f"Images: {len(images)}")
    print(f"Annotations: {len(annotations)}")
    print(f"Categories: {len(categories)}")

    return output_json_path


# ---------------------------------------------------------------------
# Direct Prediction (without SAHI)
# ---------------------------------------------------------------------

def collect_image_paths(source_dir: Path) -> list[Path]:
    image_paths = sorted(
        list(source_dir.glob("*.jpg"))
        + list(source_dir.glob("*.jpeg"))
        + list(source_dir.glob("*.png"))
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {source_dir}")

    return image_paths


def run_direct_predict(model_path: Path, source_dir: Path, gt_coco_path: Path) -> Path:
    """
    Führt direkte Inference ohne SAHI durch und konvertiert die Ergebnisse in COCO-Format.
    
    Args:
        model_path: Pfad zum YOLO-Modell
        source_dir: Verzeichnis mit Bildern für Inference
        gt_coco_path: Pfad zur Ground-Truth COCO JSON
    
    Returns:
        Pfad zur JSON-Datei mit den Vorhersagen in COCO-Format
    """
    from ultralytics import YOLO

    print("\nRunning direct YOLO inference without SAHI...")
    print(f"Model: {model_path}")
    print(f"Source directory: {source_dir}")
    
    # Lade das YOLO-Modell
    model = YOLO(str(model_path))
    
    # Sammle alle Bilder im Verzeichnis
    image_paths = collect_image_paths(source_dir)
    print(f"Found {len(image_paths)} images for inference")

    # Führe Inference durch und sammle Ergebnisse
    predictions = []
    annotation_id = 1
    
    for count, image_path in enumerate(image_paths, 1):
        file_name = image_path.name
        
        # Führe Inference durch
        results = model.predict(
            source=str(image_path),
            conf=CONF_THRESHOLD,
            device=DEVICE,
            verbose=False
        )
        
        # Verarbeite Ergebnisse
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                # Extrahiere Bounding Boxes
                for i, box in enumerate(result.boxes):
                    # YOLO gibt normalized oder pixel-coordinates zurück,
                    # abhängig von der Konfiguration. Wir verwenden xyxy (corners)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Konvertiere zu COCO-Format [x, y, width, height]
                    bbox_x = float(x1)
                    bbox_y = float(y1)
                    bbox_w = float(x2 - x1)
                    bbox_h = float(y2 - y1)
                    
                    # Kategorie-ID (YOLO gibt 0-indizierte IDs zurück, GT ist aber 1-indiziert)
                    category_id = int(box.cls[0].item()) + 1

                    # Konfidenz-Score
                    confidence = float(box.conf[0].item())
                    
                    prediction = {
                        "file_name": file_name,
                        "category_id": category_id,
                        "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
                        "area": bbox_w * bbox_h,
                        "iscrowd": 0,
                        "score": confidence,
                        "id": annotation_id
                    }
                    predictions.append(prediction)
                    annotation_id += 1
        
        if (count) % 10 == 0:
            print(f"  Processed {count}/{len(image_paths)} images...")
    
    # Speichere Vorhersagen in COCO-Format
    output_dir = OUTPUT_DIR / RUN_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_path = output_dir / "result.json"
    
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Saved {len(predictions)} predictions to {result_path}")
    
    return result_path


def run_direct_ecdet_predict(model_path: Path, source_dir: Path, gt_coco_path: Path) -> Path:
    """
    Runs direct ECDet inference without SAHI and converts results to COCO format.
    """
    model, device, transform, model_name = _load_ecdet_model(model_path)

    print("\nRunning direct ECDet inference without SAHI...")
    print(f"Model: {model_name}")
    print(f"Checkpoint: {model_path}")
    print(f"Source directory: {source_dir}")

    image_paths = collect_image_paths(source_dir)
    print(f"Found {len(image_paths)} images for inference")

    predictions = []
    annotation_id = 1

    with torch.no_grad():
        for count, image_path in enumerate(image_paths, 1):
            file_name = image_path.name

            image = Image.open(image_path).convert("RGB")
            image_width, image_height = image.size
            image_tensor = transform(image).unsqueeze(0).to(device)
            orig_target_sizes = torch.tensor([[image_width, image_height]], device=device)

            outputs = model(image_tensor, orig_target_sizes)
            output = outputs[0]

            keep = output["scores"] >= CONF_THRESHOLD
            scores = output["scores"][keep]
            labels = output["labels"][keep]
            boxes = output["boxes"][keep]

            for score, label, box in zip(scores, labels, boxes):
                x_min, y_min, x_max, y_max = box.detach().cpu().tolist()
                bbox_width = float(x_max - x_min)
                bbox_height = float(y_max - y_min)

                if bbox_width <= 0 or bbox_height <= 0:
                    continue

                prediction = {
                    "file_name": file_name,
                    "category_id": int(label.item()),
                    "bbox": [
                        float(x_min),
                        float(y_min),
                        bbox_width,
                        bbox_height,
                    ],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0,
                    "score": float(score.item()),
                    "id": annotation_id,
                }
                predictions.append(prediction)
                annotation_id += 1

            if count % 10 == 0:
                print(f"  Processed {count}/{len(image_paths)} images...")

    output_dir = OUTPUT_DIR / RUN_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / "result.json"

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    print(f"Saved {len(predictions)} predictions to {result_path}")

    return result_path


def _load_ecdet_model(model_path: Path, device: torch.device | None = None) -> tuple[torch.nn.Module, torch.device, T.Compose, str]:
    """Load ECDet with the same preprocessing and category remapping used for direct inference."""
    from src import ecdet as ecdet_module

    ecdet_model = {
        "small": ecdet_module.ECDetModel.SMALL,
        "medium": ecdet_module.ECDetModel.MEDIUM,
        "large": ecdet_module.ECDetModel.LARGE,
        "xlarge": ecdet_module.ECDetModel.XLARGE,
    }[ECDET_MODEL_NAME]

    if device is None:
        device = torch.device(DEVICE if torch.cuda.is_available() or not DEVICE.startswith("cuda") else "cpu")

    original_read_config = ecdet_module.read_config

    def read_ecdet_config(config_path: str) -> dict:
        config = original_read_config(config_path)
        use_encoder_idx = config["encoder"]["use_encoder_idx"]
        config["encoder"]["use_encoder_idx"] = [
            encoder_idx for encoder_idx in use_encoder_idx if encoder_idx is not None
        ]
        return config

    ecdet_module.read_config = read_ecdet_config
    try:
        model = ecdet_module.get_model(
            str(model_path),
            model_size=ecdet_model,
            map_location=str(device),
        ).to(device)
    finally:
        ecdet_module.read_config = original_read_config
    model.eval()

    # ECDet emits internal labels; the postprocessor maps them back to RailBench category IDs.
    model.postprocessor.remap_mscoco_category = True

    image_size = model.config["decoder"]["eval_spatial_size"]
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return model, device, transform, ecdet_model.value


def _create_ecdet_sahi_detection_model(model_path: Path, load_at_init: bool = True) -> object:
    """Create the SAHI adapter only when ECDet sliced inference is requested."""
    from sahi.models.base import DetectionModel

    class _ECDetSahiDetectionModel(DetectionModel):
        """SAHI DetectionModel adapter for this project's ECDet wrapper."""

        def load_model(self) -> None:
            if self.model_path is None:
                raise ValueError("model_path is required for ECDet SAHI inference.")

            model, device, transform, _ = _load_ecdet_model(Path(self.model_path), self.device)
            self.device = device
            self.transform = transform
            self.set_model(model)

        def set_model(self, model: torch.nn.Module, **kwargs) -> None:
            self.model = model.to(self.device)
            self.model.eval()

        def perform_inference(self, image: np.ndarray) -> None:
            if self.model is None:
                raise ValueError("Model is not loaded, load it by calling .load_model()")

            pil_image = Image.fromarray(image).convert("RGB")
            image_width, image_height = pil_image.size
            image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            orig_target_sizes = torch.tensor([[image_width, image_height]], device=self.device)

            with torch.no_grad():
                self._original_predictions = self.model(image_tensor, orig_target_sizes)

        def _create_object_prediction_list_from_original_predictions(
            self,
            shift_amount_list: list[list[int]] | None = [[0, 0]],
            full_shape_list: list[list[int]] | None = None,
        ) -> None:
            from sahi.prediction import ObjectPrediction
            from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list

            shift_amount_list = fix_shift_amount_list(shift_amount_list)
            full_shape_list = fix_full_shape_list(full_shape_list)
            object_prediction_list_per_image = []

            for image_ind, output in enumerate(self._original_predictions):
                shift_amount = shift_amount_list[image_ind]
                full_shape = None if full_shape_list is None else full_shape_list[image_ind]
                keep = output["scores"] >= self.confidence_threshold
                scores = output["scores"][keep]
                labels = output["labels"][keep]
                boxes = output["boxes"][keep]

                object_prediction_list = []
                for score, label, box in zip(scores, labels, boxes):
                    x_min, y_min, x_max, y_max = box.detach().cpu().tolist()
                    bbox = [
                        max(0.0, float(x_min)),
                        max(0.0, float(y_min)),
                        float(x_max),
                        float(y_max),
                    ]

                    if full_shape is not None:
                        bbox[0] = min(float(full_shape[1]), bbox[0])
                        bbox[1] = min(float(full_shape[0]), bbox[1])
                        bbox[2] = min(float(full_shape[1]), bbox[2])
                        bbox[3] = min(float(full_shape[0]), bbox[3])

                    if not bbox[0] < bbox[2] or not bbox[1] < bbox[3]:
                        continue

                    category_id = int(label.item())
                    object_prediction_list.append(
                        ObjectPrediction(
                            bbox=bbox,
                            category_id=category_id,
                            category_name=_ecdet_sahi_category_name(category_id),
                            shift_amount=shift_amount,
                            score=float(score.item()),
                            full_shape=full_shape,
                        )
                    )

                object_prediction_list_per_image.append(object_prediction_list)

            self._object_prediction_list_per_image = object_prediction_list_per_image

    return _ECDetSahiDetectionModel(
        model_path=str(model_path),
        device=DEVICE,
        confidence_threshold=CONF_THRESHOLD,
        load_at_init=load_at_init,
    )


# ---------------------------------------------------------------------
# SAHI Prediction
# ---------------------------------------------------------------------

def run_sahi_predict(model_path: Path, source_dir: Path, gt_coco_path: Path) -> Path:
    from sahi.predict import predict

    detection_model = None
    predict_kwargs = {
        "model_type": "ultralytics",
        "model_path": str(model_path),
        "model_device": DEVICE,
        "model_confidence_threshold": CONF_THRESHOLD,
    }

    if MODEL_BACKEND == "ecdet":
        detection_model = _create_ecdet_sahi_detection_model(model_path)
        predict_kwargs = {}

    result = predict(
        detection_model=detection_model,
        **predict_kwargs,

        source=str(source_dir),

        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
        overlap_width_ratio=OVERLAP_WIDTH_RATIO,

        postprocess_type="NMS",
        postprocess_match_metric="IOU",
        postprocess_match_threshold=POSTPROCESS_MATCH_THRESHOLD,

        dataset_json_path=str(gt_coco_path),

        project=str(OUTPUT_DIR),
        name=RUN_NAME,

        no_standard_prediction=True,

        export_pickle=False,
        export_crop=False,
        visual_bbox_thickness=2,
        visual_text_size=1,

        # WICHTIG:
        # SAHI gibt dann den tatsächlich verwendeten Output-Ordner zurück.
        return_dict=True,
    )

    if result is None or "export_dir" not in result:
        raise RuntimeError(
            "SAHI predict() did not return an export_dir. "
            "Check whether prediction finished successfully."
        )

    prediction_dir = Path(result["export_dir"])

    print(f"SAHI actual export dir: {prediction_dir}")

    candidates = [
        prediction_dir / "result.json",
        prediction_dir / "results.json",
        prediction_dir / "prediction_result.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            print(f"Using prediction JSON: {candidate}")
            return candidate

    json_files = list(prediction_dir.rglob("*.json"))

    if not json_files:
        print("\nFiles created by SAHI:")
        for path in prediction_dir.rglob("*"):
            print(path)

        raise FileNotFoundError(
            f"No prediction JSON found in actual SAHI export dir: {prediction_dir}\n"
            f"GT COCO path exists: {gt_coco_path.exists()}\n"
            f"GT COCO path: {gt_coco_path}"
        )

    print(f"Using prediction JSON: {json_files[0]}")
    return json_files[0]


# ---------------------------------------------------------------------
# Save Predictions (custom test format)
# ---------------------------------------------------------------------

CATEGORY_NAME_TO_ID = {
    "train": 1,
    "catenary_pole": 2,
    "signal_pole": 3,
    "signal": 4,
    "person": 5,
    "road_vehicle": 6,
    "bicycle": 7,
}

ECDET_SAHI_CATEGORY_NAMES = tuple(
    category_name
    for category_name, category_id in sorted(CATEGORY_NAME_TO_ID.items(), key=lambda item: item[1])
)


def _ecdet_sahi_category_name(category_id: int) -> str:
    """Return the RailBench category name for ECDet SAHI's zero-based label ids."""
    if 0 <= category_id < len(ECDET_SAHI_CATEGORY_NAMES):
        return ECDET_SAHI_CATEGORY_NAMES[category_id]
    raise KeyError(category_id)


def _clamp_score(s: float) -> float:
    try:
        s = float(s)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, s))


def _validate_bbox(bbox) -> list:
    try:
        bbox_list = list(bbox)
    except Exception:
        raise ValueError(f"bbox must be an iterable with 4 numeric values. Got: {bbox}")
    if len(bbox_list) != 4:
        raise ValueError(f"bbox must contain 4 elements [u_min, v_min, w, h]. Got: {bbox_list}")
    try:
        bbox_list = [float(x) for x in bbox_list]
    except Exception:
        raise ValueError(f"bbox elements must be numeric: {bbox_list}")
    return bbox_list


def save_predictions_coco(
    pred_json_path: Path,
    gt_coco_path: Path | None,
    out_path: Path,
    score_threshold: float = 0.0,
    category_id_offset: int | None = None,
) -> Path:
    """
    Lädt ein COCO-artiges Prediction-JSON (Liste von Dicts) und speichert es im gewünschten
    Testformat (Liste von Objekten mit keys: file_name, category_id, bbox, score).

    - pred_json_path: Pfad zur SAHI/COCO Prediction JSON (typisch: result.json)
    - gt_coco_path: Pfad zur generierten COCO GT (wird verwendet, um image_id -> file_name zu mappen).
                    Falls None oder nicht existent, wird der Image-Ordner aus config verwendet.
    - out_path: Ausgabepfad (z.B. OUTPUT_DIR / 'pred_test.json')
    - score_threshold: optional, nur Vorhersagen mit score >= threshold werden gespeichert
    - category_id_offset: optionaler Offset fuer numerische Prediction-IDs.
                          SAHI/Ultralytics liefert 0..6, RAIL-BENCH erwartet 1..7.
                          None erkennt 0-basierte IDs automatisch.
    """
    with open(pred_json_path, "r", encoding="utf-8") as f:
        pred_raw = json.load(f)

    # SAHI/COCO outputs are usually a list; falls ein dict geliefert wird, versuchen wir, die Liste zu extrahieren
    # TODO can this be removed? (Not reuqired for direct at least)
    if isinstance(pred_raw, dict):
        # häufige keys: 'predictions', 'results', 'images', 'annotations'
        for key in ("predictions", "results", "images", "annotations", "detections"):
            if key in pred_raw and isinstance(pred_raw[key], list):
                preds = pred_raw[key]
                break
        else:
            # Fallback: suche Listeneinträge in der dict values
            list_vals = [v for v in pred_raw.values() if isinstance(v, list)]
            preds = list_vals[0] if list_vals else []
    elif isinstance(pred_raw, list):
        preds = pred_raw
    else:
        raise ValueError(f"Unrecognized prediction JSON structure: {type(pred_raw)}")

    # Lade GT COCO, um image_id -> file_name zu mappen
    image_id_to_name = {}
    if gt_coco_path is not None and gt_coco_path.exists():
        with open(gt_coco_path, "r", encoding="utf-8") as f:
            gt = json.load(f)

        image_id_to_name = {img["id"]: img["file_name"] for img in gt.get("images", [])}
    else:
        raise RuntimeError("Warning: could not resolve image dir from data config")

    raw_category_ids = []
    for pred in preds:
        if not isinstance(pred, dict) or "category_id" not in pred:
            continue
        try:
            raw_category_ids.append(int(pred["category_id"]))
        except (TypeError, ValueError):
            continue

    if category_id_offset is None:
        category_id_offset = 1 if raw_category_ids and min(raw_category_ids) == 0 else 0
    if category_id_offset == 1:
        print("Detected 0-based prediction category_ids. Mapping them to 1-based RAIL-BENCH category_ids.")

    results = []
    for p in preds:
        # if preds contains the file name already
        file_name = p.get("file_name") or p.get("image_file") or None

        if file_name is None and "image_id" in p:
            file_name = image_id_to_name.get(p["image_id"])

        # Falls nur image_id vorhanden ist, mappe ggf. über GT
        if file_name is None and "image_id" in p and p.get("image_id") in image_id_to_name:
            file_name = image_id_to_name.get(p.get("image_id"))

        # Falls kein file_name, versuche 'file_name' aus Pfadfragmenten / 'image_path'
        if file_name is None:
            for key in ("image_path", "path", "file"):
                if key in p and isinstance(p[key], str):
                    file_name = Path(p[key]).name
                    break

        if file_name is None:
            # wenn weiterhin unbekannt: versuchen wir, anhand available_file_names heuristisch zu matchen
            if isinstance(p.get("image_id"), int) and p.get("image_id") in image_id_to_name:
                file_name = image_id_to_name.get(p.get("image_id"))

        if file_name is None:
            # falls immer noch unbekannt: skip
            print(f"Warning: could not determine file name for prediction entry, skipping: {p}")
            continue

        # Kategorie
        cat_id = None
        cat_id_is_numeric_prediction = False
        if "category_id" in p:
            cat_id = p["category_id"]
            cat_id_is_numeric_prediction = True
        elif "category" in p:
            cat_id = p["category"]
        if isinstance(cat_id, str):
            # wenn Name gegeben, mappe
            if cat_id.isdigit():
                cat_id = int(cat_id)
                cat_id_is_numeric_prediction = True
            elif cat_id in CATEGORY_NAME_TO_ID:
                cat_id = CATEGORY_NAME_TO_ID[cat_id]

        if not isinstance(cat_id, int):
            # manchmal ist category_name vorhanden
            cat_name = p.get("category_name") or p.get("label")
            if isinstance(cat_name, str) and cat_name in CATEGORY_NAME_TO_ID:
                cat_id = CATEGORY_NAME_TO_ID[cat_name]

        # final: falls noch nicht int, versuche zu casten
        cat_id = int(cat_id)
        if cat_id_is_numeric_prediction:
            cat_id += category_id_offset

        # bbox
        bbox = p.get("bbox") or p.get("box") or p.get("bounding_box")
        if bbox is None and "x" in p and "y" in p and "w" in p and "h" in p:
            bbox = [p["x"], p["y"], p["w"], p["h"]]

        try:
            bbox = _validate_bbox(bbox)
        except Exception as e:
            print(f"Warning: invalid bbox for {file_name}, skipping: {e}")
            continue

        score = _clamp_score(p.get("score") or p.get("confidence") or 0.0)
        if score < float(score_threshold):
            continue

        if not isinstance(cat_id, int) or not (1 <= cat_id <= 7):
            print(f"Warning: invalid category_id for {file_name}: {cat_id}. Skipping entry.")
            continue

        results.append({
            "file_name": file_name,
            "category_id": int(cat_id),
            "bbox": bbox,
            "score": float(score),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Wrote {len(results)} normalized predictions to: {out_path}")
    return out_path


# ---------------------------------------------------------------------
# Category-ID Check / Fix
# ---------------------------------------------------------------------

def fix_gt_category_ids(gt_json_path: Path, category_id_offset: int = 1) -> None:
    """
    Passt die category_ids in der Ground Truth an, um sicherzustellen, dass sie
    mit den YOLO/SAHI-Vorhersagen übereinstimmen.

    - category_id_offset: +1 verschiebt category_ids von 0-6 auf 1-7

    Prüft zuerst, ob die Anpassung bereits durchgeführt wurde (nur verschieben wenn noch 0-6).
    Modifiziert die JSON-Datei IN-PLACE (überschreibt Original).
    """
    if not gt_json_path.exists():
        raise FileNotFoundError(f"Ground-truth JSON not found: {gt_json_path}")

    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    current_category_ids = sorted({c["id"] for c in gt.get("categories", [])})

    # Prüfe, ob Anpassung bereits durchgeführt wurde
    if min(current_category_ids) >= 1:
        print(f"Ground-truth hat bereits 1-indizierte Kategorien: {current_category_ids}. Keine Anpassung nötig.")
        return

    print(f"Aktuelle Kategorien: {current_category_ids}")

    # Verschiebe alle category_ids in den Kategorien
    for cat in gt.get("categories", []):
        original_cat_id = cat["id"]
        cat["id"] = cat["id"] + category_id_offset
        print(f"  Category '{cat.get('name')}': {original_cat_id} -> {cat['id']}")

    # Verschiebe alle category_ids in den Annotationen
    annotations_updated = 0
    for ann in gt.get("annotations", []):
        if 0 <= ann["category_id"] <= 6:  # Nur verschieben, wenn im YOLO-Bereich
            ann["category_id"] = ann["category_id"] + category_id_offset
            annotations_updated += 1

    # Speichern
    with open(gt_json_path, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2)

    new_category_ids = sorted({c["id"] for c in gt.get("categories", [])})
    print(f"Ground-truth Kategorien angepasst: {annotations_updated} Annotationen aktualisiert")
    print(f"  Neue Kategorien: {new_category_ids}")


def inspect_category_ids(gt_json_path: Path, pred_json_path: Path):
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    with open(pred_json_path, "r", encoding="utf-8") as f:
        pred = json.load(f)

    gt_category_ids = sorted({c["id"] for c in gt["categories"]})
    gt_annotation_category_ids = sorted({a["category_id"] for a in gt["annotations"]})
    pred_category_ids = sorted({p["category_id"] for p in pred})

    print("\nCategory ID check:")
    print(f"GT category ids:            {gt_category_ids}")
    print(f"GT annotation category ids: {gt_annotation_category_ids}")
    print(f"Prediction category ids:    {pred_category_ids}")

    return gt_category_ids, gt_annotation_category_ids, pred_category_ids


# Hilfsfunktion zur Korrektur der Predictions
# -------

def fix_predictions_category_ids(pred_json_path: Path, gt_json_path: Path) -> Path:
    """
    Korrigiert die category_ids in den Predictions, um sie mit der GT abzustimmen.
    
    YOLO gibt 0-indizierte Kategorien zurück, aber die GT kann 1-indiziert sein.
    Diese Funktion überprüft und korrigiert die Kategorien.
    
    Returns: Pfad zur korrigierten Prediction-Datei
    """
    with open(pred_json_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)
    
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    
    gt_category_ids = {c["id"] for c in gt_data.get("categories", [])}
    
    # Verarbeite Predictions
    if isinstance(pred_data, list) and len(pred_data) > 0:
        pred_cat_ids = {p.get("category_id") for p in pred_data if isinstance(p, dict) and "category_id" in p}
        
        if not pred_cat_ids:
            print("  ⚠️  Keine category_ids in Predictions gefunden")
            return pred_json_path
        
        # Bestimme den erforderlichen Offset
        gt_min = min(gt_category_ids) if gt_category_ids else 0
        pred_min = min(pred_cat_ids)
        pred_max = max(pred_cat_ids)
        
        print(f"  Pred kategorie range: [{pred_min}, {pred_max}]")
        print(f"  GT kategorie range: [{gt_min}, {max(gt_category_ids)}]")
        print(f"  Pred categories: {sorted(pred_cat_ids)}")
        print(f"  GT categories: {sorted(gt_category_ids)}")
        
        # Überprüfe ob Shift nötig ist
        # Wenn Pred [0-6] aber GT [1-7], verschiebe um +1
        # Wenn Pred [1-7] aber GT [1-7], kein shift nötig
        # Wenn Pred [0-3] aber GT [2-4] und es sind nur 4 items, könnte es auch [0-3] sein die zu [2-4] gehören
        
        needs_shift = False
        
        # Fall 1: Predictions beginnen mit 0, GT mit 1
        if pred_min == 0 and gt_min == 1:
            needs_shift = True
            offset = 1
            print(f"  ➡️  Detecting 0-based predictions, GT is 1-based. Shift by +{offset}")
        # Fall 2: Predictions und GT haben gleichen min aber unterschiedliche Werte
        elif pred_min == gt_min and pred_cat_ids != gt_category_ids:
            # Überprüfe ob es nur eine Subset-Issue ist (z.B. nicht alle Kategorien sind vorhanden)
            if pred_cat_ids <= gt_category_ids:
                print("  ℹ️  Predictions sind eine Subset von GT (nicht alle Kategorien vorhanden)")
                needs_shift = False
            else:
                # Predictions haben Kategorien die nicht in GT sind
                print(f"  ⚠️  Predictions haben Kategorien die nicht in GT sind: {pred_cat_ids - gt_category_ids}")
                needs_shift = False
        else:
            print("  ℹ️  Keine Anpassung nötig")
        
        if needs_shift:
            # Verschiebe alle category_ids um +offset
            for p in pred_data:
                if isinstance(p, dict) and "category_id" in p:
                    p["category_id"] = int(p["category_id"]) + offset
            
            # Speichere korrigierte Predictions
            with open(pred_json_path, "w", encoding="utf-8") as f:
                json.dump(pred_data, f, indent=2)
            
            print(f"  ✓ Predictions category_ids korrigiert (+{offset})")
        
        # Verifiziere nach Korrektur
        pred_cat_ids_after = {p.get("category_id") for p in pred_data if isinstance(p, dict) and "category_id" in p}
        print(f"  Nach Korrektur: Pred categories: {sorted(pred_cat_ids_after)}")
    
    return pred_json_path


# -------

# ---------------------------------------------------------------------
# COCO Evaluation
# ---------------------------------------------------------------------

def evaluate_coco(gt_json_path: Path, pred_json_path: Path) -> dict:
    print("\n🔍 Korrigiere Predictions vor Evaluation...")
    fix_predictions_category_ids(pred_json_path, gt_json_path)
    
    coco_gt = COCO(str(gt_json_path))
    coco_dt = coco_gt.loadRes(str(pred_json_path))

    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    metrics = {
        "sahi_coco/AP50-95": float(evaluator.stats[0]),
        "sahi_coco/AP50": float(evaluator.stats[1]),
        "sahi_coco/AP75": float(evaluator.stats[2]),
        "sahi_coco/AP_small": float(evaluator.stats[3]),
        "sahi_coco/AP_medium": float(evaluator.stats[4]),
        "sahi_coco/AP_large": float(evaluator.stats[5]),
        "sahi_coco/AR_1": float(evaluator.stats[6]),
        "sahi_coco/AR_10": float(evaluator.stats[7]),
        "sahi_coco/AR_100": float(evaluator.stats[8]),
        "sahi_coco/AR_small": float(evaluator.stats[9]),
        "sahi_coco/AR_medium": float(evaluator.stats[10]),
        "sahi_coco/AR_large": float(evaluator.stats[11]),
    }

    return metrics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print(f"Run name: {RUN_NAME}")
    print(f"Using {MODEL_BACKEND} backend with {'SAHI' if USE_SAHI else 'direct'} inference method")

    if MODEL_BACKEND not in {"yolo", "ecdet"}:
        raise ValueError(f"Unsupported MODEL_BACKEND: {MODEL_BACKEND}")

    if MODEL_BACKEND == "ecdet" and not ECDET_WANDB_ARTIFACT:
        raise ValueError("Set ECDET_WANDB_ARTIFACT to the W&B artifact path for the trained ECDet checkpoint.")

    if MODEL_BACKEND == "ecdet":
        model_path = resolve_ecdet_checkpoint_path()
    else:
        model_path = download_model_from_wandb()

    if MODEL_BACKEND == "yolo":
        print_yolo_model_information(model_path)

    source_dir = resolve_split_image_dir(DATA_CONFIG_PATH, SPLIT)
    gt_coco_path = DEFAULT_GT_COCO_PATH

    if not gt_coco_path.exists():
        raise FileNotFoundError(f"Ground-truth COCO file not found: {gt_coco_path}")

    # Stelle sicher, dass die Ground Truth 1-indizierte Kategorien hat (statt 0-indiziert von YOLO)
    # Dies macht sie konsistent mit den YOLO/SAHI-Vorhersagen nach unserem Offset
    try:
        print("\nAnpassung der Ground-Truth-Kategorien (0-6 -> 1-7)...")
        fix_gt_category_ids(gt_coco_path, category_id_offset=1)
    except Exception as e:
        print(f"Warning: Could not fix GT category IDs: {e}")

    # Wähle zwischen SAHI und direkter Inference
    if USE_SAHI:
        pred_json_path = run_sahi_predict(
            model_path=model_path,
            source_dir=source_dir,
            gt_coco_path=gt_coco_path,
        )
    elif MODEL_BACKEND == "ecdet":
        pred_json_path = run_direct_ecdet_predict(
            model_path=model_path,
            source_dir=source_dir,
            gt_coco_path=gt_coco_path,
        )
    else:
        pred_json_path = run_direct_predict(
            model_path=model_path,
            source_dir=source_dir,
            gt_coco_path=gt_coco_path,
        )

    pred_test_path = None

    try:
        pred_test_path = OUTPUT_DIR / "pred_test.json"
        save_predictions_coco(
            pred_json_path,
            gt_coco_path,
            pred_test_path,
            score_threshold=0.2,
            category_id_offset=1 if USE_SAHI and MODEL_BACKEND == "yolo" else 0,
        )
    except Exception as e:
        print(f"Warning: failed to write normalized pred_test.json: {e}")

    eval_pred_path = pred_test_path if pred_test_path is not None and pred_test_path.exists() else pred_json_path
    print(f"\nInspecting category IDs for evaluation predictions: {eval_pred_path}")
    inspect_category_ids(
        gt_json_path=gt_coco_path,
        pred_json_path=eval_pred_path,
    )

   # metrics = evaluate_coco(
   #     gt_json_path=gt_coco_path,
    #    pred_json_path=pred_json_path,
   # )

    #print("\nCOCO Metrics:")
    #for key, value in metrics.items():
    #    print(f"{key}: {value:.4f}")

    #wandb.log(metrics)

    init_wandb_run()

    eval_artifact = wandb.Artifact(
        name=f"{RUN_NAME}_eval_files",
        type="evaluation",
        description=f"{'SAHI' if USE_SAHI else 'Direct'} predictions and existing COCO ground-truth for evaluation.",
    )

    eval_artifact.add_file(str(gt_coco_path))
    eval_artifact.add_file(str(eval_pred_path))
    wandb.log_artifact(eval_artifact)

    wandb.finish()

    benchmark_calculator = BenchmarkCalculator(test_pipeline=TEST_PIPELINE)
    benchmark_calculator.run_railbench_format_check(is_railbench_test=False)
    benchmark_calculator.prepare_and_run_object_eval(
        split="val",
        detector_name=RUN_NAME,
        project_name="eric-toudik-railbench",
    )

if __name__ == "__main__":
    main()

