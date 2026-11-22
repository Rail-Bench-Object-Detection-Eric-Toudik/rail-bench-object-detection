#!/usr/bin/env python3
"""Hybrid ECDet-L inference for the RAIL-BENCH Object Detection Challenge.

The hybrid model combines two ECDet-L checkpoints (input resolution 1024):

* ``direct`` checkpoint: trained on full images, applied to the whole image
  (no slicing). Its predictions are used for the class ``train`` only.
* ``sahi`` checkpoint: trained on image slices, applied with SAHI (sliced
  inference, 768 x 2048 slices). Its predictions are used for all other classes.

The result is written as ``pred_test.json`` in the format of the official
RAIL-BENCH toolkit (list of ``file_name``, ``category_id`` (1..7),
``bbox`` ([x, y, width, height] in pixels) and ``score``).

Model loading (``get_model``), preprocessing and sliced inference
(``sahi.predict.predict``) follow ``src/run_inference.py`` of this repository.
The NMS threshold of the slice merge (``--nms-threshold``) and the score
threshold of the output file (``--score-threshold``) can be set on the
command line; their defaults are defined by the constants below.

Example:
    python hybrid_inference.py --image-dir path/to/test_images
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.calculate_railbench_benchmark import BenchmarkCalculator

REPO_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------
# Classes (RAIL-BENCH category ids are one-based: 1..7)
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
CATEGORY_NAMES = tuple(
    name for name, _ in sorted(CATEGORY_NAME_TO_ID.items(), key=lambda item: item[1])
)
TRAIN_CATEGORY_ID = CATEGORY_NAME_TO_ID["train"]

# Label indexing of the ECDet checkpoints (set manually, no auto-detection):
# - "zero_based": model outputs 0..6
# - "one_based":  model outputs 1..7
# The checkpoints of this project output one-based labels.
DEFAULT_LABEL_INDEXING = "auto"

# ---------------------------------------------------------------------
# Inference parameters (as in src/run_inference.py)
# ---------------------------------------------------------------------
SLICE_HEIGHT = 2048
SLICE_WIDTH = 768
OVERLAP_HEIGHT_RATIO = 0.2
OVERLAP_WIDTH_RATIO = 0.3

# Confidence threshold of the model branches (deliberately low for mAP).
CONF_THRESHOLD = 0.001
# NMS (IOU) threshold used by SAHI to merge the predictions of overlapping slices.
DEFAULT_POSTPROCESS_MATCH_THRESHOLD = 0.5
# Predictions with a lower score are discarded when the result file is written.
DEFAULT_EXPORT_SCORE_THRESHOLD = 0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------
# Label indexing
# ---------------------------------------------------------------------
def validate_label_mode(label_mode: str) -> str:
    if label_mode not in {"auto", "zero_based", "one_based"}:
        raise ValueError(
            f"Unsupported label mode {label_mode!r}. Use 'auto', 'zero_based', or 'one_based'."
        )
    return label_mode


def detect_label_indexing(category_ids: list[int]) -> str:
    """Detect ECDet label indexing from the emitted category IDs."""
    unique_ids = set(category_ids)
    if 0 in unique_ids:
        return "zero_based"
    if unique_ids and all(1 <= category_id <= len(CATEGORY_NAMES) for category_id in unique_ids):
        return "one_based"
    raise ValueError(
        f"Could not determine label indexing from ECDet category IDs: {sorted(unique_ids)}"
    )


def resolve_auto_label_indexing(category_ids: list[int], label_indexing: str) -> str:
    label_indexing = validate_label_mode(label_indexing)
    if label_indexing == "auto":
        return detect_label_indexing(category_ids)
    return label_indexing


def to_railbench_category_id(category_id: int, label_mode: str) -> int:
    """Convert the configured label indexing to RAIL-BENCH one-based ids (1..7)."""
    label_mode = validate_label_mode(label_mode)
    max_category_id = len(CATEGORY_NAMES)

    if label_mode == "zero_based":
        if 0 <= category_id < max_category_id:
            return category_id + 1
        raise KeyError(category_id)

    if 1 <= category_id <= max_category_id:
        return category_id
    raise KeyError(category_id)


# ---------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------
def collect_image_paths(image_dir: Path) -> list[Path]:
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No images (.jpg, .jpeg, .png) found in {image_dir}")
    return image_paths


def resolve_device(requested_device: str):
    import torch

    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU (this will be slow).")
        return torch.device("cpu")
    return torch.device(requested_device)


def resolve_project_path(path: Path) -> Path:
    """Resolve CLI paths relative to the project root instead of the shell CWD."""
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved.resolve()


def resolve_image_dir(image_dir: Path, split: str) -> Path:
    """Resolve an image directory and support split directories under data/images."""
    resolved = resolve_project_path(image_dir)
    if resolved.is_dir():
        return resolved

    # Common project layout: data/images/{train,val,test}.
    data_images_split = resolve_project_path(Path("data") / "images" / split)
    if data_images_split.is_dir():
        return data_images_split

    raise FileNotFoundError(f"Image directory not found: {resolved}")


# ---------------------------------------------------------------------
# ECDet model
# ---------------------------------------------------------------------
def load_ecdet_model(checkpoint_path: Path, config_path: Path, device):
    """Load an ECDet checkpoint together with its preprocessing."""
    import torchvision.transforms as T

    from src import ecdet as ecdet_module

    original_read_config = ecdet_module.read_config

    def read_ecdet_config(_config_path: str) -> dict:
        config = original_read_config(str(config_path))
        use_encoder_idx = config["encoder"]["use_encoder_idx"]
        config["encoder"]["use_encoder_idx"] = [
            encoder_idx for encoder_idx in use_encoder_idx if encoder_idx is not None
        ]
        return config

    ecdet_module.read_config = read_ecdet_config
    try:
        model = ecdet_module.get_model(
            str(checkpoint_path),
            map_location=str(device),
        ).to(device)
    finally:
        ecdet_module.read_config = original_read_config
    model.eval()

    # ECDet emits internal labels; the postprocessor maps them back to category ids.
    model.postprocessor.remap_mscoco_category = True

    image_size = model.config["decoder"]["eval_spatial_size"]
    transform = T.Compose(
        [
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, transform


# ---------------------------------------------------------------------
# Branch 1: direct inference on the full image
# ---------------------------------------------------------------------
def predict_direct(
        checkpoint_path: Path,
        config_path: Path,
        device,
        image_paths: list[Path],
        label_indexing: str,
) -> list[dict]:
    import torch
    from PIL import Image

    model, transform = load_ecdet_model(checkpoint_path, config_path, device)
    print(f"\nDirect inference (full images): {checkpoint_path}")

    raw_predictions: list[dict] = []
    observed_category_ids: list[int] = []
    with torch.no_grad():
        for count, image_path in enumerate(image_paths, 1):
            image = Image.open(image_path).convert("RGB")
            image_width, image_height = image.size
            image_tensor = transform(image).unsqueeze(0).to(device)
            orig_target_sizes = torch.tensor([[image_width, image_height]], device=device)

            output = model(image_tensor, orig_target_sizes)[0]

            keep = output["scores"] >= CONF_THRESHOLD
            scores = output["scores"][keep]
            labels = output["labels"][keep]
            boxes = output["boxes"][keep]

            for score, label, box in zip(scores, labels, boxes):
                raw_category_id = int(label.item())
                observed_category_ids.append(raw_category_id)
                x_min, y_min, x_max, y_max = box.detach().cpu().tolist()
                bbox_width = float(x_max - x_min)
                bbox_height = float(y_max - y_min)
                if bbox_width <= 0 or bbox_height <= 0:
                    continue

                raw_predictions.append(
                    {
                        "file_name": image_path.name,
                        "raw_category_id": raw_category_id,
                        "bbox": [float(x_min), float(y_min), bbox_width, bbox_height],
                        "score": float(score.item()),
                    }
                )

            if count % 10 == 0 or count == len(image_paths):
                print(f"  Processed {count}/{len(image_paths)} images")

    if observed_category_ids:
        resolved_label_indexing = resolve_auto_label_indexing(observed_category_ids, label_indexing)
        print(f"  Detected direct label indexing: {resolved_label_indexing}")
    else:
        resolved_label_indexing = "one_based" if label_indexing == "auto" else label_indexing

    predictions = [
        {
            "file_name": prediction["file_name"],
            "category_id": to_railbench_category_id(
                prediction["raw_category_id"], resolved_label_indexing
            ),
            "bbox": prediction["bbox"],
            "score": prediction["score"],
        }
        for prediction in raw_predictions
    ]

    return predictions


# ---------------------------------------------------------------------
# Branch 2: sliced inference (SAHI)
# ---------------------------------------------------------------------
def create_ecdet_sahi_model(
        checkpoint_path: Path,
        config_path: Path,
        device,
        label_indexing: str,
):
    """Create a SAHI DetectionModel adapter for ECDet."""
    import numpy as np
    import torch
    from PIL import Image
    from sahi.models.base import DetectionModel

    class ECDetSahiDetectionModel(DetectionModel):
        """SAHI DetectionModel adapter for the ECDet wrapper of this project."""

        def load_model(self) -> None:
            if self.model_path is None:
                raise ValueError("model_path is required for ECDet SAHI inference.")

            model, transform = load_ecdet_model(Path(self.model_path), config_path, self.device)
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

            observed_category_ids = []
            for output in self._original_predictions:
                keep = output["scores"] >= self.confidence_threshold
                observed_category_ids.extend(
                    int(label.item()) for label in output["labels"][keep]
                )

            if observed_category_ids:
                resolved_label_indexing = resolve_auto_label_indexing(
                    observed_category_ids, label_indexing
                )
                print(f"  Detected SAHI label indexing: {resolved_label_indexing}")
            else:
                resolved_label_indexing = (
                    "one_based" if label_indexing == "auto" else label_indexing
                )

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

                    category_id = to_railbench_category_id(int(label.item()), resolved_label_indexing)
                    object_prediction_list.append(
                        ObjectPrediction(
                            bbox=bbox,
                            category_id=category_id,
                            category_name=CATEGORY_NAMES[category_id - 1],
                            shift_amount=shift_amount,
                            score=float(score.item()),
                            full_shape=full_shape,
                        )
                    )

                object_prediction_list_per_image.append(object_prediction_list)

            self._object_prediction_list_per_image = object_prediction_list_per_image

    return ECDetSahiDetectionModel(
        model_path=str(checkpoint_path),
        device=str(device),
        confidence_threshold=CONF_THRESHOLD,
        load_at_init=True,
    )


def write_image_manifest(image_paths: list[Path], out_path: Path) -> dict[int, str]:
    """Write the image list in COCO style, as SAHI's predict() needs image ids.

    Same structure as ``write_test_image_manifest`` in ``src/run_inference.py``
    (images with id/file_name/size, categories, no annotations / no labels).
    """
    from PIL import Image

    images = []
    for image_id, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        images.append(
            {"id": image_id, "file_name": image_path.name, "width": width, "height": height}
        )

    manifest = {
        "images": images,
        "categories": [
            {"id": category_id, "name": name} for name, category_id in CATEGORY_NAME_TO_ID.items()
        ],
        "annotations": [],
    }
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    return {image["id"]: image["file_name"] for image in images}


def find_sahi_result_json(export_dir: Path) -> Path:
    """Locate the prediction JSON written by SAHI in its export directory."""
    for name in ("result.json", "results.json", "prediction_result.json"):
        candidate = export_dir / name
        if candidate.exists():
            return candidate

    json_files = list(export_dir.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No prediction JSON found in SAHI export dir: {export_dir}")
    return json_files[0]


def load_prediction_entries(pred_json_path: Path) -> list[dict]:
    with pred_json_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for value in raw.values():
            if isinstance(value, list):
                return value
        return []
    raise ValueError(f"Unrecognized prediction JSON structure: {type(raw)}")


def predict_sahi(
        checkpoint_path: Path,
        config_path: Path,
        device,
        image_dir: Path,
        image_paths: list[Path],
        label_indexing: str,
        postprocess_match_threshold: float,
) -> list[dict]:
    """Sliced inference with ``sahi.predict.predict`` (as in ``run_sahi_predict``)."""
    import tempfile

    from sahi.predict import predict

    detection_model = create_ecdet_sahi_model(checkpoint_path, config_path, device, label_indexing)
    print(f"\nSliced inference (SAHI, {SLICE_WIDTH}x{SLICE_HEIGHT} slices): {checkpoint_path}")

    with tempfile.TemporaryDirectory(prefix="hybrid_sahi_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        manifest_path = temp_dir / "images.json"
        image_id_to_name = write_image_manifest(image_paths, manifest_path)

        result = predict(
            detection_model=detection_model,
            source=str(image_dir),
            slice_height=SLICE_HEIGHT,
            slice_width=SLICE_WIDTH,
            overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
            overlap_width_ratio=OVERLAP_WIDTH_RATIO,
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=postprocess_match_threshold,
            dataset_json_path=str(manifest_path),
            project=str(temp_dir),
            name="sahi",
            no_standard_prediction=True,
            export_pickle=False,
            export_crop=False,
            novisual=True,  # no visualization images (does not change the predictions)
            return_dict=True,
        )

        if result is None or "export_dir" not in result:
            raise RuntimeError(
                "SAHI predict() did not return an export_dir. "
                "Check whether prediction finished successfully."
            )

        entries = load_prediction_entries(find_sahi_result_json(Path(result["export_dir"])))

    predictions: list[dict] = []
    for entry in entries:
        file_name = entry.get("file_name") or image_id_to_name.get(entry.get("image_id"))
        if file_name is None:
            print(f"Warning: could not determine file name for prediction, skipping: {entry}")
            continue

        predictions.append(
            {
                "file_name": file_name,
                # The adapter already returns one-based RAIL-BENCH category ids.
                "category_id": to_railbench_category_id(int(entry["category_id"]), "one_based"),
                "bbox": entry["bbox"],
                "score": entry.get("score") or entry.get("confidence") or 0.0,
            }
        )

    return predictions


# ---------------------------------------------------------------------
# Post-processing: score filter and hybrid merge
# ---------------------------------------------------------------------
def filter_predictions(predictions: list[dict], score_threshold: float) -> list[dict]:
    """Keep valid predictions with score >= threshold and category ids 1..7."""
    filtered = []
    for prediction in predictions:
        bbox = [float(value) for value in prediction["bbox"]]
        if len(bbox) != 4:
            continue

        score = max(0.0, min(1.0, float(prediction["score"])))
        if score < score_threshold:
            continue

        category_id = int(prediction["category_id"])
        if not 1 <= category_id <= len(CATEGORY_NAMES):
            continue

        filtered.append(
            {
                "file_name": prediction["file_name"],
                "category_id": category_id,
                "bbox": bbox,
                "score": score,
            }
        )
    return filtered


def merge_hybrid_predictions(
        direct_predictions: list[dict],
        sahi_predictions: list[dict],
) -> list[dict]:
    """`train` from the direct model, all remaining classes from the SAHI model."""
    merged = [p for p in direct_predictions if p["category_id"] == TRAIN_CATEGORY_ID]
    merged.extend(p for p in sahi_predictions if p["category_id"] != TRAIN_CATEGORY_ID)
    print(
        f"\nDirect `train` predictions kept: "
        f"{sum(1 for p in direct_predictions if p['category_id'] == TRAIN_CATEGORY_ID)}"
    )
    print(
        f"SAHI non-`train` predictions kept: "
        f"{sum(1 for p in sahi_predictions if p['category_id'] != TRAIN_CATEGORY_ID)}"
    )
    return merged


# ---------------------------------------------------------------------
# Run and benchmark helpers
# ---------------------------------------------------------------------
def build_unique_run_name(split: str, nms_threshold: float, score_threshold: float) -> str:
    """Create a unique, filesystem-safe name for benchmark project and detector."""
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    if slurm_job_id:
        run_id = f"slurm{slurm_job_id}"
    else:
        run_id = datetime.now(timezone.utc).strftime("utc%Y%m%dT%H%M%SZ")

    nms_tag = str(nms_threshold).replace(".", "")
    score_tag = str(score_threshold).replace(".", "")
    return f"ecdet_hybrid_{split}_nms{nms_tag}_score{score_tag}_{run_id}"


def run_benchmark_calculator(output_path: Path, split: str, run_name: str) -> None:
    """Run the RAIL-BENCH formatting check and validation-set evaluation."""
    gt_path = None
    if split == "val":
        gt_path = REPO_ROOT / "data" / "coco_annotations" / "objects" / "annotations_val.json"

    benchmark_calculator = BenchmarkCalculator(
        test_pipeline=False,
        project_root=REPO_ROOT,
        gt_path=gt_path,
        pred_path=output_path,
    )

    format_rc = benchmark_calculator.run_railbench_format_check(
        is_railbench_test=(split == "test")
    )
    if format_rc != 0:
        raise RuntimeError(f"RAIL-BENCH format check failed with exit code {format_rc}")

    if split == "val":
        benchmark_calculator.prepare_and_run_object_eval(
            split=split,
            detector_name=run_name,
            project_name=run_name,
        )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the hybrid ECDet-L model (direct + SAHI) and write pred_test.json "
                    "for the RAIL-BENCH Object Detection toolkit.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory with the input images (.jpg, .jpeg, .png).",
    )
    parser.add_argument(
        "--direct-checkpoint",
        type=Path,
        default=REPO_ROOT / "weights" / "ecdet_l_1024_direct.pth",
        help="ECDet-L checkpoint trained on full images (used for class `train`).",
    )
    parser.add_argument(
        "--sahi-checkpoint",
        type=Path,
        default=REPO_ROOT / "weights" / "ecdet_l_1024_sahi.pth",
        help="ECDet-L checkpoint trained on image slices (used for all other classes).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "ecdet_large_1024_config.yaml",
        help="ECDet configuration file (default: config/ecdet_large_1024_config.yaml).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pred_test.json"),
        help="Output file (default: ./pred_test.json).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device, e.g. 'cuda:0' or 'cpu'. Falls back to CPU if CUDA is unavailable.",
    )
    parser.add_argument(
        "--label-indexing",
        choices=["auto", "zero_based", "one_based"],
        default=DEFAULT_LABEL_INDEXING,
        help="ECDet label indexing: 'auto' (detected, default), 'one_based' (1..7), "
             "or 'zero_based' (0..6). The output file always uses 1..7.",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=DEFAULT_POSTPROCESS_MATCH_THRESHOLD,
        help="IOU threshold of the NMS that merges the SAHI slice predictions "
             f"(default: {DEFAULT_POSTPROCESS_MATCH_THRESHOLD}).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_EXPORT_SCORE_THRESHOLD,
        help="Predictions with a lower score are not written to the output file "
             f"(default: {DEFAULT_EXPORT_SCORE_THRESHOLD}).",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        required=True,
        help="Dataset split used for inference and RAIL-BENCH reporting.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_label_mode(args.label_indexing)

    args.image_dir = resolve_image_dir(args.image_dir, args.split)
    args.direct_checkpoint = resolve_project_path(args.direct_checkpoint)
    args.sahi_checkpoint = resolve_project_path(args.sahi_checkpoint)
    args.config = resolve_project_path(args.config)
    args.output = resolve_project_path(args.output)

    if not args.image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
    for name, path in (
            ("Direct checkpoint", args.direct_checkpoint),
            ("SAHI checkpoint", args.sahi_checkpoint),
            ("Config", args.config),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")

    image_paths = collect_image_paths(args.image_dir)
    device = resolve_device(args.device)
    print(f"Found {len(image_paths)} images in {args.image_dir}")
    print(f"Device: {device}")

    direct_predictions = predict_direct(
        args.direct_checkpoint, args.config, device, image_paths, args.label_indexing
    )
    sahi_predictions = predict_sahi(
        args.sahi_checkpoint,
        args.config,
        device,
        args.image_dir,
        image_paths,
        args.label_indexing,
        args.nms_threshold,
    )

    direct_predictions = filter_predictions(direct_predictions, args.score_threshold)
    sahi_predictions = filter_predictions(sahi_predictions, args.score_threshold)
    merged_predictions = merge_hybrid_predictions(direct_predictions, sahi_predictions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(merged_predictions, file, ensure_ascii=False, indent=4)

    print(f"Wrote {len(merged_predictions)} predictions to {args.output}")
    print(
        "Check the format with the official toolkit "
        "(Benchmarks/RAILBENCH_Object): "
        f"python check_formatting.py --pred_file {args.output.resolve()}"
    )

    # ---------------------------------------------------------------------
    # Run benchmark if requested (output file ends with .json)
    # ---------------------------------------------------------------------
    if args.output.name.endswith(".json"):
        run_name = build_unique_run_name(args.split, args.nms_threshold, args.score_threshold)
        print(f"\nRunning benchmark (RAIL-BENCH) for split '{args.split}', run name: {run_name}")
        run_benchmark_calculator(args.output, args.split, run_name)


if __name__ == "__main__":
    main()
