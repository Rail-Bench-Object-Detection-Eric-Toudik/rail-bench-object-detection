from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is validated by script execution
    plt = None

try:
    import seaborn as sns
except Exception:  # pragma: no cover - seaborn is optional
    sns = None

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - optional for analysis without images
    Image = None
    ImageDraw = None
    ImageFont = None


DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_LOCALIZATION_IOU_THRESHOLD = 0.1
DEFAULT_SLICE_WIDTH = 768
DEFAULT_SLICE_HEIGHT = 2048
DEFAULT_OVERLAP_WIDTH = 0.3
DEFAULT_OVERLAP_HEIGHT = 0.3
DEFAULT_TILE_EDGE_MARGIN = 32
DEFAULT_NUM_EXAMPLES = 5

SIZE_BIN_ORDER = ["small", "medium", "large"]
TILE_RELATION_ORDER = ["none", "near", "crosses"]
EXAMPLE_CATEGORIES = [
    "completely_missed",
    "localization",
    "small_object_miss",
    "occlusion_miss",
    "tile_edge_miss",
    "false_positive",
]

FULL_EXAMPLE_CATEGORIES = [f"{category}_full" for category in EXAMPLE_CATEGORIES]


@dataclass(slots=True)
class AnnotationRecord:
    """Validated ground-truth annotation with derived analysis features."""

    gt_id: int
    image_id: int
    file_name: str
    image_width: int
    image_height: int
    category_id: int
    class_name: str
    bbox_xywh: tuple[float, float, float, float]
    bbox_xyxy: tuple[float, float, float, float]
    bbox_area: float
    relative_area: float
    relative_width: float
    relative_height: float
    occlusion: int
    ignore: int
    iscrowd: int
    crosses_tile_edge: bool = False
    near_tile_edge: bool = False
    distance_to_nearest_tile_edge: float = math.inf
    tile_relation: str = "none"
    size_bin: str = "unknown"
    matched: bool = False
    error_type: str = "completely_missed"
    matched_pred_index: int | None = None
    pred_score: float | None = None
    match_iou: float | None = None
    best_pred_index: int | None = None
    best_pred_iou: float | None = None
    best_pred_score: float | None = None
    best_pred_bbox_xywh: tuple[float, float, float, float] | None = None


@dataclass(slots=True)
class PredictionRecord:
    """Validated prediction for matching and false-positive analysis."""

    pred_index: int
    file_name: str
    image_id: int
    category_id: int
    class_name: str
    bbox_xywh: tuple[float, float, float, float]
    bbox_xyxy: tuple[float, float, float, float]
    score: float


@dataclass(slots=True)
class LoadedData:
    """Validated annotation and prediction indices."""

    images_by_id: dict[int, dict[str, Any]]
    images_by_file_name: dict[str, dict[str, Any]]
    categories_by_id: dict[int, str]
    evaluated_annotations_by_image: dict[int, list[AnnotationRecord]]
    all_annotations_by_image: dict[int, list[AnnotationRecord]]
    predictions_by_file_name: dict[str, list[PredictionRecord]]
    predictions: list[PredictionRecord]
    ignored_annotation_count: int
    crowd_annotation_count: int


def load_json(path: Path) -> Any:
    """Load JSON from ``path``."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    """Write JSON to ``path`` with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def xywh_to_xyxy(bbox: Iterable[float]) -> tuple[float, float, float, float]:
    """Convert ``[x, y, width, height]`` to ``[x1, y1, x2, y2]``."""
    x, y, width, height = [float(value) for value in bbox]
    return x, y, x + width, y + height


def bbox_area(bbox_xyxy: Iterable[float]) -> float:
    """Return the non-negative area of an ``xyxy`` box."""
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    """Return the intersection area of two ``xyxy`` boxes."""
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    """Return the IoU of two ``xyxy`` boxes in ``[0, 1]``."""
    area_a = bbox_area(box_a)
    area_b = bbox_area(box_b)
    inter = intersection(box_a, box_b)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    value = inter / union
    return float(min(1.0, max(0.0, value)))


def _validate_bbox(bbox: Any, *, context: str) -> tuple[float, float, float, float]:
    """Validate and normalize a COCO-style bbox."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"{context}: bbox must contain exactly four values")

    try:
        x, y, width, height = [float(value) for value in bbox]
    except Exception as exc:  # pragma: no cover - explicit validation path
        raise ValueError(f"{context}: bbox values must be numeric") from exc

    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"{context}: bbox width and height must be positive")

    return x, y, width, height


def load_annotations(annotations_path: Path, predictions_path: Path) -> LoadedData:
    """Load and validate annotations and predictions from the task folder."""
    payload = load_json(annotations_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {annotations_path}")

    images = payload.get("images", [])
    categories = payload.get("categories", [])
    annotations = payload.get("annotations", [])

    images_by_id: dict[int, dict[str, Any]] = {}
    images_by_file_name: dict[str, dict[str, Any]] = {}
    categories_by_id: dict[int, str] = {}

    for image in images:
        if not isinstance(image, dict):
            raise ValueError("Each image entry must be a JSON object")
        try:
            image_id = int(image["id"])
            file_name = str(image["file_name"])
            width = int(image["width"])
            height = int(image["height"])
        except Exception as exc:
            raise ValueError(f"Invalid image entry: {image}") from exc
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size for {file_name}: {width}x{height}")
        if image_id in images_by_id:
            raise ValueError(f"Duplicate image id: {image_id}")
        if file_name in images_by_file_name:
            raise ValueError(f"Duplicate file name in annotations: {file_name}")
        image_payload = {"id": image_id, "file_name": file_name, "width": width, "height": height}
        images_by_id[image_id] = image_payload
        images_by_file_name[file_name] = image_payload

    for category in categories:
        if not isinstance(category, dict):
            raise ValueError("Each category entry must be a JSON object")
        try:
            category_id = int(category["id"])
            category_name = str(category["name"])
        except Exception as exc:
            raise ValueError(f"Invalid category entry: {category}") from exc
        categories_by_id[category_id] = category_name

    all_annotations_by_image: dict[int, list[AnnotationRecord]] = defaultdict(list)
    evaluated_annotations_by_image: dict[int, list[AnnotationRecord]] = defaultdict(list)
    ignored_annotation_count = 0
    crowd_annotation_count = 0

    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError("Each annotation entry must be a JSON object")
        try:
            gt_id = int(annotation["id"])
            image_id = int(annotation["image_id"])
            category_id = int(annotation["category_id"])
        except Exception as exc:
            raise ValueError(f"Invalid annotation entry: {annotation}") from exc

        if image_id not in images_by_id:
            raise ValueError(f"Annotation {gt_id} references unknown image_id {image_id}")
        if category_id not in categories_by_id:
            raise ValueError(f"Annotation {gt_id} references unknown category_id {category_id}")

        image_info = images_by_id[image_id]
        bbox_xywh = _validate_bbox(annotation.get("bbox"), context=f"annotation {gt_id}")
        bbox_xyxy = xywh_to_xyxy(bbox_xywh)
        area_value = bbox_area(bbox_xyxy)
        if area_value <= 0.0:
            raise ValueError(f"Annotation {gt_id} has zero area")

        ignore_value = int(annotation.get("ignore", 0) or 0)
        iscrowd_value = int(annotation.get("iscrowd", 0) or 0)
        if ignore_value == 1:
            ignored_annotation_count += 1
        if iscrowd_value != 0:
            crowd_annotation_count += 1

        relative_area = area_value / float(image_info["width"] * image_info["height"])
        relative_width = bbox_xywh[2] / float(image_info["width"])
        relative_height = bbox_xywh[3] / float(image_info["height"])
        if not (0.0 <= relative_area <= 1.0):
            raise ValueError(f"Annotation {gt_id} has invalid relative area {relative_area}")
        if not (0.0 <= relative_width <= 1.0 and 0.0 <= relative_height <= 1.0):
            raise ValueError(f"Annotation {gt_id} has invalid relative size")

        record = AnnotationRecord(
            gt_id=gt_id,
            image_id=image_id,
            file_name=str(image_info["file_name"]),
            image_width=int(image_info["width"]),
            image_height=int(image_info["height"]),
            category_id=category_id,
            class_name=categories_by_id[category_id],
            bbox_xywh=bbox_xywh,
            bbox_xyxy=bbox_xyxy,
            bbox_area=area_value,
            relative_area=relative_area,
            relative_width=relative_width,
            relative_height=relative_height,
            occlusion=int(annotation.get("occlusion", 0) or 0),
            ignore=ignore_value,
            iscrowd=iscrowd_value,
        )
        all_annotations_by_image[image_id].append(record)
        if ignore_value == 0 and iscrowd_value == 0:
            evaluated_annotations_by_image[image_id].append(record)

    if not predictions_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {predictions_path}")

    predictions_payload = load_json(predictions_path)
    if not isinstance(predictions_payload, list):
        raise ValueError(f"Expected a prediction list in {predictions_path}")

    predictions: list[PredictionRecord] = []
    predictions_by_file_name: dict[str, list[PredictionRecord]] = defaultdict(list)
    for pred_index, prediction in enumerate(predictions_payload):
        if not isinstance(prediction, dict):
            raise ValueError("Each prediction entry must be a JSON object")

        file_name = prediction.get("file_name")
        if not isinstance(file_name, str):
            raise ValueError(f"Prediction {pred_index} has an invalid file_name")
        if file_name not in images_by_file_name:
            raise ValueError(f"Prediction {pred_index} references unknown file_name {file_name!r}")

        try:
            category_id = int(prediction["category_id"])
            score = float(prediction["score"])
        except Exception as exc:
            raise ValueError(f"Invalid prediction entry: {prediction}") from exc
        if category_id not in categories_by_id:
            raise ValueError(f"Prediction {pred_index} references unknown category_id {category_id}")

        bbox_xywh = _validate_bbox(prediction.get("bbox"), context=f"prediction {pred_index}")
        bbox_xyxy = xywh_to_xyxy(bbox_xywh)

        image_info = images_by_file_name[file_name]
        record = PredictionRecord(
            pred_index=pred_index,
            file_name=file_name,
            image_id=int(image_info["id"]),
            category_id=category_id,
            class_name=categories_by_id[category_id],
            bbox_xywh=bbox_xywh,
            bbox_xyxy=bbox_xyxy,
            score=score,
        )
        predictions.append(record)
        predictions_by_file_name[file_name].append(record)

    return LoadedData(
        images_by_id=images_by_id,
        images_by_file_name=images_by_file_name,
        categories_by_id=categories_by_id,
        evaluated_annotations_by_image=evaluated_annotations_by_image,
        all_annotations_by_image=all_annotations_by_image,
        predictions_by_file_name=predictions_by_file_name,
        predictions=predictions,
        ignored_annotation_count=ignored_annotation_count,
        crowd_annotation_count=crowd_annotation_count,
    )


@lru_cache(maxsize=None)
def generate_sahi_tiles(
    image_width: int,
    image_height: int,
    slice_width: int,
    slice_height: int,
    overlap_width_ratio: float,
    overlap_height_ratio: float,
) -> list[tuple[float, float, float, float]]:
    """Reconstruct SAHI-style slice boxes for one image."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    if slice_width <= 0 or slice_height <= 0:
        raise ValueError("Slice dimensions must be positive")
    if not (0.0 <= overlap_width_ratio < 1.0 and 0.0 <= overlap_height_ratio < 1.0):
        raise ValueError("Slice overlap ratios must be in [0, 1)")

    def build_starts(full_length: int, slice_length: int, overlap_ratio: float) -> list[float]:
        if slice_length >= full_length:
            return [0.0]

        step = max(1, int(slice_length * (1.0 - overlap_ratio)))
        starts = [0.0]
        current_start = 0.0
        while True:
            next_start = current_start + step
            if next_start + slice_length >= full_length:
                final_start = float(full_length - slice_length)
                if final_start > starts[-1] + 1e-9:
                    starts.append(final_start)
                break
            starts.append(float(next_start))
            current_start = next_start
        return sorted(set(starts))

    x_starts = build_starts(image_width, slice_width, overlap_width_ratio)
    y_starts = build_starts(image_height, slice_height, overlap_height_ratio)
    tiles: list[tuple[float, float, float, float]] = []
    for y_start in y_starts:
        for x_start in x_starts:
            x2 = min(float(image_width), x_start + float(slice_width))
            y2 = min(float(image_height), y_start + float(slice_height))
            tiles.append((float(x_start), float(y_start), x2, y2))
    return tiles


def calculate_tile_edge_features(
    annotation: AnnotationRecord,
    slice_width: int,
    slice_height: int,
    overlap_width_ratio: float,
    overlap_height_ratio: float,
    tile_edge_margin: int,
) -> tuple[bool, bool, float, str]:
    """Calculate tile-edge relations for one annotation."""
    tiles = generate_sahi_tiles(
        image_width=annotation.image_width,
        image_height=annotation.image_height,
        slice_width=slice_width,
        slice_height=slice_height,
        overlap_width_ratio=overlap_width_ratio,
        overlap_height_ratio=overlap_height_ratio,
    )
    vertical_edges = sorted(
        {
            float(x_coord)
            for x1, _y1, x2, _y2 in tiles
            for x_coord in (x1, x2)
            if 0.0 < x_coord < float(annotation.image_width)
        }
    )
    horizontal_edges = sorted(
        {
            float(y_coord)
            for _x1, y1, _x2, y2 in tiles
            for y_coord in (y1, y2)
            if 0.0 < y_coord < float(annotation.image_height)
        }
    )

    x1, y1, x2, y2 = annotation.bbox_xyxy
    crosses_vertical = any(x1 < edge < x2 for edge in vertical_edges)
    crosses_horizontal = any(y1 < edge < y2 for edge in horizontal_edges)
    crosses_tile_edge = crosses_vertical or crosses_horizontal

    if crosses_tile_edge:
        distance_to_nearest_tile_edge = 0.0
    else:
        distances: list[float] = []
        for edge in vertical_edges:
            if edge < x1:
                distances.append(x1 - edge)
            elif edge > x2:
                distances.append(edge - x2)
            else:
                distances.append(0.0)
        for edge in horizontal_edges:
            if edge < y1:
                distances.append(y1 - edge)
            elif edge > y2:
                distances.append(edge - y2)
            else:
                distances.append(0.0)
        distance_to_nearest_tile_edge = min(distances) if distances else math.inf

    near_tile_edge = distance_to_nearest_tile_edge <= float(tile_edge_margin)
    tile_relation = "crosses" if crosses_tile_edge else "near" if near_tile_edge else "none"
    return crosses_tile_edge, near_tile_edge, distance_to_nearest_tile_edge, tile_relation


def calculate_size_bins(annotations: list[AnnotationRecord]) -> tuple[float, float, list[AnnotationRecord]]:
    """Assign dataset-global size bins based on relative area quantiles."""
    relative_areas = np.array([annotation.relative_area for annotation in annotations], dtype=float)
    if relative_areas.size == 0:
        raise ValueError("Cannot compute size bins without evaluated annotations")

    q33 = float(np.quantile(relative_areas, 1.0 / 3.0))
    q67 = float(np.quantile(relative_areas, 2.0 / 3.0))

    for annotation in annotations:
        if annotation.relative_area <= q33:
            size_bin = "small"
        elif annotation.relative_area <= q67:
            size_bin = "medium"
        else:
            size_bin = "large"
        annotation.size_bin = size_bin

    return q33, q67, annotations


def _match_image_class(
    annotations: list[AnnotationRecord],
    predictions: list[PredictionRecord],
    iou_threshold: float,
    localization_iou_threshold: float,
) -> tuple[list[AnnotationRecord], list[int]]:
    """Match one image/class pair greedily."""
    if not annotations and not predictions:
        return [], []

    best_prediction_by_gt: dict[int, tuple[float, PredictionRecord]] = {}
    candidate_pairs: list[tuple[float, float, int, int, int, int]] = []
    for gt_index, annotation in enumerate(annotations):
        for pred_index, prediction in enumerate(predictions):
            iou_value = iou(annotation.bbox_xyxy, prediction.bbox_xyxy)
            candidate_pairs.append((iou_value, prediction.score, annotation.gt_id, prediction.pred_index, gt_index, pred_index))
            current_best = best_prediction_by_gt.get(gt_index)
            if current_best is None:
                best_prediction_by_gt[gt_index] = (iou_value, prediction)
            else:
                current_iou, current_prediction = current_best
                if iou_value > current_iou + 1e-12 or (
                    math.isclose(iou_value, current_iou) and (
                        prediction.score > current_prediction.score + 1e-12
                        or (
                            math.isclose(prediction.score, current_prediction.score)
                            and prediction.pred_index < current_prediction.pred_index
                        )
                    )
                ):
                    best_prediction_by_gt[gt_index] = (iou_value, prediction)

    candidate_pairs.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    matched_gt_indices: set[int] = set()
    matched_prediction_indices: set[int] = set()
    matched_pairs: dict[int, tuple[int, float]] = {}
    for iou_value, _score, _gt_id, _pred_index, gt_index, pred_index in candidate_pairs:
        if iou_value < iou_threshold:
            break
        if gt_index in matched_gt_indices or predictions[pred_index].pred_index in matched_prediction_indices:
            continue
        matched_gt_indices.add(gt_index)
        matched_prediction_indices.add(predictions[pred_index].pred_index)
        matched_pairs[gt_index] = (pred_index, iou_value)

    for gt_index, annotation in enumerate(annotations):
        best_prediction = best_prediction_by_gt.get(gt_index)
        if best_prediction is not None:
            best_iou, best_pred = best_prediction
            annotation.best_pred_index = best_pred.pred_index
            annotation.best_pred_iou = best_iou
            annotation.best_pred_score = best_pred.score
            annotation.best_pred_bbox_xywh = best_pred.bbox_xywh
        else:
            annotation.best_pred_index = None
            annotation.best_pred_iou = None
            annotation.best_pred_score = None
            annotation.best_pred_bbox_xywh = None

        if gt_index in matched_pairs:
            matched_pred_index, match_iou = matched_pairs[gt_index]
            matched_prediction = predictions[matched_pred_index]
            annotation.matched = True
            annotation.error_type = "hit"
            annotation.matched_pred_index = matched_prediction.pred_index
            annotation.pred_score = matched_prediction.score
            annotation.match_iou = match_iou
        else:
            annotation.matched = False
            if annotation.best_pred_iou is not None and annotation.best_pred_iou >= localization_iou_threshold:
                annotation.error_type = "localization"
            else:
                annotation.error_type = "completely_missed"
            annotation.matched_pred_index = None
            annotation.pred_score = None
            annotation.match_iou = None

    return annotations, sorted(matched_prediction_indices)


def match_predictions_to_ground_truth(
    loaded_data: LoadedData,
    iou_threshold: float,
    localization_iou_threshold: float,
) -> tuple[list[AnnotationRecord], list[PredictionRecord]]:
    """Match predictions to evaluated ground-truth annotations."""
    all_evaluated_annotations: list[AnnotationRecord] = []
    matched_prediction_indices: set[int] = set()

    for image_id, annotations in loaded_data.evaluated_annotations_by_image.items():
        annotations_by_class: dict[int, list[AnnotationRecord]] = defaultdict(list)
        for annotation in annotations:
            annotations_by_class[annotation.category_id].append(annotation)

        predictions_by_class: dict[int, list[PredictionRecord]] = defaultdict(list)
        for prediction in loaded_data.predictions_by_file_name.get(loaded_data.images_by_id[image_id]["file_name"], []):
            predictions_by_class[prediction.category_id].append(prediction)

        for category_id, class_annotations in annotations_by_class.items():
            class_predictions = predictions_by_class.get(category_id, [])
            matched_annotations, matched_prediction_indices_for_class = _match_image_class(
                class_annotations,
                class_predictions,
                iou_threshold=iou_threshold,
                localization_iou_threshold=localization_iou_threshold,
            )
            all_evaluated_annotations.extend(matched_annotations)
            matched_prediction_indices.update(matched_prediction_indices_for_class)

    unmatched_predictions = [
        prediction
        for prediction in loaded_data.predictions
        if prediction.pred_index not in matched_prediction_indices
    ]
    return all_evaluated_annotations, unmatched_predictions


def _annotation_to_row(annotation: AnnotationRecord) -> dict[str, Any]:
    """Serialize one evaluated annotation row."""
    return {
        "gt_id": annotation.gt_id,
        "image_id": annotation.image_id,
        "file_name": annotation.file_name,
        "image_width": annotation.image_width,
        "image_height": annotation.image_height,
        "category_id": annotation.category_id,
        "class_name": annotation.class_name,
        "gt_x": annotation.bbox_xywh[0],
        "gt_y": annotation.bbox_xywh[1],
        "gt_width": annotation.bbox_xywh[2],
        "gt_height": annotation.bbox_xywh[3],
        "gt_x2": annotation.bbox_xyxy[2],
        "gt_y2": annotation.bbox_xyxy[3],
        "bbox_area": annotation.bbox_area,
        "relative_area": annotation.relative_area,
        "relative_width": annotation.relative_width,
        "relative_height": annotation.relative_height,
        "size_bin": getattr(annotation, "size_bin", None),
        "occlusion": annotation.occlusion,
        "ignore": annotation.ignore,
        "iscrowd": annotation.iscrowd,
        "matched": annotation.matched,
        "error_type": annotation.error_type,
        "matched_pred_index": annotation.matched_pred_index,
        "pred_score": annotation.pred_score,
        "match_iou": annotation.match_iou,
        "best_pred_iou": annotation.best_pred_iou,
        "best_pred_score": annotation.best_pred_score,
        "best_pred_bbox": json.dumps(annotation.best_pred_bbox_xywh) if annotation.best_pred_bbox_xywh is not None else None,
        "crosses_tile_edge": annotation.crosses_tile_edge,
        "near_tile_edge": annotation.near_tile_edge,
        "distance_to_nearest_tile_edge": annotation.distance_to_nearest_tile_edge,
        "tile_relation": annotation.tile_relation,
    }


def _prediction_to_row(prediction: PredictionRecord, loaded_data: LoadedData) -> dict[str, Any]:
    """Serialize one unmatched prediction row."""
    image_info = loaded_data.images_by_file_name[prediction.file_name]
    nearest_gt = None
    nearest_gt_iou = None
    for candidate in loaded_data.all_annotations_by_image.get(prediction.image_id, []):
        candidate_iou = iou(candidate.bbox_xyxy, prediction.bbox_xyxy)
        if nearest_gt is None or candidate_iou > nearest_gt_iou + 1e-12 or (
            math.isclose(candidate_iou, nearest_gt_iou or 0.0) and candidate.gt_id < nearest_gt.gt_id
        ):
            nearest_gt = candidate
            nearest_gt_iou = candidate_iou

    nearest_gt_class = nearest_gt.class_name if nearest_gt is not None else None
    near_tile_edge = nearest_gt.near_tile_edge if nearest_gt is not None else None
    return {
        "pred_index": prediction.pred_index,
        "file_name": prediction.file_name,
        "image_id": prediction.image_id,
        "image_width": int(image_info["width"]),
        "image_height": int(image_info["height"]),
        "category_id": prediction.category_id,
        "class_name": prediction.class_name,
        "bbox": json.dumps(list(prediction.bbox_xywh)),
        "score": prediction.score,
        "best_gt_iou": nearest_gt_iou,
        "nearest_gt_class": nearest_gt_class,
        "nearest_gt_id": nearest_gt.gt_id if nearest_gt is not None else None,
        "nearest_gt_bbox": json.dumps(list(nearest_gt.bbox_xywh)) if nearest_gt is not None else None,
        "near_tile_edge": near_tile_edge,
    }


def create_object_dataframe(annotations: list[AnnotationRecord]) -> pd.DataFrame:
    """Create the object-level analysis table."""
    dataframe = pd.DataFrame([_annotation_to_row(annotation) for annotation in annotations])
    if dataframe.empty:
        raise ValueError("Object table is empty")
    return dataframe.sort_values("gt_id").reset_index(drop=True)


def create_false_positive_dataframe(
    predictions: list[PredictionRecord],
    loaded_data: LoadedData,
) -> pd.DataFrame:
    """Create the false-positive analysis table."""
    rows = [_prediction_to_row(prediction, loaded_data) for prediction in predictions]
    columns = [
        "pred_index",
        "file_name",
        "image_id",
        "image_width",
        "image_height",
        "category_id",
        "class_name",
        "bbox",
        "score",
        "best_gt_iou",
        "nearest_gt_class",
        "nearest_gt_id",
        "nearest_gt_bbox",
        "near_tile_edge",
    ]
    return pd.DataFrame(rows, columns=columns)


def aggregate_recall(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Aggregate hit/miss statistics for ``group_columns``."""
    if not group_columns:
        raise ValueError("group_columns must not be empty")

    aggregation = (
        df.groupby(group_columns, dropna=False)
        .agg(
            gt_count=("gt_id", "size"),
            hits=("matched", "sum"),
        )
        .reset_index()
    )
    aggregation["hits"] = aggregation["hits"].astype(int)
    aggregation["misses"] = aggregation["gt_count"] - aggregation["hits"]
    aggregation["recall"] = aggregation["hits"] / aggregation["gt_count"]
    return aggregation.sort_values(group_columns).reset_index(drop=True)


def _annotated_heatmap(
    pivot_recall: pd.DataFrame,
    pivot_count: pd.DataFrame,
    title: str,
    output_path: Path,
    *,
    row_order: list[Any],
    col_order: list[Any],
    xlabel: str,
    ylabel: str,
) -> None:
    """Render a heatmap with recall and sample count annotations."""
    if plt is None:
        return
    if pivot_recall.empty:
        return

    fig, ax = plt.subplots(figsize=(max(7, len(col_order) * 1.2), max(5, len(row_order) * 0.4)))
    values = pivot_recall.to_numpy(dtype=float)
    counts = pivot_count.to_numpy(dtype=float)
    masked = np.isnan(values) | np.isnan(counts) | (counts <= 0)
    display_values = np.where(masked, np.nan, values)
    image = ax.imshow(display_values, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(col_order)), [str(column) for column in col_order])
    ax.set_yticks(np.arange(len(row_order)), [str(row) for row in row_order])
    for row_index in range(display_values.shape[0]):
        for col_index in range(display_values.shape[1]):
            if masked[row_index, col_index]:
                continue
            ax.text(
                col_index,
                row_index,
                f"{display_values[row_index, col_index]:.2f}\n(n={int(counts[row_index, col_index])})",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
    ax.set_xlim(-0.5, len(col_order) - 0.5)
    ax.set_ylim(len(row_order) - 0.5, -0.5)
    fig.colorbar(image, ax=ax, label="Recall")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def create_plots(object_df: pd.DataFrame, output_dir: Path) -> None:
    """Create quantitative plots from the object-level table."""
    if plt is None:
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    class_table = aggregate_recall(object_df, ["class_name"]).sort_values("recall", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(class_table))))
    ax.barh(class_table["class_name"], class_table["recall"], color="#4472c4")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Class")
    ax.set_title("Recall by class")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "recall_by_class.png", dpi=200)
    plt.close(fig)

    size_table = aggregate_recall(object_df, ["size_bin"]).set_index("size_bin").reindex(SIZE_BIN_ORDER).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(size_table["size_bin"], size_table["recall"], color="#70ad47")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Size bin")
    ax.set_ylabel("Recall")
    ax.set_title("Recall by size")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "recall_by_size.png", dpi=200)
    plt.close(fig)

    class_size_table = aggregate_recall(object_df, ["class_name", "size_bin"])
    class_order = sorted(object_df["class_name"].unique().tolist())
    class_size_pivot_recall = (
        class_size_table.pivot(index="class_name", columns="size_bin", values="recall")
        .reindex(index=class_order, columns=SIZE_BIN_ORDER)
    )
    class_size_pivot_count = (
        class_size_table.pivot(index="class_name", columns="size_bin", values="gt_count")
        .reindex(index=class_order, columns=SIZE_BIN_ORDER)
    )
    _annotated_heatmap(
        class_size_pivot_recall,
        class_size_pivot_count,
        "Recall by class and size",
        plots_dir / "class_size_heatmap.png",
        row_order=class_order,
        col_order=SIZE_BIN_ORDER,
        xlabel="Size bin",
        ylabel="Class",
    )

    occlusion_table = aggregate_recall(object_df, ["occlusion"]).sort_values("occlusion")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(occlusion_table["occlusion"].astype(str), occlusion_table["recall"], color="#ed7d31")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Occlusion")
    ax.set_ylabel("Recall")
    ax.set_title("Recall by occlusion")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "recall_by_occlusion.png", dpi=200)
    plt.close(fig)

    class_occlusion_table = aggregate_recall(object_df, ["class_name", "occlusion"])
    occlusion_order = sorted(object_df["occlusion"].unique().tolist())
    class_occlusion_pivot_recall = (
        class_occlusion_table.pivot(index="class_name", columns="occlusion", values="recall")
        .reindex(index=class_order, columns=occlusion_order)
    )
    class_occlusion_pivot_count = (
        class_occlusion_table.pivot(index="class_name", columns="occlusion", values="gt_count")
        .reindex(index=class_order, columns=occlusion_order)
    )
    _annotated_heatmap(
        class_occlusion_pivot_recall,
        class_occlusion_pivot_count,
        "Recall by class and occlusion",
        plots_dir / "class_occlusion_heatmap.png",
        row_order=class_order,
        col_order=occlusion_order,
        xlabel="Occlusion",
        ylabel="Class",
    )

    tile_table = aggregate_recall(object_df, ["tile_relation"]).set_index("tile_relation").reindex(TILE_RELATION_ORDER).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(tile_table["tile_relation"], tile_table["recall"], color="#5b9bd5")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Tile relation")
    ax.set_ylabel("Recall")
    ax.set_title("Recall by tile relation")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "recall_by_tile_edge.png", dpi=200)
    plt.close(fig)

    class_tile_table = aggregate_recall(object_df, ["class_name", "tile_relation"])
    class_tile_pivot_recall = (
        class_tile_table.pivot(index="class_name", columns="tile_relation", values="recall")
        .reindex(index=class_order, columns=TILE_RELATION_ORDER)
    )
    class_tile_pivot_count = (
        class_tile_table.pivot(index="class_name", columns="tile_relation", values="gt_count")
        .reindex(index=class_order, columns=TILE_RELATION_ORDER)
    )
    _annotated_heatmap(
        class_tile_pivot_recall,
        class_tile_pivot_count,
        "Recall by class and tile relation",
        plots_dir / "class_tile_edge_heatmap.png",
        row_order=class_order,
        col_order=TILE_RELATION_ORDER,
        xlabel="Tile relation",
        ylabel="Class",
    )


def _quantile_example_rows(
    dataframe: pd.DataFrame,
    metric_column: str,
    num_examples: int,
    preferred_quantiles: list[float] | None = None,
) -> list[pd.Series]:
    """Select representative rows near the requested quantiles."""
    if dataframe.empty:
        return []
    if len(dataframe) <= num_examples:
        return [row for _, row in dataframe.sort_values([metric_column]).iterrows()]

    quantiles = preferred_quantiles or np.linspace(0.1, 0.9, num_examples).tolist()
    candidates = dataframe.sort_values([metric_column, "gt_id" if "gt_id" in dataframe.columns else "pred_index"]).copy()
    selected_indices: list[int] = []
    used_indices: set[int] = set()
    metric_values = candidates[metric_column].astype(float)

    for quantile in quantiles:
        target = float(metric_values.quantile(quantile))
        best_index = None
        best_distance = None
        for row_index, row in candidates.iterrows():
            if row_index in used_indices:
                continue
            distance = abs(float(row[metric_column]) - target)
            if best_distance is None or distance < best_distance - 1e-12 or (
                math.isclose(distance, best_distance) and row_index < (best_index if best_index is not None else row_index)
            ):
                best_index = row_index
                best_distance = distance
        if best_index is not None:
            used_indices.add(best_index)
            selected_indices.append(best_index)
        if len(selected_indices) >= num_examples:
            break

    if len(selected_indices) < num_examples:
        for row_index in candidates.index:
            if row_index not in used_indices:
                used_indices.add(row_index)
                selected_indices.append(row_index)
            if len(selected_indices) >= num_examples:
                break

    return [candidates.loc[index] for index in selected_indices[:num_examples]]


def select_representative_examples(
    object_df: pd.DataFrame,
    false_positive_df: pd.DataFrame,
    num_examples: int,
) -> dict[str, list[pd.Series]]:
    """Pick deterministic representative examples for each error class."""
    examples: dict[str, list[pd.Series]] = {category: [] for category in EXAMPLE_CATEGORIES}

    completely_missed = object_df[(~object_df["matched"]) & (object_df["error_type"] == "completely_missed")]
    localization = object_df[(~object_df["matched"]) & (object_df["error_type"] == "localization")]
    small_object_miss = object_df[(~object_df["matched"]) & (object_df["size_bin"] == "small")]
    occlusion_miss = object_df[(~object_df["matched"]) & (object_df["occlusion"] > 0)]
    tile_edge_miss = object_df[(~object_df["matched"]) & (object_df["crosses_tile_edge"])]

    examples["completely_missed"] = _quantile_example_rows(completely_missed, "relative_area", num_examples)
    examples["localization"] = _quantile_example_rows(localization, "best_pred_iou", num_examples)
    examples["small_object_miss"] = _quantile_example_rows(small_object_miss, "relative_area", num_examples)
    examples["occlusion_miss"] = _quantile_example_rows(occlusion_miss, "relative_area", num_examples)
    examples["tile_edge_miss"] = _quantile_example_rows(tile_edge_miss, "relative_area", num_examples)
    if not false_positive_df.empty:
        examples["false_positive"] = [
            row for _, row in false_positive_df.sort_values(["score", "pred_index"], ascending=[False, True]).head(num_examples).iterrows()
        ]

    return examples


def _load_font(font_size: int) -> ImageFont.ImageFont | None:
    """Return a readable font if Pillow is available."""
    if ImageFont is None:
        return None

    candidate_paths = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\tahoma.ttf"),
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size=font_size)
            except OSError:
                continue
    return ImageFont.load_default()


def _draw_dashed_line(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], *, fill: str, width: int, dash_length: int = 10) -> None:
    """Draw a dashed line with Pillow."""
    x1, y1 = start
    x2, y2 = end
    total_length = math.hypot(x2 - x1, y2 - y1)
    if total_length == 0:
        return
    segment_count = max(1, int(total_length // dash_length))
    for index in range(segment_count):
        start_fraction = index / segment_count
        end_fraction = min(1.0, (index + 0.5) / segment_count)
        draw.line(
            (
                x1 + (x2 - x1) * start_fraction,
                y1 + (y2 - y1) * start_fraction,
                x1 + (x2 - x1) * end_fraction,
                y1 + (y2 - y1) * end_fraction,
            ),
            fill=fill,
            width=width,
        )


def _format_example_title(row: pd.Series, *, score_label: str) -> str:
    """Format a concise multi-line example title."""
    first_line = f"{row.get('file_name', '')} | {row.get('class_name', '')} | {row.get('error_type', score_label)}"
    second_line = (
        f"IoU={row.get('match_iou', row.get('best_gt_iou', row.get('best_pred_iou', 'n/a')))} | "
        f"score={row.get('pred_score', row.get('best_pred_score', row.get('score', 'n/a')))}"
    )
    third_line = (
        f"size={row.get('size_bin', 'n/a')} | occlusion={row.get('occlusion', 'n/a')} | "
        f"tile={row.get('tile_relation', row.get('near_tile_edge', 'n/a'))}"
    )
    return "\n".join([first_line, second_line, third_line])


def _draw_text_box(draw: ImageDraw.ImageDraw, image_size: tuple[int, int], text: str, font: ImageFont.ImageFont | None) -> None:
    """Draw a wrapped text box in the top-left corner."""
    if font is None:
        return
    x_padding = 10
    y_padding = 8
    lines = text.splitlines()
    text_bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    background = [(6, 6), (min(image_size[0] - 6, 12 + text_width), min(image_size[1] - 6, 12 + text_height))]
    draw.rectangle(background, fill=(0, 0, 0))
    draw.multiline_text((6 + x_padding, 6 + y_padding), "\n".join(lines), fill=(255, 255, 0), font=font, spacing=4)


def _draw_prediction_box(
    draw: ImageDraw.ImageDraw,
    bbox_xywh: tuple[float, float, float, float],
    *,
    color: str,
    width: int,
) -> None:
    """Draw one prediction box in ``xywh`` format."""
    x, y, bbox_width, bbox_height = bbox_xywh
    draw.rectangle([(x, y), (x + bbox_width, y + bbox_height)], outline=color, width=width)


def _crop_region(image_width: int, image_height: int, boxes: list[tuple[float, float, float, float]], padding: int) -> tuple[int, int, int, int]:
    """Compute a crop rectangle around the supplied boxes."""
    x1_values = [box[0] for box in boxes]
    y1_values = [box[1] for box in boxes]
    x2_values = [box[2] for box in boxes]
    y2_values = [box[3] for box in boxes]
    min_x = max(0, int(min(x1_values) - padding))
    min_y = max(0, int(min(y1_values) - padding))
    max_x = min(image_width, int(max(x2_values) + padding))
    max_y = min(image_height, int(max(y2_values) + padding))
    if max_x <= min_x:
        max_x = min(image_width, min_x + 1)
    if max_y <= min_y:
        max_y = min(image_height, min_y + 1)
    return min_x, min_y, max_x, max_y


def _parse_xywh_bbox(value: Any) -> tuple[float, float, float, float] | None:
    """Parse an ``xywh`` bbox from JSON or sequence form."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            return None
    else:
        parsed_value = value
    if not isinstance(parsed_value, (list, tuple)) or len(parsed_value) != 4:
        return None
    try:
        x, y, width, height = [float(item) for item in parsed_value]
    except Exception:
        return None
    return x, y, width, height


def _render_example(
    image_path: Path,
    row: pd.Series,
    output_path: Path,
    *,
    object_df: pd.DataFrame,
    false_positive_df: pd.DataFrame,
    tile_edges: dict[int, tuple[list[float], list[float]]],
    example_type: str,
) -> None:
    """Render one representative example as an annotated PNG."""
    if Image is None or ImageDraw is None:
        return
    if not image_path.exists():
        return

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _load_font(26)

    def find_object_row(gt_id: int | None) -> pd.Series | None:
        if gt_id is None:
            return None
        matches = object_df[object_df["gt_id"] == gt_id]
        if matches.empty:
            return None
        return matches.iloc[0]

    gt_row = row if example_type != "false_positive" else find_object_row(row.get("nearest_gt_id"))

    boxes_to_crop: list[tuple[float, float, float, float]] = []
    if gt_row is not None:
        boxes_to_crop.append((gt_row["gt_x"], gt_row["gt_y"], gt_row["gt_x2"], gt_row["gt_y2"]))
    if example_type == "false_positive":
        pred_bbox = _parse_xywh_bbox(row.get("bbox"))
        if pred_bbox is not None:
            boxes_to_crop.append(
                (
                    pred_bbox[0],
                    pred_bbox[1],
                    pred_bbox[0] + pred_bbox[2],
                    pred_bbox[1] + pred_bbox[3],
                )
            )
    else:
        best_pred_bbox = _parse_xywh_bbox(row.get("best_pred_bbox"))
        if best_pred_bbox is not None:
            boxes_to_crop.append(
                (
                    best_pred_bbox[0],
                    best_pred_bbox[1],
                    best_pred_bbox[0] + best_pred_bbox[2],
                    best_pred_bbox[1] + best_pred_bbox[3],
                )
            )

    crop_padding = max(80, int(max((box[2] - box[0] for box in boxes_to_crop), default=0.0) * 1.5))
    crop_padding = max(crop_padding, int(max((box[3] - box[1] for box in boxes_to_crop), default=0.0) * 1.5))
    crop_left, crop_top, crop_right, crop_bottom = _crop_region(image.width, image.height, boxes_to_crop, crop_padding)
    image = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    draw = ImageDraw.Draw(image)

    if gt_row is not None:
        draw.rectangle(
            [
                (gt_row["gt_x"] - crop_left, gt_row["gt_y"] - crop_top),
                (gt_row["gt_x2"] - crop_left, gt_row["gt_y2"] - crop_top),
            ],
            outline="lime",
            width=4,
        )

    if example_type == "false_positive":
        pred_bbox = _parse_xywh_bbox(row.get("bbox"))
        if pred_bbox is None:
            return
        draw.rectangle(
            [
                (pred_bbox[0] - crop_left, pred_bbox[1] - crop_top),
                (pred_bbox[0] + pred_bbox[2] - crop_left, pred_bbox[1] + pred_bbox[3] - crop_top),
            ],
            outline="red",
            width=4,
        )
    else:
        best_pred_bbox = _parse_xywh_bbox(row.get("best_pred_bbox"))
        if best_pred_bbox is None:
            return
        draw.rectangle(
            [
                (best_pred_bbox[0] - crop_left, best_pred_bbox[1] - crop_top),
                (best_pred_bbox[0] + best_pred_bbox[2] - crop_left, best_pred_bbox[1] + best_pred_bbox[3] - crop_top),
            ],
            outline="red",
            width=4,
        )

    if example_type == "tile_edge_miss" and gt_row is not None:
        vertical_edges, horizontal_edges = tile_edges.get(int(gt_row["image_id"]), ([], []))
        for edge in vertical_edges:
            if crop_left < edge < crop_right:
                _draw_dashed_line(draw, (edge - crop_left, 0), (edge - crop_left, image.height), fill="cyan", width=2)
        for edge in horizontal_edges:
            if crop_top < edge < crop_bottom:
                _draw_dashed_line(draw, (0, edge - crop_top), (image.width, edge - crop_top), fill="cyan", width=2)

    _draw_text_box(draw, image.size, _format_example_title(row, score_label=example_type), font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _render_full_example(
    image_path: Path,
    row: pd.Series,
    output_path: Path,
    *,
    object_df: pd.DataFrame,
    loaded_data: LoadedData,
    tile_edges: dict[int, tuple[list[float], list[float]]],
    example_type: str,
) -> None:
    """Render a full-frame representative example with all GTs and predictions."""
    if Image is None or ImageDraw is None:
        return
    if not image_path.exists():
        return

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _load_font(26)
    file_name = str(row.get("file_name"))

    gt_rows = object_df[object_df["file_name"] == file_name]
    for _, gt_row in gt_rows.iterrows():
        is_focus_gt = int(gt_row["gt_id"]) == int(row.get("gt_id", -1))
        outline = "lime" if is_focus_gt else "green"
        width = 4 if is_focus_gt else 2
        draw.rectangle(
            [
                (float(gt_row["gt_x"]), float(gt_row["gt_y"])),
                (float(gt_row["gt_x2"]), float(gt_row["gt_y2"])),
            ],
            outline=outline,
            width=width,
        )

    for prediction in loaded_data.predictions_by_file_name.get(file_name, []):
        is_focus_pred = False
        if example_type == "false_positive":
            focus_pred_bbox = _parse_xywh_bbox(row.get("bbox"))
            is_focus_pred = focus_pred_bbox is not None and tuple(prediction.bbox_xywh) == tuple(focus_pred_bbox)
        else:
            focus_pred_index = row.get("matched_pred_index") if bool(row.get("matched")) else row.get("best_pred_index")
            is_focus_pred = focus_pred_index is not None and int(prediction.pred_index) == int(focus_pred_index)
        outline = "magenta" if is_focus_pred else "red"
        width = 4 if is_focus_pred else 1
        _draw_prediction_box(draw, prediction.bbox_xywh, color=outline, width=width)

    if example_type == "tile_edge_miss" and not gt_rows.empty:
        focus_image_id = int(gt_rows.iloc[0]["image_id"])
        vertical_edges, horizontal_edges = tile_edges.get(focus_image_id, ([], []))
        for edge in vertical_edges:
            if 0.0 < edge < float(image.width):
                _draw_dashed_line(draw, (edge, 0), (edge, image.height), fill="cyan", width=2)
        for edge in horizontal_edges:
            if 0.0 < edge < float(image.height):
                _draw_dashed_line(draw, (0, edge), (image.width, edge), fill="cyan", width=2)

    _draw_text_box(draw, image.size, _format_example_title(row, score_label=example_type), font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def visualize_examples(
    object_df: pd.DataFrame,
    false_positive_df: pd.DataFrame,
    loaded_data: LoadedData,
    output_dir: Path,
    image_dir: Path | None,
    num_examples: int,
    slice_width: int,
    slice_height: int,
    overlap_width_ratio: float,
    overlap_height_ratio: float,
) -> None:
    """Write representative example visualizations if images are available."""
    if image_dir is None or Image is None or ImageDraw is None:
        return

    examples_dir = output_dir / "examples"
    full_examples_dir = output_dir / "examples_full"
    examples_dir.mkdir(parents=True, exist_ok=True)
    full_examples_dir.mkdir(parents=True, exist_ok=True)
    examples = select_representative_examples(object_df, false_positive_df, num_examples)

    tile_edges: dict[int, tuple[list[float], list[float]]] = {}
    for image_id, image_info in loaded_data.images_by_id.items():
        tiles = generate_sahi_tiles(
            image_width=int(image_info["width"]),
            image_height=int(image_info["height"]),
            slice_width=slice_width,
            slice_height=slice_height,
            overlap_width_ratio=overlap_width_ratio,
            overlap_height_ratio=overlap_height_ratio,
        )
        vertical_edges = sorted(
            {
                float(x_coord)
                for x1, _y1, x2, _y2 in tiles
                for x_coord in (x1, x2)
                if 0.0 < x_coord < float(image_info["width"])
            }
        )
        horizontal_edges = sorted(
            {
                float(y_coord)
                for _x1, y1, _x2, y2 in tiles
                for y_coord in (y1, y2)
                if 0.0 < y_coord < float(image_info["height"])
            }
        )
        tile_edges[image_id] = (vertical_edges, horizontal_edges)

    for category in EXAMPLE_CATEGORIES:
        category_dir = examples_dir / category
        full_category_dir = full_examples_dir / f"{category}_full"
        category_dir.mkdir(parents=True, exist_ok=True)
        full_category_dir.mkdir(parents=True, exist_ok=True)
        for example_index, row in enumerate(examples.get(category, []), start=1):
            file_name = str(row.get("file_name"))
            image_path = image_dir / file_name
            if not image_path.exists():
                continue
            output_path = category_dir / f"{example_index:02d}_{Path(file_name).stem}.png"
            full_output_path = full_category_dir / f"{example_index:02d}_{Path(file_name).stem}.png"
            _render_example(
                image_path=image_path,
                row=row,
                output_path=output_path,
                object_df=object_df,
                false_positive_df=false_positive_df,
                tile_edges=tile_edges,
                example_type=category,
            )
            _render_full_example(
                image_path=image_path,
                row=row,
                output_path=full_output_path,
                object_df=object_df,
                loaded_data=loaded_data,
                tile_edges=tile_edges,
                example_type=category,
            )


def create_condition_table(object_df: pd.DataFrame) -> pd.DataFrame:
    """Create the optional combined-condition table."""
    conditions = {
        "small": object_df["size_bin"] == "small",
        "occluded": object_df["occlusion"] > 0,
        "crosses_tile_edge": object_df["crosses_tile_edge"],
        "small_and_occluded": (object_df["size_bin"] == "small") & (object_df["occlusion"] > 0),
        "small_and_crosses_tile_edge": (object_df["size_bin"] == "small") & object_df["crosses_tile_edge"],
        "occluded_and_crosses_tile_edge": (object_df["occlusion"] > 0) & object_df["crosses_tile_edge"],
        "small_and_occluded_and_crosses_tile_edge": (object_df["size_bin"] == "small")
        & (object_df["occlusion"] > 0)
        & object_df["crosses_tile_edge"],
    }
    rows: list[dict[str, Any]] = []
    for condition_name, condition_mask in conditions.items():
        subset = object_df.loc[condition_mask].copy()
        if subset.empty:
            continue
        rows.append(
            {
                "condition": condition_name,
                "gt_count": len(subset),
                "hits": int(subset["matched"].sum()),
                "misses": int(len(subset) - int(subset["matched"].sum())),
                "recall": float(subset["matched"].sum() / len(subset)),
            }
        )
    return pd.DataFrame(rows)


def create_summary(
    object_df: pd.DataFrame,
    false_positive_df: pd.DataFrame,
    loaded_data: LoadedData,
    *,
    iou_threshold: float,
    localization_iou_threshold: float,
    slice_width: int,
    slice_height: int,
    overlap_width_ratio: float,
    overlap_height_ratio: float,
    tile_edge_margin: int,
    size_q33: float,
    size_q67: float,
) -> dict[str, Any]:
    """Create the final JSON summary payload."""
    hits = int(object_df["matched"].sum())
    misses = int(len(object_df) - hits)
    return {
        "config": {
            "iou_threshold": iou_threshold,
            "localization_iou_threshold": localization_iou_threshold,
            "slice_width": slice_width,
            "slice_height": slice_height,
            "overlap_width": overlap_width_ratio,
            "overlap_height": overlap_height_ratio,
            "tile_edge_margin": tile_edge_margin,
        },
        "dataset": {
            "num_images": len(loaded_data.images_by_id),
            "num_annotations_total": int(sum(len(items) for items in loaded_data.all_annotations_by_image.values())),
            "num_annotations_evaluated": int(len(object_df)),
            "num_predictions": int(len(loaded_data.predictions)),
        },
        "size_thresholds": {
            "small_max_relative_area": size_q33,
            "medium_max_relative_area": size_q67,
        },
        "detection": {
            "hits": hits,
            "misses": misses,
            "recall": float(hits / len(object_df)) if len(object_df) else 0.0,
            "localization_errors": int((object_df["error_type"] == "localization").sum()),
            "completely_missed": int((object_df["error_type"] == "completely_missed").sum()),
            "false_positives": int(len(false_positive_df)),
        },
    }


def _write_tables(object_df: pd.DataFrame, false_positive_df: pd.DataFrame, output_dir: Path) -> None:
    """Write analysis tables to CSV files."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    aggregate_recall(object_df, ["class_name"]).to_csv(tables_dir / "by_class.csv", index=False)
    aggregate_recall(object_df, ["size_bin"]).to_csv(tables_dir / "by_size.csv", index=False)
    aggregate_recall(object_df, ["class_name", "size_bin"]).to_csv(tables_dir / "by_class_size.csv", index=False)
    aggregate_recall(object_df, ["occlusion"]).to_csv(tables_dir / "by_occlusion.csv", index=False)
    aggregate_recall(object_df, ["class_name", "occlusion"]).to_csv(tables_dir / "by_class_occlusion.csv", index=False)
    aggregate_recall(object_df, ["tile_relation"]).to_csv(tables_dir / "by_tile_edge.csv", index=False)
    aggregate_recall(object_df, ["class_name", "tile_relation"]).to_csv(tables_dir / "by_class_tile_edge.csv", index=False)

    condition_table = create_condition_table(object_df)
    if not condition_table.empty:
        condition_table.to_csv(tables_dir / "by_conditions.csv", index=False)

    object_df.to_csv(output_dir / "object_analysis.csv", index=False)
    false_positive_df.to_csv(output_dir / "false_positives.csv", index=False)


def _check_assertions(object_df: pd.DataFrame, false_positive_df: pd.DataFrame, loaded_data: LoadedData) -> None:
    """Run the requested plausibility assertions."""
    evaluated_gt_count = len(object_df)
    hits = int(object_df["matched"].sum())
    misses = evaluated_gt_count - hits
    assert hits + misses == evaluated_gt_count
    assert len(false_positive_df) + hits == len(loaded_data.predictions)
    assert object_df["matched_pred_index"].dropna().is_unique
    assert object_df["gt_id"].is_unique
    assert object_df["match_iou"].dropna().between(0.0, 1.0).all()
    assert object_df["best_pred_iou"].dropna().between(0.0, 1.0).all()
    assert object_df["relative_area"].between(0.0, 1.0).all()
    assert set(loaded_data.images_by_id.keys()).issuperset(loaded_data.all_annotations_by_image.keys())
    assert all(prediction.file_name in loaded_data.images_by_file_name for prediction in loaded_data.predictions)
    assert object_df["size_bin"].notna().all()
    assert object_df["size_bin"].isin(SIZE_BIN_ORDER).all()
    assert object_df["crosses_tile_edge"].isin([True, False]).all()
    assert object_df["near_tile_edge"].isin([True, False]).all()


def _print_summary(object_df: pd.DataFrame, false_positive_df: pd.DataFrame, loaded_data: LoadedData, summary: dict[str, Any]) -> None:
    """Emit a concise progress log."""
    print("Loaded:")
    print(f"- {summary['dataset']['num_images']} images")
    print(f"- {summary['dataset']['num_annotations_total']} annotations")
    print(f"- {summary['dataset']['num_predictions']} predictions")
    print(f"- {loaded_data.ignored_annotation_count} ignored annotations")
    print(f"- {loaded_data.crowd_annotation_count} crowd annotations")
    print(f"- {len(object_df)} evaluated annotations")
    print("Matching:")
    print(f"- {summary['detection']['hits']} hits")
    print(f"- {summary['detection']['misses']} misses")
    print(f"- {summary['detection']['false_positives']} false positives")
    print("Errors:")
    print(f"- {summary['detection']['localization_errors']} localization errors")
    print(f"- {summary['detection']['completely_missed']} completely missed")
    print("Properties:")
    print(
        f"- size thresholds: q33={summary['size_thresholds']['small_max_relative_area']:.6f}, "
        f"q67={summary['size_thresholds']['medium_max_relative_area']:.6f}"
    )
    print(f"- number of occluded objects: {int((object_df['occlusion'] > 0).sum())}")
    print(f"- number crossing tile edges: {int(object_df['crosses_tile_edge'].sum())}")


def run_analysis(
    annotations_path: Path,
    predictions_path: Path,
    eval_path: Path,
    output_dir: Path,
    image_dir: Path | None,
    iou_threshold: float,
    localization_iou_threshold: float,
    slice_width: int,
    slice_height: int,
    overlap_width_ratio: float,
    overlap_height_ratio: float,
    tile_edge_margin: int,
    num_examples: int,
) -> dict[str, Any]:
    """Execute the full object-detection error analysis pipeline."""
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotations_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {predictions_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_path}")

    # The stored evaluation remains the reference for AP/AP50/AP75/AR.
    _ = load_json(eval_path)

    loaded_data = load_annotations(annotations_path, predictions_path)
    all_evaluated_annotations: list[AnnotationRecord] = []
    for image_id, annotations in loaded_data.evaluated_annotations_by_image.items():
        for annotation in annotations:
            crosses_tile_edge, near_tile_edge, distance_to_edge, tile_relation = calculate_tile_edge_features(
                annotation,
                slice_width=slice_width,
                slice_height=slice_height,
                overlap_width_ratio=overlap_width_ratio,
                overlap_height_ratio=overlap_height_ratio,
                tile_edge_margin=tile_edge_margin,
            )
            annotation.crosses_tile_edge = crosses_tile_edge
            annotation.near_tile_edge = near_tile_edge
            annotation.distance_to_nearest_tile_edge = distance_to_edge
            annotation.tile_relation = tile_relation
            all_evaluated_annotations.append(annotation)

    size_q33, size_q67, all_evaluated_annotations = calculate_size_bins(all_evaluated_annotations)
    matched_annotations, unmatched_predictions = match_predictions_to_ground_truth(
        loaded_data,
        iou_threshold=iou_threshold,
        localization_iou_threshold=localization_iou_threshold,
    )

    object_df = create_object_dataframe(matched_annotations)
    false_positive_df = create_false_positive_dataframe(unmatched_predictions, loaded_data)
    _check_assertions(object_df, false_positive_df, loaded_data)

    output_dir.mkdir(parents=True, exist_ok=True)
    examples_root = output_dir / "examples"
    for category in EXAMPLE_CATEGORIES:
        (examples_root / category).mkdir(parents=True, exist_ok=True)
    _write_tables(object_df, false_positive_df, output_dir)
    create_plots(object_df, output_dir)
    visualize_examples(
        object_df=object_df,
        false_positive_df=false_positive_df,
        loaded_data=loaded_data,
        output_dir=output_dir,
        image_dir=image_dir,
        num_examples=num_examples,
        slice_width=slice_width,
        slice_height=slice_height,
        overlap_width_ratio=overlap_width_ratio,
        overlap_height_ratio=overlap_height_ratio,
    )

    summary = create_summary(
        object_df,
        false_positive_df,
        loaded_data,
        iou_threshold=iou_threshold,
        localization_iou_threshold=localization_iou_threshold,
        slice_width=slice_width,
        slice_height=slice_height,
        overlap_width_ratio=overlap_width_ratio,
        overlap_height_ratio=overlap_height_ratio,
        tile_edge_margin=tile_edge_margin,
        size_q33=size_q33,
        size_q67=size_q67,
    )
    save_json(output_dir / "summary.json", summary)
    _print_summary(object_df, false_positive_df, loaded_data, summary)
    print(f"Outputs:")
    print(f"- {output_dir}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the analysis CLI."""
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Analyse object-detection errors for stored predictions and annotations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=script_dir / "annotations_val.json",
        help="Path to the COCO-style annotations JSON.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=script_dir / "pred_val.json",
        help="Path to the prediction JSON list.",
    )
    parser.add_argument(
        "--eval",
        type=Path,
        default=script_dir / "eval_val.json",
        help="Path to the stored evaluation JSON.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory containing the original validation images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "error_analysis",
        help="Output directory for tables, plots, and examples.",
    )
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument("--localization-iou-threshold", type=float, default=DEFAULT_LOCALIZATION_IOU_THRESHOLD)
    parser.add_argument("--slice-width", type=int, default=DEFAULT_SLICE_WIDTH)
    parser.add_argument("--slice-height", type=int, default=DEFAULT_SLICE_HEIGHT)
    parser.add_argument("--overlap-width", type=float, default=DEFAULT_OVERLAP_WIDTH)
    parser.add_argument("--overlap-height", type=float, default=DEFAULT_OVERLAP_HEIGHT)
    parser.add_argument("--tile-edge-margin", type=int, default=DEFAULT_TILE_EDGE_MARGIN)
    parser.add_argument("--num-examples", type=int, default=DEFAULT_NUM_EXAMPLES)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    run_analysis(
        annotations_path=args.annotations,
        predictions_path=args.predictions,
        eval_path=args.eval,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        iou_threshold=args.iou_threshold,
        localization_iou_threshold=args.localization_iou_threshold,
        slice_width=args.slice_width,
        slice_height=args.slice_height,
        overlap_width_ratio=args.overlap_width,
        overlap_height_ratio=args.overlap_height,
        tile_edge_margin=args.tile_edge_margin,
        num_examples=args.num_examples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
