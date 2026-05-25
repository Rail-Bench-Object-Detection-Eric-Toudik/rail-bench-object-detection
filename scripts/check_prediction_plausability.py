import json
from pathlib import Path
from typing import Union


def load_json(file_path: str) -> Union[dict, list]:
    """Load JSON file."""
    with open(file_path) as f:
        return json.load(f)


def get_image_info(annotations_path: str) -> dict[str, tuple[int, int]]:
    """Extract image dimensions from ground truth annotations.

    Returns: dict mapping filename to (width, height)
    """
    data = load_json(annotations_path)
    return {img["file_name"]: (img["width"], img["height"]) for img in data["images"]}


def check_filenames(predictions: list, gt_annotations_path: str) -> dict:
    """Check if prediction filenames exist in ground truth.

    Returns: dict with results
    """
    image_info = get_image_info(gt_annotations_path)
    pred_files = {p["file_name"] for p in predictions}
    gt_files = set(image_info.keys())

    missing = pred_files - gt_files
    extra = gt_files - pred_files

    return {
        "pred_count": len(pred_files),
        "gt_count": len(gt_files),
        "missing_in_gt": list(missing),
        "not_in_predictions": list(extra),
        "pass": len(missing) == 0
    }


def check_bbox_bounds(predictions: list, gt_annotations_path: str) -> dict:
    """Check if bounding boxes are within image bounds.

    Returns: dict with invalid predictions
    """
    image_info = get_image_info(gt_annotations_path)
    invalid = []

    for pred in predictions:
        fname = pred["file_name"]
        if fname not in image_info:
            continue

        width, height = image_info[fname]
        x, y, w, h = pred["bbox"]

        # Check bounds: x, y >= 0 and x+w <= width, y+h <= height, w > 0, h > 0
        if not (0 <= x and 0 <= y and x + w <= width and y + h <= height and w > 0 and h > 0):
            invalid.append({
                "file_name": fname,
                "bbox": pred["bbox"],
                "image_size": (width, height),
                "category_id": pred.get("category_id"),
                "score": pred.get("score")
            })

    return {
        "invalid_count": len(invalid),
        "invalid_predictions": invalid,
        "pass": len(invalid) == 0
    }


def check_predictions(pred_file: str, gt_annotations: str) -> dict:
    """Run all plausibility checks on predictions."""
    predictions = load_json(pred_file)

    results = {
        "total_predictions": len(predictions),
        "filenames": check_filenames(predictions, gt_annotations),
        "bounds": check_bbox_bounds(predictions, gt_annotations),
    }

    results["passed"] = results["filenames"]["pass"] and results["bounds"]["pass"]

    return results


if __name__ == "__main__":
    import sys

    pred_file = sys.argv[1] if len(sys.argv) > 1 else "../outputs/pred_test.json"
    gt_file = sys.argv[2] if len(sys.argv) > 2 else "../data/coco_annotations/objects/annotations_val.json"

    results = check_predictions(pred_file, gt_file)
    print(json.dumps(results, indent=2))

    if not results["passed"]:
        print("\n Plausibility checks FAILED")
        sys.exit(1)
    else:
        print("\n✓ All plausibility checks PASSED")


