import copy
import json
from pathlib import Path


ROAD_VEHICLE_CLASS_ID = 5
BICYCLE_CLASS_ID = 6
SIGNAL_POLE_CLASS_ID = 2

ROAD_VEHICLE_FACTOR = 2
BICYCLE_FACTOR = 4
SIGNAL_POLE_FACTOR = 2
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "images" / "train"
LABEL_DIR = ROOT / "data" / "labels" / "train"
OUTPUT_FILE = ROOT / "data" / "train_oversampled.txt"
COCO_INPUT_FILE = ROOT / "data" / "coco_annotations" / "objects" / "annotations_train.json"
COCO_OUTPUT_FILE = (
    ROOT / "data" / "coco_annotations" / "objects" / "annotations_train_oversampled.json"
)


def repeat_factor_for(label_file: Path) -> int:
    factor = 1

    if label_file.exists():
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue

            class_id = int(parts[0])
            if class_id == ROAD_VEHICLE_CLASS_ID:
                factor = max(factor, ROAD_VEHICLE_FACTOR)
            elif class_id == BICYCLE_CLASS_ID:
                factor = max(factor, BICYCLE_FACTOR)
            elif class_id == SIGNAL_POLE_CLASS_ID:
                factor = max(factor, SIGNAL_POLE_FACTOR)

    return factor


def build_annotations_by_image_id(annotations: list[dict]) -> dict[int, list[dict]]:
    annotations_by_image_id = {}

    for annotation in annotations:
        image_id = annotation["image_id"]
        annotations_by_image_id.setdefault(image_id, []).append(annotation)

    return annotations_by_image_id


def build_oversampled_coco(image_repeat_factors: dict[str, int]) -> dict:
    coco_data = json.loads(COCO_INPUT_FILE.read_text())
    images_by_file_name = {image["file_name"]: image for image in coco_data["images"]}
    annotations_by_image_id = build_annotations_by_image_id(coco_data["annotations"])

    oversampled_images = []
    oversampled_annotations = []
    next_image_id = max(image["id"] for image in coco_data["images"]) + 1
    next_annotation_id = 1

    for file_name, repeat_factor in image_repeat_factors.items():
        source_image = images_by_file_name[file_name]
        source_image_ids = [source_image["id"]]

        for _ in range(repeat_factor - 1):
            source_image_ids.append(next_image_id)
            next_image_id += 1

        for image_id in source_image_ids:
            oversampled_image = copy.deepcopy(source_image)
            oversampled_image["id"] = image_id
            oversampled_images.append(oversampled_image)

            for annotation in annotations_by_image_id.get(source_image["id"], []):
                oversampled_annotation = copy.deepcopy(annotation)
                oversampled_annotation["id"] = next_annotation_id
                oversampled_annotation["image_id"] = image_id
                oversampled_annotations.append(oversampled_annotation)
                next_annotation_id += 1

    oversampled_coco_data = copy.deepcopy(coco_data)
    oversampled_coco_data["images"] = oversampled_images
    oversampled_coco_data["annotations"] = oversampled_annotations
    return oversampled_coco_data


def main() -> None:
    lines = []
    image_repeat_factors = {}

    for image_file in sorted(IMAGE_DIR.iterdir()):
        if image_file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_file = LABEL_DIR / f"{image_file.stem}.txt"
        repeat_factor = repeat_factor_for(label_file)
        lines.extend([str(image_file.resolve())] * repeat_factor)
        image_repeat_factors[image_file.name] = repeat_factor

    OUTPUT_FILE.write_text("\n".join(lines) + "\n")
    oversampled_coco_data = build_oversampled_coco(image_repeat_factors)
    COCO_OUTPUT_FILE.write_text(json.dumps(oversampled_coco_data, indent=4) + "\n")


if __name__ == "__main__":
    main()
