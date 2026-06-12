#!/usr/bin/env python3
import json
from pathlib import Path
import argparse


def add_area_to_annotations(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data.get("annotations", [])
    missing = 0

    for ann in annotations:
        if "area" not in ann:
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                raise ValueError(
                    f"Annotation {ann.get('id', '<no id>')} in {input_path} hat kein gültiges bbox-Feld."
                )
            # COCO bbox format: [x, y, width, height]
            ann["area"] = float(bbox[2] * bbox[3])
            missing += 1

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(
        f"{input_path.name}: {len(annotations)} Annotationen verarbeitet, "
        f"{missing} area-Werte ergänzt -> {output_path.name}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Ergänzt fehlende COCO 'area'-Felder in annotations_train.json und annotations_val.json."
    )
    parser.add_argument(
        "--train",
        default="../data/coco_annotations/objects/annotations_train.json",
        help="Pfad zu annotations_train.json",
    )
    parser.add_argument(
        "--val",
        default="../data/coco_annotations/objects/annotations_val.json",
        help="Pfad zu annotations_val.json",
    )
    parser.add_argument(
        "--out-dir",
        default="../data/coco_annotations/objects/",
        help="Zielordner für die neuen JSON-Dateien. Standard: gleicher Ordner wie Input-Dateien",
    )
    args = parser.parse_args()

    train_path = Path(args.train)
    val_path = Path(args.val)

    out_dir = Path(args.out_dir) if args.out_dir else None

    if out_dir is None:
        train_out = train_path.with_name("annotations_train_area.json")
        val_out = val_path.with_name("annotations_val_area.json")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        train_out = out_dir / "annotations_train_area.json"
        val_out = out_dir / "annotations_val_area.json"

    add_area_to_annotations(train_path, train_out)
    add_area_to_annotations(val_path, val_out)


if __name__ == "__main__":
    main()