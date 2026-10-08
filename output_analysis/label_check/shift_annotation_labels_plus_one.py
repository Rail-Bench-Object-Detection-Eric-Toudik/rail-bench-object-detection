"""Shift COCO label IDs from 0-based to 1-based indexing.

The script reads a COCO-style annotation JSON file, increments every category
ID and every annotation `category_id` by 1, and writes the result to a new file
in the same directory by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_annotation_file(path: Path) -> dict[str, Any]:
    """Load a COCO annotation JSON file."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def shift_labels_by_one(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the annotation data with label IDs shifted by one."""

    shifted_data = dict(data)

    categories = []
    for category in data.get("categories", []):
        shifted_category = dict(category)
        shifted_category["id"] = int(shifted_category["id"]) + 1
        categories.append(shifted_category)
    shifted_data["categories"] = categories

    annotations = []
    for annotation in data.get("annotations", []):
        shifted_annotation = dict(annotation)
        if "category_id" in shifted_annotation:
            shifted_annotation["category_id"] = int(shifted_annotation["category_id"]) + 1
        annotations.append(shifted_annotation)
    shifted_data["annotations"] = annotations

    return shifted_data


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""

    default_input_file = Path(__file__).with_name("val_sliced_coco.json")
    default_output_file = Path(__file__).with_name("val_sliced_coco_1to7.json")
    parser = argparse.ArgumentParser(description="Shift COCO label IDs from 0-based to 1-based indexing.")
    parser.add_argument(
        "input_file",
        nargs="?",
        default=default_input_file,
        type=Path,
        help="Path to the source annotation JSON file.",
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default=default_output_file,
        type=Path,
        help="Path to write the shifted annotation JSON file.",
    )
    return parser


def main() -> int:
    """Run the relabeling step."""

    parser = build_parser()
    args = parser.parse_args()
    data = load_annotation_file(args.input_file)
    shifted_data = shift_labels_by_one(data)

    with args.output_file.open("w", encoding="utf-8") as handle:
        json.dump(shifted_data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Wrote shifted annotations to {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
