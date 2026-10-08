"""Inspect a COCO annotation file and report its label IDs.

This script prints the category IDs declared in the annotation file and the
category IDs referenced by annotations so it is easy to confirm whether the
labels are 0-based or 1-based.
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


def summarize_labels(data: dict[str, Any]) -> str:
    """Return a human-readable summary of the labels used in the dataset."""

    categories = data.get("categories", [])
    annotations = data.get("annotations", [])

    category_ids = sorted({int(category["id"]) for category in categories})
    category_names = [
        str(category.get("name", "")) for category in sorted(categories, key=lambda item: int(item["id"]))
    ]
    annotation_category_ids = sorted(
        {int(annotation["category_id"]) for annotation in annotations if "category_id" in annotation}
    )

    if category_ids and category_ids[0] == 0:
        index_style = "0-based (0-6)"
    elif category_ids and category_ids[0] == 1:
        index_style = "1-based (1-7)"
    else:
        index_style = "non-standard"

    lines = [
        f"Category IDs: {category_ids}",
        f"Category names: {category_names}",
        f"Annotation category IDs: {annotation_category_ids}",
        f"Label indexing: {index_style}",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""

    default_annotation_file = Path(__file__).with_name("val_sliced_coco.json")
    parser = argparse.ArgumentParser(description="Check label IDs in a COCO annotation file.")
    parser.add_argument(
        "annotation_file",
        nargs="?",
        default=default_annotation_file,
        type=Path,
        help="Path to the annotation JSON file.",
    )
    return parser


def main() -> int:
    """Run the label checker."""

    parser = build_parser()
    args = parser.parse_args()
    data = load_annotation_file(args.annotation_file)
    print(summarize_labels(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
