import os
from pathlib import Path
import argparse

from sahi.slicing import slice_coco
from sahi.utils.coco import Coco, export_coco_as_yolo

def slice_coco_dataset(
    image_dir: Path,
    coco_json: Path,
    output_dir: Path,
    output_file_name: str,
    slice_size: int = 1024,
    overlap_ratio: float = 0.2,
    min_area_ratio: float = 0.1,
):
    """
    Slices COCO images + annotations into smaller tiles.

    min_area_ratio:
        If an object is cut by a tile border, SAHI keeps it only if the visible
        area in the tile is at least this fraction of the original bbox area.
        Example: 0.1 means at least 10% visible area.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    coco_dict, coco_path = slice_coco(
        coco_annotation_file_path=str(coco_json),
        image_dir=str(image_dir),
        output_dir=str(output_dir),
        output_file_name=output_file_name,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        min_area_ratio=min_area_ratio,
        ignore_negative_samples=False,
    )

    return Path(coco_path)


def prepare_dataset(
    train_images: Path,
    train_json: Path,
    val_images: Path,
    val_json: Path,
    output_dir: Path,
    slice_size: int,
    overlap_ratio: float,
    min_area_ratio: float,
):
    sliced_dir = output_dir / "sliced_coco"
    yolo_dir = output_dir / "yolo"

    print("Slicing train dataset...")
    sliced_train_json = slice_coco_dataset(
        image_dir=train_images,
        coco_json=train_json,
        output_dir=sliced_dir / "train",
        output_file_name="train_sliced",
        slice_size=slice_size,
        overlap_ratio=overlap_ratio,
        min_area_ratio=min_area_ratio,
    )

    print("Slicing val dataset...")
    sliced_val_json = slice_coco_dataset(
        image_dir=val_images,
        coco_json=val_json,
        output_dir=sliced_dir / "val",
        output_file_name="val_sliced",
        slice_size=slice_size,
        overlap_ratio=overlap_ratio,
        min_area_ratio=min_area_ratio,
    )

    print("Loading sliced COCO datasets...")
    train_coco = Coco.from_coco_dict_or_path(
        str(sliced_train_json),
        image_dir=str(sliced_dir / "train"),
        ignore_negative_samples=False,
        clip_bboxes_to_img_dims=True,
    )

    val_coco = Coco.from_coco_dict_or_path(
        str(sliced_val_json),
        image_dir=str(sliced_dir / "val"),
        ignore_negative_samples=False,
        clip_bboxes_to_img_dims=True,
    )

    print("Exporting sliced COCO dataset to YOLO format...")
    data_yaml_path = export_coco_as_yolo(
        output_dir=str(yolo_dir),
        train_coco=train_coco,
        val_coco=val_coco,
    )

    print("\nDone.")
    print(f"Sliced COCO dir: {sliced_dir}")
    print(f"YOLO dataset dir: {yolo_dir}")
    print(f"YOLO data.yaml: {data_yaml_path}")

    return data_yaml_path

if __name__ == "__main__":

    slice_size = 1024
    overlap_ratio = 0.2
    min_area_ratio = 0.2
    
    path = "../data/"
    train_json = Path(os.pah.join(path, "coco_annotations/objects/annotations_train.json"))
    val_json = Path(os.path.join(path, "coco_annotations/objects/annotations_val.json"))
    val_images = Path(os.path.join(path, "images", "val"))
    train_images = Path(os.path.join(path, "images", "train"))
    output_dir = Path(os.path.join(path, "sahi"))




    prepare_dataset(
        train_images=train_images,
        train_json=train_json,
        val_images=val_images,
        val_json=val_json,
        output_dir=output_dir,
        slice_size=args.slice_size,
        overlap_ratio=args.overlap_ratio,
        min_area_ratio=args.min_area_ratio,
    )