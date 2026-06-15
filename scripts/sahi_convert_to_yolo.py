import os
from pathlib import Path

import json
import shutil
import platform
import yaml
from sahi.slicing import slice_coco
from sahi.utils.coco import Coco, export_coco_as_yolo

def slice_coco_dataset(
    image_dir: Path,
    coco_json: Path,
    output_dir: Path,
    output_file_name: str,
    slice_width: int = 1024,
    slice_height: int = 1024,
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
        output_coco_annotation_file_name=output_file_name,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        min_area_ratio=min_area_ratio,
        ignore_negative_samples=False,
    )

    return Path(coco_path)


def remap_coco_categories_to_zero_based(coco_json_path: Path) -> None:
    """Remap COCO category ids to YOLO class indices 0..n-1."""
    with open(coco_json_path, "r", encoding="utf-8") as fh:
        coco_obj = json.load(fh)

    categories = sorted(coco_obj.get("categories", []), key=lambda cat: cat["id"])
    category_id_map = {cat["id"]: new_id for new_id, cat in enumerate(categories)}

    if not category_id_map:
        raise ValueError(f"No categories found in '{coco_json_path}'")

    for category in categories:
        category["id"] = category_id_map[category["id"]]

    for annotation in coco_obj.get("annotations", []):
        category_id = annotation.get("category_id")
        if category_id not in category_id_map:
            raise ValueError(
                f"Annotation references unknown category_id {category_id} in '{coco_json_path}'"
            )
        annotation["category_id"] = category_id_map[category_id]

    coco_obj["categories"] = categories

    with open(coco_json_path, "w", encoding="utf-8") as fh:
        json.dump(coco_obj, fh)


def prepare_dataset(
    train_images: Path,
    train_json: Path,
    val_images: Path,
    val_json: Path,
    output_dir: Path,
    slice_height: int,
    slice_width: int,
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
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_ratio=overlap_ratio,
        min_area_ratio=min_area_ratio,
    )

    # Minimal sanitization: remove annotations with missing/empty bboxes
    def _sanitize_coco_file(coco_json_path: Path) -> int:
        """Remove annotations with missing/invalid bboxes.

        Returns the number of removed annotations.
        """
        try:
            with open(coco_json_path, "r", encoding="utf-8") as fh:
                coco_obj = json.load(fh)
        except Exception:
            return 0

        anns = coco_obj.get("annotations", [])
        clean_anns = []
        for ann in anns:
            bbox = ann.get("bbox")
            # keep only annotations where bbox is a list/tuple of length 4
            if not isinstance(bbox, (list, tuple)):
                continue
            if len(bbox) != 4:
                continue
            # validate all bbox values are numeric and finite
            try:
                bbox_vals = [float(v) for v in bbox]
                # check for NaN or inf
                if any(v != v or v == float('inf') or v == float('-inf') for v in bbox_vals):
                    continue
                # discard degenerate bboxes (zero width or height)
                if bbox_vals[2] <= 0 or bbox_vals[3] <= 0:
                    continue
            except (ValueError, TypeError):
                continue
            clean_anns.append(ann)

        removed = len(anns) - len(clean_anns)
        if removed > 0:
            coco_obj["annotations"] = clean_anns
            with open(coco_json_path, "w", encoding="utf-8") as fh:
                json.dump(coco_obj, fh)
            print(f"Sanitized '{coco_json_path}': removed {removed} invalid annotations")

        return removed

    removed_train = _sanitize_coco_file(sliced_train_json)
    if removed_train > 0:
        print(f"Total removed from train sliced json: {removed_train}")
    remap_coco_categories_to_zero_based(sliced_train_json)

    print("Slicing val dataset...")
    sliced_val_json = slice_coco_dataset(
        image_dir=val_images,
        coco_json=val_json,
        output_dir=sliced_dir / "val",
        output_file_name="val_sliced",
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_ratio=overlap_ratio,
        min_area_ratio=min_area_ratio,
    )

    # sanitize val as well
    removed_val = _sanitize_coco_file(sliced_val_json)
    if removed_val > 0:
        print(f"Total removed from val sliced json: {removed_val}")
    remap_coco_categories_to_zero_based(sliced_val_json)

    print("Loading sliced COCO datasets...")
    try:
        train_coco = Coco.from_coco_dict_or_path(
            str(sliced_train_json),
            image_dir=str(sliced_dir / "train"),
            ignore_negative_samples=False,
            clip_bboxes_to_img_dims=False,
        )
    except ValueError as e:
        if "unpack" in str(e).lower():
            print(f"Warning: bbox error during train load, retrying...")
            # re-sanitize more aggressively and try again
            _sanitize_coco_file(sliced_train_json)
            train_coco = Coco.from_coco_dict_or_path(
                str(sliced_train_json),
                image_dir=str(sliced_dir / "train"),
                ignore_negative_samples=False,
                clip_bboxes_to_img_dims=False,
            )
        else:
            raise

    try:
        val_coco = Coco.from_coco_dict_or_path(
            str(sliced_val_json),
            image_dir=str(sliced_dir / "val"),
            ignore_negative_samples=False,
            clip_bboxes_to_img_dims=False,
        )
    except ValueError as e:
        if "unpack" in str(e).lower():
            print(f"Warning: bbox error during val load, retrying...")
            # re-sanitize more aggressively and try again
            _sanitize_coco_file(sliced_val_json)
            val_coco = Coco.from_coco_dict_or_path(
                str(sliced_val_json),
                image_dir=str(sliced_dir / "val"),
                ignore_negative_samples=False,
                clip_bboxes_to_img_dims=False,
            )
        else:
            raise

    print("Exporting sliced COCO dataset to YOLO format...")
    try:
        data_yaml_path = export_coco_as_yolo(
            output_dir=str(yolo_dir),
            train_coco=train_coco,
            val_coco=val_coco,
        )
    except OSError as e:
        # On Windows creating symlinks may require elevated privileges (WinError 1314).
        # If symlink creation fails, retry by copying files instead.
        win_err = getattr(e, "winerror", None)
        if platform.system() == "Windows" and win_err == 1314:
            print("Warning: insufficient privileges to create symlinks on Windows. Falling back to copying files.")
            # remove partially created output dir
            try:
                shutil.rmtree(yolo_dir)
            except Exception:
                pass

            # monkeypatch os.symlink to copy as fallback during export
            _orig_symlink = os.symlink

            def _copy_symlink(src, dst):
                try:
                    # try hard link first
                    os.link(src, dst)
                except Exception:
                    shutil.copy2(src, dst)

            os.symlink = _copy_symlink
            try:
                data_yaml_path = export_coco_as_yolo(
                    output_dir=str(yolo_dir),
                    train_coco=train_coco,
                    val_coco=val_coco,
                )
            finally:
                # restore original
                os.symlink = _orig_symlink
        else:
            raise

    print("\nDone.")
    print(f"Sliced COCO dir: {sliced_dir}")
    print(f"YOLO dataset dir: {yolo_dir}")
    print(f"YOLO data.yaml: {data_yaml_path}")

    return data_yaml_path

if __name__ == "__main__":

    config_path = Path(__file__).resolve().parents[1] / "config" / "run_config_sahi_basic_augmentations.yaml"
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    slice_height = config["slice_height"]
    slice_width = config["slice_width"]
    overlap_ratio = config["overlap_height_ratio"]
    min_area_ratio = config["min_area_ratio"]
    
    path = "../data/"
    train_json = Path(os.path.join(path, "coco_annotations/objects/annotations_train.json"))
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
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_ratio=overlap_ratio,
        min_area_ratio=min_area_ratio,
    )
