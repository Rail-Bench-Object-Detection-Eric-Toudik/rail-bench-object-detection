from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any
import tkinter as tk

from PIL import Image, ImageDraw, ImageFont, ImageTk


def _load_label_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidate_paths = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\tahoma.ttf"),
        Path(ImageFont.__file__).resolve().parent / "fonts" / "DejaVuSans.ttf",
        Path(ImageFont.__file__).resolve().parent / "DejaVuSans.ttf",
    ]

    for font_path in candidate_paths:
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size=font_size)
            except OSError:
                continue

    return ImageFont.load_default()


def _show_images_in_single_window(images: list[Image.Image], titles: list[str]) -> None:
    if not images:
        return

    root = tk.Tk()
    root.title("COCO Bounding Box Viewer")
    root.minsize(640, 480)

    image_label = tk.Label(root)
    image_label.pack(padx=8, pady=8, fill="both", expand=True)

    info_label = tk.Label(root)
    info_label.pack(padx=8, pady=(0, 8))

    current_index = 0
    resize_job: str | None = None
    last_root_size: tuple[int, int] | None = None
    last_render_signature: tuple[int, int, int] | None = None

    def _fit_to_window(pil_image: Image.Image) -> Image.Image:
        available_width = image_label.winfo_width()
        available_height = image_label.winfo_height()

        if available_width <= 1 or available_height <= 1:
            available_width = max(root.winfo_width() - 24, 1)
            available_height = max(root.winfo_height() - 120, 1)

        original_width, original_height = pil_image.size
        scale = min(available_width / original_width, available_height / original_height)
        scale = max(scale, 0.01)

        resized_width = max(int(original_width * scale), 1)
        resized_height = max(int(original_height * scale), 1)
        return pil_image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    def _render(index: int) -> None:
        nonlocal last_render_signature
        pil_image = _fit_to_window(images[index])
        render_signature = (index, pil_image.width, pil_image.height)
        if render_signature == last_render_signature:
            return

        photo = ImageTk.PhotoImage(pil_image)
        image_label.configure(image=photo)
        image_label.image = photo
        info_label.configure(text=f"{index + 1}/{len(images)} - {titles[index]}")
        last_render_signature = render_signature

    def _next_image(_event: Any | None = None) -> None:
        nonlocal current_index
        current_index = (current_index + 1) % len(images)
        _render(current_index)

    def _prev_image(_event: Any | None = None) -> None:
        nonlocal current_index
        current_index = (current_index - 1) % len(images)
        _render(current_index)

    button_row = tk.Frame(root)
    button_row.pack(padx=8, pady=(0, 8))
    tk.Button(button_row, text="Previous", command=_prev_image, width=12).pack(side="left", padx=4)
    tk.Button(button_row, text="Next", command=_next_image, width=12).pack(side="left", padx=4)

    root.bind("<Left>", _prev_image)
    root.bind("<Right>", _next_image)
    root.bind("p", _prev_image)
    root.bind("n", _next_image)
    root.bind("q", lambda _event: root.destroy())

    def _on_resize(_event: Any) -> None:
        nonlocal resize_job, last_root_size
        if _event.widget is not root:
            return

        new_size = (root.winfo_width(), root.winfo_height())
        if new_size == last_root_size:
            return
        last_root_size = new_size

        if resize_job is not None:
            root.after_cancel(resize_job)
        resize_job = root.after(60, _render, current_index)

    root.bind("<Configure>", _on_resize)

    _render(current_index)
    root.mainloop()


def show_sample_images_with_bboxes(
    annotations_path: str | Path,
    images_dir: str | Path,
    num_images: int = 50,
    seed: int = 42,
    label_font_size: int = 24,
    show: bool = True,
) -> list[Image.Image]:
    """Load a COCO-style annotation file and visualize sampled images with bounding boxes.

    Args:
        annotations_path: Path to a COCO-style JSON file.
        images_dir: Directory that contains the image files referenced by `file_name`.
        num_images: Number of images to sample and draw.
        seed: Random seed for reproducible sampling.
        label_font_size: Font size for category labels drawn on top of the boxes.
        show: If True, open one window to browse rendered images with next/previous controls.

    Returns:
        List of PIL images with rendered bounding boxes and category labels.
    """
    annotations_path = Path(annotations_path)
    images_dir = Path(images_dir)

    with annotations_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images: list[dict[str, Any]] = coco.get("images", [])
    annotations: list[dict[str, Any]] = coco.get("annotations", [])
    categories: list[dict[str, Any]] = coco.get("categories", [])

    if not images:
        raise ValueError(f"No images found in annotation file: {annotations_path}")

    categories_by_id = {cat["id"]: cat.get("name", str(cat["id"])) for cat in categories}
    label_font = _load_label_font(label_font_size)
    label_padding = max(4, label_font_size // 4)
    outline_width = max(2, label_font_size // 12)

    annotations_by_image_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        annotations_by_image_id[ann["image_id"]].append(ann)

    rng = random.Random(seed)
    sample_count = min(num_images, len(images))
    sampled_images = rng.sample(images, k=sample_count)

    rendered_images: list[Image.Image] = []
    sampled_titles: list[str] = []
    for image_info in sampled_images:
        image_path = images_dir / image_info["file_name"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image referenced in annotations does not exist: {image_path}")

        image = Image.open(image_path).convert("RGB")
        drawer = ImageDraw.Draw(image)

        for ann in annotations_by_image_id.get(image_info["id"], []):
            x, y, w, h = ann["bbox"]
            x2 = x + w
            y2 = y + h
            category_name = categories_by_id.get(ann["category_id"], str(ann["category_id"]))

            drawer.rectangle([(x, y), (x2, y2)], outline="red", width=outline_width)

            text_bbox = drawer.textbbox((0, 0), category_name, font=label_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            text_x = max(0, min(int(x), image.width - text_width - 2 * label_padding))
            text_y = int(y) - text_height - 2 * label_padding - 2
            if text_y < 0:
                text_y = min(int(y2) + 2, image.height - text_height - 2 * label_padding)
            text_y = max(0, text_y)

            background_box = [
                (text_x, text_y),
                (min(image.width, text_x + text_width + 2 * label_padding), min(image.height, text_y + text_height + 2 * label_padding)),
            ]
            drawer.rectangle(background_box, fill="black")
            drawer.text(
                (text_x + label_padding, text_y + label_padding),
                category_name,
                fill="yellow",
                font=label_font,
            )

        rendered_images.append(image)
        sampled_titles.append(str(image_info["file_name"]))

    if show:
        _show_images_in_single_window(rendered_images, sampled_titles)

    return rendered_images


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show sample images with COCO bounding boxes.")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/annotations/objects/annotations_train.json"),
        help="Path to COCO JSON annotations file.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/images/train"),
        help="Directory containing dataset images.",
    )
    parser.add_argument("--num-images", type=int, default=50, help="How many random images to draw.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument(
        "--label-font-size",
        type=int,
        default=24,
        help="Font size for the class labels drawn on each bounding box.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Render in memory only (useful for quick validation in CI/headless environments).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    rendered = show_sample_images_with_bboxes(
        annotations_path=args.annotations,
        images_dir=args.images_dir,
        num_images=args.num_images,
        seed=args.seed,
        label_font_size=args.label_font_size,
        show=not args.no_show,
    )
    print(f"Rendered {len(rendered)} image(s) with bounding boxes.")


if __name__ == "__main__":
    main()




