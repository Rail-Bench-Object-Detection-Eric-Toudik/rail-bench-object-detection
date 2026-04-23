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
    last_render_signature: tuple[int, int, int, float, int, int, int, int] | None = None
    zoom_factor = 1.0
    min_zoom = 0.2
    max_zoom = 8.0
    zoom_step = 1.2
    pan_x = 0.0
    pan_y = 0.0
    drag_last_position: tuple[int, int] | None = None

    def _get_available_size() -> tuple[int, int]:
        available_width = image_label.winfo_width()
        available_height = image_label.winfo_height()

        if available_width <= 1 or available_height <= 1:
            available_width = max(root.winfo_width() - 24, 1)
            available_height = max(root.winfo_height() - 120, 1)
        return available_width, available_height

    def _render(index: int) -> None:
        nonlocal last_render_signature, pan_x, pan_y
        source_image = images[index]
        available_width, available_height = _get_available_size()

        original_width, original_height = source_image.size
        fit_scale = min(available_width / original_width, available_height / original_height)
        scale = max(min(fit_scale * zoom_factor, max_zoom), 0.01)
        resized_width = max(int(original_width * scale), 1)
        resized_height = max(int(original_height * scale), 1)
        zoomed_image = source_image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

        max_pan_x = max(zoomed_image.width - available_width, 0)
        max_pan_y = max(zoomed_image.height - available_height, 0)
        pan_x = max(0.0, min(pan_x, float(max_pan_x)))
        pan_y = max(0.0, min(pan_y, float(max_pan_y)))

        if max_pan_x > 0 or max_pan_y > 0:
            left = int(round(pan_x))
            top = int(round(pan_y))
            right = min(left + available_width, zoomed_image.width)
            bottom = min(top + available_height, zoomed_image.height)
            left = max(0, right - available_width)
            top = max(0, bottom - available_height)
            pil_image = zoomed_image.crop((left, top, right, bottom))
        else:
            pil_image = zoomed_image

        render_signature = (
            index,
            pil_image.width,
            pil_image.height,
            round(zoom_factor, 4),
            int(round(pan_x)),
            int(round(pan_y)),
            available_width,
            available_height,
        )
        if render_signature == last_render_signature:
            return

        photo = ImageTk.PhotoImage(pil_image)
        image_label.configure(image=photo)
        image_label.image = photo
        info_label.configure(
            text=(
                f"{index + 1}/{len(images)} - {titles[index]} - "
                f"Zoom {zoom_factor:.2f}x - Pan {int(round(pan_x))},{int(round(pan_y))}"
            )
        )
        last_render_signature = render_signature

    def _set_zoom(new_zoom: float) -> None:
        nonlocal zoom_factor, pan_x, pan_y
        zoom_factor = max(min_zoom, min(max_zoom, new_zoom))
        if zoom_factor <= 1.0:
            pan_x = 0.0
            pan_y = 0.0
        _render(current_index)

    def _zoom_in(_event: Any | None = None) -> None:
        _set_zoom(zoom_factor * zoom_step)

    def _zoom_out(_event: Any | None = None) -> None:
        _set_zoom(zoom_factor / zoom_step)

    def _zoom_reset(_event: Any | None = None) -> None:
        _set_zoom(1.0)

    def _next_image(_event: Any | None = None) -> None:
        nonlocal current_index, pan_x, pan_y
        current_index = (current_index + 1) % len(images)
        pan_x = 0.0
        pan_y = 0.0
        _render(current_index)

    def _prev_image(_event: Any | None = None) -> None:
        nonlocal current_index, pan_x, pan_y
        current_index = (current_index - 1) % len(images)
        pan_x = 0.0
        pan_y = 0.0
        _render(current_index)

    def _start_drag(event: Any) -> None:
        nonlocal drag_last_position
        if zoom_factor <= 1.0:
            return
        drag_last_position = (int(event.x_root), int(event.y_root))
        image_label.configure(cursor="fleur")

    def _drag_image(event: Any) -> None:
        nonlocal drag_last_position, pan_x, pan_y
        if zoom_factor <= 1.0 or drag_last_position is None:
            return

        current_position = (int(event.x_root), int(event.y_root))
        delta_x = current_position[0] - drag_last_position[0]
        delta_y = current_position[1] - drag_last_position[1]
        drag_last_position = current_position

        # Move viewport opposite to drag direction to emulate grab-and-move.
        pan_x -= float(delta_x)
        pan_y -= float(delta_y)
        _render(current_index)

    def _end_drag(_event: Any) -> None:
        nonlocal drag_last_position
        drag_last_position = None
        image_label.configure(cursor="")

    button_row = tk.Frame(root)
    button_row.pack(padx=8, pady=(0, 8))
    tk.Button(button_row, text="Previous", command=_prev_image, width=12).pack(side="left", padx=4)
    tk.Button(button_row, text="Next", command=_next_image, width=12).pack(side="left", padx=4)
    tk.Button(button_row, text="Zoom In", command=_zoom_in, width=12).pack(side="left", padx=4)
    tk.Button(button_row, text="Zoom Out", command=_zoom_out, width=12).pack(side="left", padx=4)
    tk.Button(button_row, text="Reset Zoom", command=_zoom_reset, width=12).pack(side="left", padx=4)

    root.bind("<Left>", _prev_image)
    root.bind("<Right>", _next_image)
    root.bind("<KeyPress-plus>", _zoom_in)
    root.bind("<KeyPress-equal>", _zoom_in)
    root.bind("<KeyPress-minus>", _zoom_out)
    root.bind("<KeyPress-0>", _zoom_reset)
    root.bind("p", _prev_image)
    root.bind("n", _next_image)
    root.bind("q", lambda _event: root.destroy())

    def _on_mousewheel(event: Any) -> None:
        delta = getattr(event, "delta", 0)
        if delta > 0:
            _zoom_in()
        elif delta < 0:
            _zoom_out()

    root.bind("<MouseWheel>", _on_mousewheel)
    root.bind("<Button-4>", lambda _event: _zoom_in())
    root.bind("<Button-5>", lambda _event: _zoom_out())
    image_label.bind("<ButtonPress-1>", _start_drag)
    image_label.bind("<B1-Motion>", _drag_image)
    image_label.bind("<ButtonRelease-1>", _end_drag)

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
    image_id: int | None = None,
) -> list[Image.Image]:
    """Load a COCO-style annotation file and visualize sampled images with bounding boxes.

    Args:
        annotations_path: Path to a COCO-style JSON file.
        images_dir: Directory that contains the image files referenced by `file_name`.
        num_images: Number of images to sample and draw.
        seed: Random seed for reproducible sampling.
        label_font_size: Font size for category labels drawn on top of the boxes.
        show: If True, open one window to browse rendered images with next/previous controls.
        image_id: If provided, render only the image with this COCO image id.

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
    images_by_id = {img["id"]: img for img in images}
    label_font = _load_label_font(label_font_size)
    label_padding = max(4, label_font_size // 4)
    outline_width = max(2, label_font_size // 12)

    annotations_by_image_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        annotations_by_image_id[ann["image_id"]].append(ann)

    if image_id is not None:
        selected_image = images_by_id.get(image_id)
        if selected_image is None:
            raise ValueError(f"Image id {image_id} not found in annotation file: {annotations_path}")
        sampled_images = [selected_image]
    else:
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
            occlusion = ann.get("occlusion", ann.get("occluded", "unknown"))
            label_text = f"{category_name}  occ:{occlusion}"

            drawer.rectangle([(x, y), (x2, y2)], outline="red", width=outline_width)

            text_bbox = drawer.textbbox((0, 0), label_text, font=label_font)
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
                label_text,
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
    parser.add_argument(
        "--image-id",
        type=int,
        default=None,
        help="Render only this COCO image id (from the selected annotations/images split).",
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
        image_id=args.image_id,
    )
    print(f"Rendered {len(rendered)} image(s) with bounding boxes.")


if __name__ == "__main__":
    main()




