import  os
import json
from typing import Optional, List, Dict, Any, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def letterbox(img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
    """ Resize with unchanged aspect ratio using padding
    """
    shape = img.shape[:2]  # current shape (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1]) # r is the scaling factor
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1] #
    dw /= 2
    dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def coco_box_to_yolo(box: List[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """# COCO: [x_min, y_min, width, height] to (x_center, y_center, w, h)  """
    x, y, w, h = box
    cx = x + w / 2
    cy = y + h / 2
    return cx / img_w, cy / img_h, w / img_w, h / img_h

class CocoYOLODataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        annotations_json: str,
        img_size: int = 640,
        classes_to_keep: Optional[List[int]] = None,  # keep subset of category ids if needed
    ):
        """
        Load image and bounding box information from JSONS
        """
        self.images_dir = images_dir
        self.annotations_json = annotations_json
        self.img_size = img_size

        with open(self.annotations_json, 'r', encoding='utf-8') as f:
            coco = json.load(f)

        # map image id -> file info
        self.id2img = {img['id']: img for img in coco.get('images', [])}
        # group annotations by image id
        self.ann_by_img = {}
        for ann in coco.get('annotations', []):
            img_id = ann['image_id']
            if classes_to_keep and ann.get('category_id') not in classes_to_keep:
                continue
            self.ann_by_img.setdefault(img_id, []).append(ann)

        # numpy list of image dicts
        self.images = []
        for img_id, info in self.id2img.items():
            file_path = os.path.join(self.images_dir, info['file_name'])
            if os.path.exists(file_path):
                self.images.append(info)
            else:
                # try without path mismatch
                alt = os.path.join(self.images_dir, os.path.basename(info['file_name']))
                if os.path.exists(alt):
                    info = info.copy()
                    info['file_name'] = os.path.basename(info['file_name'])
                    self.images.append(info)
                else:
                    # missing file: skip
                    print(f"-- Missing file {info['file_name']}")
                    continue

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        info = self.images[idx]
        img_path = os.path.join(self.images_dir, info['file_name'])
        img = cv2.imread(img_path)  # BGR
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        # Get annotations for this image (COCO format)
        anns = self.ann_by_img.get(info['id'], [])

        # Convert boxes to YOLO normalized [class, x_center, y_center, w, h]
        targets = []
        for ann in anns:
            bbox = ann['bbox']  # [x_min, y_min, w, h]
            cls = ann['category_id']
            # Here we assume they are contiguous and start at 0 or 1. If starting at 1, convert to 0-based:
            if cls > 0:
                cls0 = cls - 1
            else:
                cls0 = cls
            x_c, y_c, w_n, h_n = coco_box_to_yolo(bbox, orig_w, orig_h)
            targets.append([cls0, x_c, y_c, w_n, h_n])
        targets = np.array(targets, dtype=np.float32) if len(targets) > 0 else np.zeros((0,5), dtype=np.float32)

        # Letterbox resize image to img_size and adjust boxes accordingly
        img_l, r, (dw, dh) = letterbox(img, new_shape=(self.img_size, self.img_size))
        # adjust boxes: boxes were normalized before; we must convert back to pixel coords, apply scale+pad, then renormalize
        if targets.shape[0] > 0:
            # x_center,y_center,w,h normalized -> absolute on orig image
            x_c = targets[:,1] * orig_w
            y_c = targets[:,2] * orig_h
            w_abs = targets[:,3] * orig_w
            h_abs = targets[:,4] * orig_h

            # apply scale r and pad (dw,dh)
            x_c = x_c * r + dw
            y_c = y_c * r + dh
            w_abs = w_abs * r
            h_abs = h_abs * r

            # normalize to new image size
            x_c = x_c / self.img_size
            y_c = y_c / self.img_size
            w_n = w_abs / self.img_size
            h_n = h_abs / self.img_size

            targets[:,1] = x_c
            targets[:,2] = y_c
            targets[:,3] = w_n
            targets[:,4] = h_n

        # HWC -> CHW and to tensor
        img_l = img_l.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_l).permute(2, 0, 1)  # (C, H, W)
        targets_tensor = torch.from_numpy(targets)  # (N,5) or (0,5)

        return img_tensor, targets_tensor

def yolo_collate_fn(batch):
    """
    Collate that stacks images and concatenates targets adding image index column.
    Returns:
      images: tensor (B,3,H,W)
      targets: tensor (M,6) with columns [img_idx, class, x, y, w, h]
    """
    imgs = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    imgs = torch.stack(imgs, 0)
    if any(t.numel() for t in targets):
        t_list = []
        for i, t in enumerate(targets):
            if t.numel() == 0:
                continue
            # prepend image index
            img_idx_col = torch.full((t.shape[0], 1), i, dtype=t.dtype)
            t_list.append(torch.cat((img_idx_col, t), dim=1))
        targets = torch.cat(t_list, dim=0) if len(t_list) > 0 else torch.zeros((0,6))
    else:
        targets = torch.zeros((0,6))
    return imgs, targets