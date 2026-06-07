"""
train.py

Fine-tune ECDet (EdgeCrafter) on a COCO-format dataset using your custom
get_model() and ECDetModel enum.

This script:
- Loads ECDet_M (or any ECDetModel enum)
- Uses your get_model() abstraction
- Implements the 3-phase freeze schedule
- Uses WandB with WANDB_API_KEY and with wandb.init(...)
- Logs to src/wandb and logs/
- Supports single GPU now, cloud later
- Uses COCO evaluation (mAP)
"""

import argparse
import os
import time
import json
import random
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torch.cuda.amp import autocast, GradScaler

import wandb

from ecdet import get_model, ECDetModel

# COCO evaluation
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dirs(paths: List[str]):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def collate_fn(batch):
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets


def build_transforms(image_size: int, is_train: bool = True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])


class CocoWrapper(CocoDetection):
    """COCO dataset wrapper returning (image, target_list)."""

    def __init__(self, root, annFile, transforms=None):
        super().__init__(root, annFile)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        if self._transforms:
            img = self._transforms(img)
        return img, target


# ---------------------------------------------------------
# Freeze schedule helpers
# ---------------------------------------------------------

def freeze_backbone(model: nn.Module):
    if hasattr(model, "backbone"):
        for p in model.backbone.parameters():
            p.requires_grad = False


def unfreeze_backbone(model: nn.Module):
    if hasattr(model, "backbone"):
        for p in model.backbone.parameters():
            p.requires_grad = True


def partial_unfreeze_encoder(model: nn.Module, last_n: int = 4):
    """Unfreeze last N encoder children."""
    if not hasattr(model, "encoder"):
        return

    children = list(model.encoder.children())
    if not children:
        # fallback: unfreeze entire encoder
        for p in model.encoder.parameters():
            p.requires_grad = True
        return

    # Freeze all first
    for p in model.encoder.parameters():
        p.requires_grad = False

    # Unfreeze last N
    for module in children[-last_n:]:
        for p in module.parameters():
            p.requires_grad = True


def get_param_groups(model: nn.Module, lr_head: float, lr_backbone: float, weight_decay: float):
    """Separate LR for backbone vs head."""
    head_params = []
    backbone_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone") or name.startswith("encoder"):
            backbone_params.append(p)
        elif name.startswith("decoder") or name.startswith("postprocessor"):
            head_params.append(p)
        else:
            other_params.append(p)

    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": lr_head})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone})
    if other_params:
        groups.append({"params": other_params, "lr": lr_head})

    for g in groups:
        g.setdefault("weight_decay", weight_decay)

    return groups


# ---------------------------------------------------------
# COCO evaluation
# ---------------------------------------------------------

def coco_evaluate(model, dataloader, device, output_dir, image_size):
    model.eval()
    results = []

    with torch.no_grad():
        for images, targets in dataloader:
            batch = torch.stack(images).to(device)
            outputs = model(batch, orig_target_sizes=[t for t in targets])

            if isinstance(outputs, dict):
                outputs = [outputs]

            for out, t in zip(outputs, targets):
                if len(t) > 0 and "image_id" in t[0]:
                    image_id = int(t[0]["image_id"])
                else:
                    image_id = -1

                boxes = out.get("boxes") or out.get("pred_boxes")
                scores = out.get("scores") or out.get("pred_scores")
                labels = out.get("labels") or out.get("pred_labels")

                if boxes is None:
                    continue

                boxes = boxes.cpu().numpy()
                scores = scores.cpu().numpy()
                labels = labels.cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(score)
                    })

    res_file = os.path.join(output_dir, f"coco_results_{int(time.time())}.json")
    with open(res_file, "w") as f:
        json.dump(results, f)

    try:
        coco_gt = dataloader.dataset.coco
        coco_dt = coco_gt.loadRes(res_file)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        mAP = float(coco_eval.stats[0])
    except Exception as e:
        print("COCO evaluation failed:", e)
        mAP = 0.0

    model.train()
    return mAP


# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------

def train_loop(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    make_dirs([args.output_dir, args.wandb_dir, args.logs_dir])

    # Dataset
    train_tf = build_transforms(args.image_size, True)
    val_tf = build_transforms(args.image_size, False)

    train_dataset = CocoWrapper(
        root=os.path.join(args.data_root, "train"),
        annFile=os.path.join(args.data_root, args.train_ann),
        transforms=train_tf
    )
    val_dataset = CocoWrapper(
        root=os.path.join(args.data_root, "val"),
        annFile=os.path.join(args.data_root, args.val_ann),
        transforms=val_tf
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )

    # Model config
    model_cfg = {
        "backbone": {"image_size": args.image_size},
        "encoder": {},
        "decoder": {},
        "postprocessor": {"num_classes": args.num_classes}
    }

    # Load model using YOUR get_model()
    model_enum = ECDetModel(args.model_repo)
    print(f"Loading pretrained ECDet model: {model_enum.value}")
    model = get_model(model_cfg, model_enum)
    model.to(device)

    # Freeze schedule phase 1
    freeze_backbone(model)

    # Optimizer
    param_groups = get_param_groups(model, args.lr_head, args.lr_backbone, args.weight_decay)
    optimizer = optim.AdamW(param_groups)
    scaler = GradScaler(enabled=not args.no_amp)

    # LR scheduler
    total_steps = len(train_loader) * args.epochs

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.1415926535))).item()

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # WandB
    if args.use_wandb:
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        with wandb.init(project=args.wandb_project, dir=args.wandb_dir, config=vars(args)):
            _train(args, model, train_loader, val_loader, optimizer, scheduler, scaler, device)
    else:
        _train(args, model, train_loader, val_loader, optimizer, scheduler, scaler, device)


def _train(args, model, train_loader, val_loader, optimizer, scheduler, scaler, device):
    best_map = -1.0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        # Freeze schedule
        if epoch == args.freeze_phase_1_epochs:
            print(f"Epoch {epoch}: partial unfreeze")
            partial_unfreeze_encoder(model, args.partial_unfreeze_last_n)
            optimizer = optim.AdamW(get_param_groups(model, args.lr_head, args.lr_backbone, args.weight_decay))
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, scheduler.lr_lambdas[0])

        if epoch == args.partial_unfreeze_until:
            print(f"Epoch {epoch}: full unfreeze")
            unfreeze_backbone(model)
            optimizer = optim.AdamW(get_param_groups(model, args.lr_head * 0.5, args.lr_backbone * 0.5, args.weight_decay))
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, scheduler.lr_lambdas[0])

        # Training
        for step, (images, targets) in enumerate(train_loader):
            batch = torch.stack(images).to(device)

            optimizer.zero_grad()
            with autocast(enabled=not args.no_amp):
                outputs = model(batch, orig_target_sizes=[t for t in targets])

                if isinstance(outputs, dict) and "loss" in outputs:
                    loss = outputs["loss"]
                elif isinstance(outputs, dict):
                    loss = sum(v for k, v in outputs.items() if k.startswith("loss"))
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if step % args.log_interval == 0:
                print(f"[Epoch {epoch}][{step}/{len(train_loader)}] loss={loss.item():.4f}")
                if wandb.run:
                    wandb.log({"loss": loss.item(), "epoch": epoch}, step=global_step)

        print(f"Epoch {epoch} finished, avg_loss={epoch_loss/len(train_loader):.4f}")

        # Save last checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_map": best_map,
        }
        torch.save(ckpt, os.path.join(args.output_dir, "last_checkpoint.pth"))

        # Evaluation
        if (epoch + 1) % args.eval_interval_epochs == 0:
            mAP = coco_evaluate(model, val_loader, device, args.output_dir, args.image_size)
            print(f"Validation mAP: {mAP:.4f}")
            if wandb.run:
                wandb.log({"val/mAP": mAP, "epoch": epoch}, step=global_step)

            if mAP > best_map:
                best_map = mAP
                torch.save(ckpt, os.path.join(args.output_dir, "best_checkpoint.pth"))
                print(f"New best model saved with mAP={best_map:.4f}")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-repo", default="ECDet_M")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-ann", default="annotations/instances_train.json")
    parser.add_argument("--val-ann", default="annotations/instances_val.json")

    parser.add_argument("--output-dir", default="logs")
    parser.add_argument("--wandb-dir", default="src/wandb")
    parser.add_argument("--logs-dir", default="logs")

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)

    parser.add_argument("--freeze-phase-1-epochs", type=int, default=50)
    parser.add_argument("--partial-unfreeze-until", type=int, default=150)
    parser.add_argument("--partial-unfreeze-last-n", type=int, default=4)

    parser.add_argument("--eval-interval-epochs", type=int, default=1)
    parser.add_argument("--use-wandb", action="store_true", default=True)
    parser.add_argument("--wandb-project", default="Rail-Bench-Object-Detection-Eric-Toudik")

    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--num-classes", type=int, default=7)

    return parser.parse_args()


# ---------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    make_dirs([args.output_dir, args.wandb_dir, args.logs_dir])
    train_loop(args)
