"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

import datetime
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import wandb
from dotenv import load_dotenv

from ..misc import dist_utils, stats
from ..optim.lr_scheduler import FlatCosineLRScheduler
from ._solver import BaseSolver
from .ec_engine import evaluate, train_one_epoch


class ECSolver(BaseSolver):

    def _wandb_config(self, job_type: str) -> dict[str, object]:
        cfg = self.cfg
        config = {
            'task': getattr(cfg, 'task', None),
            'job_type': job_type,
            'output_dir': str(self.output_dir),
            'resume': getattr(cfg, 'resume', None),
            'tuning': getattr(cfg, 'tuning', None),
            'device': getattr(cfg, 'device', None),
            'seed': getattr(cfg, 'seed', None),
            'last_epoch': getattr(cfg, 'last_epoch', None),
        }

        for attr in ('epochs', 'clip_max_norm', 'print_freq', 'checkpoint_freq', 'sync_bn', 'find_unused_parameters'):
            value = getattr(cfg, attr, None)
            if value is not None:
                config[attr] = value

        for attr in ('optimizer', 'lr_scheduler', 'lr_warmup_scheduler', 'criterion', 'postprocessor', 'ema', 'scaler'):
            value = getattr(cfg, attr, None)
            if value is not None:
                config[attr] = type(value).__name__

        return config

    def _init_wandb(self, job_type: str) -> Any | None:
        if not dist_utils.is_main_process():
            return None

        load_dotenv()
        api_key = os.getenv('WANDB_API_KEY')
        if api_key:
            wandb.login(key=api_key)
        else:
            wandb.login()

        project = getattr(self.cfg, 'wandb_project', None) or 'Rail-Bench-Object-Detection-EdgeCrafter'
        run_name = getattr(self.cfg, 'wandb_run_name', None)
        if run_name is None:
            run_name = f'{job_type}_{self.cfg.task}_{self.output_dir.name}'

        return wandb.init(
            project=project,
            name=run_name,
            job_type=job_type,
            config=self._wandb_config(job_type),
        )

    def _log_model_artifact(self, run: Any, artifact_name: str, checkpoint_paths: Sequence[Path]) -> None:
        if run is None or not dist_utils.is_main_process():
            return

        artifact = wandb.Artifact(name=artifact_name, type='model')
        added = False
        for checkpoint_path in checkpoint_paths:
            if checkpoint_path and checkpoint_path.exists():
                artifact.add_file(str(checkpoint_path))
                added = True

        if added:
            logged_artifact = run.log_artifact(artifact)
            logged_artifact.wait()

    def fit(self, ):
        self.train()
        args = self.cfg
        wandb_run = self._init_wandb('train')
        artifact_id = getattr(wandb_run, 'id', wandb_run.name) if wandb_run is not None else None

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)
        
        stop_aug_epoch = self.train_dataloader.dataset._transforms.stop_epoch  # epoch to stop augmentation
        if args.lrsheduler is not None:
            no_aug_epochs = args.epochs - stop_aug_epoch
            flat_epochs = self.train_dataloader.dataset._transforms.mosaic_epoch if args.flat_epoch is None else args.flat_epoch
            iter_per_epoch = len(self.train_dataloader)
            warmup_iter = min(args.warmup_iter, 3 * iter_per_epoch)  
            
            print(f'FlatCosineLRScheduler with flat_epochs: {flat_epochs}, no_aug_epochs: {no_aug_epochs}, warmup_iter: {args.warmup_iter}')
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch, total_epochs=args.epochs, 
                                                warmup_iter=warmup_iter, flat_epochs=flat_epochs, no_aug_epochs=no_aug_epochs)
            self.self_lr_scheduler = True
        else:
            self.self_lr_scheduler = False

        top1 = 0
        best_stat = {'epoch': -1, }
        training_completed = False
        # evaluate again before resume training
        total_time = 0.0
        try:
            if self.last_epoch > 0:
                module = self.ema.module if self.ema else self.model
                # compute a monotonic wandb step based on completed training iterations
                if hasattr(self, 'train_dataloader') and self.train_dataloader is not None:
                    train_steps = len(self.train_dataloader)
                    wandb_step = (self.last_epoch + 1) * train_steps
                else:
                    wandb_step = self.last_epoch
                test_stats, coco_evaluator = evaluate(
                    module,
                    self.criterion,
                    self.postprocessor,
                    self.val_dataloader,
                    self.evaluator,
                    self.device,
                    wandb_run=wandb_run,
                    step=wandb_step,
                )
                for k in test_stats:
                    best_stat['epoch'] = self.last_epoch
                    best_stat[k] = test_stats[k][0]
                    top1 = test_stats[k][0]
                    print(f'best_stat: {best_stat}')

                if wandb_run is not None and dist_utils.is_main_process():
                    wandb_run.summary['best_epoch'] = self.last_epoch
                    for k, v in best_stat.items():
                        if k != 'epoch':
                            wandb_run.summary[f'best/{k}'] = v

            best_stat_print = best_stat.copy()
            start_time = time.time()
            start_epoch = self.last_epoch + 1
            for epoch in range(start_epoch, args.epochs):

                self.train_dataloader.set_epoch(epoch)
                # self.train_dataloader.dataset.set_epoch(epoch)
                if dist_utils.is_dist_available_and_initialized():
                    self.train_dataloader.sampler.set_epoch(epoch)

                if epoch == stop_aug_epoch:
                    if dist_utils.is_dist_available_and_initialized():
                        torch.distributed.barrier()
                    self.load_resume_state(str(self.output_dir / 'best.pth'))
                    self.last_epoch = epoch - 1

                train_stats = train_one_epoch(
                    self.self_lr_scheduler,
                    self.lr_scheduler,
                    self.model,
                    self.criterion,
                    self.train_dataloader,
                    self.optimizer,
                    self.device,
                    epoch,
                    max_norm=args.clip_max_norm,
                    print_freq=args.print_freq,
                    ema=self.ema,
                    scaler=self.scaler,
                    lr_warmup_scheduler=self.lr_warmup_scheduler,
                    writer=self.writer,
                    wandb_run=wandb_run,
                )

                if not self.self_lr_scheduler:  # update by epoch
                    if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                        self.lr_scheduler.step()

                self.last_epoch += 1

                if self.output_dir and epoch < stop_aug_epoch:
                    checkpoint_paths = [self.output_dir / 'last.pth']
                    # extra checkpoint before LR drop and every 100 epochs
                    if (epoch + 1) % args.checkpoint_freq == 0:
                        checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                    for checkpoint_path in checkpoint_paths:
                        dist_utils.save_on_master(self.state_dict(), checkpoint_path)

                module = self.ema.module if self.ema else self.model
                # compute a monotonic wandb step for evaluation logging (after the epoch's training steps)
                if hasattr(self, 'train_dataloader') and self.train_dataloader is not None:
                    train_steps = len(self.train_dataloader)
                    wandb_step = (epoch + 1) * train_steps
                else:
                    wandb_step = epoch
                test_stats, coco_evaluator = evaluate(
                    module,
                    self.criterion,
                    self.postprocessor,
                    self.val_dataloader,
                    self.evaluator,
                    self.device,
                    wandb_run=wandb_run,
                    step=wandb_step,
                )

                for k in test_stats:
                    if self.writer and dist_utils.is_main_process():
                        for i, v in enumerate(test_stats[k]):
                            self.writer.add_scalar(f'Test/{k}_{i}', v, epoch)

                    if k in best_stat:
                        best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                        best_stat[k] = max(best_stat[k], test_stats[k][0])
                    else:
                        best_stat['epoch'] = epoch
                        best_stat[k] = test_stats[k][0]

                    if best_stat[k] > top1:
                        best_stat_print['epoch'] = epoch
                        top1 = best_stat[k]
                        if self.output_dir:
                            best_checkpoint_path = self.output_dir / 'best.pth'
                            dist_utils.save_on_master(self.state_dict(), best_checkpoint_path)

                    best_stat_print[k] = max(best_stat[k], top1)
                    print(f'best_stat: {best_stat_print}')  # global best

                if wandb_run is not None and dist_utils.is_main_process():
                    wandb_run.summary['epoch'] = epoch
                    wandb_run.summary['best_epoch'] = best_stat['epoch']
                    for k, v in best_stat.items():
                        if k != 'epoch':
                            wandb_run.summary[f'best/{k}'] = v

                log_stats = {
                    **{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                    'epoch': epoch,
                    'n_parameters': n_parameters
                }

                if self.output_dir and dist_utils.is_main_process():
                    with (self.output_dir / "log.txt").open("a") as f:
                        f.write(json.dumps(log_stats) + "\n")

                    # for evaluation logs
                    if coco_evaluator is not None:
                        (self.output_dir / 'eval').mkdir(exist_ok=True)
                        if self.iou_type in coco_evaluator.coco_eval:
                            filenames = ['latest.pth']
                            if epoch % 50 == 0:
                                filenames.append(f'{epoch:03}.pth')
                            for name in filenames:
                                torch.save(coco_evaluator.coco_eval[self.iou_type].eval,
                                        self.output_dir / "eval" / name)
                if torch.cuda.is_available():  # Just for clearing up GPU memory. You can remove it if you have enough GPU memory.
                    torch.cuda.empty_cache()

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))
            training_completed = True

            if training_completed and wandb_run is not None:
                self._log_model_artifact(wandb_run, f'{artifact_id}_model', [self.output_dir / 'best.pth'])
                wandb_run.summary['training_time'] = total_time
                wandb_run.summary['training_time_hms'] = total_time_str
        finally:
            if wandb_run is not None:
                wandb.finish()


    def val(self, ):
        self.eval()

        wandb_run = self._init_wandb('eval')

        try:
            module = self.ema.module if self.ema else self.model
            # compute a monotonic wandb step for evaluation logging
            if hasattr(self, 'train_dataloader') and self.train_dataloader is not None:
                train_steps = len(self.train_dataloader)
                wandb_step = (self.last_epoch + 1) * train_steps
            else:
                wandb_step = self.last_epoch
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                wandb_run=wandb_run,
                step=wandb_step,
            )

            if self.output_dir:
                eval_path = self.output_dir / 'eval.pth'
                dist_utils.save_on_master(coco_evaluator.coco_eval[self.iou_type].eval, eval_path)
                if wandb_run is not None:
                    artifact_id = getattr(wandb_run, 'id', wandb_run.name)
                    self._log_model_artifact(wandb_run, f'{artifact_id}_eval', [eval_path])

            if wandb_run is not None and dist_utils.is_main_process():
                wandb_run.summary['eval_epoch'] = self.last_epoch
                for k, values in test_stats.items():
                    if isinstance(values, (list, tuple)):
                        for i, value in enumerate(values):
                            wandb_run.summary[f'eval/{k}_{i}'] = value
                    else:
                        wandb_run.summary[f'eval/{k}'] = values
        finally:
            if wandb_run is not None:
                wandb.finish()

        return
