import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
import time

import mlflow
import sys
import csv
import os

import logging

logger = logging.getLogger(__name__)


def _move_to_device(x, device: str, non_blocking: bool = True):
    """Recursively move tensors (or collections of tensors) to a device."""
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=non_blocking)
    if isinstance(x, dict):
        return {k: _move_to_device(v, device, non_blocking) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [_move_to_device(v, device, non_blocking) for v in x]
        return type(x)(t)
    return x

# NOTE: bunch of flags for tf32 precision
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# NOTE: bunch of flags for torch.compile() for faster training
torch._inductor.config.conv_1x1_as_mm = True # NOTE: treats 1x1 convolutions as faster matrix multiplications
torch._inductor.config.coordinate_descent_tuning = True # NOTE: enables advanced tuning to find the best kernel
torch._inductor.config.epilogue_fusion = True # NOTE: for matmul + bias operations
torch._inductor.config.coordinate_descent_check_all_directions = True # NOTE: makes the tuning search even harder (longer compile, but faster result)

class Trainer():
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer = None, scheduler: _LRScheduler = None, compile: bool = False):
        self.model = model
        self.model_name = model._get_name() # NOTE: get the name of a Model for expiriments convinience
        self.optimizer = optimizer or optim.AdamW(model.parameters(), lr=3e-4)
        self.scheduler = scheduler
        self.device = next(self.model.parameters()).device.type
        self.compile = compile

        if self.compile:
            # optimize the model for the fastest training
            self.model = torch.compile(self.model, mode="max-autotune", fullgraph=True)

    @staticmethod
    def accuracy(logits, y):
        return (logits.argmax(1) == y).float().mean().item()
    
    def save(self, path: str, include_optimizer: bool = True, include_scheduler: bool = True):
        state = {
            "model_name": self.model_name,
            "model_state": self.model.state_dict(),
            "compile": self.compile
        }
        if include_optimizer:
            state["optimizer_state"] = self.optimizer.state_dict()
        if include_scheduler and self.scheduler is not None:
            state["scheduler_state"] = self.scheduler.state_dict()
        tmp_path = path + ".tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
        return path
    
    def create_submission(self,
                          test_loader: DataLoader,
                          class_names,
                          submission_path: str = "submission.csv"):
        self.model.eval()
        rows = []
        with torch.inference_mode():
            for xb, ids in test_loader:
                xb = _move_to_device(xb, self.device)
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    out = self.model(xb)
                logits = out[0] if isinstance(out, tuple) else out
                preds = logits.argmax(1).tolist()
                for sample_id, pred_idx in zip(ids, preds):
                    rows.append((sample_id, class_names[pred_idx]))

        tmp_path = submission_path + ".tmp"
        with open(tmp_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "label"])
            writer.writerows(rows)
        os.replace(tmp_path, submission_path)

        logger.info(f"Submission written to {submission_path} ({len(rows)} rows)")
        return submission_path

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        # In DataBricks MLflow we have the next experiment naming
        # e.g. /Users/<user_email>/<experiment_name>
        exp = mlflow.set_experiment(experiment_name=f"/Users/artemserbulov117@gmail.com/9 Topics {self.model_name}")

        with mlflow.start_run(experiment_id=exp.experiment_id):
            for epoch in range(1, epochs + 1):
                t0 = time.time()
                self.model.train()
                train_loss = 0.0
                train_acc = 0.0
                for xb, yb in train_loader:
                    xb = _move_to_device(xb, self.device)
                    yb = _move_to_device(yb, self.device)
                    self.optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=self.device, dtype=torch.bfloat16): # NOTE: one line of code for bfloat16 oprations
                        logits, loss = self.model(xb, yb)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # NOTE: for not shocking the model too bad because of a bad batch
                    self.optimizer.step()
                    
                    # Step scheduler per batch (for warmup schedulers)
                    if self.scheduler is not None:
                        self.scheduler.step()
                    
                    train_loss += loss.item() * yb.size(0)
                    train_acc += (logits.argmax(1) == yb).float().sum().item()
                
                self.model.eval()
                val_loss = 0.0
                val_acc = 0.0
                with torch.inference_mode():
                    for xb, yb in val_loader:
                        xb = _move_to_device(xb, self.device)
                        yb = _move_to_device(yb, self.device)
                        with torch.autocast(device_type=self.device, dtype=torch.bfloat16): # NOTE: one line of code for bfloat16 oprations
                            logits, loss = self.model(xb, yb)
                        val_loss += loss.item() * yb.size(0)
                        val_acc += (logits.argmax(1) == yb).float().sum().item()

                torch.cuda.synchronize() # NOTE: wait for the GPU to perform all operations
                t1 = time.time()
                dt = (t1 - t0) # NOTE: time difference in seconds

                n_tr, n_va = len(train_loader.dataset), len(val_loader.dataset)
                train_loss /= n_tr; train_acc /= n_tr
                val_loss /= n_va; val_acc /= n_va

                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                }, step=epoch)

                logger.info(f"Epoch {epoch:02d} | train_acc={train_acc:.3f} val_acc={val_acc:.3f} | time/sec={dt:.3f}")

                current_lr = self.optimizer.param_groups[0]["lr"]
                mlflow.log_metric("lr", current_lr, step=epoch)