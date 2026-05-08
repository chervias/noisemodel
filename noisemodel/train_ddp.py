"""
Multi-GPU training loop for CMBNoiseAutoencoder (DistributedDataParallel).

This module is the DDP analog of train.py.  The public interface is identical:

    from .train_ddp import train
    train(cfg)

It must be launched via torchrun, which sets LOCAL_RANK / RANK / WORLD_SIZE:

    torchrun --nproc_per_node=4 train_script_ddp.py --config config.yaml

Changes versus train.py
-----------------------
1.  setup_ddp() / cleanup_ddp()  — initialise NCCL process group.
2.  DistributedSampler            — each GPU sees a non-overlapping shard.
3.  DDP(model)                    — gradient averaging across GPUs.
4.  Linear LR scaling             — lr * world_size (standard rule).
5.  rank-0 gate                   — only rank 0 logs, writes CSV, saves ckpts.
6.  Cache barrier                 — rank 0 populates disk cache, then barrier,
                                    so ranks 1-3 never race on .tmp files.
7.  save/load_checkpoint          — uses model.module.state_dict() to strip DDP.
8.  reduce_mean()                 — averages scalar loss across GPUs for logging.
9.  evaluate()                    — runs on rank 0 only, passes model.module to
                                    training_step (avoids DDP on inference path).
10. train_sampler.set_epoch()     — required for correct per-epoch shuffling.
"""

import os
import math
import time
import logging
import numpy as np
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR

from .io import (LATDataset, make_dataloader, cmb_collate_fn,
                 load_config, save_config, apply_cli_overrides)
from .model import CMBNoiseAutoencoder, training_step
from .log import setup_logging, CSVLogger

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp():
    """
    Initialise NCCL process group from env vars set by torchrun:
    LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT.
    Returns (local_rank, rank, world_size).
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # Add device_id to silence the warnings
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),  # ← add this
    )
    return local_rank, dist.get_rank(), dist.get_world_size()

def cleanup_ddp():
    dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def reduce_mean(tensor: torch.Tensor) -> float:
    """Sum a scalar tensor across all ranks and return the mean as a float."""
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / dist.get_world_size()).item()


# ---------------------------------------------------------------------------
# Checkpoint helpers  (override train.py versions to handle DDP wrapper)
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, scheduler,
                    epoch, global_step, best_val_loss, cfg):
    """Save checkpoint — unwraps DDP so the file is identical to single-GPU."""
    torch.save({
        "epoch":           epoch,
        "global_step":     global_step,
        "best_val_loss":   best_val_loss,
        "model_state":     model.module.state_dict(),   # unwrap DDP
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "config":          cfg,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load checkpoint — works whether model is DDP-wrapped or bare."""
    ckpt   = torch.load(path, map_location="cpu")
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt


def rotate_checkpoints(ckpt_dir: Path, prefix: str, keep: int):
    ckpts = sorted(ckpt_dir.glob(f"{prefix}_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    for old in ckpts[:-keep]:
        old.unlink()
        log.info(f"Deleted old checkpoint: {old.name}")


# ---------------------------------------------------------------------------
# Evaluation  (rank 0 only — passes model.module to avoid DDP on eval path)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, amp, max_batches=None):
    """
    Mean NLL loss over loader.  Always called on rank 0 only.
    Passes model.module to training_step so DDP is not involved in inference.
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    bare = model.module   # unwrapped CMBNoiseAutoencoder

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        with torch.no_grad():
            loss = training_step(bare, batch, device, amp=amp)
        total_loss += loss.item()
        n_batches  += 1

    model.train()
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: dict, only_cache: bool = False):
    # ---- DDP initialisation --------------------------------------------------
    local_rank, rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # Different seed per rank so random crops differ across GPUs each step
    torch.manual_seed(cfg["seed"] + rank)
    np.random.seed(cfg["seed"]   + rank)

    # ---- Directories and logging (rank 0 only) --------------------------------
    output_dir = Path(cfg["output_dir"])
    ckpt_dir   = output_dir / "checkpoints"

    if is_main(rank):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir    = setup_logging(output_dir)
        save_config(cfg, output_dir / "config_used.yaml")
        csv_logger = CSVLogger(log_dir / "train_log.csv")
        log.info(f"World size : {world_size} GPUs")
        log.info(f"GPU {rank}  : {torch.cuda.get_device_name(local_rank)}")

    # ---- Disk cache — rank 0 populates, others wait --------------------------
    # Without this barrier, ranks 1-3 would race rank 0 on the same .tmp files
    # the first time a dataset is built.
    if is_main(rank):
        log.info("Rank 0: populating disk cache for all splits...")
        _cache_kwargs = dict(
            context               = cfg["context"],
            preprocess            = cfg["preprocess"],
            band                  = cfg["band"],
            chunk_size            = cfg["chunk_size"],
            normalize_tod         = cfg["normalize_tod"],
            normalize_focal_plane = cfg["normalize_focal_plane"],
            preload               = False,
        )
        for key in ("list_of_obs_train", "list_of_obs_val", "list_of_obs_test"):
            if cfg.get(key, ""):
                LATDataset(list_of_obs=cfg[key],
                           cache_dir=cfg.get("cache_dir", "cache"),
                           **_cache_kwargs)
        log.info("Rank 0: disk cache ready.")

    dist.barrier()   # ranks 1-3 wait here until rank 0 finishes caching

    if only_cache:
        if is_main(rank):
            log.info("Only cache requested, we are done.")
        cleanup_ddp()
        return False

    # ---- Dataloaders ---------------------------------------------------------
    _dataset_kwargs = dict(
        context               = cfg["context"],
        preprocess            = cfg["preprocess"],
        band                  = cfg["band"],
        chunk_size            = cfg["chunk_size"],
        normalize_tod         = cfg["normalize_tod"],
        normalize_focal_plane = cfg["normalize_focal_plane"],
        preload               = cfg["preload"],
        cache_dir             = cfg.get("cache_dir", "cache"),
    )

    # Training: DistributedSampler gives every rank a non-overlapping shard
    train_dataset = LATDataset(
        list_of_obs = cfg["list_of_obs_train"],
        **_dataset_kwargs,
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas = world_size,
        rank         = rank,
        shuffle      = True,
        drop_last    = True,   # keeps all ranks at the same batch count
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size         = cfg["batch_size"],
        sampler            = train_sampler,
        num_workers        = cfg["num_workers"],
        collate_fn         = cmb_collate_fn,
        pin_memory         = True,
        prefetch_factor    = cfg["prefetch_factor"] if cfg["num_workers"] > 0 else None,
        persistent_workers = cfg["num_workers"] > 0,
    )

    # Validation: rank 0 only, full val set, plain sequential loader
    if is_main(rank):
        val_loader = make_dataloader(
            list_of_obs     = cfg["list_of_obs_val"],
            batch_size      = cfg["batch_size"],
            shuffle         = False,
            num_workers     = cfg["num_workers"],
            pin_memory      = True,
            prefetch_factor = cfg["prefetch_factor"],
            random_crop     = False,
            **_dataset_kwargs,
        )
        log.info(f"Train observations : {len(train_dataset)} "
                 f"({len(train_dataset) // world_size} per GPU)")
        log.info(f"Val   observations : {len(val_loader.dataset)}")
        log.info(f"Train batches/GPU/epoch: {len(train_loader)}")

    # ---- Model ---------------------------------------------------------------
    base_model = CMBNoiseAutoencoder(
        nbin     = cfg["nbin"],
        nmode    = cfg["nmode"],
        fmin     = cfg["fmin"],
        fmax     = cfg["fmax"],
        d_model  = cfg["d_model"],
        d_latent = cfg["d_latent"],
        d_hidden = cfg["d_hidden"],
        n_heads  = cfg["n_heads"],
        n_layers = cfg["n_layers"],
        dropout  = cfg["dropout"],
    ).to(device)

    model = DDP(base_model, device_ids=[local_rank])

    if is_main(rank):
        n_params = sum(p.numel() for p in base_model.parameters()
                       if p.requires_grad)
        log.info(f"Model parameters: {n_params:,}")

    # ---- Optimiser -----------------------------------------------------------
    # Linear LR scaling: effective batch = batch_size * world_size,
    # so scale LR proportionally.
    effective_lr = cfg["lr"]
    optimizer = AdamW(
        model.parameters(),
        lr           = effective_lr,
        weight_decay = cfg["weight_decay"],
    )
    if is_main(rank):
        log.info(f"Effective LR: {effective_lr:.2e} "
                 f"(base {cfg['lr']:.2e} x {world_size} GPUs)")

    # ---- LR Schedule ---------------------------------------------------------
    total_steps  = len(train_loader) * cfg["n_epochs"]
    warmup_steps = len(train_loader) * cfg["warmup_epochs"]

    if cfg["schedule"] == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr           = effective_lr,
            total_steps      = total_steps,
            pct_start        = cfg["warmup_epochs"] / cfg["n_epochs"],
            anneal_strategy  = "cos",
            div_factor       = 10.0,
            final_div_factor = effective_lr / max(cfg["lr_min"], 1e-12),
        )
    elif cfg["schedule"] == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max   = total_steps - warmup_steps,
            eta_min = cfg["lr_min"],
        )
    else:
        scheduler = None

    def warmup_lr(step: int):
        if step < warmup_steps and cfg["schedule"] == "cosine":
            scale = (step + 1) / max(warmup_steps, 1)
            for pg in optimizer.param_groups:
                pg["lr"] = effective_lr * scale

    # ---- AMP scaler ----------------------------------------------------------
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])

    # ---- Resume --------------------------------------------------------------
    best_val_loss = math.inf
    global_step   = 0
    start_epoch   = 0

    latest_ckpt = ckpt_dir / "latest.pt"
    if latest_ckpt.exists():
        if is_main(rank):
            log.info(f"Resuming from {latest_ckpt}")
        ckpt = load_checkpoint(latest_ckpt, model, optimizer, scheduler)
        start_epoch   = ckpt["epoch"] + 1
        global_step   = ckpt["global_step"]
        best_val_loss = ckpt["best_val_loss"]
        # Reset LR to current effective_lr in case config changed since checkpoint
        for pg in optimizer.param_groups:
            pg["lr"] = effective_lr
        if is_main(rank):
            log.info(f"  Resumed at epoch {start_epoch}, step {global_step}, "
                     f"best_val_loss={best_val_loss:.4f}")

    # ---- Epoch loop ----------------------------------------------------------
    if is_main(rank):
        log.info("Starting training...")
    model.train()

    for epoch in range(start_epoch, cfg["n_epochs"]):
        # Required: makes shuffling differ per epoch across all ranks
        train_sampler.set_epoch(epoch)

        epoch_loss  = 0.0
        epoch_steps = 0
        epoch_start = time.time()

        for batch in train_loader:
            warmup_lr(global_step)
            optimizer.zero_grad()

            # We cannot use training_step(model, ...) here because DDP does
            # not expose model.loss().  Instead we inline the forward pass
            # through the DDP wrapper (which handles gradient sync) and call
            # model.module.loss() on the underlying CMBNoiseAutoencoder.
            tod         = batch["tod"].to(device, non_blocking=True)
            focal_plane = batch["focal_plane"].to(device, non_blocking=True)
            det_mask    = batch["det_mask"].to(device, non_blocking=True)
            srate       = batch["srate"].to(device, non_blocking=True)

            with torch.autocast(device.type, enabled=cfg["amp"]):
                out  = model(tod, focal_plane, det_mask, srate)
                loss = model.module.loss(out, det_mask)
                if not torch.isfinite(loss) or loss.item()>15000:
                    log.warning(
                        f"  Anomaly detected (loss={loss.item():.0f}) at step {global_step}. "
                        f"Skipping batch to prevent gradient explosion.")
                    optimizer.zero_grad()
                    continue

            scaler.scale(loss).backward()

            if cfg["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])

            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = (scaler.get_scale() == scale_before)

            if scheduler is not None and optimizer_stepped:
                if cfg["schedule"] == "onecycle":
                    scheduler.step()
                elif cfg["schedule"] == "cosine" and global_step >= warmup_steps:
                    scheduler.step()

            # Average loss across GPUs for logging
            mean_loss    = reduce_mean(loss.detach())
            epoch_loss  += mean_loss
            epoch_steps += 1
            global_step += 1

            if is_main(rank):
                current_lr = optimizer.param_groups[0]["lr"]

                if global_step % cfg["log_every_n_steps"] == 0:
                    log.info(
                        f"epoch {epoch:3d}  step {global_step:6d}  "
                        f"train_loss={mean_loss:.4f}  lr={current_lr:.2e}"
                    )
                    csv_logger.write({
                        "epoch": epoch, "global_step": global_step,
                        "train_loss": mean_loss, "lr": current_lr, "val_loss": "",
                    })

                if global_step % cfg["val_every_n_steps"] == 0:
                    val_loss = evaluate(model, val_loader, device, cfg["amp"],
                                        max_batches=20)
                    log.info(f"  [mid-epoch val]  step {global_step:6d}  "
                             f"val_loss={val_loss:.4f}")
                    csv_logger.write({
                        "epoch": epoch, "global_step": global_step,
                        "train_loss": "", "lr": current_lr, "val_loss": val_loss,
                    })
                    model.train()

        # ------ End of epoch --------------------------------------------------
        epoch_time = time.time() - epoch_start
        mean_train = epoch_loss / max(epoch_steps, 1)

        if is_main(rank):
            full_val_every = cfg.get("full_val_every_n_epochs", 10)
            if (epoch + 1) % full_val_every == 0:
                val_loss = evaluate(model, val_loader, device, cfg["amp"])  # uncapped
            else:
                val_loss = evaluate(model, val_loader, device, cfg["amp"],
                        max_batches=cfg.get("val_max_batches", 30))
            model.train()

            log.info(
                f"Epoch {epoch:3d} done | "
                f"train={mean_train:.4f}  val={val_loss:.4f}  "
                f"time={epoch_time:.0f}s"
            )
            csv_logger.write({
                "epoch": epoch, "global_step": global_step,
                "train_loss": mean_train,
                "lr": optimizer.param_groups[0]["lr"],
                "val_loss": val_loss,
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler,
                                epoch, global_step, best_val_loss, cfg)
                log.info(f"  *** New best val_loss={best_val_loss:.4f} — saved best.pt")

            save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, scheduler,
                            epoch, global_step, best_val_loss, cfg)

            if (epoch + 1) % cfg["checkpoint_every_n_epochs"] == 0:
                ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
                save_checkpoint(ckpt_path, model, optimizer, scheduler,
                                epoch, global_step, best_val_loss, cfg)
                log.info(f"  Saved epoch checkpoint: {ckpt_path.name}")
                rotate_checkpoints(ckpt_dir, "epoch", cfg["keep_last_n_checkpoints"])

        dist.barrier()   # all ranks sync before starting next epoch

    # ---- Test ----------------------------------------------------------------
    if is_main(rank) and cfg.get("list_of_obs_test", ""):
        log.info("Running test set evaluation with best checkpoint...")
        load_checkpoint(ckpt_dir / "best.pt", model)
        test_loader = make_dataloader(
            list_of_obs     = cfg["list_of_obs_test"],
            batch_size      = cfg["batch_size"],
            shuffle         = False,
            num_workers     = cfg["num_workers"],
            pin_memory      = True,
            prefetch_factor = cfg["prefetch_factor"],
            **_dataset_kwargs,
        )
        log.info(f"Test observations: {len(test_loader.dataset)}")
        test_loss = evaluate(model, test_loader, device, cfg["amp"])
        log.info(f"Test loss: {test_loss:.4f}")
        csv_logger.write({
            "epoch": "test", "global_step": global_step,
            "train_loss": "", "lr": "", "val_loss": test_loss,
        })

    if is_main(rank):
        csv_logger.close()
        log.info(f"Training complete. Outputs in: {output_dir}")

    cleanup_ddp()
    return best_val_loss