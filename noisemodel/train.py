"""
Training code
"""
import numpy as np
import math
import time
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import logging

from .io import make_dataloader, load_config, save_config, apply_cli_overrides
from .model import CMBNoiseAutoencoder, training_step
from .log import setup_logging, CSVLogger

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: CMBNoiseAutoencoder,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    cfg: dict,
):
    torch.save(
        {
            "epoch":          epoch,
            "global_step":    global_step,
            "best_val_loss":  best_val_loss,
            "model_state":    model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "config":         cfg,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: CMBNoiseAutoencoder,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt


def rotate_checkpoints(ckpt_dir: Path, prefix: str, keep: int):
    """Delete oldest epoch checkpoints, keeping only the last `keep`."""
    ckpts = sorted(ckpt_dir.glob(f"{prefix}_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    for old in ckpts[:-keep]:
        old.unlink()
        log.info(f"Deleted old checkpoint: {old.name}")

def train(cfg: dict, only_cache: bool = False):
    # ---- Setup ----------------------------------------------------------------
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    output_dir = Path(cfg["output_dir"])
    ckpt_dir   = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir    = setup_logging(output_dir)

    save_config(cfg, output_dir / "config_used.yaml")
    csv_logger = CSVLogger(log_dir / "train_log.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ---- Dataloaders ----------------------------------------------------------
    log.info("Building dataloaders...")

    shared_loader_kwargs = dict(
        context               = cfg["context"],
        preprocess            = cfg["preprocess"],
        band                  = cfg["band"],
        downsample            = cfg["downsample"],
        chunk_size            = cfg["chunk_size"],
        normalize_tod         = cfg["normalize_tod"],
        normalize_focal_plane = cfg["normalize_focal_plane"],
        num_workers           = cfg["num_workers"],
        num_cache_workers     = cfg["num_cache_workers"],
        pin_memory            = cfg["pin_memory"] and device.type == "cuda",
        prefetch_factor       = cfg["prefetch_factor"],
        preload               = cfg["preload"],
    )

    train_loader = make_dataloader(
        list_of_obs = cfg["list_of_obs_train"],
        batch_size  = cfg["batch_size"],
        shuffle     = True,
        **shared_loader_kwargs,
    )
    val_loader = make_dataloader(
        list_of_obs = cfg["list_of_obs_val"],
        batch_size  = cfg["batch_size"],
        shuffle     = False,
        **shared_loader_kwargs,
    )
    if only_cache:
        log.info(f"Only cache requested, we are done")
        return False

    log.info(f"Train observations : {len(train_loader.dataset)}")
    log.info(f"Val   observations : {len(val_loader.dataset)}")
    log.info(f"Train batches/epoch: {len(train_loader)}")

    # ---- Model ----------------------------------------------------------------
    model = CMBNoiseAutoencoder(
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

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {n_params:,}")

    # ---- Optimiser ------------------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr           = cfg["lr"],
        weight_decay = cfg["weight_decay"],
    )

    # ---- LR Schedule ----------------------------------------------------------
    total_steps   = len(train_loader) * cfg["n_epochs"]
    warmup_steps  = len(train_loader) * cfg["warmup_epochs"]

    if cfg["schedule"] == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr      = cfg["lr"],
            total_steps = total_steps,
            pct_start   = cfg["warmup_epochs"] / cfg["n_epochs"],
            anneal_strategy = "cos",
            div_factor  = 10.0,
            final_div_factor = cfg["lr"] / max(cfg["lr_min"], 1e-12),
        )
    elif cfg["schedule"] == "cosine":
        # Linear warmup then cosine decay — handled manually below
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max  = total_steps - warmup_steps,
            eta_min = cfg["lr_min"],
        )
    else:
        scheduler = None

    def warmup_lr(step: int):
        """Scale LR linearly from lr/100 to lr over warmup_steps."""
        if step < warmup_steps and cfg["schedule"] == "cosine":
            scale = (step + 1) / max(warmup_steps, 1)
            for pg in optimizer.param_groups:
                pg["lr"] = cfg["lr"] * scale

    # ---- AMP scaler -----------------------------------------------------------
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["amp"] and device.type == "cuda")

    # ---- Training state -------------------------------------------------------
    best_val_loss  = math.inf
    global_step    = 0
    start_epoch    = 0

    # Resume from checkpoint if one exists
    latest_ckpt = ckpt_dir / "latest.pt"
    if latest_ckpt.exists():
        log.info(f"Resuming from {latest_ckpt}")
        ckpt = load_checkpoint(latest_ckpt, model, optimizer, scheduler)
        start_epoch   = ckpt["epoch"] + 1
        global_step   = ckpt["global_step"]
        best_val_loss = ckpt["best_val_loss"]
        # Reset LR to current effective_lr in case config changed since checkpoint
        for pg in optimizer.param_groups:
            pg["lr"] = effective_lr
        log.info(f"  Resumed at epoch {start_epoch}, step {global_step}, "
                 f"best_val_loss={best_val_loss:.4f}")

    # ---- Epoch loop -----------------------------------------------------------
    log.info("Starting training...")
    model.train()

    for epoch in range(start_epoch, cfg["n_epochs"]):
        epoch_loss   = 0.0
        epoch_steps  = 0
        epoch_start  = time.time()

        for batch in train_loader:
            # ------ Warmup LR (cosine schedule only) -------------------------
            warmup_lr(global_step)

            # ------ Forward + loss -------------------------------------------
            optimizer.zero_grad()

            loss = training_step(model, batch, device, amp=cfg["amp"])

            # ------ Backward -------------------------------------------------
            scaler.scale(loss).backward()

            if cfg["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])

            # Track scale before step so we can detect whether the optimizer
            # actually ran.  With AMP, scaler.step() silently skips the
            # optimizer when gradients are inf/nan (common in the first few
            # steps), which triggers a spurious scheduler warning.
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = (scaler.get_scale() == scale_before)

            # Step schedule only when the optimizer actually ran
            if scheduler is not None and optimizer_stepped:
                if cfg["schedule"] == "onecycle":
                    scheduler.step()
                elif cfg["schedule"] == "cosine" and global_step >= warmup_steps:
                    scheduler.step()

            # ------ Bookkeeping ----------------------------------------------
            loss_val    = loss.item()
            epoch_loss += loss_val
            epoch_steps += 1
            global_step += 1

            current_lr = optimizer.param_groups[0]["lr"]

            # Log every N steps
            if global_step % cfg["log_every_n_steps"] == 0:
                log.info(
                    f"epoch {epoch:3d}  step {global_step:6d}  "
                    f"train_loss={loss_val:.4f}  lr={current_lr:.2e}"
                )
                csv_logger.write({
                    "epoch":       epoch,
                    "global_step": global_step,
                    "train_loss":  loss_val,
                    "lr":          current_lr,
                    "val_loss":    "",
                })

            # Mid-epoch validation
            if global_step % cfg["val_every_n_steps"] == 0:
                val_loss = evaluate(
                    model, val_loader, device, cfg["amp"],
                    max_batches=20,   # quick check — full val runs at epoch end
                )
                log.info(
                    f"  [mid-epoch val]  step {global_step:6d}  "
                    f"val_loss={val_loss:.4f}"
                )
                csv_logger.write({
                    "epoch":       epoch,
                    "global_step": global_step,
                    "train_loss":  "",
                    "lr":          current_lr,
                    "val_loss":    val_loss,
                })
                model.train()

        # ------ End-of-epoch --------------------------------------------------
        epoch_time   = time.time() - epoch_start
        mean_train   = epoch_loss / max(epoch_steps, 1)

        # Full validation pass
        val_loss = evaluate(model, val_loader, device, cfg["amp"])
        model.train()

        log.info(
            f"Epoch {epoch:3d} done | "
            f"train={mean_train:.4f}  val={val_loss:.4f}  "
            f"time={epoch_time:.0f}s"
        )
        csv_logger.write({
            "epoch":       epoch,
            "global_step": global_step,
            "train_loss":  mean_train,
            "lr":          optimizer.param_groups[0]["lr"],
            "val_loss":    val_loss,
        })

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                ckpt_dir / "best.pt",
                model, optimizer, scheduler,
                epoch, global_step, best_val_loss, cfg,
            )
            log.info(f"  *** New best val_loss={best_val_loss:.4f} — saved best.pt")

        # Save latest (for resuming)
        save_checkpoint(
            ckpt_dir / "latest.pt",
            model, optimizer, scheduler,
            epoch, global_step, best_val_loss, cfg,
        )

        # Periodic epoch checkpoint
        if (epoch + 1) % cfg["checkpoint_every_n_epochs"] == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(
                ckpt_path,
                model, optimizer, scheduler,
                epoch, global_step, best_val_loss, cfg,
            )
            log.info(f"  Saved epoch checkpoint: {ckpt_path.name}")
            rotate_checkpoints(ckpt_dir, "epoch", cfg["keep_last_n_checkpoints"])

    # ---- Test set evaluation --------------------------------------------------
    if cfg.get("list_of_obs_test", ""):
        log.info("Running test set evaluation with best checkpoint...")
        load_checkpoint(ckpt_dir / "best.pt", model)
        test_loader = make_dataloader(
            list_of_obs = cfg["list_of_obs_test"],
            batch_size  = cfg["batch_size"],
            shuffle     = False,
            **shared_loader_kwargs,
        )
        log.info(f"Test observations: {len(test_loader.dataset)}")
        test_loss = evaluate(model, test_loader, device, cfg["amp"])
        log.info(f"Test loss: {test_loss:.4f}")
        csv_logger.write({
            "epoch":       "test",
            "global_step": global_step,
            "train_loss":  "",
            "lr":          "",
            "val_loss":    test_loss,
        })

    csv_logger.close()
    log.info(f"Training complete. Outputs in: {output_dir}")
    return best_val_loss

# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: CMBNoiseAutoencoder,
    loader,
    device: torch.device,
    amp: bool,
    max_batches: Optional[int] = None,
) -> float:
    """
    Run the model over `loader` and return mean NLL loss.

    Parameters
    ----------
    max_batches : int or None
        Cap the number of batches (useful for quick mid-epoch val checks).
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        with torch.no_grad():
            loss = training_step(model, batch, device, amp=amp)

        total_loss += loss.item()
        n_batches  += 1

    model.train()
    return total_loss / max(n_batches, 1)
