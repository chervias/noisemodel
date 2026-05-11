import argparse
import logging
import torch
import sys
from pathlib import Path

from noisemodel.io import load_config, make_dataloader
from noisemodel.model import CMBNoiseAutoencoder, training_step

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

def evaluate_test_set(cfg_path, only_cache=False, num_cache_workers=None):
    # Load config
    cfg = load_config(cfg_path)
    
    # Check if test dataset is defined
    test_list = cfg.get("list_of_obs_test", "")
    if not test_list:
        raise ValueError("Please define 'list_of_obs_test' in your config.yaml.")
        
    # Resolution priority: CLI argument -> config.yaml -> fallback to standard num_workers
    config_workers = cfg.get("num_cache_workers", cfg.get("num_workers", 1))
    workers = num_cache_workers if num_cache_workers is not None else config_workers

    log.info(f"Preparing test dataset from {test_list} ...")
    log.info(f"Using {workers} workers for disk caching.")
    
    # 1. Create the test dataloader (This handles the caching automatically)
    test_loader = make_dataloader(
        list_of_obs           = test_list,
        context               = cfg["context"],
        preprocess            = cfg["preprocess"],
        band                  = cfg["band"],
        downsample            = cfg.get("downsample", None),
        num_cache_workers     = workers,  # <--- Now respects the config!
        cache_dir             = str(Path(cfg["output_dir"]) / "cache"),
        preload               = False,
        batch_size            = cfg["batch_size"],
        chunk_size            = None,   
        normalize_tod         = cfg["normalize_tod"],
        normalize_focal_plane = cfg["normalize_focal_plane"],
        shuffle               = False,  
        num_workers           = cfg["num_workers"],
        pin_memory            = cfg["pin_memory"],
    )
    
    if only_cache:
        log.info("Test set disk caching complete! Exiting due to --only-cache.")
        return

    # 2. Initialise the bare model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    log.info("Initialising model...")
    
    model = CMBNoiseAutoencoder(
        nmode     = cfg["nmode"],
        bin_edges = cfg.get("bin_edges", None),
        nbin      = cfg.get("nbin", None),
        fmin      = cfg.get("fmin", None),
        fmax      = cfg.get("fmax", None),
        d_model  = cfg["d_model"],
        d_latent = cfg["d_latent"],
        d_hidden = cfg["d_hidden"],
        n_heads  = cfg["n_heads"],
        n_layers = cfg["n_layers"],
        dropout  = cfg["dropout"],
        normalize_tod = cfg.get("normalize_tod", True),
        window = cfg.get("window", 2.0),
    ).to(device)
    
    # 3. Load the best weights
    ckpt_path = Path(cfg["output_dir"]) / "checkpoints" / "best.pt"
    log.info(f"Loading weights from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)    
    # Strip prefixes added by torch.compile() or DDP
    state_dict = ckpt["model_state"]
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k.replace("_orig_mod.", "").replace("module.", "")
        clean_state_dict[clean_key] = v
    model.load_state_dict(clean_state_dict)
    model.eval()
    
    log.info(f"Evaluating {len(test_loader.dataset)} test observations...")
    
    # 4. Evaluation Loop
    total_loss = 0.0
    n_batches = 0
    
    for batch_idx, batch in enumerate(test_loader):
        with torch.no_grad():
            loss = training_step(model, batch, device, amp=cfg.get("amp", False))
            
        total_loss += loss.item()
        n_batches += 1
        log.info(f"  Batch [{batch_idx+1}/{len(test_loader)}] | batch NLL: {loss.item():.4f} | mean NLL: {total_loss / n_batches:.4f}")

    mean_loss = total_loss / max(n_batches, 1)
    log.info(f"\n========================================")
    log.info(f"Test Evaluation Complete!")
    log.info(f"Final Test NLL Loss: {mean_loss:.4f}")
    log.info(f"========================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config.yaml")
    parser.add_argument("--only-cache", action="store_true", help="Build the cache for the test set and exit")
    parser.add_argument("--num-cache-workers", type=int, default=None, help="Override the number of CPU workers for caching via CLI")
    args = parser.parse_args()
    
    evaluate_test_set(args.config, only_cache=args.only_cache, num_cache_workers=args.num_cache_workers)