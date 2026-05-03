"""
Training script for CMBNoiseAutoencoder on real SO LAT data.

Usage
-----
    python train_script.py --config config.yaml

All options can be set in a YAML config file (see make_default_config())
or overridden on the command line, e.g.:

    python train_script.py --config config.yaml --lr 3e-4 --batch-size 4

Outputs (all written to --output-dir):
    checkpoints/best.pt          — best validation loss
    checkpoints/epoch_{N}.pt     — periodic epoch checkpoints
    logs/train_log.csv           — loss per step
    config_used.yaml             — full resolved config (for reproducibility)
"""

import os
import sys
import argparse
import logging

import yaml

import noisemodel

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train CMBNoiseAutoencoder on SO LAT data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",      required=True, help="Path to YAML config file")
    p.add_argument("--lr",          type=float,    help="Override learning rate")
    p.add_argument("--batch-size",  type=int,      dest="batch_size",
                   help="Override batch size")
    p.add_argument("--n-epochs",    type=int,      dest="n_epochs",
                   help="Override number of epochs")
    p.add_argument("--output-dir",  type=str,      dest="output_dir",
                   help="Override output directory")
    p.add_argument("--seed",        type=int,      help="Override random seed")
    p.add_argument("--print-config", action="store_true",
                   help="Print resolved config and exit")
    return p.parse_args()


def make_example_config(path: str = "config.yaml"):
    """Write an example config file to disk — edit before running."""
    cfg = noisemodel.make_default_config()
    cfg.update({
        "list_of_obs_train": "/path/to/train_obs.txt",
        "list_of_obs_val":   "/path/to/val_obs.txt",
        "list_of_obs_test":  "/path/to/test_obs.txt",
        "context":           "/path/to/context.yaml",
        "preprocess":        "/path/to/preprocess.yaml",
        "output_dir":        "runs/exp001",
    })
    save_config(cfg, path)
    print(f"Example config written to {path}")


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.config):
        # First-time helper: write an example config if the file doesn't exist
        print(f"Config file not found: {args.config}")
        print("Writing example config — edit it then re-run.")
        make_example_config(args.config)
        sys.exit(0)

    cfg = noisemodel.load_config(args.config)
    cfg = noisemodel.apply_cli_overrides(cfg, args)

    if args.print_config:
        print(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
        sys.exit(0)

    # Validate required fields
    required = ["list_of_obs_train", "list_of_obs_val", "context", "preprocess"]
    missing  = [k for k in required if not cfg.get(k)]
    if missing:
        print(f"ERROR: missing required config fields: {missing}")
        sys.exit(1)

    noisemodel.train(cfg)