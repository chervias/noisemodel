"""
Entry point for multi-GPU training of CMBNoiseAutoencoder.

This script is the DDP analog of train_script.py.  It must be launched via
torchrun so that LOCAL_RANK / RANK / WORLD_SIZE are set correctly:

    torchrun --nproc_per_node=4 train_script_ddp.py --config config.yaml

For a SLURM job on Perlmutter use the provided submit_perlmutter.sh.

CLI flags are identical to train_script.py; --only-cache is also supported.
"""

import os
import sys
import argparse
import logging

import yaml

import noisemodel
from noisemodel.train_ddp import train

log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-GPU training of CMBNoiseAutoencoder on SO LAT data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",       required=True, help="Path to YAML config file")
    p.add_argument("--lr",           type=float,    help="Override learning rate")
    p.add_argument("--batch-size",   type=int,      dest="batch_size",
                   help="Override batch size (per GPU)")
    p.add_argument("--n-epochs",     type=int,      dest="n_epochs",
                   help="Override number of epochs")
    p.add_argument("--output-dir",   type=str,      dest="output_dir",
                   help="Override output directory")
    p.add_argument("--seed",         type=int,      help="Override random seed")
    p.add_argument("--print-config", action="store_true",
                   help="Print resolved config and exit")
    p.add_argument("--only-cache",   action="store_true",
                   help="Only populate the disk cache, then exit")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    cfg = noisemodel.load_config(args.config)
    cfg = noisemodel.apply_cli_overrides(cfg, args)

    if args.print_config:
        # Only rank 0 should print — torchrun starts all ranks so guard here
        if int(os.environ.get("RANK", "0")) == 0:
            print(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
        sys.exit(0)

    required = ["list_of_obs_train", "list_of_obs_val", "context", "preprocess"]
    missing  = [k for k in required if not cfg.get(k)]
    if missing:
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"ERROR: missing required config fields: {missing}")
        sys.exit(1)

    if args.only_cache:
        train(cfg, only_cache=True)
    else:
        train(cfg)