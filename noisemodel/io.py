"""
Dataset and DataLoader for CMB noise model learning with NmatDetvecs.

Each observation is preprocessed once and cached to disk as a .npz file
containing the full-length signal-subtracted TOD, focal plane positions, and
sample rate.  Subsequent epochs load from the cache, skipping the expensive
sotodlib preprocessing pipeline entirely.

If preload=True, all cached observations are additionally loaded into RAM at
construction time, so __getitem__ never touches disk during training.

What is stored in the cache
---------------------------
  tod:          [ndet, nsamp]  float32  — desloped, windowed+unwindowed TOD
  focal_plane:  [ndet, 2]     float32  — (xi, eta) detector positions
  srate:        scalar float            — sample rate in Hz

What is NOT stored (computed fresh per __getitem__ call)
--------------------------------------------------------
  - Random crop  (preserves time-axis data augmentation across epochs)
  - Per-detector variance normalization  (depends on the crop)
  - Focal plane normalization  (applied after loading, uses dataset-wide stats)

Variable ndet and nsamp are handled via:
  - ndet:  padding to max_ndet in the batch, with a boolean detector mask
  - nsamp: random cropping to a fixed chunk_size in __getitem__
"""

import os
import hashlib
import logging
import warnings
import argparse
import yaml
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from sotodlib import core, mapmaking
from sotodlib.preprocess import preprocess_util as pp_util
from sotodlib.site_pipeline.utils.config import _get_config
from pixell import utils as putils, fft

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: Path, sub_id: str) -> Path:
    safe = hashlib.md5(str(sub_id).encode()).hexdigest()
    return cache_dir / f"{safe}.pt"

def _load_from_cache(path: Path) -> dict:
    # Use weights_only=True for security and slightly faster loading
    return torch.load(path, weights_only=True)

def _save_to_cache(path: Path, tod: np.ndarray, focal_plane: np.ndarray,
                   srate: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    # We use a .tmp.pt extension for the atomic write
    tmp_path = path.with_name(path.stem + ".tmp.pt")
    # Convert arrays directly to PyTorch tensors before saving. 
    # This completely eliminates the numpy -> torch conversion during dataloading.
    data_to_save = {
        "tod": torch.from_numpy(tod.astype(np.float32)),
        "focal_plane": torch.from_numpy(focal_plane.astype(np.float32)),
        "srate": torch.tensor(srate, dtype=torch.float32)
    }
    # Save the dictionary using torch.save
    torch.save(data_to_save, tmp_path)
    # Rename atomically to the final path
    tmp_path.rename(path)

def _preprocess_and_cache_one(sub_id, cache_dir, context_path, preproc_path, window, downsample=None):
    """Module-level so it can be pickled by ProcessPoolExecutor."""
    context = core.Context(context_path)
    preproc = _get_config(preproc_path)

    meta = context.get_meta(sub_id)
    obs, _ = pp_util.load_and_preprocess(sub_id, preproc, context=context, meta=meta)

    # Steps that mirror the mapmaker
    obs.restrict('dets', obs.dets.vals[obs.det_info.wafer.type == 'OPTC'])
    mapmaking.fix_boresight_glitches(obs)
    putils.deslope(obs.signal, w=5, inplace=True)
    obs.signal = obs.signal.astype(np.float32)
    putils.deslope(obs.signal, w=5, inplace=True)
    # here we downsample
    if downsample is not None:
        obs = mapmaking.downsample_obs(obs, downsample)
    srate = (obs.samps.count - 1) / (obs.timestamps[-1] - obs.timestamps[0])
    tod = obs.signal.astype(np.float32)          # [ndet, nsamp]

    # Check if we have less than 50 detectors
    if tod.shape[0] < 50:
        raise ValueError("Less than 50 detectors left")

    # Check for nans in the TOD, if so we skip this sub_id
    if np.isnan(tod).any():
        raise ValueError("NaNs detected in TOD")

    fp = np.array(
        [obs.focal_plane.xi, obs.focal_plane.eta],
        dtype=np.float32,
    ).T                                          # [ndet, 2]
    _save_to_cache(_cache_path(Path(cache_dir), sub_id), tod, fp, srate)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LATDataset(Dataset):
    """
    Dataset of LAT observations for noise model learning.

    Each item represents one observation and returns:
      tod          Tensor [ndet, chunk_size]  — cropped, optionally normalised
      focal_plane  Tensor [ndet, 2]           — detector (xi, eta) positions
      srate        Tensor scalar              — sample rate in Hz
      ndet         int                        — true (unpadded) detector count
      nsamp        int                        — true (unpadded) sample count
      obs_id       str                        — observation identifier

    Parameters
    ----------
    list_of_obs : str
        Path to the observation list file passed to mapmaking.get_subids.
    context : str
        Path to the sotodlib context YAML.
    preprocess : str
        Path to the preprocessing config.
    band : str
        Frequency band filter (e.g. 'f090').
    downsample : int
        Downsample by this factor.
    cache_dir : str or Path
        Directory where preprocessed .npz files are stored.  Created
        automatically if it does not exist.  Multiple datasets (train / val /
        test) can safely share the same cache_dir — filenames are keyed by
        sub_id hash so there are no collisions.
    preload : bool
        If True, load all cached observations into RAM during __init__.
        Safe only when the dataset fits in memory; speeds up training by
        eliminating all disk I/O after the first epoch.
    chunk_size : int or None
        If set, randomly crop each TOD to this many samples per __getitem__
        call.  Acts as time-axis data augmentation.  If None the full TOD is
        returned (variable length — collate_fn pads).
    normalize_tod : bool
        Normalise each detector's TOD chunk to unit variance.  Applied after
        cropping so the scale is consistent with the chunk used.
    normalize_focal_plane : bool
        Normalise focal plane coordinates to [-1, 1] using a dataset-wide
        bounding box computed from metadata.
    window : float
        Window half-width in seconds, applied (and then unapplied) to the TOD
        before caching — matches the windowing done in the mapmaker.
    """

    def __init__(
        self,
        list_of_obs: str,
        context: str,
        preprocess: str,
        band: Optional[str] = 'f090',
        downsample: Optional[int] = None,
        num_cache_workers: Optional[int] = 1,
        cache_dir: Optional[str] = 'cache',
        preload: Optional[bool] = False,
        chunk_size: Optional[int] = 360_000,    # 30 min @ 200 Hz
        random_crop: Optional[bool] = True,
        normalize_tod: bool = True,
        normalize_focal_plane: bool = True,
        window: float = 2.0,
    ):
        self.context      = core.Context(context)
        self.context_path = context
        self.preproc      = _get_config(preprocess)
        self.preproc_path = preprocess
        self.band         = band
        self.cache_dir    = Path(cache_dir)
        self.preload      = preload
        self.chunk_size     = chunk_size
        self.random_crop  = random_crop
        self.normalize_tod          = normalize_tod
        self.normalize_focal_plane  = normalize_focal_plane
        self.window = window
        self.downsample = downsample
        self.num_cache_workers = num_cache_workers

        # Resolve observation list
        self.sub_ids = mapmaking.get_subids(list_of_obs, context=self.context)
        self.sub_ids = mapmaking.filter_subids(self.sub_ids, bands=self.band)
        log.info(f"LATDataset: {len(self.sub_ids)} observations after filtering")

        # --- Focal plane normalisation stats (Cached) ---
        self._fp_offset = None
        self._fp_scale  = None
        if self.normalize_focal_plane:
            import hashlib
            import json
            # Create a unique hash for this exact set of observations
            id_str = "".join(sorted(self.sub_ids)).encode('utf-8')
            dset_hash = hashlib.md5(id_str).hexdigest()[:12]
            # Ensure cache directory exists using the existing Path object
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            stats_file = self.cache_dir / f"fp_stats_{dset_hash}.json"
            if stats_file.exists():
                log.info(f"Loading cached focal plane stats from {stats_file.name}...")
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                    self._fp_offset = np.array(stats["offset"], dtype=np.float32)
                    self._fp_scale  = np.array(stats["scale"], dtype=np.float32)
            else:
                log.info("Computing focal plane stats (this will be cached for future runs)...")
                self._compute_focal_plane_stats()
                
                # Save the results to disk so we never have to compute this again
                with open(stats_file, 'w') as f:
                    json.dump({
                        "offset": self._fp_offset.tolist(),
                        "scale":  self._fp_scale.tolist()
                    }, f)
        # ------------------------------------------------

        # Populate the disk cache for any observations not yet cached
        self._populate_disk_cache()

        # Optionally load everything into RAM
        self._mem_cache: dict = {}
        if self.preload:
            self._load_into_memory()

    # ------------------------------------------------------------------
    # Focal plane normalisation
    # ------------------------------------------------------------------

    def _compute_focal_plane_stats(self):
        """
        Dataset-wide focal plane bounding box for normalisation.
        Reads metadata only (no TOD loading) so it is fast.
        """
        all_xy = []
        for sub_id in self.sub_ids:
            meta = self.context.get_meta(sub_id)
            fp = np.array(
                [meta.focal_plane.xi, meta.focal_plane.eta],
                dtype=np.float32,
            ).T                        # [ndet, 2]
            all_xy.append(fp)
        all_xy = np.concatenate(all_xy, axis=0)   # [total_dets, 2]
        fp_min = all_xy.min(axis=0)
        fp_max = all_xy.max(axis=0)
        self._fp_offset = (fp_max + fp_min) / 2.0
        self._fp_scale  = (fp_max - fp_min) / 2.0
        self._fp_scale  = np.where(self._fp_scale == 0, 1.0, self._fp_scale)

    def _normalize_fp(self, fp: np.ndarray) -> np.ndarray:
        if self._fp_offset is None:
            return fp
        return (fp - self._fp_offset) / self._fp_scale

    # ------------------------------------------------------------------
    # Disk cache population
    # ------------------------------------------------------------------

    def _preprocess_obs(self, sub_id) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Run the full sotodlib preprocessing pipeline for one observation.

        Returns
        -------
        tod          [ndet, nsamp]  float32
        focal_plane  [ndet, 2]     float32
        srate        float
        """
        meta = self.context.get_meta(sub_id)
        obs, _ = pp_util.load_and_preprocess(
            sub_id, self.preproc, context=self.context, meta=meta
        )

        # Steps that mirror the mapmaker
        obs.restrict('dets', obs.dets.vals[obs.det_info.wafer.type == 'OPTC'])
        mapmaking.fix_boresight_glitches(obs)
        putils.deslope(obs.signal, w=5, inplace=True)
        obs.signal = obs.signal.astype(np.float32)
        putils.deslope(obs.signal, w=5, inplace=True)
        # here we downsample
        if self.downsample is not None:
            obs = mapmaking.downsample_obs(obs, self.downsample)
        srate = (obs.samps.count - 1) / (obs.timestamps[-1] - obs.timestamps[0])
        tod = obs.signal.astype(np.float32)          # [ndet, nsamp]

        # Check if we have less than 50 detectors
        if tod.shape[0] < 50:
            raise ValueError("Less than 50 detectors left")

        # Check for nans in the TOD, if so we skip this sub_id
        if np.isnan(tod).any():
            raise ValueError("NaNs detected in TOD")

        fp = np.array(
            [obs.focal_plane.xi, obs.focal_plane.eta],
            dtype=np.float32,
        ).T                                          # [ndet, 2]

        return tod, fp, float(srate)

    def _populate_disk_cache(self):
        """
        For every observation not yet on disk, run the preprocessing pipeline
        and write a .npz cache file.  Already-cached observations are skipped.
        """
        missing = [
            sub_id for sub_id in self.sub_ids
            if not _cache_path(self.cache_dir, sub_id).exists()
        ]
        if not missing:
            log.info("Disk cache: all observations already cached.")
            return

        log.info(f"Disk cache: preprocessing {len(missing)} observations "
                 f"using {self.num_cache_workers} workers...")

        with ProcessPoolExecutor(max_workers=self.num_cache_workers) as pool:
            futures = {
                pool.submit(_preprocess_and_cache_one,
                            sub_id, self.cache_dir,
                            self.context_path, self.preproc_path,
                            self.window, downsample=self.downsample): sub_id
                for sub_id in missing
            }
            for i, future in enumerate(as_completed(futures)):
                sub_id = futures[future]
                try:
                    future.result()
                    log.info(f"  [{i+1}/{len(missing)}] cached {sub_id}")
                except Exception as e:
                    log.warning(f"Failed to preprocess {sub_id}: {e}. Skipping.")

    # ------------------------------------------------------------------
    # In-memory preload
    # ------------------------------------------------------------------

    def _load_into_memory(self):
        """Load all disk-cached observations into self._mem_cache."""
        log.info(f"Preloading {len(self.sub_ids)} observations into RAM...")
        n_loaded = 0
        for idx, sub_id in enumerate(self.sub_ids):
            path = _cache_path(self.cache_dir, sub_id)
            if not path.exists():
                warnings.warn(f"Cache file missing for {sub_id}, skipping preload.")
                continue
            self._mem_cache[idx] = _load_from_cache(path)
            n_loaded += 1
        log.info(f"Preloaded {n_loaded}/{len(self.sub_ids)} observations into RAM.")

    # ------------------------------------------------------------------
    # Core load — from memory or disk
    # ------------------------------------------------------------------

    def _load_obs(self, idx: int) -> dict:
        """
        Return the raw (uncropped, unnormalised) cached data for observation idx.
        Checks the in-memory cache first, falls back to disk.
        """
        if idx in self._mem_cache:
            return self._mem_cache[idx]

        path = _cache_path(self.cache_dir, self.sub_ids[idx])
        if not path.exists():
            # Cache miss after __init__ — can happen if _populate_disk_cache
            # skipped this obs due to a preprocessing error.  Try again now.
            warnings.warn(
                f"Cache miss at __getitem__ time for {self.sub_ids[idx]}. "
                f"Attempting to preprocess on the fly."
            )
            tod, fp, srate = self._preprocess_obs(self.sub_ids[idx])
            _save_to_cache(path, tod, fp, srate)

        return _load_from_cache(path)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sub_ids)

    def __getitem__(self, idx: int) -> dict:
        while True:
            try:
                data  = self._load_obs(idx)
                tod   = data["tod"]                          # [ndet, nsamp]  float32
                fp    = data["focal_plane"]                  # [ndet, 2]      float32
                srate = data["srate"]
                ndet, nsamp = tod.shape

                # Failsafe: check cached data just in case bad data slipped through
                if tod.shape[0] < 50:
                    raise ValueError("Cached TOD has < 50 detectors")
                if np.isnan(tod).any():
                    raise ValueError("NaNs found in cached TOD")

                # Sucess, we break the while loop
                break
            except Exception as e:
                log.warning(f"Skipping index {idx} ({self.sub_ids[idx]}) due to error: {e}")
                # Pick a brand new random index and try again
                idx = np.random.randint(0, len(self.sub_ids))

        # --- Random crop (time-axis data augmentation) --------------------
        if self.chunk_size is not None:
            if nsamp <= self.chunk_size:
                chunk = tod
            else:
                if self.random_crop:
                    t0    = np.random.randint(0, nsamp - self.chunk_size)
                else:
                    # Deterministic center crop for validation
                    t0 = (nsamp - self.chunk_size) // 2
                chunk = tod[:, t0 : t0 + self.chunk_size]
        else:
            chunk = tod

        # --- Focal plane normalisation ------------------------------------
        if self.normalize_focal_plane:
            fp = self._normalize_fp(fp)
        # the tod normalization, final deslope and windowing used to happen here
        # we moved them to the model so they can be done efficiently on gpu

        return {
            "tod":         chunk,
            "focal_plane": fp.clone(),
            "srate":       srate,
            "ndet":        ndet,
            "nsamp":       chunk.shape[1],
            "obs_id":      str(self.sub_ids[idx]),
        }


# ---------------------------------------------------------------------------
# Collate function — pads variable ndet across observations in a batch
# ---------------------------------------------------------------------------

def cmb_collate_fn(batch: list) -> dict:
    """
    Pad variable-ndet observations to the maximum detector count in the batch.

    Produces
    --------
    tod          [B, max_ndet, max_nsamp]  float32  — zero-padded
    focal_plane  [B, max_ndet, 2]         float32  — zero-padded
    det_mask     [B, max_ndet]            bool     — True = real detector
    srate        [B]
    ndet         [B]  int64
    nsamp        [B]  int64
    obs_ids      list[str]
    """
    max_ndet  = max(item["ndet"]  for item in batch)
    max_nsamp = max(item["nsamp"] for item in batch)
    B = len(batch)

    tod_out = torch.zeros(B, max_ndet, max_nsamp, dtype=torch.float32)
    fp_out  = torch.zeros(B, max_ndet, 2,         dtype=torch.float32)
    mask    = torch.zeros(B, max_ndet,             dtype=torch.bool)

    for i, item in enumerate(batch):
        nd = item["ndet"]
        ns = item["nsamp"]
        tod_out[i, :nd, :ns] = item["tod"]
        fp_out [i, :nd, :]   = item["focal_plane"]
        mask   [i, :nd]      = True

    return {
        "tod":         tod_out,
        "focal_plane": fp_out,
        "det_mask":    mask,
        "srate":       torch.stack([item["srate"] for item in batch]),
        "ndet":        torch.tensor([item["ndet"]  for item in batch], dtype=torch.long),
        "nsamp":       torch.tensor([item["nsamp"] for item in batch], dtype=torch.long),
        "obs_ids":     [item["obs_id"] for item in batch],
    }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_dataloader(
    list_of_obs: str,
    context: str,
    preprocess: str,
    band: Optional[str] = 'f090',
    downsample: Optional[int] = None,
    num_cache_workers: Optional[int] = 1,
    cache_dir: str = 'cache',
    preload: bool = False,
    batch_size: int = 4,
    chunk_size: Optional[int] = 360_000,
    random_crop: Optional[bool] = True,
    normalize_tod: bool = True,
    normalize_focal_plane: bool = True,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """
    Build a DataLoader for the CMB noise model dataset.

    Parameters
    ----------
    list_of_obs : str
        Path to the observation list file.
    context : str
        Path to the sotodlib context YAML.
    preprocess : str
        Path to the preprocessing config.
    band : str
        Frequency band (e.g. 'f090').
    cache_dir : str
        Directory for preprocessed .npz cache files.  Train, val, and test
        loaders can share the same cache_dir safely.
    preload : bool
        Load all observations into RAM after caching.  Requires sufficient
        memory but eliminates all disk I/O during training.
    batch_size : int
        Observations per batch.
    chunk_size : int or None
        TOD samples per observation (random crop).  None returns full TODs.
    normalize_tod : bool
        Normalise each detector TOD chunk to unit variance.
    normalize_focal_plane : bool
        Normalise focal plane coords to [-1, 1].
    shuffle : bool
        Shuffle between epochs (should be True for train, False for val/test).
    num_workers : int
        Worker processes for background loading.  Set 0 for debugging.
    pin_memory : bool
        Pin CPU tensors for faster GPU transfer.
    prefetch_factor : int
        Batches to prefetch per worker.
    """
    dataset = LATDataset(
        list_of_obs           = list_of_obs,
        context               = context,
        preprocess            = preprocess,
        band                  = band,
        downsample            = downsample,
        num_cache_workers     = num_cache_workers,
        cache_dir             = cache_dir,
        preload               = preload,
        chunk_size            = chunk_size,
        random_crop           = random_crop,
        normalize_tod         = normalize_tod,
        normalize_focal_plane = normalize_focal_plane,
    )
    return DataLoader(
        dataset,
        batch_size        = batch_size,
        shuffle           = shuffle,
        num_workers       = num_workers,
        collate_fn        = cmb_collate_fn,
        pin_memory        = pin_memory,
        prefetch_factor   = prefetch_factor if num_workers > 0 else None,
        persistent_workers = num_workers > 0,
    )

# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    cfg = make_default_config()
    with open(path) as f:
        overrides = yaml.safe_load(f)
    if overrides:
        cfg.update(overrides)
    return cfg


def save_config(cfg: dict, path: str):
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Override config values with explicit CLI arguments."""
    cli_map = {
        "lr":          "lr",
        "batch_size":  "batch_size",
        "n_epochs":    "n_epochs",
        "output_dir":  "output_dir",
        "seed":        "seed",
    }
    for attr, key in cli_map.items():
        val = getattr(args, attr, None)
        if val is not None:
            cfg[key] = val
    return cfg

# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

def make_default_config() -> dict:
    return {
        # --- Data ---
        "list_of_obs_train": "",          # required
        "list_of_obs_val":   "",          # required
        "list_of_obs_test":  "",          # optional — evaluated once at the end
        "context":           "",          # required
        "preprocess":        "",          # required
        "band":              "f090",
        "chunk_size":        360_000,     # 30 min at 200 Hz
        "normalize_tod":     True,
        "normalize_focal_plane": True,
        "num_workers":       4,
        "prefetch_factor":   2,
        "preload":         False,

        # --- Model ---
        "nmode":    20,
        "d_model":  128,
        "d_latent": 64,
        "d_hidden": 256,
        "n_heads":  4,
        "n_layers": 3,
        "dropout":  0.1,

        # --- Optimiser ---
        "lr":            3e-4,
        "weight_decay":  1e-5,
        "grad_clip":     1.0,       # max gradient norm; 0 to disable

        # --- Schedule ---
        # "onecycle" ramps up then down over the whole training run.
        # "cosine" decays from lr to lr_min over n_epochs.
        # "none" keeps lr constant.
        "schedule":      "cosine",
        "lr_min":        1e-6,      # floor for cosine schedule
        "warmup_epochs": 2,         # linear warmup before schedule kicks in

        # --- Training ---
        "batch_size":        4,
        "n_epochs":          50,
        "val_every_n_steps": 200,   # run validation every N training steps
        "log_every_n_steps": 20,    # print/log loss every N steps

        # --- Checkpointing ---
        "output_dir":         "runs/exp001",
        "checkpoint_every_n_epochs": 5,
        "keep_last_n_checkpoints":   3,   # older ones are deleted

        # --- Misc ---
        "seed":      42,
        "amp":       True,          # automatic mixed precision (float16 on GPU)
        "pin_memory": True,
    }