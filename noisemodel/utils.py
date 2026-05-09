# ---------------------------------------------------------------------------
# Spectral utilities
# ---------------------------------------------------------------------------
import numpy as np
import torch
from typing import Optional

def torch_deslope(tod: torch.Tensor, w: int = 5) -> torch.Tensor:
    """
    Remove a linear trend connecting the start and end of the TOD on the GPU.
    tod: [B, ndet, nsamp]
    """
    nsamp = tod.shape[-1]
    # Average the first w and last w samples
    start_val = tod[..., :w].mean(dim=-1, keepdim=True)
    end_val = tod[..., -w:].mean(dim=-1, keepdim=True)
    # Create a linear ramp from 0 to 1
    ramp = torch.linspace(0, 1, steps=nsamp, device=tod.device, dtype=tod.dtype)
    ramp = ramp.view(1, 1, nsamp)
    # Subtract the trend: start_val + (end_val - start_val) * ramp
    trend = start_val + (end_val - start_val) * ramp
    return tod - trend

def torch_tukey_window(window_length: int, alpha: float = 0.1, device="cpu") -> torch.Tensor:
    """
    Generate a Tukey (tapered cosine) window directly on the GPU.

    alpha: Fraction of the window inside the cosine tapered region.
           alpha=0 is a rectangular window, alpha=1 is a Hann window.
    """
    if alpha <= 0:
        return torch.ones(window_length, device=device)
    elif alpha >= 1.0:
        return torch.hann_window(window_length, periodic=False, device=device)
    taper_pts = int(alpha * window_length / 2)
    window = torch.ones(window_length, device=device)
    if taper_pts > 0:
        # Generate a Hann window to use for the smooth tapered edges
        hann = torch.hann_window(2 * taper_pts, periodic=False, device=device)
        # Apply the left and right halves of the Hann window to our edges
        window[:taper_pts] = hann[:taper_pts]
        window[-taper_pts:] = hann[-taper_pts:]
    return window

def torch_apply_window(tod: torch.Tensor, window: float = 2.0, srate: float=200.0) -> torch.Tensor:
    """
    Apply a cosine taper (Tukey window) to the edges of the TOD on the GPU.
    tod: [B, ndet, nsamp]
    window in seconds
    srate in Hz
    """
    nsamp = tod.shape[-1]
    window_frac = (window*srate)/nsamp
    # Generate the window directly on the same GPU as the TOD
    window_ = torch_tukey_window(nsamp, alpha=window_frac, device=tod.device).to(tod.dtype)
    # Multiply the TOD by the window (broadcasts automatically across B and ndet)
    return tod * window_.view(1, 1, nsamp)

def make_freq_bins(
    srate: float,
    nsamp: int,
    nbin: int,
    fmin: float = 0.01,
    fmax: Optional[float] = None,
) -> torch.Tensor:
    """
    Build log-spaced frequency bin edges matching NmatDetvecs convention.

    Returns
    -------
    edges : Tensor [nbin + 1]  — bin edges in units of FFT bin indices
    """
    if fmax is None:
        fmax = srate / 2.0
    freqs = torch.fft.rfftfreq(nsamp, d=1.0 / srate)  # [nfreq]
    fmin  = max(fmin, freqs[1].item())                 # can't go below df
    fmax  = min(fmax, freqs[-1].item())
    edges = torch.exp(
        torch.linspace(np.log(fmin), np.log(fmax), nbin + 1)
    )
    return edges  # [nbin + 1]

def bin_power_spectrum(
    ftod: torch.Tensor,            # [B, ndet, nfreq]  complex
    edges: torch.Tensor,           # [nbin + 1]  freq edges (Hz)
    freqs: torch.Tensor,           # [nfreq]  Hz
    mask: torch.Tensor,            # [B, ndet]  bool
    eps: float = 1e-30,
) -> torch.Tensor:
    """
    Compute mean power per detector per log-spaced frequency bin.

    Returns
    -------
    psd : Tensor [B, ndet, nbin]  — mean power (real, positive)
    """
    B, ndet, nfreq = ftod.shape
    nbin = len(edges) - 1
    psd = torch.zeros(B, ndet, nbin, device=ftod.device, dtype=torch.float32)

    for b in range(nbin):
        lo, hi = edges[b].item(), edges[b + 1].item()
        idx = (freqs >= lo) & (freqs < hi)
        if idx.sum() == 0:
            continue
        # Mean power in bin b, per detector
        power = ftod[:, :, idx].abs().pow(2).mean(dim=-1)  # [B, ndet]
        psd[:, :, b] = power

    # Zero out padded detectors so they never affect statistics
    psd = psd * mask.unsqueeze(-1)
    return psd  # [B, ndet, nbin]