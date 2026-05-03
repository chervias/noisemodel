# ---------------------------------------------------------------------------
# Spectral utilities
# ---------------------------------------------------------------------------
import numpy as np
import torch
from typing import Optional


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