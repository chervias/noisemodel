"""
Self-supervised noise model autoencoder for CMB mapmaking.

Architecture overview
---------------------
The model takes a signal-subtracted TOD [B, ndet, nsamp] and focal plane
positions [B, ndet, 2] and predicts the parameters of a Woodbury noise model:

    N_b = diag(D_b) + V_b V_b^T

where:
  D  [B, nbin, ndet]        — uncorrelated noise power per detector per freq bin
  V  [B, nbin, ndet, nmode] — correlated mode amplitudes  (factored as vecs * sqrt(E))

vecs [B, ndet, nmode]  are the spatial mode shapes (shared across all bins)
E    [B, nbin, nmode]  are the per-bin mode amplitudes

This exactly mirrors the parametrisation of NmatDetvecs, but the mode shapes
and amplitudes are predicted by a neural network rather than estimated by PCA.

The model is trained with the self-supervised Gaussian log-likelihood loss,
computed efficiently WITHOUT ever forming the dense [ndet, ndet] covariance:

    L_b = sum_d log(D_b_d) + log det(M_b)
        + tr(D_b^{-1} Chat_b) - tr(M_b^{-1} W_b W_b^H / |F_b|)

where M_b = I_k + V_b^T diag(1/D_b) V_b  is a [k x k] matrix (cheap)
      W_b = (diag(1/D_b) V_b)^T ftod_b    is a [k x |F_b|] matrix

All bottleneck operations are O(ndet * nmode) rather than O(ndet^2).

Encoder
-------
1. FFT the TOD and bin into log-spaced frequency bins  →  per-detector log-PSD
   [B, ndet, nbin]
2. Embed each detector: [log-PSD | focal-plane (x,y)]  →  d_model via Linear
3. N transformer layers with masked self-attention over the detector axis
   (detectors attend to each other, learning cross-detector correlations)
4. Masked mean-pool over detectors  →  global latent z  [B, d_latent]
   (this is the bottleneck: forces a compact atmospheric-state representation)

Decoder
-------
5. Broadcast z back to each detector, concatenate with local PSD embedding
6. MLP per detector  →  three output heads:
     D_head   [B, ndet, nbin]   softplus   (positive uncorrelated power)
     vecs_head [B, ndet, nmode]            (spatial mode shapes)
     E_head   [B, nbin, nmode]  softplus   (positive per-bin amplitudes)
7. Assemble  V[b, d, :] = vecs[d, :] * sqrt(E[b, :])

The factored V = vecs * sqrt(E) is an inductive bias matching NmatDetvecs:
mode spatial shapes are continuous across frequency, only amplitudes vary.

Inputs from the dataloader (io.py)
-----------------------------------
  tod        [B, max_ndet, max_nsamp]  float32  — zero-padded
  focal_plane [B, max_ndet, 2]        float32  — zero-padded
  det_mask   [B, max_ndet]            bool     — True = real detector
  srate      [B]                      float32  — samples per second
  ndet       [B]                      int64    — unpadded detector counts
  nsamp      [B]                      int64    — unpadded sample counts
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .utils import *

# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def fourier_encode(x, num_bands=4):
    x = x.unsqueeze(-1)
    scales = 2.0 ** torch.arange(num_bands, device=x.device, dtype=x.dtype)
    x = x * scales * math.pi
    return torch.cat([x.sin(), x.cos()], dim=-1).flatten(-2)

class DetectorTransformerEncoder(nn.Module):
    """
    Transformer encoder that operates over the *detector* axis.
 
    Each detector is a token.  Its initial embedding is formed from its
    local log-PSD features and focal-plane position.  Self-attention layers
    then allow each detector to gather information from all others, learning
    the cross-detector correlation structure.
 
    A final masked mean-pool produces a single global vector z [B, d_latent]
    that represents the atmospheric state for this observation — this is the
    autoencoder bottleneck.
 
    Parameters
    ----------
    nbin : int
        Number of frequency bins (PSD feature length per detector).
    d_model : int
        Internal transformer width.
    d_latent : int
        Bottleneck latent dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer encoder layers.
    dropout : float
    """
 
    def __init__(
        self,
        nbin: int,
        d_model: int = 128,
        d_latent: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Per-detector input: log-PSD (nbin) + focal-plane xy (2)
        # we are encoding the focal plane into 16 now
        self.input_proj = nn.Linear(nbin + 16, d_model)
 
        # Stack of standard TransformerEncoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,   # expects [B, seq, d_model]
            norm_first=True,    # pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
 
        # Project pooled representation to bottleneck
        self.pool_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_latent),
        )
 
    def forward(
        self,
        log_psd: torch.Tensor,     # [B, ndet, nbin]
        focal_plane: torch.Tensor, # [B, ndet, 2]
        det_mask: torch.Tensor,    # [B, ndet]  True = real
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        z       : [B, d_latent]    — global bottleneck vector
        h       : [B, ndet, d_model] — per-detector hidden states (for decoder)
        """
        # Build per-detector token embeddings
        fp_encoded = fourier_encode(focal_plane, num_bands=4) # turns 2 coords into 16 features
        x = torch.cat([log_psd, fp_encoded], dim=-1) # [B, ndet, nbin+2]
        x = self.input_proj(x)                           # [B, ndet, d_model]
 
        # TransformerEncoder uses src_key_padding_mask where True = *ignore*
        # det_mask is True = real, so we invert it
        padding_mask = ~det_mask  # [B, ndet]  True = padded token
 
        h = self.transformer(x, src_key_padding_mask=padding_mask)  # [B, ndet, d_model]
 
        # Masked mean pool → global latent z
        mask_f = det_mask.unsqueeze(-1).float()           # [B, ndet, 1]
        z = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1)  # [B, d_model]
        z = self.pool_proj(z)                             # [B, d_latent]
 
        return z, h


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class NoiseDecoder(nn.Module):
    """
    Decode the global latent z + per-detector hidden states into noise model
    parameters (D, vecs, E).

    The output noise model is:

        N_b = diag(D_b) + V_b V_b^T
        V_b[:, d, :] = vecs[d, :] * sqrt(E[b, :])     (factored form)

    Parameters
    ----------
    nbin : int
        Number of frequency bins.
    nmode : int
        Number of correlated modes.
    d_model : int
        Width of the encoder hidden states passed in.
    d_latent : int
        Bottleneck dimension.
    d_hidden : int
        Width of the decoder MLP.
    """

    def __init__(
        self,
        nbin: int,
        nmode: int,
        d_model: int = 128,
        d_latent: int = 64,
        d_hidden: int = 256,
    ):
        super().__init__()
        self.nbin = nbin
        self.nmode = nmode

        # Fuse global z with per-detector encoder hidden state
        self.fuse = nn.Sequential(
            nn.Linear(d_model + d_latent, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
        )

        # Head 1: D — uncorrelated power per detector per bin
        # Output [ndet, nbin], activated with softplus to guarantee > 0
        self.D_head = nn.Linear(d_hidden, nbin)

        # Head 2: vecs — spatial mode shapes, shared across frequency bins
        # Output [ndet, nmode], no activation (normalised later)
        self.vecs_head = nn.Linear(d_hidden, nmode)

        # Head 3: E — per-bin mode amplitudes
        # Input is just z (global), output [nbin, nmode], softplus > 0
        self.E_head = nn.Sequential(
            nn.Linear(d_latent, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, nbin * nmode),
        )

    def forward(
        self,
        z: torch.Tensor,           # [B, d_latent]
        h: torch.Tensor,           # [B, ndet, d_model]
        det_mask: torch.Tensor,    # [B, ndet]  True = real
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        D    : [B, nbin, ndet]    — uncorrelated noise power (positive)
        vecs : [B, ndet, nmode]   — spatial mode shapes
        E    : [B, nbin, nmode]   — per-bin mode amplitudes (positive)
        """
        B, ndet, _ = h.shape

        # Broadcast z to every detector, then fuse with per-detector features
        z_exp = z.unsqueeze(1).expand(-1, ndet, -1)         # [B, ndet, d_latent]
        fused = self.fuse(torch.cat([h, z_exp], dim=-1))    # [B, ndet, d_hidden]

        # D: [B, ndet, nbin]  → rearrange to [B, nbin, ndet]
        D = F.softplus(self.D_head(fused))                   # [B, ndet, nbin]
        D = D.permute(0, 2, 1)                               # [B, nbin, ndet]

        # vecs: [B, ndet, nmode]  — L2-normalise each mode across real detectors
        vecs = self.vecs_head(fused)                         # [B, ndet, nmode]
        # Normalise using only real detectors to avoid padded-det influence
        mask_f = det_mask.unsqueeze(-1).float()              # [B, ndet, 1]
        norm = (vecs.pow(2) * mask_f).sum(dim=1, keepdim=True).sqrt() + 1e-8
        vecs = vecs / norm                                   # [B, ndet, nmode]

        # E: [B, nbin, nmode]
        E = F.softplus(self.E_head(z))                       # [B, nbin*nmode]
        E = E.view(B, self.nbin, self.nmode)                 # [B, nbin, nmode]

        # Zero out padded detectors in D and vecs
        D    = D    * det_mask.unsqueeze(1).float()           # [B, nbin, ndet]
        vecs = vecs * mask_f                                  # [B, ndet, nmode]

        return D, vecs, E


# ---------------------------------------------------------------------------
# Loss — self-supervised Gaussian log-likelihood via Woodbury
# ---------------------------------------------------------------------------

def woodbury_nll_loss(
    ftod: torch.Tensor,    # [B, ndet, nfreq]  complex  (rfft output)
    freqs: torch.Tensor,   # [nfreq]  Hz
    edges: torch.Tensor,   # [nbin+1] Hz  — bin edges
    D: torch.Tensor,       # [B, nbin, ndet]  positive
    vecs: torch.Tensor,    # [B, ndet, nmode]
    E: torch.Tensor,       # [B, nbin, nmode]  positive
    det_mask: torch.Tensor,# [B, ndet]  True = real
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute the mean Gaussian negative log-likelihood per frequency bin,
    using the Woodbury / matrix-determinant lemma to stay O(ndet * nmode).

    The model covariance for bin b is:
        N_b = diag(D_b) + V_b V_b^T          V_b = vecs * sqrt(E_b)

    The NLL decomposes as:
        L_b = log_det_term + trace_term

    log_det_term (via matrix-determinant lemma):
        = sum_d log(D_b[d]) + log det(I_k + V_b^T diag(1/D_b) V_b)

    trace_term (via Woodbury):
        = tr(diag(1/D_b) Chat_b) - tr(M_b^{-1} W_b W_b^H / |F_b|)

    where M_b = I_k + V_b^T diag(1/D_b) V_b    [k x k]
          W_b = (diag(1/D_b) V_b)^T ftod_b      [k x |F_b|]

    Chat_b is never formed explicitly; only its diagonal and
    projections onto V_b are needed.

    Returns
    -------
    loss : scalar — mean NLL per bin (averaged over batch and bins)
    """
    B, ndet, nfreq = ftod.shape
    nbin = D.shape[1]
    device = ftod.device

    total_loss = torch.zeros(1, device=device)
    n_bins_used = 0

    for b in range(nbin):
        lo = edges[b].item()
        hi = edges[b + 1].item()
        idx = (freqs >= lo) & (freqs < hi)
        n_f = idx.sum().item()
        if n_f == 0:
            continue

        # ftod in this bin: [B, ndet, n_f]  complex
        ft_b = ftod[:, :, idx]

        D_b    = D[:, b, :]          # [B, ndet]   positive
        E_b    = E[:, b, :]          # [B, nmode]  positive
        # V_b = vecs * sqrt(E_b): [B, ndet, nmode]
        V_b    = vecs * E_b.unsqueeze(1).sqrt()   # [B, ndet, nmode]

        D_b_clamped = D_b.clamp(min=1e-3)

        #iD_b   = 1.0 / (D_b + eps)  # [B, ndet]
        iD_b = 1.0 / D_b_clamped
        iD_V   = iD_b.unsqueeze(-1) * V_b  # [B, ndet, nmode]  = diag(1/D) V

        # M = I + V^T diag(1/D) V  [B, nmode, nmode]
        M = torch.eye(vecs.shape[-1], device=device).unsqueeze(0) \
            + V_b.transpose(-1, -2) @ iD_V           # [B, nmode, nmode]

        # ------ log-det term ------
        # sum_d log D_b[d]  (only real detectors)
        log_D_sum = D_b_clamped.log() * det_mask.float()
#        log_D_sum = (D_b + eps).log() * det_mask.float()  # [B, ndet]
        log_D_sum = log_D_sum.sum(dim=-1)                  # [B]

        # log det(M) via Cholesky for numerical stability
        try:
            L_chol = torch.linalg.cholesky(M)
            log_det_M = 2.0 * L_chol.diagonal(dim1=-1, dim2=-2).log().sum(-1)  # [B]
        except Exception:
            # fallback: slogdet
            sign, log_det_M = torch.linalg.slogdet(M)
            log_det_M = log_det_M * sign.clamp(min=0)

        log_det_term = log_D_sum + log_det_M  # [B]

        # ------ trace term ------
        # tr(diag(1/D) Chat) = sum_d (1/D_b[d]) * (1/n_f) * sum_f |ft_b[d,f]|^2
        #                    = sum_d (1/D_b[d]) * psd_b[d]
        psd_b = ft_b.abs().pow(2).mean(dim=-1)    # [B, ndet]  mean power in bin
        trace_iD_C = (iD_b * psd_b * det_mask.float()).sum(dim=-1)  # [B]

        # W = (diag(1/D) V)^T @ ftod_b  [B, nmode, n_f]  (complex)
        # iD_V: [B, ndet, nmode] → transpose: [B, nmode, ndet]
        # ft_b: [B, ndet, n_f]
        W = iD_V.transpose(-1, -2).to(ft_b.dtype) @ ft_b  # [B, nmode, n_f]

        # WW^H / n_f  [B, nmode, nmode]  (Hermitian, real part suffices)
        WW = (W @ W.conj().transpose(-1, -2)).real / n_f   # [B, nmode, nmode]

        # tr(M^{-1} WW^H / n_f)  — solve M x = WW, take trace
        try:
            MiWW = torch.linalg.solve(M, WW)  # [B, nmode, nmode]
        except Exception:
            MiWW = torch.linalg.lstsq(M, WW).solution
        trace_woodbury = MiWW.diagonal(dim1=-1, dim2=-2).sum(-1)  # [B]

        trace_term = trace_iD_C - trace_woodbury  # [B]

        # Accumulate: NLL = log_det + trace, minimised at N = Chat
        bin_loss = (log_det_term + trace_term).mean()
        total_loss = total_loss + bin_loss
        n_bins_used += 1

    return total_loss / max(n_bins_used, 1)


# ---------------------------------------------------------------------------
# Full autoencoder model
# ---------------------------------------------------------------------------

class CMBNoiseAutoencoder(nn.Module):
    """
    Self-supervised noise model autoencoder for CMB mapmaking.
 
    Predicts Woodbury noise model parameters (D, vecs, E) for a given
    signal-subtracted TOD and focal plane configuration.
 
    Parameters
    ----------
    nbin : int
        Number of log-spaced frequency bins covering [fmin, fmax].
    nmode : int
        Number of correlated atmospheric modes (rank of V_b V_b^T).
    fmin : float
        Minimum frequency for noise model bins (Hz).
    fmax : float or None
        Maximum frequency for noise model bins (Hz).
        If None, uses srate/2.
    d_model : int
        Transformer hidden dimension.
    d_latent : int
        Bottleneck latent dimension. Smaller = more regularisation.
    d_hidden : int
        Decoder MLP hidden dimension.
    n_heads : int
        Number of self-attention heads.
    n_layers : int
        Number of transformer encoder layers.
    dropout : float
        Dropout probability.
    """
 
    def __init__(
        self,
        nbin: int = 48,
        nmode: int = 20,
        fmin: float = 0.16,
        fmax: Optional[float] = None,
        d_model: int = 128,
        d_latent: int = 64,
        d_hidden: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.nbin   = nbin
        self.nmode  = nmode
        self.fmin   = fmin
        self.fmax   = fmax
 
        self.encoder = DetectorTransformerEncoder(
            nbin=nbin,
            d_model=d_model,
            d_latent=d_latent,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.decoder = NoiseDecoder(
            nbin=nbin,
            nmode=nmode,
            d_model=d_model,
            d_latent=d_latent,
            d_hidden=d_hidden,
        )
 
    def _get_spectral_features(
        self,
        tod: torch.Tensor,       # [B, ndet, nsamp]  (may be padded)
        det_mask: torch.Tensor,  # [B, ndet]
        srate: torch.Tensor,     # [B]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        FFT the TOD and compute log-PSD features.
 
        Returns
        -------
        ftod    : [B, ndet, nfreq]  complex — full one-sided FFT
        log_psd : [B, ndet, nbin]   float   — log mean power per bin per detector
        freqs   : [nfreq]           float   — frequency axis (Hz), built from
                                              median srate across the batch
        edges   : [nbin+1]          float   — bin edges (Hz)
        """
        B, ndet, nsamp = tod.shape
 
        # Use median srate for building common freq grid across the batch.
        # Individual srates in SO are very close (~200 Hz) so this is fine.
        srate_val = srate.median().item()
        freqs = torch.fft.rfftfreq(nsamp, d=1.0 / srate_val).to(tod.device)
        edges = make_freq_bins(
            srate_val, nsamp, self.nbin, self.fmin,
            self.fmax if self.fmax is not None else srate_val / 2.0
        ).to(tod.device)
 
        # FFT along the time axis
        ftod = torch.fft.rfft(tod, dim=-1)  # [B, ndet, nfreq]
 
        # Bin the power spectrum
        psd = bin_power_spectrum(ftod, edges, freqs, det_mask)  # [B, ndet, nbin]
 
        # Log-scale PSD (better dynamic range for the network)
        log_psd = (psd + 1e-30).log()
 
        return ftod, log_psd, freqs, edges
 
    def forward(
        self,
        tod: torch.Tensor,           # [B, max_ndet, max_nsamp]
        focal_plane: torch.Tensor,   # [B, max_ndet, 2]
        det_mask: torch.Tensor,      # [B, max_ndet]  bool
        srate: torch.Tensor,         # [B]
    ) -> dict:
        """
        Forward pass.
 
        Returns a dict with:
          D     : [B, nbin, ndet]    — predicted uncorrelated noise power
          vecs  : [B, ndet, nmode]   — predicted spatial mode shapes
          E     : [B, nbin, nmode]   — predicted per-bin mode amplitudes
          V     : [B, nbin, ndet, nmode] — assembled V = vecs * sqrt(E)
          z     : [B, d_latent]      — bottleneck latent vector
          ftod  : [B, ndet, nfreq]   — complex FFT (for loss computation)
          freqs : [nfreq]            — frequency axis (Hz)
          edges : [nbin+1]           — bin edges (Hz)
        """
        # 1. Spectral feature extraction
        ftod, log_psd, freqs, edges = self._get_spectral_features(
            tod, det_mask, srate
        )
 
        # 2. Encode: detector-axis transformer → z, h
        z, h = self.encoder(log_psd, focal_plane, det_mask)
 
        # 3. Decode: z + h → D, vecs, E
        D, vecs, E = self.decoder(z, h, det_mask)
 
        # 4. Assemble V_b = vecs * sqrt(E_b)  [B, nbin, ndet, nmode]
        sqrt_E = E.sqrt().unsqueeze(2)                     # [B, nbin, 1, nmode]
        V = vecs.unsqueeze(1) * sqrt_E                     # [B, nbin, ndet, nmode]
 
        return {
            "D":      D,       # [B, nbin, ndet]
            "vecs":   vecs,    # [B, ndet, nmode]
            "E":      E,       # [B, nbin, nmode]
            "V":      V,       # [B, nbin, ndet, nmode]
            "z":      z,       # [B, d_latent]
            "ftod":   ftod,    # [B, ndet, nfreq]  complex
            "freqs":  freqs,   # [nfreq]
            "edges":  edges,   # [nbin+1]
        }
 
    def loss(
        self,
        out: dict,
        det_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the self-supervised Gaussian NLL loss from a forward() output.
 
        Parameters
        ----------
        out : dict — output from forward()
        det_mask : [B, ndet]  bool
 
        Returns
        -------
        loss : scalar tensor
        """
        return woodbury_nll_loss(
            ftod     = out["ftod"],
            freqs    = out["freqs"],
            edges    = out["edges"],
            D        = out["D"],
            vecs     = out["vecs"],
            E        = out["E"],
            det_mask = det_mask,
        )

# ---------------------------------------------------------------------------
# Training step (call from your training loop)
# ---------------------------------------------------------------------------

def training_step(
    model: CMBNoiseAutoencoder,
    batch: dict,
    device: torch.device,
    amp: bool = False,
) -> torch.Tensor:
    """
    Single training step given a batch from the LATDataset dataloader.

    Parameters
    ----------
    model  : CMBNoiseAutoencoder
    batch  : dict from cmb_collate_fn — keys: tod, focal_plane, det_mask, srate, ...
    device : torch.device
    amp    : bool — enable automatic mixed precision (float16 on CUDA)

    Returns
    -------
    loss : scalar tensor with grad
    """
    tod         = batch["tod"].to(device, non_blocking=True)
    focal_plane = batch["focal_plane"].to(device, non_blocking=True)
    det_mask    = batch["det_mask"].to(device, non_blocking=True)
    srate       = batch["srate"].to(device, non_blocking=True)

    with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
        out  = model(tod, focal_plane, det_mask, srate)
        loss = model.loss(out, det_mask)
    return loss