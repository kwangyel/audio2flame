"""
Audio2FLAME model and loss.

A per-subject, non-autoregressive Audio2FLAME regressor:

    raw waveform (16 kHz)
        -> Wav2Vec2 encoder (fine-tuned end-to-end)
        -> linear interpolation from ~50 Hz to the video fps
        -> Transformer encoder + sinusoidal positional encoding
        -> linear head -> FLAME params (exp[100] + jaw[6] = 106)

The audio CNN feature extractor of Wav2Vec2 is frozen by default (it is a
low-level acoustic front-end and rarely benefits from per-subject fine-tuning),
while the transformer layers of Wav2Vec2 are fine-tuned.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model


def align_to_fps(features: torch.Tensor, n_frames: int) -> torch.Tensor:
    """Linearly resample a [B, T, C] feature sequence along time to ``n_frames``.

    Wav2Vec2 produces ~50 features per second; the FLAME tracking is at the video
    frame rate (e.g. 30 fps). This resamples the audio feature timeline onto the
    target frame timeline so the two are aligned 1:1.
    """
    if features.shape[1] == n_frames:
        return features
    # F.interpolate expects [B, C, T]
    x = features.transpose(1, 2)
    x = F.interpolate(x, size=n_frames, mode="linear", align_corners=True)
    return x.transpose(1, 2)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class Audio2Flame(nn.Module):
    def __init__(
        self,
        out_dim: int = 106,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        w2v_name: str = "facebook/wav2vec2-base-960h",
        freeze_feature_extractor: bool = True,
    ):
        super().__init__()
        self.out_dim = out_dim

        self.w2v = Wav2Vec2Model.from_pretrained(w2v_name)
        # The Wav2Vec2 config disables SpecAugment-style masking at eval; we also
        # disable it during training because per-subject data is small.
        self.w2v.config.apply_spec_augment = False
        if freeze_feature_extractor:
            self.w2v.feature_extractor._freeze_parameters()

        w2v_dim = self.w2v.config.hidden_size
        self.proj = nn.Linear(w2v_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, wav: torch.Tensor, n_frames: int) -> torch.Tensor:
        """wav: [B, num_samples] float waveform at 16 kHz. Returns [B, n_frames, out_dim]."""
        feat = self.w2v(wav).last_hidden_state          # [B, T_audio, w2v_dim]
        feat = align_to_fps(feat, n_frames)             # [B, n_frames, w2v_dim]
        x = self.proj(feat)
        x = self.pos_enc(x)
        x = self.dropout(x)
        x = self.encoder(x)
        return self.head(x)                             # [B, n_frames, out_dim]

    def param_groups(self, backbone_lr: float, head_lr: float):
        """Two param groups: a small lr for the Wav2Vec2 backbone, larger for the rest."""
        backbone, rest = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (backbone if name.startswith("w2v.") else rest).append(p)
        return [
            {"params": backbone, "lr": backbone_lr},
            {"params": rest, "lr": head_lr},
        ]


class SmoothFlameLoss(nn.Module):
    """Position MSE (expr + weighted jaw) plus a light temporal velocity term.

    All terms operate in the normalized target space. ``w_vel`` is intentionally
    small: large velocity weights over-smooth and mute lip articulation.
    """

    def __init__(self, expr_dim: int = 100, w_jaw: float = 2.0, w_vel: float = 2.0):
        super().__init__()
        self.expr_dim = expr_dim
        self.w_jaw = w_jaw
        self.w_vel = w_vel
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        loss_expr = self.mse(pred[..., : self.expr_dim], target[..., : self.expr_dim])
        loss_jaw = self.mse(pred[..., self.expr_dim :], target[..., self.expr_dim :])

        pred_vel = pred[:, 1:] - pred[:, :-1]
        target_vel = target[:, 1:] - target[:, :-1]
        loss_vel = self.mse(pred_vel, target_vel)

        total = loss_expr + self.w_jaw * loss_jaw + self.w_vel * loss_vel
        components = {
            "expr": loss_expr.detach(),
            "jaw": loss_jaw.detach(),
            "vel": loss_vel.detach(),
        }
        return total, components
