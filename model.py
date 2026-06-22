"""
Audio2FLAME model and loss.

A per-subject, non-autoregressive Audio2FLAME regressor:

    raw waveform (16 kHz)
        -> Wav2Vec2 encoder (fully frozen)
        -> linear interpolation from ~50 Hz to the video fps
        -> 2-layer BiLSTM
        -> MLP head -> FLAME params (exp[100] + eyelids[2] + jaw[6] = 108)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

EXPR_DIM = 100
EYELID_DIM = 2
JAW_DIM = 6
DEFAULT_OUT_DIM = EXPR_DIM + EYELID_DIM + JAW_DIM


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


class Audio2Flame(nn.Module):
    def __init__(
        self,
        out_dim: int = DEFAULT_OUT_DIM,
        bilstm_hidden: int = 256,
        bilstm_layers: int = 2,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
        w2v_name: str = "facebook/wav2vec2-base-960h",
    ):
        super().__init__()
        self.out_dim = out_dim

        self.w2v = Wav2Vec2Model.from_pretrained(w2v_name)
        self.w2v.config.apply_spec_augment = False
        for param in self.w2v.parameters():
            param.requires_grad = False
        self.w2v.eval()

        w2v_dim = self.w2v.config.hidden_size
        self.bilstm = nn.LSTM(
            input_size=w2v_dim,
            hidden_size=bilstm_hidden,
            num_layers=bilstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if bilstm_layers > 1 else 0.0,
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * bilstm_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, out_dim),
        )

    def forward(self, wav: torch.Tensor, n_frames: int) -> torch.Tensor:
        """wav: [B, num_samples] float waveform at 16 kHz. Returns [B, n_frames, out_dim]."""
        with torch.no_grad():
            feat = self.w2v(wav).last_hidden_state          # [B, T_audio, w2v_dim]
        feat = align_to_fps(feat, n_frames)                 # [B, n_frames, w2v_dim]
        x, _ = self.bilstm(feat)
        return self.mlp(x)                                  # [B, n_frames, out_dim]

    def train(self, mode: bool = True):
        super().train(mode)
        self.w2v.eval()
        return self


class SmoothFlameLoss(nn.Module):
    """Position MSE (expr + weighted eyelids + weighted jaw) plus a temporal velocity term.

    All terms operate in the normalized target space. ``w_vel`` is intentionally
    small: large velocity weights over-smooth and mute lip articulation.
    """

    def __init__(
        self,
        expr_dim: int = EXPR_DIM,
        eyelid_dim: int = EYELID_DIM,
        w_eyelids: float = 1.0,
        w_jaw: float = 2.0,
        w_vel: float = 2.0,
    ):
        super().__init__()
        self.expr_dim = expr_dim
        self.eyelid_dim = eyelid_dim
        self.jaw_start = expr_dim + eyelid_dim
        self.w_eyelids = w_eyelids
        self.w_jaw = w_jaw
        self.w_vel = w_vel
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        loss_expr = self.mse(pred[..., : self.expr_dim], target[..., : self.expr_dim])
        loss_eyelids = self.mse(
            pred[..., self.expr_dim : self.jaw_start],
            target[..., self.expr_dim : self.jaw_start],
        )
        loss_jaw = self.mse(pred[..., self.jaw_start :], target[..., self.jaw_start :])

        pred_vel = pred[:, 1:] - pred[:, :-1]
        target_vel = target[:, 1:] - target[:, :-1]
        loss_vel = self.mse(pred_vel, target_vel)

        total = (
            loss_expr
            + self.w_eyelids * loss_eyelids
            + self.w_jaw * loss_jaw
            + self.w_vel * loss_vel
        )
        components = {
            "expr": loss_expr.detach(),
            "eyelids": loss_eyelids.detach(),
            "jaw": loss_jaw.detach(),
            "vel": loss_vel.detach(),
        }
        return total, components
