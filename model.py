"""
Audio2FLAME model and loss.

A per-subject, non-autoregressive Audio2FLAME regressor:

    raw waveform (16 kHz)
        -> Wav2Vec2 encoder (CNN frozen, transformer fine-tuned)
        -> 2-layer BiLSTM at ~50 Hz
        -> linear interpolation from ~50 Hz to the video fps
        -> separate MLP heads -> FLAME params (exp[100] + eyelids[2] + jaw[6] = 108)
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


def _make_head(in_dim: int, out_dim: int, hidden: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


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
        self.expr_dim = EXPR_DIM
        self.eyelid_dim = EYELID_DIM
        self.jaw_dim = JAW_DIM

        self.w2v = Wav2Vec2Model.from_pretrained(w2v_name)
        self.w2v.config.apply_spec_augment = False
        self.w2v.feature_extractor._freeze_parameters()

        w2v_dim = self.w2v.config.hidden_size
        feat_dim = 2 * bilstm_hidden
        self.bilstm = nn.LSTM(
            input_size=w2v_dim,
            hidden_size=bilstm_hidden,
            num_layers=bilstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if bilstm_layers > 1 else 0.0,
        )
        self.head_exp = _make_head(feat_dim, EXPR_DIM, mlp_hidden, dropout)
        self.head_eyelids = _make_head(feat_dim, EYELID_DIM, mlp_hidden, dropout)
        self.head_jaw = _make_head(feat_dim, JAW_DIM, mlp_hidden, dropout)

    def forward(self, wav: torch.Tensor, n_frames: int) -> torch.Tensor:
        """wav: [B, num_samples] float waveform at 16 kHz. Returns [B, n_frames, out_dim]."""
        feat = self.w2v(wav).last_hidden_state          # [B, T_audio, w2v_dim]
        x, _ = self.bilstm(feat)
        x = align_to_fps(x, n_frames)                   # [B, n_frames, 2*h]
        exp = self.head_exp(x)
        eyelids = self.head_eyelids(x)
        jaw = self.head_jaw(x)
        return torch.cat([exp, eyelids, jaw], dim=-1)   # [B, n_frames, out_dim]

    def train(self, mode: bool = True):
        super().train(mode)
        self.w2v.feature_extractor.eval()
        return self

    def param_groups(self, w2v_lr: float, head_lr: float):
        """W2V transformer encoder at ``w2v_lr``; BiLSTM + heads at ``head_lr``."""
        w2v_params, head_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("w2v."):
                w2v_params.append(param)
            else:
                head_params.append(param)
        return [
            {"params": w2v_params, "lr": w2v_lr},
            {"params": head_params, "lr": head_lr},
        ]


class SmoothFlameLoss(nn.Module):
    """Position MSE (expr + weighted eyelids + weighted jaw) plus velocity and acceleration."""

    def __init__(
        self,
        expr_dim: int = EXPR_DIM,
        eyelid_dim: int = EYELID_DIM,
        w_eyelids: float = 1.0,
        w_jaw: float = 2.0,
        w_vel: float = 2.0,
        w_acc: float = 1.0,
    ):
        super().__init__()
        self.expr_dim = expr_dim
        self.eyelid_dim = eyelid_dim
        self.jaw_start = expr_dim + eyelid_dim
        self.w_eyelids = w_eyelids
        self.w_jaw = w_jaw
        self.w_vel = w_vel
        self.w_acc = w_acc
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

        loss_acc = pred.new_tensor(0.0)
        if pred.shape[1] > 2:
            pred_acc = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
            target_acc = target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]
            loss_acc = self.mse(pred_acc, target_acc)

        total = (
            loss_expr
            + self.w_eyelids * loss_eyelids
            + self.w_jaw * loss_jaw
            + self.w_vel * loss_vel
            + self.w_acc * loss_acc
        )
        components = {
            "expr": loss_expr.detach(),
            "eyelids": loss_eyelids.detach(),
            "jaw": loss_jaw.detach(),
            "vel": loss_vel.detach(),
            "acc": loss_acc.detach(),
        }
        return total, components
