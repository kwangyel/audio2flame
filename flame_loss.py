"""Metrical-compatible FLAME lip vertex loss for Audio2FLAME training."""

from __future__ import annotations

import os
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from data import load_frame, to_numpy
from model import EXPR_DIM, EYELID_DIM, JAW_DIM

EYES_DIM = 12


def _discover_flame_lmk_path(metrical_repo: str, flame_dir: str) -> str:
    candidates = [
        os.path.join(flame_dir, "FLAME_lmk_embedding.npz"),
        os.path.join(flame_dir, "landmark_embedding.npy"),
        os.path.join(metrical_repo, "data", "landmark_embedding.npy"),
        os.path.join(metrical_repo, "data", "FLAME2020", "landmark_embedding.npy"),
        os.path.join(metrical_repo, "flame", "data", "landmark_embedding.npy"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Could not find FLAME landmark embedding. Pass --flame_lmk explicitly. "
        f"Tried: {candidates}"
    )


def load_lip_indices(flame_dir: str) -> np.ndarray:
    masks_path = os.path.join(flame_dir, "FLAME_masks.pkl")
    if not os.path.isfile(masks_path):
        raise FileNotFoundError(f"FLAME_masks.pkl not found in {flame_dir}")
    with open(masks_path, "rb") as f:
        masks = pickle.load(f, encoding="latin1")
    lips = masks["lips"] if isinstance(masks, dict) else masks.lips
    return np.asarray(lips, dtype=np.int64)


def load_shape_from_frame(ref_frame: str) -> np.ndarray:
    frame = load_frame(ref_frame)
    flame = frame["flame"] if isinstance(frame.get("flame"), dict) else frame
    shape = to_numpy(flame["shape"]).reshape(-1).astype(np.float32)
    if shape.shape[0] != 300:
        raise ValueError(f"Expected shape dim 300 in {ref_frame}, got {shape.shape[0]}")
    return shape


class FlameLipLoss(nn.Module):
    """MSE on metrical FLAME lip vertices between predicted and GT motion."""

    def __init__(
        self,
        metrical_repo: str,
        flame_dir: str,
        ref_frame: str,
        flame_lmk: str = "",
        expr_dim: int = EXPR_DIM,
        eyelid_dim: int = EYELID_DIM,
        jaw_dim: int = JAW_DIM,
    ):
        super().__init__()
        self.expr_dim = expr_dim
        self.eyelid_dim = eyelid_dim
        self.jaw_dim = jaw_dim
        self.jaw_start = expr_dim + eyelid_dim

        metrical_repo = os.path.abspath(metrical_repo)
        flame_dir = os.path.abspath(flame_dir)
        if metrical_repo not in sys.path:
            sys.path.insert(0, metrical_repo)

        lmk_path = flame_lmk or _discover_flame_lmk_path(metrical_repo, flame_dir)
        geom_path = os.path.join(flame_dir, "generic_model.pkl")
        if not os.path.isfile(geom_path):
            raise FileNotFoundError(f"generic_model.pkl not found in {flame_dir}")

        lip_idx = load_lip_indices(flame_dir)
        self.register_buffer("lip_idx", torch.from_numpy(lip_idx).long(), persistent=False)

        shape = load_shape_from_frame(ref_frame)
        self.register_buffer("shape", torch.from_numpy(shape).float().unsqueeze(0), persistent=False)

        cwd = os.getcwd()
        try:
            os.chdir(metrical_repo)
            from flame.FLAME import FLAME

            config = SimpleNamespace(
                flame_geom_path=geom_path,
                flame_lmk_path=lmk_path,
                num_shape_params=300,
                num_exp_params=expr_dim,
            )
            self.flame = FLAME(config)
        finally:
            os.chdir(cwd)

        for param in self.flame.parameters():
            param.requires_grad = False
        self.flame.eval()

        self._metrical_repo = metrical_repo
        self.mse = nn.MSELoss()

    def _identity_cameras(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        return torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)

    def _denorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean

    def _split_params(self, motion: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        motion = self._denorm(motion, mean, std)
        b, t, _ = motion.shape
        flat = motion.reshape(b * t, -1)
        exp = flat[:, : self.expr_dim]
        eyelids = flat[:, self.expr_dim : self.jaw_start]
        jaw = flat[:, self.jaw_start :]
        return exp, eyelids, jaw, b, t

    def _vertices(
        self,
        exp: torch.Tensor,
        eyelids: torch.Tensor,
        jaw: torch.Tensor,
        eyes: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        n = exp.shape[0]
        shape = self.shape.to(device=device, dtype=dtype).expand(n, -1)
        cameras = self._identity_cameras(n, device, dtype)

        cwd = os.getcwd()
        try:
            os.chdir(self._metrical_repo)
            verts, _, _ = self.flame(
                shape_params=shape,
                cameras=cameras,
                expression_params=exp,
                jaw_pose_params=jaw,
                eye_pose_params=eyes,
                eyelid_params=eyelids,
            )
        finally:
            os.chdir(cwd)
        return verts

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        eyes_gt: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """pred/target: normalized [B, T, D]. eyes_gt: [B, T, 12]. mean/std: [D]."""
        device = pred.device
        dtype = pred.dtype
        mean = mean.to(device=device, dtype=dtype)
        std = std.to(device=device, dtype=dtype)

        pred_exp, pred_eyelids, pred_jaw, b, t = self._split_params(pred, mean, std)
        tgt_exp, tgt_eyelids, tgt_jaw, _, _ = self._split_params(target, mean, std)
        eyes = eyes_gt.reshape(b * t, EYES_DIM).to(device=device, dtype=dtype)

        with torch.no_grad():
            verts_gt = self._vertices(tgt_exp, tgt_eyelids, tgt_jaw, eyes, device, torch.float32)

        verts_pred = self._vertices(
            pred_exp.float(), pred_eyelids.float(), pred_jaw.float(), eyes.float(),
            device, torch.float32,
        )
        lip_idx = self.lip_idx.to(device)
        loss = self.mse(verts_pred[:, lip_idx], verts_gt[:, lip_idx])
        return loss.to(dtype)
