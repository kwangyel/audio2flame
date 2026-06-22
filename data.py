"""Shared data loading for Audio2FLAME training and inference."""

import glob
import os
import pickle
import re

import numpy as np
import torch

from model import EYELID_DIM, EXPR_DIM, JAW_DIM


def load_frame(path: str):
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        with open(path, "rb") as f:
            data = pickle.load(f)
    return data


def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _flame_dict(frame_data: dict) -> dict:
    if isinstance(frame_data.get("flame"), dict):
        return frame_data["flame"]
    return frame_data


def _frame_sort_key(path: str) -> int:
    name = os.path.basename(path)
    match = re.search(r"(\d+)", name)
    if match:
        return int(match.group(1))
    return 0


def load_targets_from_frames(frames_dir: str) -> np.ndarray:
    """Load exp, eyelids, and jaw from Metrical tracker .frame files.

    Returns [T, EXPR_DIM + EYELID_DIM + JAW_DIM] float32 array.
    """
    pattern = os.path.join(frames_dir, "*.frame")
    paths = sorted(glob.glob(pattern), key=_frame_sort_key)
    if not paths:
        raise FileNotFoundError(f"No .frame files found in {frames_dir}")

    rows = []
    for path in paths:
        flame = _flame_dict(load_frame(path))
        exp = to_numpy(flame["exp"]).reshape(-1).astype(np.float32)
        eyelids = to_numpy(flame["eyelids"]).reshape(-1).astype(np.float32)
        jaw = to_numpy(flame["jaw"]).reshape(-1).astype(np.float32)
        if exp.shape[0] != EXPR_DIM:
            raise ValueError(f"{path}: expected exp dim {EXPR_DIM}, got {exp.shape[0]}")
        if eyelids.shape[0] != EYELID_DIM:
            raise ValueError(f"{path}: expected eyelids dim {EYELID_DIM}, got {eyelids.shape[0]}")
        if jaw.shape[0] != JAW_DIM:
            raise ValueError(f"{path}: expected jaw dim {JAW_DIM}, got {jaw.shape[0]}")
        rows.append(np.concatenate([exp, eyelids, jaw], axis=0))

    return np.stack(rows, axis=0)


def build_targets_from_npy(expr_path: str, jaw_path: str, eyelids_path: str) -> np.ndarray:
    """Load targets from separate npy files. Returns [T, 108] float32."""
    expr = np.load(expr_path).astype(np.float32).reshape(-1, EXPR_DIM)
    jaw = np.load(jaw_path).astype(np.float32).reshape(-1, JAW_DIM)
    eyelids = np.load(eyelids_path).astype(np.float32).reshape(-1, EYELID_DIM)

    lengths = [len(expr), len(jaw), len(eyelids)]
    T = min(lengths)
    if len(set(lengths)) != 1:
        print(
            f"[!] expr ({len(expr)}), jaw ({len(jaw)}), eyelids ({len(eyelids)}) "
            f"lengths differ; truncating to {T}."
        )
    return np.concatenate([expr[:T], eyelids[:T], jaw[:T]], axis=-1)
