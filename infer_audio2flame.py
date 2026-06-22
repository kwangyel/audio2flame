"""
Audio2FLAME inference.

Runs a trained model on a new WAV and produces driving FLAME parameters:

  - exp[100] + eyelids[2] + jaw[6] are predicted from audio (overlap-add over windows),
  - eyes[12] (gaze) are added procedurally or copied from reference,
  - shape/tex/sh and camera/opencv metadata are copied from a reference .frame.

Outputs per-frame .frame files (drop-in for the metrical-tracker format used to
drive the Gaussian avatar) plus a combined predicted_params.npz.

Example:
  python infer_audio2flame.py --audio new_speech.wav --ckpt audio2flame_best.pth \
      --stats flame_stats.npz --ref_frame /content/combined/00005.frame \
      --out_dir driven_frames
"""

import argparse
import copy
import os

import numpy as np
import torch

from data import load_frame, to_numpy
from model import Audio2Flame, EYELID_DIM, EXPR_DIM, JAW_DIM
from train_audio2flame import SR, load_audio, normalize_waveform, samples_for_frames


# ---------------------------------------------------------------------------
# Overlap-add prediction over the whole clip
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_full(model, wav, n_frames, fps, out_dim, win_frames, overlap, device):
    model.eval()
    preds = torch.zeros(n_frames, out_dim)
    weights = torch.zeros(n_frames)
    step = max(1, win_frames - overlap)

    starts = list(range(0, max(1, n_frames - win_frames + 1), step))
    if starts[-1] != n_frames - win_frames:
        starts.append(max(0, n_frames - win_frames))

    for f0 in starts:
        f1 = min(f0 + win_frames, n_frames)
        nf = f1 - f0
        a0 = samples_for_frames(f0, fps, SR)
        a1 = samples_for_frames(f1, fps, SR)
        audio = wav[a0:a1]
        if audio.shape[0] < 1:
            continue
        audio = normalize_waveform(audio).unsqueeze(0).to(device)
        out = model(audio, nf)[0].cpu()              # [nf, out_dim]

        # Hann blending so window seams cross-fade smoothly.
        if nf > 1:
            w = torch.hann_window(nf, periodic=False) + 1e-3
        else:
            w = torch.ones(nf)
        preds[f0:f1] += out * w.unsqueeze(1)
        weights[f0:f1] += w

    weights = weights.clamp(min=1e-6)
    return (preds / weights.unsqueeze(1)).numpy()


# ---------------------------------------------------------------------------
# Procedural eyes
# ---------------------------------------------------------------------------
def make_gaze_drift(n_frames, dim, amp, fps, rng, smooth_sec=0.5):
    if amp <= 0:
        return np.zeros((n_frames, dim), dtype=np.float32)
    smooth = max(1, int(round(smooth_sec * fps)))
    kernel = np.ones(smooth, dtype=np.float32) / smooth
    noise = rng.standard_normal((n_frames, dim)).astype(np.float32)
    out = np.stack(
        [np.convolve(noise[:, c], kernel, mode="same") for c in range(dim)], axis=1
    )
    scale = np.abs(out).max() + 1e-6
    return (out / scale) * amp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    stats = np.load(args.stats)
    mean = stats["mean"].astype(np.float32)
    std = stats["std"].astype(np.float32)
    expr_dim = int(stats.get("expr_dim", EXPR_DIM))
    eyelid_dim = int(stats.get("eyelid_dim", EYELID_DIM))
    jaw_dim = int(stats.get("jaw_dim", JAW_DIM))
    fps = int(args.fps if args.fps else int(stats["fps"]))
    out_dim = mean.shape[0]
    jaw_start = expr_dim + eyelid_dim

    ckpt = torch.load(args.ckpt, map_location="cpu")
    margs = ckpt.get("args", {})
    model = Audio2Flame(
        out_dim=ckpt.get("out_dim", out_dim),
        bilstm_hidden=margs.get("bilstm_hidden", args.bilstm_hidden),
        bilstm_layers=margs.get("bilstm_layers", args.bilstm_layers),
        mlp_hidden=margs.get("mlp_hidden", args.mlp_hidden),
        dropout=margs.get("dropout", args.dropout),
        w2v_name=margs.get("w2v_name", args.w2v_name),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    print(
        f"Loaded checkpoint {args.ckpt} (out_dim={out_dim}, "
        f"expr={expr_dim}, eyelids={eyelid_dim}, jaw={jaw_dim})"
    )

    wav = load_audio(args.audio, SR)
    n_frames = int(round(wav.shape[0] / SR * fps))
    print(f"Audio {wav.shape[0] / SR:.1f}s -> {n_frames} frames at {fps} fps")

    win_frames = int(round(args.win_sec * fps))
    overlap = int(round(args.overlap_sec * fps))
    pred_norm = predict_full(model, wav, n_frames, fps, out_dim, win_frames, overlap, device)
    pred = pred_norm * std + mean                      # un-normalize -> [n_frames, out_dim]

    pred_exp = pred[:, :expr_dim].astype(np.float32)
    pred_eyelids = pred[:, expr_dim:jaw_start].astype(np.float32)
    pred_jaw = pred[:, jaw_start:].astype(np.float32)

    # --- reference frame for identity/appearance + neutral eyes ---
    ref = load_frame(args.ref_frame)
    is_nested = isinstance(ref.get("flame", None), dict)
    ref_flame = ref["flame"] if is_nested else ref
    ref_eyes = to_numpy(ref_flame["eyes"]).reshape(-1).astype(np.float32)
    eyes_dim = ref_eyes.shape[0]

    # --- procedural gaze (optional) ---
    rng = np.random.default_rng(args.seed)
    gaze = make_gaze_drift(n_frames, eyes_dim, args.gaze_amp, fps, rng)

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(
        os.path.join(args.out_dir, "predicted_params.npz"),
        exp=pred_exp,
        eyelids=pred_eyelids,
        jaw=pred_jaw,
    )

    for i in range(n_frames):
        frame = copy.deepcopy(ref)
        flame = frame["flame"] if is_nested else frame

        flame["exp"] = pred_exp[i][None, :]
        flame["eyelids"] = pred_eyelids[i][None, :].astype(np.float32)
        flame["jaw"] = pred_jaw[i][None, :]

        eyes_i = ref_eyes + gaze[i]
        flame["eyes"] = eyes_i[None, :].astype(np.float32)

        if "frame_id" in frame:
            frame["frame_id"] = f"{i:05d}"
        torch.save(frame, os.path.join(args.out_dir, f"{i:05d}.frame"))

    print(f"Wrote {n_frames} .frame files + predicted_params.npz to {args.out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Audio2FLAME inference")
    p.add_argument("--audio", required=True)
    p.add_argument("--ckpt", default="audio2flame_best.pth")
    p.add_argument("--stats", default="flame_stats.npz")
    p.add_argument("--ref_frame", required=True,
                   help="A training .frame to copy shape/tex/sh/camera and neutral eyes from")
    p.add_argument("--out_dir", default="driven_frames")
    p.add_argument("--fps", type=int, default=0, help="0 = use fps stored in stats")
    p.add_argument("--win_sec", type=float, default=4.0)
    p.add_argument("--overlap_sec", type=float, default=1.0)
    p.add_argument("--gaze_amp", type=float, default=0.0,
                   help="Subtle idle gaze drift amplitude (0 disables)")
    p.add_argument("--seed", type=int, default=0)
    # fallbacks if checkpoint lacks args
    p.add_argument("--w2v_name", default="facebook/wav2vec2-base-960h")
    p.add_argument("--bilstm_hidden", type=int, default=256)
    p.add_argument("--bilstm_layers", type=int, default=2)
    p.add_argument("--mlp_hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
