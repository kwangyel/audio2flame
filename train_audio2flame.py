"""
Per-subject Audio2FLAME training.

Fine-tunes Wav2Vec2 end-to-end and regresses FLAME exp[100] + jaw[6] from raw
audio. Trains on short windows of a single subject's synchronized
(audio, FLAME-tracking) recording.

Inputs:
  --audio       a single WAV synced to frame 0 of the tracking
  --expr        expressions_x.npy  -> [T, 100]
  --jaw         jaw_x.npy          -> [T, 6]

Outputs:
  audio2flame_best.pth   best checkpoint (by temporal validation loss)
  flame_stats.npz        per-channel target mean/std (for un-normalization)

Example:
  python train_audio2flame.py --audio subject.wav --expr expressions_x.npy \
      --jaw jaw_x.npy --fps 30 --epochs 100 --batch_size 8
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import Audio2Flame, SmoothFlameLoss

SR = 16000


# ---------------------------------------------------------------------------
# Audio helpers (shared with inference)
# ---------------------------------------------------------------------------
def load_audio(path: str, target_sr: int = SR) -> torch.Tensor:
    """Load an audio file as a mono 1-D float tensor at ``target_sr``."""
    import torchaudio

    wav, sr = torchaudio.load(path)            # [C, N]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0).contiguous()         # [N]


def normalize_waveform(w: torch.Tensor) -> torch.Tensor:
    """Zero-mean / unit-variance, matching Wav2Vec2's per-input normalization."""
    return (w - w.mean()) / (w.std() + 1e-7)


def samples_for_frames(n_frames: int, fps: int, sr: int = SR) -> int:
    return int(round(n_frames * sr / fps))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class Audio2FlameWindows(Dataset):
    """Sliding windows over a single recording.

    Returns (waveform_window, normalized_target_window). All audio windows share
    a fixed sample length so they collate into a batch directly.
    """

    def __init__(self, wav, targets_norm, start_frames, win_frames, fps, sr=SR):
        self.wav = wav
        self.targets = targets_norm
        self.start_frames = start_frames
        self.win_frames = win_frames
        self.fps = fps
        self.sr = sr
        self.win_samples = samples_for_frames(win_frames, fps, sr)

    def __len__(self):
        return len(self.start_frames)

    def __getitem__(self, idx):
        f0 = self.start_frames[idx]
        f1 = f0 + self.win_frames

        a0 = samples_for_frames(f0, self.fps, self.sr)
        a1 = a0 + self.win_samples
        audio = self.wav[a0:a1]
        if audio.shape[0] < self.win_samples:        # pad the final window
            audio = F.pad(audio, (0, self.win_samples - audio.shape[0]))
        audio = normalize_waveform(audio)

        target = self.targets[f0:f1]
        return audio, target


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def build_targets(expr_path: str, jaw_path: str) -> np.ndarray:
    expr = np.load(expr_path).astype(np.float32)   # [T, 100]
    jaw = np.load(jaw_path).astype(np.float32)     # [T, 6]
    expr = expr.reshape(expr.shape[0], -1)
    jaw = jaw.reshape(jaw.shape[0], -1)
    T = min(len(expr), len(jaw))
    if len(expr) != len(jaw):
        print(f"[!] expr ({len(expr)}) and jaw ({len(jaw)}) lengths differ; truncating to {T}.")
    targets = np.concatenate([expr[:T], jaw[:T]], axis=-1)
    return targets


def temporal_split(n_frames, win_frames, stride, val_frac, gap_frames):
    """Split frame timeline into non-overlapping train / val regions, then make
    window start indices within each region so no window crosses the boundary."""
    train_end = int(n_frames * (1.0 - val_frac))
    val_start = min(n_frames, train_end + gap_frames)

    train_starts = list(range(0, max(0, train_end - win_frames) + 1, stride))
    val_starts = list(range(val_start, max(val_start, n_frames - win_frames) + 1, stride))
    return train_starts, val_starts, train_end, val_start


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device, expr_dim, std):
    model.eval()
    total_loss = 0.0
    mae_expr_sum, mae_jaw_sum, n = 0.0, 0.0, 0
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device)
    for audio, target in loader:
        audio = audio.to(device)
        target = target.to(device)
        pred = model(audio, target.shape[1])
        loss, _ = criterion(pred, target)
        total_loss += loss.item()

        # Per-component MAE in original (un-normalized) units.
        diff = (pred - target).abs() * std_t
        mae_expr_sum += diff[..., :expr_dim].mean().item()
        mae_jaw_sum += diff[..., expr_dim:].mean().item()
        n += 1
    return total_loss / max(1, n), mae_expr_sum / max(1, n), mae_jaw_sum / max(1, n)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- data ---
    targets = build_targets(args.expr, args.jaw)        # [T, 106]
    expr_arr = np.load(args.expr)
    expr_dim = int(expr_arr.reshape(expr_arr.shape[0], -1).shape[1])
    out_dim = targets.shape[1]
    n_frames = targets.shape[0]
    print(f"Frames: {n_frames} | target dim: {out_dim} (expr={expr_dim}, jaw={out_dim - expr_dim})")

    wav = load_audio(args.audio, SR)
    expected = samples_for_frames(n_frames, args.fps, SR)
    if wav.shape[0] < expected:
        wav = F.pad(wav, (0, expected - wav.shape[0]))
    else:
        wav = wav[:expected]
    print(f"Audio samples: {wav.shape[0]} (~{wav.shape[0] / SR:.1f}s at {SR} Hz)")

    win_frames = int(round(args.win_sec * args.fps))
    stride = max(1, int(round(args.stride_sec * args.fps)))

    train_starts, val_starts, train_end, val_start = temporal_split(
        n_frames, win_frames, stride, args.val_frac, gap_frames=win_frames
    )
    if not train_starts:
        raise ValueError("No training windows; reduce --win_sec or provide more data.")
    print(f"Train region: [0, {train_end}) -> {len(train_starts)} windows")
    print(f"Val region:   [{val_start}, {n_frames}) -> {len(val_starts)} windows")

    # --- normalization (train frames only) ---
    train_frame_mask = np.zeros(n_frames, dtype=bool)
    train_frame_mask[:train_end] = True
    mean = targets[train_frame_mask].mean(axis=0)
    std = targets[train_frame_mask].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    mean = mean.astype(np.float32)
    np.savez(args.stats_out, mean=mean, std=std, expr_dim=expr_dim, fps=args.fps)
    print(f"Saved normalization stats -> {args.stats_out}")

    targets_norm = torch.from_numpy((targets - mean) / std).float()

    train_ds = Audio2FlameWindows(wav, targets_norm, train_starts, win_frames, args.fps)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = None
    if val_starts:
        val_ds = Audio2FlameWindows(wav, targets_norm, val_starts, win_frames, args.fps)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        )
    else:
        print("[!] No validation windows (recording too short for the split). "
              "Checkpoints will be saved on training loss.")

    # --- model / optim ---
    model = Audio2Flame(
        out_dim=out_dim, d_model=args.d_model, n_layers=args.n_layers,
        w2v_name=args.w2v_name, freeze_feature_extractor=True,
    ).to(device)
    criterion = SmoothFlameLoss(expr_dim=expr_dim, w_jaw=args.w_jaw, w_vel=args.w_vel)
    optimizer = torch.optim.AdamW(
        model.param_groups(backbone_lr=args.backbone_lr, head_lr=args.head_lr),
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = max(1, int(args.warmup_frac * total_steps))
    try:
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    except Exception:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_metric = float("inf")
    epochs_no_improve = 0

    print("\n--- Training ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{args.epochs}", leave=False)
        for audio, target in pbar:
            audio = audio.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(audio, target.shape[1])
                loss, comps = criterion(pred, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             jaw=f"{comps['jaw'].item():.4f}")

        train_loss = running / max(1, len(train_loader))

        if val_loader is not None:
            val_loss, mae_expr, mae_jaw = evaluate(
                model, val_loader, criterion, device, expr_dim, std
            )
            print(f"Epoch {epoch:03d} | train {train_loss:.5f} | val {val_loss:.5f} "
                  f"| MAE expr {mae_expr:.4f} | MAE jaw {mae_jaw:.4f}")
            current = val_loss
        else:
            print(f"Epoch {epoch:03d} | train {train_loss:.5f}")
            current = train_loss

        if current < best_metric - args.min_delta:
            best_metric = current
            epochs_no_improve = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "out_dim": out_dim,
                    "expr_dim": expr_dim,
                },
                args.ckpt_out,
            )
            print(f"  -> saved best ({best_metric:.5f}) to {args.ckpt_out}")
        else:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for "
                      f"{args.patience} epochs).")
                break

    print(f"\nDone. Best metric: {best_metric:.5f}")


def parse_args():
    p = argparse.ArgumentParser(description="Per-subject Audio2FLAME trainer")
    # data
    p.add_argument("--audio", required=True, help="WAV synced to frame 0 of tracking")
    p.add_argument("--expr", default="expressions_x.npy")
    p.add_argument("--jaw", default="jaw_x.npy")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--win_sec", type=float, default=4.0)
    p.add_argument("--stride_sec", type=float, default=1.0)
    p.add_argument("--val_frac", type=float, default=0.15)
    # model
    p.add_argument("--w2v_name", default="facebook/wav2vec2-base-960h")
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=6)
    # loss
    p.add_argument("--w_jaw", type=float, default=2.0)
    p.add_argument("--w_vel", type=float, default=2.0)
    # optim
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--backbone_lr", type=float, default=1e-5)
    p.add_argument("--head_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=15, help="0 disables early stopping")
    p.add_argument("--min_delta", type=float, default=1e-5)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    # outputs
    p.add_argument("--ckpt_out", default="audio2flame_best.pth")
    p.add_argument("--stats_out", default="flame_stats.npz")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
