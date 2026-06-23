"""
Per-subject Audio2FLAME training.

Regresses FLAME exp[100] + eyelids[2] + jaw[6] from raw audio using a partially
fine-tuned Wav2Vec2 encoder, 2-layer BiLSTM (before fps align), and separate MLP
heads. Optional metrical FLAME lip vertex loss.

Example:
  python train_audio2flame.py --audio subject.wav --frames_dir combined/ \
      --ref_frame combined/00000.frame \
      --metrical_repo /path/to/metrical-tracker-parallel \
      --flame_dir /path/to/metrical-tracker-parallel/data/FLAME2020 \
      --fps 30 --epochs 100
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data import (
    build_targets_from_npy,
    load_eyes_from_frames,
    load_eyes_from_npy,
    load_targets_from_frames,
)
from model import Audio2Flame, EYELID_DIM, EXPR_DIM, JAW_DIM, SmoothFlameLoss

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

    Returns (waveform_window, normalized_target_window, eyes_window).
    """

    def __init__(self, wav, targets_norm, eyes, start_frames, win_frames, fps, sr=SR):
        self.wav = wav
        self.targets = targets_norm
        self.eyes = eyes
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
        if audio.shape[0] < self.win_samples:
            audio = F.pad(audio, (0, self.win_samples - audio.shape[0]))
        audio = normalize_waveform(audio)

        target = self.targets[f0:f1]
        eyes_win = self.eyes[f0:f1]
        return audio, target, eyes_win


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def build_targets(args) -> np.ndarray:
    if args.frames_dir:
        print(f"Loading targets from {args.frames_dir}")
        return load_targets_from_frames(args.frames_dir)
    if not args.eyelids:
        raise ValueError(
            "NPY mode requires --eyelids. Use --frames_dir for Metrical .frame files."
        )
    return build_targets_from_npy(args.expr, args.jaw, args.eyelids)


def build_eyes(args, n_frames: int) -> np.ndarray:
    if args.frames_dir:
        print(f"Loading eyes from {args.frames_dir}")
        return load_eyes_from_frames(args.frames_dir)
    if args.eyes:
        return load_eyes_from_npy(args.eyes, n_frames)
    print("[!] No eyes data found; using zeros for lip-loss eye context.")
    return np.zeros((n_frames, 12), dtype=np.float32)


def temporal_split(n_frames, win_frames, stride, val_frac, gap_frames):
    """Split frame timeline into non-overlapping train / val regions."""
    train_end = int(n_frames * (1.0 - val_frac))
    val_start = min(n_frames, train_end + gap_frames)

    train_starts = list(range(0, max(0, train_end - win_frames) + 1, stride))
    val_starts = list(range(val_start, max(val_start, n_frames - win_frames) + 1, stride))
    return train_starts, val_starts, train_end, val_start


def build_lip_loss(args, device):
    if args.w_lip <= 0:
        return None
    if not args.metrical_repo or not args.flame_dir or not args.ref_frame:
        raise ValueError(
            "Lip loss requires --metrical_repo, --flame_dir, and --ref_frame "
            "(or set --w_lip 0 to disable)."
        )
    from flame_loss import FlameLipLoss

    lip_loss = FlameLipLoss(
        metrical_repo=args.metrical_repo,
        flame_dir=args.flame_dir,
        ref_frame=args.ref_frame,
        flame_lmk=args.flame_lmk,
    ).to(device)
    print("Enabled FLAME lip vertex loss.")
    return lip_loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device, expr_dim, eyelid_dim, std):
    model.eval()
    total_loss = 0.0
    mae_expr_sum, mae_eyelids_sum, mae_jaw_sum, n = 0.0, 0.0, 0.0, 0
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device)
    jaw_start = expr_dim + eyelid_dim
    for audio, target, _eyes in loader:
        audio = audio.to(device)
        target = target.to(device)
        pred = model(audio, target.shape[1])
        loss, _ = criterion(pred, target)
        total_loss += loss.item()

        diff = (pred - target).abs() * std_t
        mae_expr_sum += diff[..., :expr_dim].mean().item()
        mae_eyelids_sum += diff[..., expr_dim:jaw_start].mean().item()
        mae_jaw_sum += diff[..., jaw_start:].mean().item()
        n += 1
    return (
        total_loss / max(1, n),
        mae_expr_sum / max(1, n),
        mae_eyelids_sum / max(1, n),
        mae_jaw_sum / max(1, n),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    targets = build_targets(args)
    expr_dim = EXPR_DIM
    eyelid_dim = EYELID_DIM
    jaw_dim = JAW_DIM
    out_dim = targets.shape[1]
    n_frames = targets.shape[0]
    print(
        f"Frames: {n_frames} | target dim: {out_dim} "
        f"(expr={expr_dim}, eyelids={eyelid_dim}, jaw={jaw_dim})"
    )

    eyes = build_eyes(args, n_frames)
    if len(eyes) != n_frames:
        n = min(len(eyes), n_frames)
        print(f"[!] eyes ({len(eyes)}) and targets ({n_frames}) differ; truncating to {n}.")
        eyes = eyes[:n]
        targets = targets[:n]
        n_frames = n

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

    train_frame_mask = np.zeros(n_frames, dtype=bool)
    train_frame_mask[:train_end] = True
    mean = targets[train_frame_mask].mean(axis=0)
    std = targets[train_frame_mask].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    mean = mean.astype(np.float32)
    np.savez(
        args.stats_out,
        mean=mean,
        std=std,
        expr_dim=expr_dim,
        eyelid_dim=eyelid_dim,
        jaw_dim=jaw_dim,
        fps=args.fps,
    )
    print(f"Saved normalization stats -> {args.stats_out}")

    targets_norm = torch.from_numpy((targets - mean) / std).float()
    eyes_t = torch.from_numpy(eyes).float()
    mean_t = torch.from_numpy(mean).float().to(device)
    std_t = torch.from_numpy(std).float().to(device)

    train_ds = Audio2FlameWindows(wav, targets_norm, eyes_t, train_starts, win_frames, args.fps)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = None
    if val_starts:
        val_ds = Audio2FlameWindows(wav, targets_norm, eyes_t, val_starts, win_frames, args.fps)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        )
    else:
        print("[!] No validation windows. Checkpoints will be saved on training loss.")

    model = Audio2Flame(
        out_dim=out_dim,
        bilstm_hidden=args.bilstm_hidden,
        bilstm_layers=args.bilstm_layers,
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout,
        w2v_name=args.w2v_name,
    ).to(device)
    criterion = SmoothFlameLoss(
        expr_dim=expr_dim,
        eyelid_dim=eyelid_dim,
        w_eyelids=args.w_eyelids,
        w_jaw=args.w_jaw,
        w_vel=args.w_vel,
        w_acc=args.w_acc,
    )
    lip_loss = build_lip_loss(args, device)

    optimizer = torch.optim.AdamW(
        model.param_groups(w2v_lr=args.w2v_lr, head_lr=args.head_lr),
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
        for audio, target, eyes in pbar:
            audio = audio.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            eyes = eyes.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(audio, target.shape[1])
                loss, comps = criterion(pred, target)

            if lip_loss is not None:
                loss_lip = lip_loss(pred.float(), target.float(), eyes.float(), mean_t, std_t)
                loss = loss + args.w_lip * loss_lip
                comps["lip"] = loss_lip.detach()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += loss.item()
            postfix = {
                "loss": f"{loss.item():.4f}",
                "jaw": f"{comps['jaw'].item():.4f}",
                "acc": f"{comps['acc'].item():.4f}",
            }
            if "lip" in comps:
                postfix["lip"] = f"{comps['lip'].item():.4f}"
            pbar.set_postfix(postfix)

        train_loss = running / max(1, len(train_loader))

        if val_loader is not None:
            val_loss, mae_expr, mae_eyelids, mae_jaw = evaluate(
                model, val_loader, criterion, device, expr_dim, eyelid_dim, std,
            )
            print(
                f"Epoch {epoch:03d} | train {train_loss:.5f} | val {val_loss:.5f} "
                f"| MAE expr {mae_expr:.4f} | MAE eyelids {mae_eyelids:.4f} "
                f"| MAE jaw {mae_jaw:.4f}"
            )
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
                    "eyelid_dim": eyelid_dim,
                    "jaw_dim": jaw_dim,
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
    p.add_argument("--frames_dir", default="", help="Directory of Metrical .frame files")
    p.add_argument("--expr", default="expressions_x.npy")
    p.add_argument("--eyelids", default="", help="Eyelids npy (required if not using --frames_dir)")
    p.add_argument("--jaw", default="jaw_x.npy")
    p.add_argument("--eyes", default="", help="Eyes npy [T,12] for lip loss in npy mode")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--win_sec", type=float, default=4.0)
    p.add_argument("--stride_sec", type=float, default=1.0)
    p.add_argument("--val_frac", type=float, default=0.15)
    # model
    p.add_argument("--w2v_name", default="facebook/wav2vec2-base-960h")
    p.add_argument("--bilstm_hidden", type=int, default=256)
    p.add_argument("--bilstm_layers", type=int, default=2)
    p.add_argument("--mlp_hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    # loss
    p.add_argument("--w_eyelids", type=float, default=1.0)
    p.add_argument("--w_jaw", type=float, default=2.0)
    p.add_argument("--w_vel", type=float, default=2.0)
    p.add_argument("--w_acc", type=float, default=1.0)
    p.add_argument("--w_lip", type=float, default=1.0, help="0 disables FLAME lip vertex loss")
    p.add_argument("--metrical_repo", default="", help="Path to metrical-tracker-parallel checkout")
    p.add_argument("--flame_dir", default="", help="Path to FLAME2020/ (generic_model.pkl, masks)")
    p.add_argument("--ref_frame", default="", help="Reference .frame for subject shape (lip loss)")
    p.add_argument("--flame_lmk", default="", help="Optional FLAME landmark embedding path")
    # optim
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--w2v_lr", type=float, default=1e-5)
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
