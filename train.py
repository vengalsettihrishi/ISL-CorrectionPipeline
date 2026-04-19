"""
train.py — Training loop for ISL recognition (minGRU + CTC).

Features:
    - CTC loss with zero_infinity=True for numerical stability
    - AdamW optimizer with weight decay
    - ReduceLROnPlateau scheduler
    - Gradient clipping (max_norm=1.0) — essential for CTC
    - Mixed-precision training (torch.cuda.amp) on GPU
    - Early stopping with patience=15, tracking validation WER
    - Best model checkpointing by validation WER
    - Reproducible seeding
    - Detailed per-epoch logging

Usage:
    python train.py
    python train.py --epochs 50 --batch_size 16 --lr 0.001
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import Config
from model.isl_model import ISLModel
from data.dataset import create_dataloaders, collate_fn
from data.label_encoder import LabelEncoder


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# CTC decoding (greedy)
# ---------------------------------------------------------------------------

def greedy_ctc_decode(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    blank_id: int = 0,
) -> List[List[int]]:
    """
    Greedy CTC decoding: argmax → collapse → remove blanks.

    Args:
        log_probs: (batch, T, vocab_size) log-probabilities.
        lengths:   (batch,) actual sequence lengths.
        blank_id:  CTC blank token ID.

    Returns:
        List of decoded token ID sequences.
    """
    predictions = log_probs.argmax(dim=-1)  # (B, T)
    decoded = []

    for b in range(predictions.size(0)):
        seq_len = lengths[b].item()
        raw = predictions[b, :seq_len].tolist()

        # Collapse consecutive duplicates
        collapsed = []
        prev = None
        for token in raw:
            if token != prev:
                collapsed.append(token)
            prev = token

        # Remove blanks
        filtered = [t for t in collapsed if t != blank_id]
        decoded.append(filtered)

    return decoded


# ---------------------------------------------------------------------------
# WER computation
# ---------------------------------------------------------------------------

def edit_distance(ref: List[int], hyp: List[int]) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    return dp[n][m]


def compute_wer(
    references: List[List[int]],
    hypotheses: List[List[int]],
) -> float:
    """
    Compute Word Error Rate (WER).

    WER = sum(edit_distances) / sum(reference_lengths)

    Args:
        references: List of reference token ID sequences.
        hypotheses: List of hypothesis token ID sequences.

    Returns:
        WER as a float (0.0 = perfect, 1.0 = 100% error).
    """
    total_edits = 0
    total_ref_len = 0

    for ref, hyp in zip(references, hypotheses):
        total_edits += edit_distance(ref, hyp)
        total_ref_len += max(len(ref), 1)  # avoid division by zero

    return total_edits / max(total_ref_len, 1)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    config: Config,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Returns:
        Average total/CTC/auxiliary losses and reliability statistics.
    """
    model.train()
    total_loss = 0.0
    total_ctc_loss = 0.0
    total_mask_loss = 0.0
    total_smooth_loss = 0.0
    total_pi = 0.0
    total_samples = 0

    for batch in loader:
        # Skip empty batches (all samples filtered out by collate_fn)
        if batch is None:
            continue
        features, labels, feat_lengths, label_lengths = batch
        features = features.to(device)
        labels = labels.to(device)
        feat_lengths = feat_lengths.to(device)
        label_lengths = label_lengths.to(device)

        optimizer.zero_grad()

        # Mixed precision forward
        use_amp = config.use_mixed_precision and device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=use_amp):
            log_probs, output_lengths, aux = model(
                features,
                feat_lengths,
                return_aux=True,
            )

            # CTC requires (T, B, V) -- transpose
            log_probs_ctc = log_probs.permute(1, 0, 2)  # (T, B, V)

            # CTC loss
            ctc_loss = criterion(
                log_probs_ctc,
                labels,
                output_lengths,
                label_lengths,
            )
            mask_loss = config.tup_lambda * aux["tup"]["activity_loss"]
            smooth_loss = config.tup_smooth_lambda * aux["tup"]["smooth_loss"]
            loss = ctc_loss + mask_loss + smooth_loss

        # Skip batch if loss is invalid
        if torch.isnan(loss) or torch.isinf(loss):
            continue

        # Backward
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip_max_norm
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip_max_norm
            )
            optimizer.step()

        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        total_ctc_loss += ctc_loss.item() * batch_size
        total_mask_loss += mask_loss.item() * batch_size
        total_smooth_loss += smooth_loss.item() * batch_size
        total_pi += aux["tup"]["pi"].mean().item() * batch_size
        total_samples += batch_size

    denom = max(total_samples, 1)
    return {
        "loss": total_loss / denom,
        "ctc_loss": total_ctc_loss / denom,
        "mask_loss": total_mask_loss / denom,
        "smooth_loss": total_smooth_loss / denom,
        "mean_pi": total_pi / denom,
    }


# ---------------------------------------------------------------------------
# Validation step
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Config,
) -> Tuple[Dict[str, float], float]:
    """
    Validate model.

    Returns:
        avg_metrics: Average total / auxiliary losses.
        wer:      Word Error Rate.
    """
    model.eval()
    total_loss = 0.0
    total_ctc_loss = 0.0
    total_mask_loss = 0.0
    total_smooth_loss = 0.0
    total_pi = 0.0
    total_samples = 0
    all_refs = []
    all_hyps = []

    for batch in loader:
        if batch is None:
            continue
        features, labels, feat_lengths, label_lengths = batch
        features = features.to(device)
        labels = labels.to(device)
        feat_lengths = feat_lengths.to(device)
        label_lengths = label_lengths.to(device)

        log_probs, output_lengths, aux = model(
            features,
            feat_lengths,
            return_aux=True,
        )

        # CTC loss
        log_probs_ctc = log_probs.permute(1, 0, 2)
        ctc_loss = criterion(
            log_probs_ctc, labels, output_lengths, label_lengths,
        )
        mask_loss = config.tup_lambda * aux["tup"]["activity_loss"]
        smooth_loss = config.tup_smooth_lambda * aux["tup"]["smooth_loss"]
        loss = ctc_loss + mask_loss + smooth_loss

        if not (torch.isnan(loss) or torch.isinf(loss)):
            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_ctc_loss += ctc_loss.item() * batch_size
            total_mask_loss += mask_loss.item() * batch_size
            total_smooth_loss += smooth_loss.item() * batch_size
            total_pi += aux["tup"]["pi"].mean().item() * batch_size
            total_samples += batch_size

        # Greedy decode for WER
        hyps = greedy_ctc_decode(log_probs, output_lengths, blank_id=0)
        all_hyps.extend(hyps)

        # Extract reference sequences (remove padding)
        for b in range(labels.size(0)):
            ref_len = label_lengths[b].item()
            ref = labels[b, :ref_len].tolist()
            all_refs.append(ref)

    denom = max(total_samples, 1)
    avg_loss = {
        "loss": total_loss / denom,
        "ctc_loss": total_ctc_loss / denom,
        "mask_loss": total_mask_loss / denom,
        "smooth_loss": total_smooth_loss / denom,
        "mean_pi": total_pi / denom,
    }
    wer = compute_wer(all_refs, all_hyps)

    return avg_loss, wer


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: Config) -> Dict:
    """
    Full training pipeline.

    Returns:
        Dict with training history and best metrics.
    """
    set_seed(config.seed)
    device = config.device
    print(f"Using device: {device}")

    # --- Data ---
    print("\nLoading data...")
    train_loader, val_loader, test_loader, encoder = create_dataloaders(config)
    vocab_size = encoder.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    # --- Model ---
    model = ISLModel.from_config(config, vocab_size).to(device)
    model.print_model_summary()

    # Verify constraints
    assert model.count_parameters() < 500_000, \
        f"Model too large: {model.count_parameters():,} params (limit: 500K)"
    assert model.model_size_mb() < 2.0, \
        f"Model too large: {model.model_size_mb():.3f} MB (limit: 2.0 MB)"

    # --- Loss, optimizer, scheduler ---
    criterion = nn.CTCLoss(blank=config.ctc_blank_id, zero_infinity=True)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=(config.use_mixed_precision and device.type == "cuda"),
    )

    # --- Checkpointing ---
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    best_val_wer = float("inf")
    best_val_loss = float("inf")
    patience_counter = 0
    history = {
        "train_loss": [], "train_ctc_loss": [], "train_mask_loss": [],
        "train_smooth_loss": [], "train_mean_pi": [],
        "val_loss": [], "val_ctc_loss": [], "val_mask_loss": [],
        "val_smooth_loss": [], "val_mean_pi": [],
        "val_wer": [], "lr": [],
    }

    print(f"\n{'='*80}")
    print(f"Starting training: {config.epochs} epochs, batch_size={config.batch_size}")
    print(f"{'='*80}\n")

    for epoch in range(1, config.epochs + 1):
        start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, scaler, config,
        )

        # Validate
        val_metrics, val_wer = validate(
            model, val_loader, criterion, device, config,
        )

        # Scheduler
        scheduler.step(val_wer)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start

        # Log
        history["train_loss"].append(train_metrics["loss"])
        history["train_ctc_loss"].append(train_metrics["ctc_loss"])
        history["train_mask_loss"].append(train_metrics["mask_loss"])
        history["train_smooth_loss"].append(train_metrics["smooth_loss"])
        history["train_mean_pi"].append(train_metrics["mean_pi"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_ctc_loss"].append(val_metrics["ctc_loss"])
        history["val_mask_loss"].append(val_metrics["mask_loss"])
        history["val_smooth_loss"].append(val_metrics["smooth_loss"])
        history["val_mean_pi"].append(val_metrics["mean_pi"])
        history["val_wer"].append(val_wer)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch:3d}/{config.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"(ctc={train_metrics['ctc_loss']:.4f}, "
            f"mask={train_metrics['mask_loss']:.4f}, "
            f"pi={train_metrics['mean_pi']:.3f}) | "
            f"Val Loss: {val_metrics['loss']:.4f} "
            f"(ctc={val_metrics['ctc_loss']:.4f}, "
            f"mask={val_metrics['mask_loss']:.4f}, "
            f"pi={val_metrics['mean_pi']:.3f}) | "
            f"Val WER: {val_wer*100:.2f}% | "
            f"LR: {current_lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        # Checkpointing (by WER)
        if val_wer < best_val_wer:
            best_val_wer = val_wer
            best_val_loss = val_metrics["loss"]
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_metrics["loss"],
                "val_wer": val_wer,
                "vocab_size": vocab_size,
                "config": {
                    "input_dim": config.feature_dim,
                    "hidden_size": config.hidden_size,
                    "num_layers": config.num_gru_layers,
                    "dropout": config.dropout,
                    "blank_id": config.ctc_blank_id,
                    "enable_velocity_temperature": config.enable_velocity_temperature,
                    "velocity_temperature_init": config.velocity_temperature_init,
                    "enable_tup": config.enable_tup,
                    "tup_blank_bias": config.tup_blank_bias,
                    "tup_temperature": config.tup_temperature,
                    "tup_hard_mask": config.tup_hard_mask,
                    "tup_threshold": config.tup_threshold,
                },
            }
            torch.save(checkpoint, ckpt_dir / "best_model.pth")
            print(f"  -> Saved best model (WER: {val_wer*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(patience={config.patience})"
                )
                break

    # Save training history
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save vocabulary alongside checkpoint
    encoder.save(str(ckpt_dir / "vocab.json"))

    print(f"\n{'='*80}")
    print(f"Training complete.")
    print(f"Best validation WER:  {best_val_wer*100:.2f}%")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*80}")

    return {
        "best_val_wer": best_val_wer,
        "best_val_loss": best_val_loss,
        "history": history,
        "vocab_size": vocab_size,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train ISL recognition model (minGRU + CTC)"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable mixed precision training")

    args = parser.parse_args()

    config = Config()

    # Override config with CLI arguments
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.hidden_size is not None:
        config.hidden_size = args.hidden_size
    if args.num_layers is not None:
        config.num_gru_layers = args.num_layers
    if args.dropout is not None:
        config.dropout = args.dropout
    if args.device is not None:
        config.device_preference = args.device
    if args.seed is not None:
        config.seed = args.seed
    if args.no_amp:
        config.use_mixed_precision = False

    train(config)


if __name__ == "__main__":
    main()
