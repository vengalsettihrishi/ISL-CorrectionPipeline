"""
train.py — Training loop for the ISL classifier.

Features:
    - Early stopping with patience
    - Learning rate scheduling (ReduceLROnPlateau)
    - Best model checkpointing
    - Per-epoch train/val loss and accuracy logging
    - Deterministic seeding for reproducibility

Usage:
    python train.py --data_dir ./processed_landmarks --epochs 50
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import PipelineConfig, ModelConfig, TrainConfig
from dataset import create_dataloaders
from model import build_model


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device(preference: str = "auto") -> torch.device:
    """Select compute device."""
    if preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(preference)

    print(f"Using device: {device}")
    return device


# ---------------------------------------------------------------------------
# Training and validation steps
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping — prevents exploding gradients in LSTMs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * sequences.size(0)
        predicted = logits.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        logits = model(sequences)
        loss = criterion(logits, labels)

        total_loss += loss.item() * sequences.size(0)
        predicted = logits.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    data_dir: str,
    train_config: TrainConfig,
    model_config: ModelConfig,
) -> Dict:
    """
    Full training pipeline.

    Returns:
        dict with training history and best metrics.
    """
    set_seed(train_config.seed)
    device = get_device(train_config.device)

    # --- Data ---
    train_loader, val_loader, label_map = create_dataloaders(
        data_dir, train_config
    )

    # Update model config with actual number of classes
    model_config.num_classes = len(label_map)

    # --- Model ---
    model = build_model(model_config).to(device)

    # --- Loss, optimizer, scheduler ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # --- Checkpointing ---
    ckpt_dir = Path(train_config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Training ---
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }

    print(f"\n{'='*60}")
    print(f"Starting training: {train_config.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, train_config.epochs + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        elapsed = time.time() - start

        # Log
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:3d}/{train_config.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"{elapsed:.1f}s"
        )

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "model_config": {
                    "input_size": model_config.input_size,
                    "hidden_size": model_config.hidden_size,
                    "num_layers": model_config.num_layers,
                    "dropout": model_config.dropout,
                    "bidirectional": model_config.bidirectional,
                    "num_classes": model_config.num_classes,
                },
                "label_map": label_map,
            }
            torch.save(checkpoint, ckpt_dir / "best_model.pth")
            print(f"  → Saved best model (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= train_config.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={train_config.patience})")
                break

    # Save training history
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"{'='*60}")

    return {
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "history": history,
        "label_map": label_map,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train ISL sign classifier")
    parser.add_argument("--data_dir", type=str, default="./processed_landmarks")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--seq_length", type=int, default=30)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    train_config = TrainConfig(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )

    model_config = ModelConfig(
        hidden_size=args.hidden_size,
    )

    train(args.data_dir, train_config, model_config)


if __name__ == "__main__":
    main()
