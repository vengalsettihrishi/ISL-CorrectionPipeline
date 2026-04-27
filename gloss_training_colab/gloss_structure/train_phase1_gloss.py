"""
Phase 1: train encoder + gloss CTC head on a small gloss dataset.

Example:
    python -m gloss_structure.train_phase1_gloss ^
      --manifest data_gloss/gloss.csv ^
      --pose_dir data_gloss/poses ^
      --out_dir checkpoints/gloss_structure_phase1 ^
      --epochs 60 --batch_size 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from gloss_structure.data import (
    CTCVocabulary,
    PoseCTCDataset,
    compute_norm_stats,
    ctc_collate,
    read_manifest,
    split_records,
)
from gloss_structure.metrics import error_rate, greedy_decode
from gloss_structure.model import TwoHeadCTCModel


def save_checkpoint(model, vocab, args, path):
    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "gloss_head": model.gloss_head.state_dict(),
            "model": model.state_dict(),
            "gloss_vocab_size": vocab.size,
            "input_dim": 450,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "blank_id": vocab.blank_id,
            "max_frames": args.max_frames,
        },
        path,
    )


def nonblank_activity_loss(log_probs, lengths, blank_id, target, weight):
    """Penalize blank-only collapse by encouraging some non-blank frames."""
    if weight <= 0:
        return log_probs.new_zeros(())
    nonblank = 1.0 - log_probs.exp()[..., blank_id]
    mask = torch.arange(nonblank.size(1), device=nonblank.device)[None, :] < lengths[:, None]
    if mask.sum() == 0:
        return log_probs.new_zeros(())
    mean_nonblank = nonblank[mask].mean()
    return weight * (mean_nonblank - target).square()


def run_epoch(model, loader, criterion, optimizer, device, args):
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        if batch is None:
            continue
        features, labels, feat_lens, label_lens, _, _ = batch
        features = features.to(device)
        labels = labels.to(device)
        feat_lens = feat_lens.to(device)
        label_lens = label_lens.to(device)

        optimizer.zero_grad()
        log_probs = model.gloss_log_probs(features)
        ctc_loss = criterion(log_probs.permute(1, 0, 2), labels, feat_lens, label_lens)
        activity_loss = nonblank_activity_loss(
            log_probs,
            feat_lens,
            model.blank_id,
            args.activity_target,
            args.activity_loss_weight,
        )
        loss = ctc_loss + activity_loss
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * features.size(0)
            total_items += features.size(0)
    return total_loss / max(total_items, 1)


@torch.no_grad()
def validate(model, loader, criterion, vocab, device):
    model.eval()
    total_loss = 0.0
    total_items = 0
    total_nonblank = 0.0
    total_frames = 0
    total_ref_tokens = 0
    total_hyp_tokens = 0
    refs = []
    hyps = []
    for batch in loader:
        if batch is None:
            continue
        features, labels, feat_lens, label_lens, _, _ = batch
        features = features.to(device)
        labels = labels.to(device)
        feat_lens = feat_lens.to(device)
        label_lens = label_lens.to(device)
        log_probs = model.gloss_log_probs(features)
        loss = criterion(log_probs.permute(1, 0, 2), labels, feat_lens, label_lens)
        if torch.isfinite(loss):
            total_loss += loss.item() * features.size(0)
            total_items += features.size(0)
        nonblank = 1.0 - log_probs.exp()[..., vocab.blank_id]
        mask = torch.arange(nonblank.size(1), device=device)[None, :] < feat_lens[:, None]
        total_nonblank += nonblank[mask].sum().item()
        total_frames += int(mask.sum().item())

        batch_hyps = greedy_decode(log_probs.cpu(), feat_lens.cpu(), vocab.blank_id)
        hyps.extend(batch_hyps)
        total_hyp_tokens += sum(len(h) for h in batch_hyps)
        for b in range(labels.size(0)):
            ref = labels[b, : int(label_lens[b])].cpu().tolist()
            refs.append(ref)
            total_ref_tokens += len(ref)
    return {
        "loss": total_loss / max(total_items, 1),
        "token_er": error_rate(refs, hyps),
        "mean_nonblank": total_nonblank / max(total_frames, 1),
        "pred_ref_ratio": total_hyp_tokens / max(total_ref_tokens, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pose_dir", required=True)
    parser.add_argument("--label_column", default="gloss")
    parser.add_argument("--out_dir", default="checkpoints/gloss_structure_phase1")
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--blank_bias_init",
        type=float,
        default=-2.0,
        help="Initial bias for CTC blank logit. Negative values reduce blank collapse.",
    )
    parser.add_argument(
        "--activity_loss_weight",
        type=float,
        default=0.05,
        help="Weight for non-blank activity regularizer used during gloss training.",
    )
    parser.add_argument(
        "--activity_target",
        type=float,
        default=0.25,
        help="Target average non-blank probability over real frames.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument(
        "--train_all",
        action="store_true",
        help="Use all gloss clips for Phase 1 training and monitoring.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="Save epoch_NNN.pth every N epochs. Use 0 to disable.",
    )
    parser.add_argument(
        "--monitor_metric",
        choices=["token_er", "loss"],
        default="token_er",
        help="Metric used for best_gloss_model.pth.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    records = read_manifest(args.manifest, args.label_column)
    if args.train_all:
        split = split_records(records, train_ratio=1.0, val_ratio=0.0, seed=args.seed)
        split.train = list(records)
        split.val = list(records)
        split.test = []
        print("Using all records for Phase 1 training/monitoring.")
    else:
        split = split_records(records, seed=args.seed)
    vocab = CTCVocabulary(token_mode="word")
    vocab.build([r["text"] for r in records])
    vocab.save(str(out_dir / "gloss_vocab.json"))

    mean, std = compute_norm_stats(split.train, args.pose_dir)
    np.savez(out_dir / "norm_stats.npz", mean=mean, std=std)

    train_ds = PoseCTCDataset(split.train, args.pose_dir, vocab, args.max_frames, mean, std)
    val_ds = PoseCTCDataset(split.val, args.pose_dir, vocab, args.max_frames, mean, std)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=ctc_collate)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, collate_fn=ctc_collate)

    model = TwoHeadCTCModel(
        input_dim=450,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        gloss_vocab_size=vocab.size,
        english_vocab_size=2,
    ).to(device)
    with torch.no_grad():
        model.gloss_head.bias[vocab.blank_id].fill_(args.blank_bias_init)
    criterion = nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_er = float("inf")
    best_val_loss = float("inf")
    best_metric = float("inf")
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, args)
        val_metrics = validate(model, val_loader, criterion, vocab, device)
        val_loss = val_metrics["loss"]
        val_er = val_metrics["token_er"]
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_token_er": val_er,
            "mean_nonblank": val_metrics["mean_nonblank"],
            "pred_ref_ratio": val_metrics["pred_ref_ratio"],
        })
        print(
            f"epoch {epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_token_er={val_er:.4f} "
            f"mean_nonblank={val_metrics['mean_nonblank']:.3f} "
            f"pred/ref={val_metrics['pred_ref_ratio']:.2f} "
            f"time={time.time() - start:.1f}s"
        )

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(model, vocab, args, out_dir / f"epoch_{epoch:03d}.pth")

        current_metric = val_er if args.monitor_metric == "token_er" else val_loss
        if current_metric < best_metric or (
            current_metric == best_metric and val_loss < best_val_loss
        ):
            best_metric = current_metric
            best_er = val_er
            best_val_loss = val_loss
            stale = 0
            save_checkpoint(model, vocab, args, out_dir / "best_gloss_model.pth")
            print(
                f"  saved best checkpoint, val_token_er={best_er:.4f}, "
                f"metric={best_metric:.4f}"
            )
        else:
            stale += 1
            if stale >= args.patience:
                print("early stopping")
                break

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
