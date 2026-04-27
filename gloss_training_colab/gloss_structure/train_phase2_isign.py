"""
Phase 2: train English CTC on iSign with frozen gloss structural guidance.

Example:
    python -m gloss_structure.train_phase2_isign ^
      --manifest data_iSign/iSign_v1.1.csv ^
      --pose_dir data_iSign/poses ^
      --gloss_checkpoint checkpoints/gloss_structure_phase1/best_gloss_model.pth ^
      --gloss_norm_stats checkpoints/gloss_structure_phase1/norm_stats.npz ^
      --out_dir checkpoints/gloss_structure_phase2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from gloss_structure.data import (
    CTCVocabulary,
    PoseCTCDataset,
    ctc_collate,
    read_manifest,
    split_records,
)
from gloss_structure.metrics import error_rate, greedy_decode
from gloss_structure.model import TwoHeadCTCModel, freeze, unfreeze


def structural_loss(english_log_probs, gloss_log_probs, lengths, blank_id=0):
    r = 1.0 - english_log_probs.exp()[..., blank_id]
    q = (1.0 - gloss_log_probs.exp()[..., blank_id]).detach()
    mask = torch.arange(r.size(1), device=r.device)[None, :] < lengths[:, None]
    return F.binary_cross_entropy(r[mask], q[mask])


def set_stage(model, stage):
    freeze(model.gloss_head)
    if stage == "head":
        freeze(model.encoder)
        unfreeze(model.english_head)
    elif stage == "full":
        unfreeze(model.encoder)
        unfreeze(model.english_head)
    else:
        raise ValueError(stage)


def train_epoch(model, loader, criterion, optimizer, device, lam):
    model.train()
    totals = {"loss": 0.0, "ctc": 0.0, "struct": 0.0, "items": 0}
    for batch in loader:
        if batch is None:
            continue
        features, labels, feat_lens, label_lens, _, _ = batch
        features = features.to(device)
        labels = labels.to(device)
        feat_lens = feat_lens.to(device)
        label_lens = label_lens.to(device)

        optimizer.zero_grad()
        encoded = model.encode(features)
        english_lp = model.english_log_probs(features, encoded)
        gloss_lp = model.gloss_log_probs(features, encoded)
        ctc = criterion(english_lp.permute(1, 0, 2), labels, feat_lens, label_lens)
        struct = structural_loss(english_lp, gloss_lp, feat_lens, model.blank_id)
        loss = ctc + lam * struct
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            n = features.size(0)
            totals["loss"] += loss.item() * n
            totals["ctc"] += ctc.item() * n
            totals["struct"] += struct.item() * n
            totals["items"] += n
    denom = max(totals["items"], 1)
    return {k: totals[k] / denom for k in ["loss", "ctc", "struct"]}


@torch.no_grad()
def validate(model, loader, criterion, vocab, device, lam):
    model.eval()
    totals = {"loss": 0.0, "ctc": 0.0, "struct": 0.0, "items": 0}
    refs, hyps = [], []
    for batch in loader:
        if batch is None:
            continue
        features, labels, feat_lens, label_lens, _, _ = batch
        features = features.to(device)
        labels = labels.to(device)
        feat_lens = feat_lens.to(device)
        label_lens = label_lens.to(device)
        encoded = model.encode(features)
        english_lp = model.english_log_probs(features, encoded)
        gloss_lp = model.gloss_log_probs(features, encoded)
        ctc = criterion(english_lp.permute(1, 0, 2), labels, feat_lens, label_lens)
        struct = structural_loss(english_lp, gloss_lp, feat_lens, model.blank_id)
        loss = ctc + lam * struct
        if torch.isfinite(loss):
            n = features.size(0)
            totals["loss"] += loss.item() * n
            totals["ctc"] += ctc.item() * n
            totals["struct"] += struct.item() * n
            totals["items"] += n
        hyps.extend(greedy_decode(english_lp.cpu(), feat_lens.cpu(), vocab.blank_id))
        for b in range(labels.size(0)):
            refs.append(labels[b, : int(label_lens[b])].cpu().tolist())
    denom = max(totals["items"], 1)
    metrics = {k: totals[k] / denom for k in ["loss", "ctc", "struct"]}
    metrics["token_er"] = error_rate(refs, hyps)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data_iSign/iSign_v1.1.csv")
    parser.add_argument("--pose_dir", default="data_iSign/poses")
    parser.add_argument("--label_column", default="text")
    parser.add_argument("--gloss_checkpoint", required=True)
    parser.add_argument("--gloss_norm_stats", required=True)
    parser.add_argument("--out_dir", default="checkpoints/gloss_structure_phase2")
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--english_token_mode", choices=["char", "word"], default="char")
    parser.add_argument("--lambda_struct", type=float, default=0.05)
    parser.add_argument("--head_epochs", type=int, default=5)
    parser.add_argument("--full_epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.gloss_checkpoint, map_location="cpu")
    stats = np.load(args.gloss_norm_stats)
    mean, std = stats["mean"], stats["std"]

    records = read_manifest(args.manifest, args.label_column)
    split = split_records(records, seed=args.seed)
    vocab = CTCVocabulary(args.english_token_mode)
    vocab.build([r["text"] for r in split.train])
    vocab.save(str(out_dir / "english_vocab.json"))

    train_ds = PoseCTCDataset(split.train, args.pose_dir, vocab, args.max_frames, mean, std)
    val_ds = PoseCTCDataset(split.val, args.pose_dir, vocab, args.max_frames, mean, std)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=ctc_collate)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, collate_fn=ctc_collate)

    model = TwoHeadCTCModel(
        input_dim=ckpt["input_dim"],
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt["dropout"],
        gloss_vocab_size=ckpt["gloss_vocab_size"],
        english_vocab_size=vocab.size,
        blank_id=ckpt["blank_id"],
    ).to(device)
    model.encoder.load_state_dict(ckpt["encoder"])
    model.gloss_head.load_state_dict(ckpt["gloss_head"])

    criterion = nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)
    history = []
    best_er = float("inf")
    stale = 0

    stages = [("head", args.head_epochs), ("full", args.full_epochs)]
    for stage, epochs in stages:
        set_stage(model, stage)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=1e-4,
        )
        for epoch in range(1, epochs + 1):
            start = time.time()
            train_m = train_epoch(
                model, train_loader, criterion, optimizer,
                device, args.lambda_struct if stage == "full" else 0.0,
            )
            val_m = validate(
                model, val_loader, criterion, vocab,
                device, args.lambda_struct if stage == "full" else 0.0,
            )
            row = {"stage": stage, "epoch": epoch, "train": train_m, "val": val_m}
            history.append(row)
            print(
                f"{stage} epoch {epoch:03d} "
                f"train_loss={train_m['loss']:.4f} "
                f"val_loss={val_m['loss']:.4f} "
                f"val_token_er={val_m['token_er']:.4f} "
                f"struct={val_m['struct']:.4f} time={time.time() - start:.1f}s"
            )
            if val_m["token_er"] < best_er:
                best_er = val_m["token_er"]
                stale = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "english_vocab_size": vocab.size,
                        "gloss_vocab_size": ckpt["gloss_vocab_size"],
                        "input_dim": ckpt["input_dim"],
                        "hidden_size": ckpt["hidden_size"],
                        "num_layers": ckpt["num_layers"],
                        "dropout": ckpt["dropout"],
                        "blank_id": ckpt["blank_id"],
                        "lambda_struct": args.lambda_struct,
                        "english_token_mode": args.english_token_mode,
                    },
                    out_dir / "best_english_struct_model.pth",
                )
                print(f"  saved best checkpoint, val_token_er={best_er:.4f}")
            elif stage == "full":
                stale += 1
                if stale >= args.patience:
                    print("early stopping")
                    break

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()

