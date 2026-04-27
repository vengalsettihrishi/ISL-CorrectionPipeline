"""
Debug a gloss CTC checkpoint and dataset.

This script is intentionally diagnostic, not a training script. It answers:

1. Are labels and vocab sane?
2. Are feature shapes/lengths compatible with CTC?
3. Is the model predicting blanks everywhere?
4. What does greedy decoding produce for real samples?
5. Can the model overfit one sample? If not, the setup/data/model is broken.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from gloss_structure.data import (
    CTCVocabulary,
    PoseCTCDataset,
    compute_norm_stats,
    ctc_collate,
    read_manifest,
)
from gloss_structure.metrics import greedy_decode
from gloss_structure.model import TwoHeadCTCModel


def decode_ids(vocab: CTCVocabulary, ids):
    return vocab.decode([int(x) for x in ids])


@torch.no_grad()
def inspect_predictions(model, loader, vocab, device, max_batches=1):
    model.eval()
    shown = 0
    for batch in loader:
        if batch is None:
            continue
        features, labels, feat_lens, label_lens, video_ids, texts = batch
        features = features.to(device)
        feat_lens = feat_lens.to(device)
        log_probs = model.gloss_log_probs(features)
        probs = log_probs.exp()
        decoded = greedy_decode(log_probs.cpu(), feat_lens.cpu(), vocab.blank_id)

        for b in range(features.size(0)):
            length = int(feat_lens[b].item())
            ref_ids = labels[b, : int(label_lens[b])].tolist()
            hyp_ids = decoded[b]
            blank_mean = probs[b, :length, vocab.blank_id].mean().item()
            nonblank_mean = 1.0 - blank_mean
            top_ids = probs[b, :length].mean(dim=0).topk(min(8, vocab.size)).indices.cpu().tolist()

            print("=" * 80)
            print(f"video_id:       {video_ids[b]}")
            print(f"frames:         {length}")
            print(f"label_len:      {int(label_lens[b])}")
            print(f"reference text: {texts[b]}")
            print(f"reference ids:  {ref_ids}")
            print(f"decoded text:   {decode_ids(vocab, hyp_ids)!r}")
            print(f"decoded ids:    {hyp_ids}")
            print(f"decoded len:    {len(hyp_ids)}")
            print(f"mean blank p:   {blank_mean:.4f}")
            print(f"mean nonblank:  {nonblank_mean:.4f}")
            print("top avg tokens:")
            for tid in top_ids:
                print(f"  {tid:4d} {vocab.id_to_token.get(tid, '<missing>')!r}")
            shown += 1
        max_batches -= 1
        if max_batches <= 0:
            break
    if shown == 0:
        print("No valid batches to inspect.")


def summarize_dataset(records, pose_dir):
    pose_dir = Path(pose_dir)
    frame_lengths = []
    label_lengths = []
    missing = []
    bad_shapes = []
    for rec in records:
        path = pose_dir / f"{rec['video_id']}.npy"
        if not path.exists():
            missing.append(rec["video_id"])
            continue
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] not in {225, 450}:
            bad_shapes.append((rec["video_id"], arr.shape))
            continue
        frame_lengths.append(arr.shape[0])
        label_lengths.append(len(rec["text"].split()))

    print("=" * 80)
    print("DATASET SUMMARY")
    print(f"records:       {len(records)}")
    print(f"missing poses: {len(missing)}")
    print(f"bad shapes:    {len(bad_shapes)}")
    if frame_lengths:
        print(
            "frames:        "
            f"min={min(frame_lengths)} mean={np.mean(frame_lengths):.1f} max={max(frame_lengths)}"
        )
    if label_lengths:
        print(
            "label words:   "
            f"min={min(label_lengths)} mean={np.mean(label_lengths):.1f} max={max(label_lengths)}"
        )
    if missing[:10]:
        print("missing examples:", missing[:10])
    if bad_shapes[:10]:
        print("bad shape examples:", bad_shapes[:10])


def load_or_build_vocab(records, vocab_path=None):
    if vocab_path and Path(vocab_path).exists():
        return CTCVocabulary.load(vocab_path)
    vocab = CTCVocabulary(token_mode="word")
    vocab.build([r["text"] for r in records])
    return vocab


def load_model(args, vocab, device):
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model = TwoHeadCTCModel(
            input_dim=ckpt["input_dim"],
            hidden_size=ckpt["hidden_size"],
            num_layers=ckpt["num_layers"],
            dropout=ckpt["dropout"],
            gloss_vocab_size=ckpt["gloss_vocab_size"],
            english_vocab_size=2,
            blank_id=ckpt["blank_id"],
        )
        model.encoder.load_state_dict(ckpt["encoder"])
        model.gloss_head.load_state_dict(ckpt["gloss_head"])
    else:
        model = TwoHeadCTCModel(
            input_dim=450,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            gloss_vocab_size=vocab.size,
            english_vocab_size=2,
            blank_id=vocab.blank_id,
        )
        with torch.no_grad():
            model.gloss_head.bias[vocab.blank_id].fill_(args.blank_bias_init)
    return model.to(device)


def overfit_one_sample(args, records, vocab, mean, std, device):
    print("=" * 80)
    print("OVERFIT ONE SAMPLE TEST")
    print("Goal: decoded text should eventually match the reference.")
    dataset = PoseCTCDataset(records[:1], args.pose_dir, vocab, args.max_frames, mean, std)
    if len(dataset) == 0:
        print("Could not build one-sample dataset.")
        return
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=ctc_collate)
    model = load_model(args, vocab, device)
    model.train()
    criterion = nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.overfit_lr, weight_decay=0.0)

    for step in range(1, args.overfit_steps + 1):
        batch = next(iter(loader))
        features, labels, feat_lens, label_lens, _, _ = batch
        features = features.to(device)
        labels = labels.to(device)
        feat_lens = feat_lens.to(device)
        label_lens = label_lens.to(device)
        optimizer.zero_grad()
        log_probs = model.gloss_log_probs(features)
        loss = criterion(log_probs.permute(1, 0, 2), labels, feat_lens, label_lens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.print_every == 0:
            decoded = greedy_decode(log_probs.detach().cpu(), feat_lens.cpu(), vocab.blank_id)[0]
            ref = labels[0, : int(label_lens[0])].cpu().tolist()
            blank_p = log_probs.exp()[0, : int(feat_lens[0]), vocab.blank_id].mean().item()
            print(
                f"step={step:04d} loss={loss.item():.4f} "
                f"blank_p={blank_p:.3f} pred_len={len(decoded)} ref_len={len(ref)} "
                f"pred={decode_ids(vocab, decoded)!r} ref={decode_ids(vocab, ref)!r}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pose_dir", required=True)
    parser.add_argument("--label_column", default="gloss")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--vocab", default=None)
    parser.add_argument("--norm_stats", default=None)
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--blank_bias_init", type=float, default=-1.0)
    parser.add_argument("--inspect_batches", type=int, default=2)
    parser.add_argument("--overfit_one", action="store_true")
    parser.add_argument("--overfit_steps", type=int, default=500)
    parser.add_argument("--overfit_lr", type=float, default=1e-3)
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    records = read_manifest(args.manifest, args.label_column)
    summarize_dataset(records, args.pose_dir)

    vocab = load_or_build_vocab(records, args.vocab)
    print("=" * 80)
    print("VOCAB SUMMARY")
    print(f"vocab size: {vocab.size}")
    print("first tokens:", list(vocab.token_to_id.items())[:20])

    if args.norm_stats and Path(args.norm_stats).exists():
        stats = np.load(args.norm_stats)
        mean, std = stats["mean"], stats["std"]
    else:
        mean, std = compute_norm_stats(records, args.pose_dir)

    dataset = PoseCTCDataset(records, args.pose_dir, vocab, args.max_frames, mean, std)
    loader = DataLoader(dataset, args.batch_size, shuffle=False, collate_fn=ctc_collate)
    model = load_model(args, vocab, device)

    if args.checkpoint:
        inspect_predictions(model, loader, vocab, device, args.inspect_batches)
    else:
        print("No checkpoint supplied; skipping prediction inspection.")

    if args.overfit_one:
        overfit_one_sample(args, records, vocab, mean, std, device)


if __name__ == "__main__":
    main()

