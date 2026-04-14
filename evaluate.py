"""
evaluate.py -- Evaluation for ISL recognition (minGRU + CTC).

Implements:
    1. Greedy CTC decoding (argmax -> collapse -> remove blanks)
    2. Metrics:
       - Word Error Rate (WER) / Token Error Rate (TER)
       - Sequence accuracy (exact match %)
       - Per-class token recall using bag-of-counts (not set presence)
    3. Detailed evaluation report with examples

The per-class metric uses bag-of-counts: for each sample, count occurrences
of each token in reference and hypothesis, credit min(ref_count, hyp_count)
as matched. This handles repeated tokens correctly and does not inflate
accuracy by treating "token appears anywhere" as correct.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint ./checkpoints/best_model.pth
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from model.isl_model import ISLModel
from data.dataset import create_dataloaders, collate_fn
from data.label_encoder import LabelEncoder


# ---------------------------------------------------------------------------
# CTC Decoding
# ---------------------------------------------------------------------------

def greedy_ctc_decode(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    blank_id: int = 0,
) -> List[List[int]]:
    """
    Greedy CTC decoding.

    Steps:
        1. Argmax at each frame -> raw prediction sequence
        2. Collapse consecutive identical tokens
        3. Remove blank tokens (ID 0)

    Args:
        log_probs: (batch, T, vocab_size) -- log-probabilities from model.
        lengths:   (batch,) -- actual sequence lengths.
        blank_id:  CTC blank token ID.

    Returns:
        List of decoded token ID sequences (one per batch element).
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
# Metrics
# ---------------------------------------------------------------------------

def edit_distance(ref: List[int], hyp: List[int]) -> int:
    """
    Compute Levenshtein edit distance between two sequences.

    Args:
        ref: Reference (ground truth) sequence.
        hyp: Hypothesis (predicted) sequence.

    Returns:
        Integer edit distance.
    """
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
                    dp[i - 1][j],       # deletion
                    dp[i][j - 1],       # insertion
                    dp[i - 1][j - 1],   # substitution
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
        references: List of ground-truth token ID sequences.
        hypotheses: List of predicted token ID sequences.

    Returns:
        WER as a float (0.0 = perfect).
    """
    total_edits = 0
    total_ref_len = 0

    for ref, hyp in zip(references, hypotheses):
        total_edits += edit_distance(ref, hyp)
        total_ref_len += max(len(ref), 1)

    return total_edits / max(total_ref_len, 1)


def compute_sequence_accuracy(
    references: List[List[int]],
    hypotheses: List[List[int]],
) -> float:
    """
    Compute exact sequence match accuracy.

    Args:
        references: Ground-truth sequences.
        hypotheses: Predicted sequences.

    Returns:
        Fraction of sequences that match exactly (0.0 to 1.0).
    """
    correct = sum(1 for r, h in zip(references, hypotheses) if r == h)
    return correct / max(len(references), 1)


def compute_per_class_recall(
    references: List[List[int]],
    hypotheses: List[List[int]],
    encoder: LabelEncoder,
    top_k: int = 20,
) -> Dict[str, Dict]:
    """
    Compute per-class token recall for the top-K most frequent tokens.

    Uses bag-of-counts (NOT set presence) to handle repeated tokens
    correctly. For each sample:
      - Count occurrences of each token in reference and hypothesis.
      - Credit min(ref_count, hyp_count) as matched.
      - Recall = total_matched / total_ref_occurrences.

    This avoids the overcounting problem where "token appears anywhere
    in hypothesis" would be scored as correct for all reference occurrences.

    Args:
        references:  Ground-truth sequences.
        hypotheses:  Predicted sequences.
        encoder:     LabelEncoder for ID -> word mapping.
        top_k:       Number of top classes to report.

    Returns:
        Dict mapping class name -> {token_id, ref_total, matched, recall}.
    """
    # Count global token frequencies in references
    global_freq = Counter()
    for ref in references:
        global_freq.update(ref)

    # Get top-K most frequent tokens (exclude blank=0 and unk=1)
    top_tokens = [
        token_id for token_id, _ in global_freq.most_common(top_k + 2)
        if token_id > 1  # skip blank and unk
    ][:top_k]

    # Compute per-class recall using bag-of-counts
    token_stats: Dict[int, Dict] = {
        t: {"ref_total": 0, "matched": 0} for t in top_tokens
    }

    for ref, hyp in zip(references, hypotheses):
        ref_counts = Counter(ref)
        hyp_counts = Counter(hyp)

        for token_id in top_tokens:
            rc = ref_counts.get(token_id, 0)
            hc = hyp_counts.get(token_id, 0)
            if rc > 0:
                token_stats[token_id]["ref_total"] += rc
                # Credit at most ref_count matches
                token_stats[token_id]["matched"] += min(rc, hc)

    # Convert to readable format
    result = {}
    for token_id in top_tokens:
        stats = token_stats[token_id]
        word = encoder.id2word.get(token_id, f"<id:{token_id}>")
        recall = stats["matched"] / max(stats["ref_total"], 1)
        result[word] = {
            "token_id": token_id,
            "ref_total": stats["ref_total"],
            "matched": stats["matched"],
            "recall": round(recall, 4),
        }

    return result


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: ISLModel,
    loader: DataLoader,
    encoder: LabelEncoder,
    device: torch.device,
    config: Config,
) -> Dict:
    """
    Run full evaluation on a dataset.

    Args:
        model:   Trained ISL model.
        loader:  DataLoader for the evaluation set.
        encoder: LabelEncoder for decoding.
        device:  Compute device.
        config:  Pipeline configuration.

    Returns:
        Dict with all evaluation metrics and sample predictions.
    """
    model.eval()
    criterion = nn.CTCLoss(blank=config.ctc_blank_id, zero_infinity=True)

    total_loss = 0.0
    total_samples = 0
    all_refs: List[List[int]] = []
    all_hyps: List[List[int]] = []

    for batch in loader:
        # Skip empty batches (all samples filtered out)
        if batch is None:
            continue

        features, labels, feat_lengths, label_lengths = batch
        features = features.to(device)
        labels = labels.to(device)
        feat_lengths = feat_lengths.to(device)
        label_lengths = label_lengths.to(device)

        # Forward
        log_probs, output_lengths = model(features, feat_lengths)

        # CTC loss
        log_probs_ctc = log_probs.permute(1, 0, 2)  # (T, B, V)
        loss = criterion(
            log_probs_ctc, labels, output_lengths, label_lengths,
        )

        if not (torch.isnan(loss) or torch.isinf(loss)):
            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        # Decode
        hyps = greedy_ctc_decode(log_probs, output_lengths, blank_id=0)
        all_hyps.extend(hyps)

        for b in range(labels.size(0)):
            ref_len = label_lengths[b].item()
            ref = labels[b, :ref_len].tolist()
            all_refs.append(ref)

    # Compute metrics
    avg_loss = total_loss / max(total_samples, 1)
    wer = compute_wer(all_refs, all_hyps)
    ter = wer  # TER = WER at token level (same computation)
    seq_accuracy = compute_sequence_accuracy(all_refs, all_hyps)
    per_class = compute_per_class_recall(all_refs, all_hyps, encoder, top_k=20)

    # Sample predictions for inspection
    samples = []
    for i in range(min(10, len(all_refs))):
        ref_text = encoder.decode(all_refs[i])
        hyp_text = encoder.decode(all_hyps[i])
        samples.append({
            "reference": ref_text,
            "hypothesis": hyp_text,
            "ref_ids": all_refs[i],
            "hyp_ids": all_hyps[i],
            "correct": all_refs[i] == all_hyps[i],
        })

    results = {
        "loss": round(avg_loss, 4),
        "wer": round(wer, 4),
        "ter": round(ter, 4),
        "sequence_accuracy": round(seq_accuracy, 4),
        "total_samples": len(all_refs),
        "label_type": encoder.label_type,
        "per_class_recall": per_class,
        "samples": samples,
    }

    return results


def print_evaluation_report(results: Dict) -> None:
    """Print a formatted evaluation report (ASCII-safe)."""
    print()
    print("=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)
    print(f"  Label type:          {results.get('label_type', 'unknown')}")
    print(f"  Total samples:       {results['total_samples']}")
    print(f"  CTC Loss:            {results['loss']:.4f}")
    print(f"  Word Error Rate:     {results['wer']*100:.2f}%")
    print(f"  Token Error Rate:    {results.get('ter', results.get('ger', 0))*100:.2f}%")
    print(f"  Sequence Accuracy:   {results['sequence_accuracy']*100:.2f}%")

    if results.get("label_type") == "english":
        print()
        print("  NOTE: Labels are English words, NOT ISL glosses.")
        print("  This is English-word CTC, not gloss recognition.")
        print("  Grammar correction stage would be redundant.")

    # Per-class recall
    per_class = results.get("per_class_recall", {})
    if per_class:
        print()
        print(f"  {'Class':<20} {'Ref Total':>10} {'Matched':>8} {'Recall':>8}")
        print(f"  {'-'*48}")
        for cls_name, stats in sorted(
            per_class.items(), key=lambda x: -x[1]["recall"]
        ):
            print(
                f"  {cls_name:<20} {stats['ref_total']:>10} "
                f"{stats['matched']:>8} {stats['recall']*100:>7.1f}%"
            )

    # Sample predictions
    samples = results.get("samples", [])
    if samples:
        print()
        print("  Sample Predictions:")
        print(f"  {'-'*48}")
        for i, s in enumerate(samples[:5]):
            mark = "OK" if s["correct"] else "XX"
            print(f"  [{mark}] Ref: {s['reference']}")
            print(f"        Hyp: {s['hypothesis']}")
            print()

    print("=" * 70)


# ---------------------------------------------------------------------------
# Load model from checkpoint
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str,
    config: Config,
    device: torch.device,
) -> Tuple[ISLModel, Dict]:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        config:          Pipeline configuration.
        device:          Compute device.

    Returns:
        model:      Loaded ISLModel.
        checkpoint: Raw checkpoint dict.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_cfg = checkpoint.get("config", {})
    vocab_size = checkpoint.get("vocab_size", 50)

    model = ISLModel(
        input_dim=model_cfg.get("input_dim", config.feature_dim),
        hidden_size=model_cfg.get("hidden_size", config.hidden_size),
        vocab_size=vocab_size,
        num_layers=model_cfg.get("num_layers", config.num_gru_layers),
        dropout=model_cfg.get("dropout", config.dropout),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded from {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")
    return model, checkpoint


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ISL recognition model"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="./checkpoints/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["val", "test"],
        help="Which split to evaluate on",
    )
    parser.add_argument(
        "--output", type=str, default="./eval_results.json",
        help="Path to save evaluation results JSON",
    )
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    config = Config()
    if args.device:
        config.device_preference = args.device
    device = config.device

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, encoder = create_dataloaders(config)
    loader = test_loader if args.split == "test" else val_loader
    print(f"Evaluating on {args.split} set")

    # Load model
    model, checkpoint = load_model(args.checkpoint, config, device)
    model.print_model_summary()

    # Evaluate
    results = evaluate(model, loader, encoder, device, config)
    print_evaluation_report(results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
