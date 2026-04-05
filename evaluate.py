"""
evaluate.py — Evaluate a trained ISL classifier.

Reports:
    - Overall accuracy
    - Macro and weighted F1 scores
    - Per-class precision, recall, F1
    - Confusion matrix saved as CSV
    - Inference latency benchmark

Usage:
    python evaluate.py --data_dir ./processed_landmarks --checkpoint ./checkpoints/best_model.pth
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import ModelConfig, TrainConfig
from dataset import ISLDataset, discover_samples
from model import ISLClassifier


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """
    Load a trained model from checkpoint.

    Returns:
        model, label_map
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct model config from saved values
    saved_config = checkpoint["model_config"]
    model_config = ModelConfig(
        input_size=saved_config["input_size"],
        hidden_size=saved_config["hidden_size"],
        num_layers=saved_config["num_layers"],
        dropout=saved_config["dropout"],
        bidirectional=saved_config["bidirectional"],
        num_classes=saved_config["num_classes"],
    )

    model = ISLClassifier(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    label_map = checkpoint["label_map"]

    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"  Val accuracy at save: {checkpoint['val_acc']:.4f}")
    print(f"  Classes: {len(label_map)}")

    return model, label_map


@torch.no_grad()
def collect_predictions(
    model: ISLClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple:
    """
    Run inference on all samples and collect predictions + labels.

    Returns:
        all_preds: np.ndarray of predicted class indices
        all_labels: np.ndarray of true class indices
        all_confidences: np.ndarray of prediction confidence scores
    """
    all_preds = []
    all_labels = []
    all_confidences = []

    for sequences, labels in loader:
        sequences = sequences.to(device)
        logits = model(sequences)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted = torch.max(probs, dim=1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_confidences.extend(confidence.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_confidences),
    )


def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    label_map: Dict[str, int],
) -> Dict:
    """
    Compute classification metrics without sklearn dependency.

    Returns dict with overall accuracy, per-class precision/recall/f1,
    and macro/weighted averages.
    """
    num_classes = len(label_map)
    idx_to_name = {v: k for k, v in label_map.items()}

    # Overall accuracy
    accuracy = (preds == labels).mean()

    # Per-class metrics
    per_class = {}
    precisions, recalls, f1s = [], [], []
    supports = []

    for cls_idx in range(num_classes):
        cls_name = idx_to_name.get(cls_idx, f"class_{cls_idx}")

        tp = ((preds == cls_idx) & (labels == cls_idx)).sum()
        fp = ((preds == cls_idx) & (labels != cls_idx)).sum()
        fn = ((preds != cls_idx) & (labels == cls_idx)).sum()
        support = (labels == cls_idx).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[cls_name] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": int(support),
        }

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    # Macro average (treat all classes equally)
    macro_f1 = np.mean(f1s)
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)

    # Weighted average (weight by class support)
    total_support = sum(supports)
    weights = [s / total_support for s in supports]
    weighted_f1 = sum(f * w for f, w in zip(f1s, weights))

    return {
        "accuracy": round(float(accuracy), 4),
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class,
    }


def build_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Build a confusion matrix. Rows = true, Cols = predicted."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(labels, preds):
        cm[true][pred] += 1
    return cm


def benchmark_latency(
    model: ISLClassifier,
    device: torch.device,
    input_shape: tuple = (1, 30, 225),
    n_runs: int = 100,
) -> Dict:
    """
    Benchmark single-sample inference latency.

    This simulates real-time inference where you process one gesture
    at a time, which is the deployment scenario.
    """
    model.eval()
    dummy = torch.randn(*input_shape).to(device)

    # Warmup
    for _ in range(10):
        _ = model(dummy)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model(dummy)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    return {
        "mean_ms": round(np.mean(times), 2),
        "std_ms": round(np.std(times), 2),
        "min_ms": round(np.min(times), 2),
        "max_ms": round(np.max(times), 2),
        "p95_ms": round(np.percentile(times, 95), 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate ISL classifier")
    parser.add_argument("--data_dir", type=str, default="./processed_landmarks")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="./eval_results.json")

    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    # Load model
    model, label_map = load_model(args.checkpoint, device)

    # Load data — use ALL data for evaluation reporting
    # (in practice you'd use a held-out test set)
    file_paths, labels, _ = discover_samples(args.data_dir)
    eval_dataset = ISLDataset(file_paths, labels, augment=False)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"\nEvaluating on {len(eval_dataset)} samples...")

    # Collect predictions
    preds, true_labels, confidences = collect_predictions(model, eval_loader, device)

    # Metrics
    metrics = compute_metrics(preds, true_labels, label_map)

    print(f"\n{'='*50}")
    print(f"Overall Accuracy:   {metrics['accuracy']:.4f}")
    print(f"Macro F1:           {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:        {metrics['weighted_f1']:.4f}")
    print(f"Macro Precision:    {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:       {metrics['macro_recall']:.4f}")
    print(f"{'='*50}")

    # Per-class results
    print(f"\n{'Class':<20} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>5}")
    print("-" * 45)
    for cls_name, cls_metrics in sorted(metrics["per_class"].items()):
        print(
            f"{cls_name:<20} "
            f"{cls_metrics['precision']:>6.3f} "
            f"{cls_metrics['recall']:>6.3f} "
            f"{cls_metrics['f1']:>6.3f} "
            f"{cls_metrics['support']:>5d}"
        )

    # Confusion matrix
    cm = build_confusion_matrix(preds, true_labels, len(label_map))
    idx_to_name = {v: k for k, v in label_map.items()}
    cm_path = Path(args.output).parent / "confusion_matrix.csv"

    with open(cm_path, "w") as f:
        header = ",".join([""] + [idx_to_name.get(i, str(i)) for i in range(len(label_map))])
        f.write(header + "\n")
        for i in range(len(label_map)):
            row = [idx_to_name.get(i, str(i))] + [str(cm[i][j]) for j in range(len(label_map))]
            f.write(",".join(row) + "\n")
    print(f"\nConfusion matrix saved to {cm_path}")

    # Latency benchmark
    print("\nBenchmarking inference latency...")
    latency = benchmark_latency(model, device)
    print(f"  Mean: {latency['mean_ms']:.2f}ms ± {latency['std_ms']:.2f}ms")
    print(f"  P95:  {latency['p95_ms']:.2f}ms")

    # Save all results
    results = {
        "metrics": metrics,
        "latency": latency,
        "mean_confidence": round(float(confidences.mean()), 4),
        "num_samples": len(eval_dataset),
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
