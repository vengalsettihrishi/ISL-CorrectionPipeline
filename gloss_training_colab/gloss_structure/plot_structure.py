"""
Plot q_t = 1 - P(gloss_blank at frame t) for one sample.

Example:
    python -m gloss_structure.plot_structure ^
      --checkpoint checkpoints/gloss_structure_phase1/best_gloss_model.pth ^
      --norm_stats checkpoints/gloss_structure_phase1/norm_stats.npz ^
      --pose_file data_gloss/poses/sample_001.npy ^
      --out_png results/sample_001_qt.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from gloss_structure.data import load_pose_features
from gloss_structure.model import TwoHeadCTCModel


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--norm_stats", required=True)
    parser.add_argument("--pose_file", required=True)
    parser.add_argument("--out_png", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    stats = np.load(args.norm_stats)
    features, length = load_pose_features(
        Path(args.pose_file),
        ckpt["max_frames"],
        stats["mean"],
        stats["std"],
    )

    model = TwoHeadCTCModel(
        input_dim=ckpt["input_dim"],
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt["dropout"],
        gloss_vocab_size=ckpt["gloss_vocab_size"],
        english_vocab_size=2,
        blank_id=ckpt["blank_id"],
    ).to(args.device)
    model.encoder.load_state_dict(ckpt["encoder"])
    model.gloss_head.load_state_dict(ckpt["gloss_head"])
    model.eval()

    x = torch.from_numpy(features).unsqueeze(0).float().to(args.device)
    log_probs = model.gloss_log_probs(x)
    q = 1.0 - log_probs.exp()[0, :length, ckpt["blank_id"]].cpu().numpy()

    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.plot(q)
    plt.ylim(0, 1)
    plt.xlabel("frame")
    plt.ylabel("q_t = 1 - P(blank)")
    plt.title("Gloss structural signal")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=160)
    print(f"saved {args.out_png}")


if __name__ == "__main__":
    main()

