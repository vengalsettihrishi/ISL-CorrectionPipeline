"""
feature_builder.py — Build final feature tensors for the ISL recognition model.

Pipeline per sample:
    1. landmarks (T, 225)
    2. velocity  (T, 225) via velocity.py
    3. concatenate → (T, 450)
    4. normalize: per-feature zero-mean, unit-variance (stats from training set)
    5. pad or truncate to max_length → (max_length, 450)

The normalization statistics (mean, std per feature) are computed once on the
training set and saved to disk. At inference time they are loaded and applied
to new sequences.

Input:
    np.ndarray of shape (T, 225) — raw landmarks from landmark_loader.py.

Output:
    np.ndarray of shape (max_length, 450), dtype float32.
    int — actual (pre-padding) sequence length (needed by CTC loss).

Usage:
    python -m data.feature_builder
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config
from data.velocity import compute_velocity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_features(
    landmarks: np.ndarray,
    config: Config,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """
    Build the full 450-dim feature vector from raw landmarks.

    Steps:
        1. Compute velocity from landmarks.
        2. Concatenate [landmarks, velocity] → (T, 450).
        3. Optionally normalize with provided mean/std.
        4. Pad (with zeros) or truncate to config.max_seq_length.

    Args:
        landmarks: np.ndarray (T, 225) — raw landmark coordinates.
        config:    Pipeline configuration.
        mean:      Per-feature mean, shape (450,). None = skip normalization.
        std:       Per-feature std, shape (450,). None = skip normalization.

    Returns:
        features:        np.ndarray (max_seq_length, 450), float32.
        actual_length:   int — original sequence length before padding
                         (capped at max_seq_length). Needed by CTC loss.
    """
    # 1. Velocity
    velocity = compute_velocity(landmarks)  # (T, 225)

    # 2. Concatenate
    features = np.concatenate([landmarks, velocity], axis=1)  # (T, 450)
    assert features.shape[1] == config.feature_dim, (
        f"Expected feature_dim={config.feature_dim}, got {features.shape[1]}"
    )

    # 3. Normalize (if stats provided)
    if mean is not None and std is not None:
        features = normalize(features, mean, std, config.norm_eps)

    # 4. Pad / truncate
    features, actual_length = pad_or_truncate(features, config.max_seq_length)

    return features, actual_length


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Apply per-feature zero-mean unit-variance normalization.

    Args:
        features: (T, D) feature matrix.
        mean:     (D,) per-feature mean computed on training set.
        std:      (D,) per-feature std computed on training set.
        eps:      Small constant to prevent division by zero.

    Returns:
        (T, D) normalized features.
    """
    return (features - mean) / (std + eps)


def compute_norm_stats(
    all_features: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-feature mean and std from a list of feature sequences.

    This should be called ONLY on training set sequences to avoid data
    leakage. The statistics are then saved and reused for val/test/inference.

    Args:
        all_features: List of arrays, each of shape (T_i, D). Variable
                      lengths are fine — statistics are pooled across all
                      frames from all sequences.

    Returns:
        mean: np.ndarray (D,) — per-feature mean.
        std:  np.ndarray (D,) — per-feature standard deviation.
    """
    # Stack all frames from all sequences
    all_frames = np.concatenate(all_features, axis=0)  # (N_total, D)
    mean = all_frames.mean(axis=0).astype(np.float32)  # (D,)
    std = all_frames.std(axis=0).astype(np.float32)    # (D,)
    return mean, std


def save_norm_stats(
    mean: np.ndarray,
    std: np.ndarray,
    path: str,
) -> None:
    """
    Save normalization statistics to disk as .npz.

    Args:
        mean: (D,) per-feature mean.
        std:  (D,) per-feature std.
        path: File path for the .npz file.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=mean, std=std)
    logger.info(f"Normalization stats saved to {path}")


def load_norm_stats(
    path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load normalization statistics from disk.

    Args:
        path: Path to the .npz file.

    Returns:
        mean: (D,) per-feature mean.
        std:  (D,) per-feature std.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Normalization stats not found at {path}. "
            "Run compute_norm_stats on the training set first."
        )
    data = np.load(path)
    return data["mean"], data["std"]


# ---------------------------------------------------------------------------
# Padding / truncation
# ---------------------------------------------------------------------------

def pad_or_truncate(
    features: np.ndarray,
    max_length: int,
) -> Tuple[np.ndarray, int]:
    """
    Pad with zeros or truncate a feature sequence to a fixed length.

    Args:
        features:   (T, D) feature array.
        max_length: Target sequence length.

    Returns:
        padded:        (max_length, D) padded/truncated array.
        actual_length: min(T, max_length) — the real length before padding.
    """
    T, D = features.shape
    actual_length = min(T, max_length)

    if T >= max_length:
        # Truncate
        return features[:max_length].copy(), actual_length
    else:
        # Pad with zeros
        padded = np.zeros((max_length, D), dtype=np.float32)
        padded[:T] = features
        return padded, actual_length


# ---------------------------------------------------------------------------
# Standalone verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = Config()
    print("=" * 60)
    print("feature_builder.py — verification")
    print("=" * 60)

    # Synthetic landmarks
    T = 50
    landmarks = np.random.randn(T, cfg.landmark_dim).astype(np.float32)

    # Build without normalization
    features, actual_len = build_features(landmarks, cfg)
    print(f"  Input landmarks:    ({T}, {cfg.landmark_dim})")
    print(f"  Output features:    {features.shape}")
    print(f"  Actual length:      {actual_len}")
    assert features.shape == (cfg.max_seq_length, cfg.feature_dim)
    assert actual_len == T

    # Compute and apply normalization
    raw_features = []
    for _ in range(10):
        lm = np.random.randn(T, cfg.landmark_dim).astype(np.float32)
        vel = compute_velocity(lm)
        raw_features.append(np.concatenate([lm, vel], axis=1))

    mean, std = compute_norm_stats(raw_features)
    print(f"  Norm mean shape:    {mean.shape}")
    print(f"  Norm std shape:     {std.shape}")
    assert mean.shape == (cfg.feature_dim,)
    assert std.shape == (cfg.feature_dim,)

    # Build with normalization
    features_norm, actual_len_norm = build_features(
        landmarks, cfg, mean=mean, std=std
    )
    print(f"  Normalized feats:   {features_norm.shape}")

    # Test save/load
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    tmp_path = tmp.name
    tmp.close()
    save_norm_stats(mean, std, tmp_path)
    loaded_mean, loaded_std = load_norm_stats(tmp_path)
    os.unlink(tmp_path)

    assert np.allclose(mean, loaded_mean), "Mean mismatch after save/load"
    assert np.allclose(std, loaded_std), "Std mismatch after save/load"

    # Test truncation
    long_lm = np.random.randn(500, cfg.landmark_dim).astype(np.float32)
    feat_trunc, len_trunc = build_features(long_lm, cfg)
    assert feat_trunc.shape == (cfg.max_seq_length, cfg.feature_dim)
    assert len_trunc == cfg.max_seq_length

    print("=" * 60)
    print("[PASS] feature_builder.py OK")
