"""
augmentation.py — Data augmentation transforms for ISL landmark sequences.

Augmentations simulate real-world variation to improve model robustness:

1. Gaussian noise: simulates MediaPipe detection jitter (std=0.005, 50% prob).
2. Temporal jitter: randomly drop/duplicate 1-3 frames, resample back (30% prob).
3. Speed perturbation: stretch/compress by 0.8x-1.2x (30% prob).
4. Random landmark dropout: zero out 10-20% of landmarks in random frames (30% prob).

All augmentations operate on raw landmark arrays BEFORE feature building
(velocity computation, normalization, padding). This ensures velocity
captures the augmented motion.

Input:
    np.ndarray of shape (T, 225) — raw landmarks.

Output:
    np.ndarray of shape (T', 225) — augmented landmarks (T' may differ due
    to speed perturbation; caller handles padding).

Usage:
    python -m data.augmentation
"""

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config


# ---------------------------------------------------------------------------
# Individual augmentation functions
# ---------------------------------------------------------------------------

def gaussian_noise(
    landmarks: np.ndarray,
    std: float = 0.005,
    prob: float = 0.5,
) -> np.ndarray:
    """
    Add Gaussian noise to landmark coordinates.

    Only adds noise to non-zero values (preserves "not detected" signal).
    Simulates natural jitter in MediaPipe landmark detection.

    Args:
        landmarks: (T, 225) landmark array.
        std:       Standard deviation of the Gaussian noise.
        prob:      Probability of applying this augmentation.

    Returns:
        (T, 225) augmented landmarks (or unmodified copy if not applied).
    """
    result = landmarks.copy()
    if np.random.random() >= prob:
        return result

    noise = np.random.normal(0, std, result.shape).astype(np.float32)
    # Only add noise where landmarks were actually detected
    mask = result != 0
    result[mask] += noise[mask]
    return result


def temporal_jitter(
    landmarks: np.ndarray,
    max_frames: int = 3,
    prob: float = 0.3,
) -> np.ndarray:
    """
    Randomly drop and duplicate frames, then resample back to original length.

    Simulates small temporal inconsistencies in video capture.

    Args:
        landmarks:  (T, 225) landmark array.
        max_frames: Maximum number of frames to drop/duplicate (1 to max_frames).
        prob:       Probability of applying this augmentation.

    Returns:
        (T, 225) augmented landmarks (same length as input).
    """
    if np.random.random() >= prob:
        return landmarks.copy()

    T = landmarks.shape[0]
    if T < 5:
        return landmarks.copy()

    seq_list = list(landmarks)

    # Drop random frames (but keep at least 3)
    n_drop = min(np.random.randint(1, max_frames + 1), T - 3)
    drop_indices = sorted(
        np.random.choice(len(seq_list), n_drop, replace=False),
        reverse=True,
    )
    for idx in drop_indices:
        seq_list.pop(idx)

    # Duplicate random frames
    n_dup = np.random.randint(1, max_frames + 1)
    for _ in range(n_dup):
        idx = np.random.randint(0, len(seq_list))
        seq_list.insert(idx, seq_list[idx].copy())

    # Resample back to original length via linear interpolation
    modified = np.array(seq_list, dtype=np.float32)
    if len(modified) == T:
        return modified

    return _resample(modified, T)


def speed_perturbation(
    landmarks: np.ndarray,
    speed_min: float = 0.8,
    speed_max: float = 1.2,
    prob: float = 0.3,
) -> np.ndarray:
    """
    Stretch or compress the temporal axis to simulate different signing speeds.

    A speed factor < 1.0 stretches (slower signing, more frames).
    A speed factor > 1.0 compresses (faster signing, fewer frames).

    Args:
        landmarks: (T, 225) landmark array.
        speed_min: Minimum speed factor (0.8 = 80% speed = stretched).
        speed_max: Maximum speed factor (1.2 = 120% speed = compressed).
        prob:      Probability of applying this augmentation.

    Returns:
        (T', 225) augmented landmarks where T' = int(T / speed_factor).
        Note: T' differs from T; the caller handles padding/truncation.
    """
    if np.random.random() >= prob:
        return landmarks.copy()

    T = landmarks.shape[0]
    if T < 3:
        return landmarks.copy()

    speed_factor = np.random.uniform(speed_min, speed_max)
    new_T = max(3, int(round(T / speed_factor)))

    return _resample(landmarks, new_T)


def landmark_dropout(
    landmarks: np.ndarray,
    drop_min: float = 0.1,
    drop_max: float = 0.2,
    prob: float = 0.3,
) -> np.ndarray:
    """
    Randomly zero out a fraction of landmarks in random frames.

    Simulates occasional detection failures where MediaPipe loses
    track of certain landmarks.

    Args:
        landmarks: (T, 225) landmark array.
        drop_min:  Minimum fraction of landmarks to zero out (per frame).
        drop_max:  Maximum fraction of landmarks to zero out (per frame).
        prob:      Probability of applying this augmentation.

    Returns:
        (T, 225) augmented landmarks with some values zeroed out.
    """
    result = landmarks.copy()
    if np.random.random() >= prob:
        return result

    T, D = result.shape

    # Select a random subset of frames to apply dropout to
    n_frames = max(1, int(T * np.random.uniform(0.1, 0.4)))
    frame_indices = np.random.choice(T, n_frames, replace=False)

    for fi in frame_indices:
        # Determine dropout fraction for this frame
        drop_frac = np.random.uniform(drop_min, drop_max)
        n_drop = max(1, int(D * drop_frac))
        drop_features = np.random.choice(D, n_drop, replace=False)
        result[fi, drop_features] = 0.0

    return result


# ---------------------------------------------------------------------------
# Combined augmentation pipeline
# ---------------------------------------------------------------------------

def augment(
    landmarks: np.ndarray,
    config: Config,
) -> np.ndarray:
    """
    Apply the full augmentation pipeline to a landmark sequence.

    The augmentations are applied in order:
        1. Gaussian noise
        2. Temporal jitter
        3. Speed perturbation
        4. Landmark dropout

    Each augmentation is applied independently with its own probability.
    The resulting sequence may have a different length (due to speed
    perturbation), which is handled downstream by pad_or_truncate.

    Args:
        landmarks: (T, 225) raw landmark array.
        config:    Pipeline configuration with augmentation settings.

    Returns:
        (T', 225) augmented landmarks. T' may differ from T.
    """
    if not config.augment:
        return landmarks.copy()

    result = landmarks

    # 1. Gaussian noise
    result = gaussian_noise(
        result,
        std=config.aug_noise_std,
        prob=config.aug_noise_prob,
    )

    # 2. Temporal jitter
    result = temporal_jitter(
        result,
        max_frames=config.aug_temporal_jitter_frames,
        prob=config.aug_temporal_jitter_prob,
    )

    # 3. Speed perturbation
    result = speed_perturbation(
        result,
        speed_min=config.aug_speed_min,
        speed_max=config.aug_speed_max,
        prob=config.aug_speed_perturb_prob,
    )

    # 4. Landmark dropout
    result = landmark_dropout(
        result,
        drop_min=config.aug_landmark_dropout_min,
        drop_max=config.aug_landmark_dropout_max,
        prob=config.aug_landmark_dropout_prob,
    )

    return result


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _resample(
    sequence: np.ndarray,
    target_length: int,
) -> np.ndarray:
    """
    Resample a sequence to a target length via linear interpolation.

    Args:
        sequence:      (T, D) array.
        target_length: Desired number of frames.

    Returns:
        (target_length, D) resampled array.
    """
    T, D = sequence.shape
    if T == target_length:
        return sequence.copy()

    indices = np.linspace(0, T - 1, target_length)
    resampled = np.zeros((target_length, D), dtype=np.float32)

    for i, idx in enumerate(indices):
        lower = int(np.floor(idx))
        upper = min(lower + 1, T - 1)
        frac = idx - lower
        resampled[i] = sequence[lower] * (1 - frac) + sequence[upper] * frac

    return resampled


# ---------------------------------------------------------------------------
# Standalone verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = Config()
    print("=" * 60)
    print("augmentation.py — verification")
    print("=" * 60)

    T, D = 100, cfg.landmark_dim
    landmarks = np.random.randn(T, D).astype(np.float32)

    # Test individual augmentations
    noisy = gaussian_noise(landmarks, std=0.005, prob=1.0)
    assert noisy.shape == landmarks.shape
    diff = np.abs(noisy - landmarks).max()
    print(f"  Gaussian noise max diff:  {diff:.6f}")

    jittered = temporal_jitter(landmarks, max_frames=3, prob=1.0)
    assert jittered.shape == landmarks.shape
    print(f"  Temporal jitter shape:    {jittered.shape}")

    sped = speed_perturbation(landmarks, speed_min=0.8, speed_max=1.2, prob=1.0)
    print(f"  Speed perturbation shape: {sped.shape} (was {landmarks.shape})")
    assert sped.shape[1] == D

    dropped = landmark_dropout(landmarks, drop_min=0.1, drop_max=0.2, prob=1.0)
    assert dropped.shape == landmarks.shape
    n_zeros = (dropped == 0).sum()
    print(f"  Landmark dropout zeros:   {n_zeros} / {T * D}")

    # Test full pipeline
    augmented = augment(landmarks, cfg)
    print(f"  Full pipeline output:     {augmented.shape}")
    assert augmented.shape[1] == D

    # Test with augment=False
    cfg_no_aug = Config(augment=False)
    not_augmented = augment(landmarks, cfg_no_aug)
    assert np.array_equal(not_augmented, landmarks), "Should be unchanged"
    print("  Augment=False test:       PASS")

    # Reproducibility test
    np.random.seed(42)
    aug1 = augment(landmarks, cfg)
    np.random.seed(42)
    aug2 = augment(landmarks, cfg)
    assert np.array_equal(aug1, aug2), "Same seed should give same result"
    print("  Reproducibility test:     PASS")

    print("=" * 60)
    print("[PASS] augmentation.py OK")
