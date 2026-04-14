"""
velocity.py -- Compute frame-to-frame velocity features for landmark sequences.

Velocity captures the *motion dynamics* of sign language, which is often more
discriminative than raw positions. For example, "hello" and "goodbye" may
have similar hand positions but very different motion patterns.

Velocity at frame t:  v_t = x_t - x_{t-1}
First frame velocity:  v_0 = zeros(225)

Input:
    np.ndarray of shape (T, 225) -- raw landmark coordinates.

Output:
    np.ndarray of shape (T, 225) -- velocity (frame-to-frame difference).

Usage:
    python -m data.velocity
"""

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config


def compute_velocity(landmarks: np.ndarray) -> np.ndarray:
    """
    Compute frame-to-frame velocity from a landmark sequence.

    For each frame t > 0:  v_t = x_t - x_{t-1}
    For frame 0:            v_0 = zeros(landmark_dim)

    This captures the instantaneous motion at each frame. Positive values
    mean the landmark moved in the positive direction, negative means it
    moved in the negative direction.

    Args:
        landmarks: np.ndarray of shape (T, D) where T is the number of
                   frames and D is the landmark dimension (225).

    Returns:
        np.ndarray of shape (T, D), dtype float32. The velocity sequence.
        Same shape as input.

    Raises:
        ValueError: If landmarks has fewer than 2 dimensions or T < 1.
    """
    if landmarks.ndim != 2:
        raise ValueError(
            f"Expected 2D array (T, D), got shape {landmarks.shape}"
        )

    T, D = landmarks.shape
    if T == 0:
        raise ValueError("Empty landmark sequence (T=0)")

    velocity = np.zeros_like(landmarks, dtype=np.float32)

    # v_0 = zeros (no previous frame to diff against)
    # v_t = x_t - x_{t-1} for t >= 1
    if T > 1:
        velocity[1:] = landmarks[1:] - landmarks[:-1]

    return velocity


def compute_velocity_batch(
    batch: np.ndarray,
) -> np.ndarray:
    """
    Compute velocity for a batch of landmark sequences.

    Args:
        batch: np.ndarray of shape (B, T, D) -- batch of landmark sequences.

    Returns:
        np.ndarray of shape (B, T, D) -- batch of velocity sequences.
    """
    if batch.ndim != 3:
        raise ValueError(
            f"Expected 3D array (B, T, D), got shape {batch.shape}"
        )

    velocities = np.zeros_like(batch, dtype=np.float32)
    if batch.shape[1] > 1:
        velocities[:, 1:, :] = batch[:, 1:, :] - batch[:, :-1, :]

    return velocities


# ---------------------------------------------------------------------------
# Standalone verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = Config()
    print("=" * 60)
    print("velocity.py -- verification")
    print("=" * 60)

    T = 50
    D = cfg.landmark_dim  # 225

    # Create synthetic landmarks with known motion
    landmarks = np.zeros((T, D), dtype=np.float32)
    for t in range(T):
        landmarks[t] = t * 0.01  # Linear motion

    velocity = compute_velocity(landmarks)

    print(f"  Landmarks shape: {landmarks.shape}")
    print(f"  Velocity shape:  {velocity.shape}")
    print(f"  v_0 (should be zeros): max={velocity[0].max():.6f}")
    print(f"  v_1 (should be ~0.01): mean={velocity[1].mean():.6f}")
    print(f"  v_t uniform (should be ~0.01): mean={velocity[2:].mean():.6f}")

    # Assertions
    assert velocity.shape == (T, D), f"Shape mismatch: {velocity.shape}"
    assert np.allclose(velocity[0], 0.0), "v_0 should be zeros"
    assert np.allclose(velocity[1:], 0.01, atol=1e-6), "v_t should be 0.01"

    # Test batch
    batch = np.random.randn(4, T, D).astype(np.float32)
    batch_vel = compute_velocity_batch(batch)
    assert batch_vel.shape == (4, T, D), f"Batch shape mismatch: {batch_vel.shape}"
    assert np.allclose(batch_vel[:, 0, :], 0.0), "Batch v_0 should be zeros"

    print("=" * 60)
    print("[PASS] velocity.py OK")
