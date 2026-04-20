"""
config.py -- Central configuration for the iSign ISL recognition pipeline.

All hyperparameters, paths, feature dimensions, normalization settings,
augmentation probabilities, and device selection live here.

Usage:
    from config import Config
    cfg = Config()
    print(cfg.feature_dim)   # 450
    print(cfg.device)        # auto-detected torch.device
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


def _auto_device() -> torch.device:
    """Auto-detect the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class Config:
    """Single source of truth for every tunable knob in the pipeline."""

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    data_dir: str = "./data_iSign"
    """Root directory for raw/processed iSign data."""

    csv_path: str = "./data_iSign/iSign_v1.1.csv"
    """Path to the iSign CSV file with video IDs and English translations."""

    pose_dir: str = "./data_iSign/poses"
    """Directory containing pre-extracted MediaPipe .npy pose files."""

    video_dir: str = "./data_iSign/videos"
    """Directory containing raw video files (fallback if poses unavailable)."""

    checkpoint_dir: str = "./checkpoints"
    """Directory to save model checkpoints."""

    vocab_path: str = "./data_iSign/vocab.json"
    """Path to save/load the vocabulary JSON file."""

    norm_stats_path: str = "./data_iSign/norm_stats.npz"
    """Path to save/load per-feature normalization statistics (mean, std)."""

    # ------------------------------------------------------------------
    # Feature specification
    # ------------------------------------------------------------------
    hand_landmarks: int = 21
    """Number of landmarks per hand from MediaPipe Hands."""

    pose_landmarks: int = 33
    """Number of landmarks from MediaPipe Pose."""

    coords_per_landmark: int = 3
    """Coordinates per landmark: x, y, z."""

    @property
    def left_hand_dim(self) -> int:
        """Dimension of left hand features: 21 × 3 = 63."""
        return self.hand_landmarks * self.coords_per_landmark

    @property
    def right_hand_dim(self) -> int:
        """Dimension of right hand features: 21 × 3 = 63."""
        return self.hand_landmarks * self.coords_per_landmark

    @property
    def pose_dim(self) -> int:
        """Dimension of pose features: 33 × 3 = 99."""
        return self.pose_landmarks * self.coords_per_landmark

    @property
    def landmark_dim(self) -> int:
        """Total landmark dimension per frame: 63 + 63 + 99 = 225."""
        return self.left_hand_dim + self.right_hand_dim + self.pose_dim

    @property
    def feature_dim(self) -> int:
        """Final feature dimension per frame: [landmarks, velocity] = 450."""
        return self.landmark_dim * 2  # landmarks + velocity

    # ------------------------------------------------------------------
    # Sequence settings
    # ------------------------------------------------------------------
    max_seq_length: int = 300
    """Maximum sequence length in frames. Pad or truncate to this."""

    # ------------------------------------------------------------------
    # MediaPipe extraction settings (fallback when poses unavailable)
    # ------------------------------------------------------------------
    max_num_hands: int = 2
    min_hand_detection_confidence: float = 0.5
    min_hand_tracking_confidence: float = 0.5
    min_pose_detection_confidence: float = 0.5
    min_pose_tracking_confidence: float = 0.5

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    normalize: bool = True
    """Whether to apply per-feature zero-mean unit-variance normalization."""

    norm_eps: float = 1e-8
    """Epsilon to prevent division by zero during normalization."""

    # ------------------------------------------------------------------
    # Augmentation probabilities and settings
    # ------------------------------------------------------------------
    augment: bool = True
    """Whether to apply data augmentation during training."""

    aug_noise_prob: float = 0.5
    """Probability of adding Gaussian noise to landmarks."""

    aug_noise_std: float = 0.005
    """Standard deviation for Gaussian noise augmentation."""

    aug_temporal_jitter_prob: float = 0.3
    """Probability of applying temporal jitter."""

    aug_temporal_jitter_frames: int = 3
    """Maximum number of frames to randomly drop/duplicate."""

    aug_speed_perturb_prob: float = 0.3
    """Probability of applying speed perturbation."""

    aug_speed_min: float = 0.8
    """Minimum speed factor for speed perturbation."""

    aug_speed_max: float = 1.2
    """Maximum speed factor for speed perturbation."""

    aug_landmark_dropout_prob: float = 0.3
    """Probability of applying random landmark dropout."""

    aug_landmark_dropout_min: float = 0.1
    """Minimum fraction of landmarks to zero out per frame."""

    aug_landmark_dropout_max: float = 0.2
    """Maximum fraction of landmarks to zero out per frame."""

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 15
    seed: int = 42

    # ------------------------------------------------------------------
    # Data splits
    # ------------------------------------------------------------------
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # ------------------------------------------------------------------
    # Model architecture (minGRU)
    # ------------------------------------------------------------------
    hidden_size: int = 128
    """Hidden dimension for minGRU layers and projection."""

    num_gru_layers: int = 3
    """Number of stacked minGRU layers."""

    dropout: float = 0.1
    """Dropout probability between layers."""

    grad_clip_max_norm: float = 1.0
    """Max gradient norm for clipping (essential for CTC stability)."""

    use_mixed_precision: bool = True
    """Enable torch.cuda.amp mixed-precision training on GPU."""

    enable_velocity_temperature: bool = False
    """Enable motion-conditioned gate temperature inside minGRU."""

    velocity_temperature_init: float = 0.5
    """Initial value for the learnable motion temperature scale."""

    enable_tup: bool = False
    """Enable Temporal Uncertainty Propagation (TUP) masking."""

    tup_lambda: float = 0.01
    """Weight for the TUP activity regularizer."""

    tup_smooth_lambda: float = 0.01
    """Weight for temporal smoothness regularization on frame reliability."""

    tup_target_activity: float = 0.6
    """Target mean frame reliability for TUP regularization."""

    tup_temperature: float = 0.67
    """Relaxation temperature for Binary Concrete / ST masking."""

    tup_hard_mask: bool = True
    """Use straight-through hard masks for TUP during training/inference."""

    tup_threshold: float = 0.5
    """Inference threshold for converting frame reliability to hard masks."""

    tup_blank_bias: float = 4.0
    """Bias added to the blank logit for uncertain frames."""

    sentence_accept_threshold: float = 0.6
    """Sentence confidence threshold above which fallback is skipped."""

    span_uncertainty_threshold: float = 0.45
    """Span uncertainty threshold that marks a region as fallback-eligible."""

    word_accept_threshold: float = 0.6
    """Confidence threshold required to accept word-level fallback output."""

    spell_accept_threshold: float = 0.55
    """Confidence threshold required to accept fingerspelling fallback output."""

    min_fallback_frames: int = 3
    """Minimum span length before the fallback controller will route it."""

    # ------------------------------------------------------------------
    # CTC
    # ------------------------------------------------------------------
    ctc_blank_id: int = 0
    """Token ID reserved for the CTC blank symbol."""

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------
    num_workers: int = 2
    pin_memory: bool = True

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device_preference: str = "auto"
    """Set to 'auto', 'cpu', 'cuda', or 'mps'."""

    @property
    def device(self) -> torch.device:
        """Resolved torch.device based on preference."""
        if self.device_preference == "auto":
            return _auto_device()
        return torch.device(self.device_preference)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def ensure_dirs(self) -> None:
        """Create all required directories if they don't exist."""
        for d in [self.data_dir, self.pose_dir, self.video_dir,
                  self.checkpoint_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        assert self.max_seq_length > 0, "max_seq_length must be positive"
        assert 0 < self.train_ratio + self.val_ratio + self.test_ratio <= 1.0, \
            "Split ratios must sum to ≤ 1.0"
        assert self.landmark_dim == 225, \
            f"Expected landmark_dim=225, got {self.landmark_dim}"
        assert self.feature_dim == 450, \
            f"Expected feature_dim=450, got {self.feature_dim}"


# ---------------------------------------------------------------------------
# Standalone verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = Config()
    print("=" * 60)
    print("Config verification")
    print("=" * 60)
    print(f"  landmark_dim   = {cfg.landmark_dim}")
    print(f"  feature_dim    = {cfg.feature_dim}")
    print(f"  max_seq_length = {cfg.max_seq_length}")
    print(f"  batch_size     = {cfg.batch_size}")
    print(f"  device         = {cfg.device}")
    print(f"  vocab_path     = {cfg.vocab_path}")
    print(f"  norm_stats     = {cfg.norm_stats_path}")
    print(f"  ctc_blank_id   = {cfg.ctc_blank_id}")
    print("=" * 60)
    print("[PASS] config.py OK")
