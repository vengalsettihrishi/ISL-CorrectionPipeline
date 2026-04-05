"""
config.py — Central configuration for the ISL recognition pipeline.

All hyperparameters, paths, and constants live here.
Nothing is hardcoded in other modules.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class LandmarkConfig:
    """Settings for MediaPipe landmark extraction."""

    # MediaPipe detection confidence thresholds
    min_hand_detection_confidence: float = 0.5
    min_hand_tracking_confidence: float = 0.5
    min_pose_detection_confidence: float = 0.5
    min_pose_tracking_confidence: float = 0.5

    max_num_hands: int = 2

    # Frame sampling
    # Target number of frames to extract per video.
    # Videos longer or shorter than this will be resampled.
    target_seq_length: int = 30

    # Feature dimensions (calculated, do not change)
    # Per hand: 21 landmarks × 3 coords = 63
    # Two hands: 63 × 2 = 126
    # Pose: 33 landmarks × 3 coords = 99
    # Total: 126 + 99 = 225
    hand_landmarks_per_hand: int = 21
    pose_landmarks: int = 33
    coords_per_landmark: int = 3

    @property
    def features_per_frame(self) -> int:
        return (
            self.hand_landmarks_per_hand * self.coords_per_landmark * 2  # both hands
            + self.pose_landmarks * self.coords_per_landmark              # pose
        )


@dataclass
class ModelConfig:
    """Settings for the LSTM classifier."""

    input_size: int = 225        # Must match LandmarkConfig.features_per_frame
    hidden_size: int = 128       # LSTM hidden units
    num_layers: int = 2          # LSTM layers
    dropout: float = 0.3         # Dropout between LSTM layers
    bidirectional: bool = True   # Bidirectional LSTM
    num_classes: int = 50        # Number of sign classes (adjust to your dataset)


@dataclass
class TrainConfig:
    """Settings for training."""

    # Data
    data_dir: str = "./processed_landmarks"
    val_split: float = 0.2       # Fraction of data for validation
    test_split: float = 0.0      # Set >0 if you want a held-out test set

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    patience: int = 10           # Early stopping patience

    # Reproducibility
    seed: int = 42

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_best_only: bool = True

    # Device
    device: str = "auto"  # "auto", "cpu", "cuda", or "mps"


@dataclass
class PipelineConfig:
    """Top-level config combining all sub-configs."""

    landmark: LandmarkConfig = field(default_factory=LandmarkConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
