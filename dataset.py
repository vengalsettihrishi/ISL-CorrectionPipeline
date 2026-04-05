"""
dataset.py — PyTorch Dataset for preprocessed landmark sequences.

Loads .npy files produced by landmark_extractor.py and provides them
as (sequence_tensor, label) pairs for training.

Also handles train/val splitting with stratification to ensure each
class is represented proportionally in both splits.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import TrainConfig


class ISLDataset(Dataset):
    """
    Dataset of preprocessed landmark sequences.

    Each sample is a .npy file with shape (seq_length, features_per_frame).
    Labels are inferred from the directory structure: class_name/file.npy
    """

    def __init__(
        self,
        file_paths: List[Path],
        labels: List[int],
        augment: bool = False,
    ):
        """
        Args:
            file_paths: List of paths to .npy files.
            labels: Corresponding integer labels.
            augment: Whether to apply data augmentation.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sequence = np.load(str(self.file_paths[idx]))  # (seq_len, features)

        if self.augment:
            sequence = self._apply_augmentation(sequence)

        tensor = torch.from_numpy(sequence).float()
        label = self.labels[idx]

        return tensor, label

    def _apply_augmentation(self, sequence: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to a landmark sequence.

        These augmentations simulate real-world variation:
        - Gaussian noise: simulates MediaPipe detection jitter
        - Temporal scaling: simulates signing at different speeds
        - Landmark dropout: simulates occasional detection failures
        """
        seq = sequence.copy()

        # 1. Gaussian noise on landmark coordinates (50% chance)
        if np.random.random() < 0.5:
            noise = np.random.normal(0, 0.01, seq.shape).astype(np.float32)
            # Only add noise to non-zero values (preserve "not detected" signal)
            mask = seq != 0
            seq[mask] += noise[mask]

        # 2. Temporal jitter — randomly drop and duplicate frames (30% chance)
        if np.random.random() < 0.3:
            seq = self._temporal_jitter(seq)

        # 3. Random landmark dropout — zero out a few landmarks (20% chance)
        if np.random.random() < 0.2:
            n_drop = np.random.randint(1, 4)  # drop 1-3 frames' landmarks
            drop_indices = np.random.choice(seq.shape[0], n_drop, replace=False)
            # Zero out a random subset of features in those frames
            for di in drop_indices:
                feature_mask = np.random.random(seq.shape[1]) < 0.3
                seq[di][feature_mask] = 0.0

        return seq

    @staticmethod
    def _temporal_jitter(sequence: np.ndarray) -> np.ndarray:
        """Randomly duplicate or remove 1-2 frames, then resample back."""
        seq_list = list(sequence)
        n = len(seq_list)

        # Remove 1-2 random frames
        n_remove = min(np.random.randint(1, 3), n - 3)
        for _ in range(n_remove):
            idx = np.random.randint(0, len(seq_list))
            seq_list.pop(idx)

        # Duplicate 1-2 random frames
        n_dup = np.random.randint(1, 3)
        for _ in range(n_dup):
            idx = np.random.randint(0, len(seq_list))
            seq_list.insert(idx, seq_list[idx].copy())

        # Resample back to original length using linear interpolation
        modified = np.array(seq_list, dtype=np.float32)
        if len(modified) == n:
            return modified

        indices = np.linspace(0, len(modified) - 1, n)
        result = np.zeros_like(sequence)
        for i, idx in enumerate(indices):
            lower = int(np.floor(idx))
            upper = min(lower + 1, len(modified) - 1)
            frac = idx - lower
            result[i] = modified[lower] * (1 - frac) + modified[upper] * frac

        return result


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def discover_samples(data_dir: str) -> Tuple[List[Path], List[int], Dict[str, int]]:
    """
    Walk the processed data directory and collect all .npy files with labels.

    Returns:
        file_paths: list of Path objects
        labels: list of integer labels
        label_map: dict of class_name -> int
    """
    data_path = Path(data_dir)
    label_map_file = data_path / "label_map.json"

    if not label_map_file.exists():
        raise FileNotFoundError(
            f"label_map.json not found in {data_dir}. "
            "Run landmark_extractor.py first."
        )

    with open(label_map_file) as f:
        label_map = json.load(f)

    file_paths = []
    labels = []

    for class_name, class_idx in label_map.items():
        class_dir = data_path / class_name
        if not class_dir.exists():
            continue
        for npy_file in sorted(class_dir.glob("*.npy")):
            file_paths.append(npy_file)
            labels.append(class_idx)

    return file_paths, labels, label_map


def create_dataloaders(
    data_dir: str,
    config: TrainConfig,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Create train and validation DataLoaders with stratified splitting.

    Returns:
        train_loader, val_loader, label_map
    """
    file_paths, labels, label_map = discover_samples(data_dir)

    # Stratified split: ensure proportional class distribution
    np.random.seed(config.seed)
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []

    # Group by class
    class_files: Dict[int, List[int]] = {}
    for i, label in enumerate(labels):
        class_files.setdefault(label, []).append(i)

    for class_idx, indices in class_files.items():
        np.random.shuffle(indices)
        n_val = max(1, int(len(indices) * config.val_split))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        for i in train_indices:
            train_paths.append(file_paths[i])
            train_labels.append(labels[i])
        for i in val_indices:
            val_paths.append(file_paths[i])
            val_labels.append(labels[i])

    print(f"Train samples: {len(train_paths)}, Val samples: {len(val_paths)}")
    print(f"Classes: {len(label_map)}")

    train_dataset = ISLDataset(train_paths, train_labels, augment=True)
    val_dataset = ISLDataset(val_paths, val_labels, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, label_map
