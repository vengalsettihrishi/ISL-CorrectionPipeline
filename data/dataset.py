"""
dataset.py -- PyTorch Dataset for the iSign ISL recognition pipeline.

This module provides:

1. ISignDataset: A PyTorch Dataset yielding
   (features, labels, feature_length, label_length) tuples, compatible
   with CTC loss.

2. collate_fn: Pads batches to the same length for DataLoader.
   Filters out invalid samples (missing landmarks, empty labels)
   to guarantee valid CTC input/target pairs.

3. Stratified train/val/test split by video_id: keeps all segments of a
   video in the same split to prevent data leakage.

4. create_dataloaders: End-to-end function to build DataLoaders from
   the iSign CSV + pose directory.

Usage:
    python -m data.dataset
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config
from data.landmark_loader import load_landmarks
from data.feature_builder import (
    build_features,
    compute_norm_stats,
    save_norm_stats,
    load_norm_stats,
)
from data.velocity import compute_velocity
from data.label_encoder import LabelEncoder, parse_isign_csv
from data.augmentation import augment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class ISignDataset(Dataset):
    """
    PyTorch Dataset for iSign ISL recognition with CTC loss.

    Each sample yields:
        features:       Tensor (max_seq_length, feature_dim=450)
        labels:         Tensor (label_length,) -- integer token IDs
        feature_length: int -- actual sequence length (pre-padding)
        label_length:   int -- number of tokens in the label

    Invalid samples (missing landmarks, empty labels) are pre-filtered
    during construction to guarantee every sample is a valid CTC pair.
    """

    def __init__(
        self,
        records: List[Dict[str, str]],
        encoder: LabelEncoder,
        config: Config,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        do_augment: bool = False,
    ):
        """
        Args:
            records:     List of {'video_id': ..., 'text': ...} dicts.
            encoder:     Trained LabelEncoder for text -> integer encoding.
            config:      Pipeline configuration.
            mean:        Per-feature mean (450,) for normalization. None = skip.
            std:         Per-feature std (450,) for normalization. None = skip.
            do_augment:  Whether to apply data augmentation.
        """
        self.encoder = encoder
        self.config = config
        self.mean = mean
        self.std = std
        self.do_augment = do_augment

        # --- Pre-filter: keep only records with available landmarks
        # and non-empty labels to guarantee valid CTC pairs ---
        self.records = []
        filtered_missing = 0
        filtered_empty_label = 0

        for rec in records:
            # Check labels first (cheap)
            labels = encoder.encode(rec["text"])
            if len(labels) == 0:
                filtered_empty_label += 1
                continue

            # Check landmark file exists
            pose_path = Path(config.pose_dir) / f"{rec['video_id']}.npy"
            if not pose_path.exists():
                filtered_missing += 1
                continue

            self.records.append(rec)

        total_filtered = filtered_missing + filtered_empty_label
        if total_filtered > 0:
            logger.warning(
                f"Filtered {total_filtered}/{len(records)} invalid samples: "
                f"{filtered_missing} missing landmarks, "
                f"{filtered_empty_label} empty labels"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Load and process a single sample.

        Returns:
            features:       (max_seq_length, 450) float32 tensor.
            labels:         (label_length,) int64 tensor.
            feature_length: int -- actual frames before padding.
            label_length:   int -- number of tokens in the label.
        """
        record = self.records[idx]
        video_id = record["video_id"]
        text = record["text"]

        # --- Load landmarks ---
        landmarks = load_landmarks(video_id, self.config)
        if landmarks is None:
            # Safety net: should not happen due to pre-filtering, but
            # return a clearly invalid sample that collate_fn will drop.
            features = np.zeros(
                (self.config.max_seq_length, self.config.feature_dim),
                dtype=np.float32,
            )
            return (
                torch.from_numpy(features),
                torch.tensor([], dtype=torch.long),
                0,  # feature_length=0 signals invalid
                0,  # label_length=0 signals invalid
            )

        # --- Augmentation (training only) ---
        if self.do_augment:
            landmarks = augment(landmarks, self.config)

        # --- Build features ---
        features, feature_length = build_features(
            landmarks, self.config, mean=self.mean, std=self.std
        )

        # --- Encode labels ---
        labels = self.encoder.encode(text)

        # CTC requires: feature_length >= label_length
        # If not, truncate labels (rare edge case with very short clips)
        if feature_length < len(labels) and feature_length > 0:
            labels = labels[:feature_length]

        return (
            torch.from_numpy(features).float(),
            torch.tensor(labels, dtype=torch.long),
            feature_length,
            len(labels),
        )


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int, int]],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Collate function for DataLoader that pads label sequences.

    Filters out invalid samples (feature_length=0 or label_length=0)
    to guarantee valid CTC input/target pairs in every batch.

    Args:
        batch: List of (features, labels, feat_len, label_len) tuples.

    Returns:
        features:        (B, max_seq_length, 450) float32 tensor.
        labels:          (B, max_label_len) int64 tensor (0-padded).
        feature_lengths: (B,) int32 tensor.
        label_lengths:   (B,) int32 tensor.
        None if all samples in the batch are invalid.
    """
    # Filter out invalid samples
    valid_batch = [
        (f, l, fl, ll) for f, l, fl, ll in batch
        if fl > 0 and ll > 0
    ]

    if len(valid_batch) == 0:
        return None

    features_list, labels_list, feat_lens, label_lens = zip(*valid_batch)

    # Stack features (already same shape)
    features = torch.stack(features_list)  # (B, T, 450)

    # Pad labels to max label length in batch
    max_label_len = max(label_lens)
    padded_labels = torch.zeros(
        len(labels_list), max_label_len, dtype=torch.long
    )
    for i, (lbl, ll) in enumerate(zip(labels_list, label_lens)):
        padded_labels[i, :ll] = lbl[:ll]

    feature_lengths = torch.tensor(feat_lens, dtype=torch.int32)
    label_lengths = torch.tensor(label_lens, dtype=torch.int32)

    return features, padded_labels, feature_lengths, label_lengths


# ---------------------------------------------------------------------------
# Stratified splitting by video_id
# ---------------------------------------------------------------------------

def stratified_split_by_video_id(
    records: List[Dict[str, str]],
    config: Config,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Split records into train/val/test sets by video_id.

    All segments of the same video stay in the same split to prevent
    data leakage, as recommended by the iSign authors.

    The assignment is deterministic: each video_id is hashed to a float
    in [0, 1) and assigned to a split based on the configured ratios.

    Args:
        records: List of {'video_id': ..., 'text': ...} dicts.
        config:  Pipeline configuration with split ratios.

    Returns:
        train_records, val_records, test_records.
    """
    # Get unique video IDs
    video_ids = sorted(set(r["video_id"] for r in records))

    # Deterministic assignment via hashing
    train_ids = set()
    val_ids = set()
    test_ids = set()

    train_cutoff = config.train_ratio
    val_cutoff = config.train_ratio + config.val_ratio

    for vid in video_ids:
        # Hash video_id to a deterministic float in [0, 1)
        h = hashlib.md5(vid.encode()).hexdigest()
        bucket = int(h[:8], 16) / (16 ** 8)

        if bucket < train_cutoff:
            train_ids.add(vid)
        elif bucket < val_cutoff:
            val_ids.add(vid)
        else:
            test_ids.add(vid)

    # Split records
    train_records = [r for r in records if r["video_id"] in train_ids]
    val_records = [r for r in records if r["video_id"] in val_ids]
    test_records = [r for r in records if r["video_id"] in test_ids]

    logger.info(
        f"Split: {len(train_records)} train, "
        f"{len(val_records)} val, {len(test_records)} test "
        f"({len(train_ids)} / {len(val_ids)} / {len(test_ids)} unique videos)"
    )

    return train_records, val_records, test_records


# ---------------------------------------------------------------------------
# Normalization stats computation (training set only)
# ---------------------------------------------------------------------------

def compute_training_norm_stats(
    train_records: List[Dict[str, str]],
    config: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-feature normalization statistics from the training set.

    Args:
        train_records: Training set records.
        config:        Pipeline configuration.

    Returns:
        mean: (450,) per-feature mean.
        std:  (450,) per-feature std.
    """
    logger.info("Computing normalization statistics from training set...")
    all_features = []

    for i, record in enumerate(train_records):
        landmarks = load_landmarks(record["video_id"], config)
        if landmarks is None:
            continue

        velocity = compute_velocity(landmarks)
        features = np.concatenate([landmarks, velocity], axis=1)
        all_features.append(features)

        if (i + 1) % 1000 == 0:
            logger.info(f"  Processed {i + 1}/{len(train_records)} samples")

    if not all_features:
        logger.warning("No features found -- returning zero mean/unit std")
        return (
            np.zeros(config.feature_dim, dtype=np.float32),
            np.ones(config.feature_dim, dtype=np.float32),
        )

    mean, std = compute_norm_stats(all_features)
    save_norm_stats(mean, std, config.norm_stats_path)
    return mean, std


# ---------------------------------------------------------------------------
# End-to-end DataLoader creation
# ---------------------------------------------------------------------------

def create_dataloaders(
    config: Config,
) -> Tuple[DataLoader, DataLoader, DataLoader, LabelEncoder]:
    """
    Create train, validation, and test DataLoaders from the iSign dataset.

    Steps:
        1. Parse CSV to get records (auto-detects gloss vs english labels).
        2. Build LabelEncoder (vocabulary).
        3. Stratified split by video_id.
        4. Compute/load normalization stats (training set only).
        5. Create Datasets (with pre-filtering of invalid samples).
        6. Create DataLoaders.

    Args:
        config: Pipeline configuration.

    Returns:
        train_loader, val_loader, test_loader, encoder.
    """
    # 1. Parse CSV (returns label_type: "gloss" or "english")
    records, label_type = parse_isign_csv(config.csv_path)

    # 2. Build vocabulary
    encoder = LabelEncoder(config, label_type=label_type)
    sentences = [r["text"] for r in records]
    encoder.build_vocab(sentences)
    encoder.save()

    # 3. Split by video_id
    train_records, val_records, test_records = stratified_split_by_video_id(
        records, config
    )

    # 4. Normalization stats
    norm_stats_path = Path(config.norm_stats_path)
    if norm_stats_path.exists():
        logger.info(f"Loading normalization stats from {config.norm_stats_path}")
        mean, std = load_norm_stats(config.norm_stats_path)
    else:
        mean, std = compute_training_norm_stats(train_records, config)

    # 5. Create datasets (with pre-filtering)
    train_dataset = ISignDataset(
        train_records, encoder, config,
        mean=mean, std=std, do_augment=True,
    )
    val_dataset = ISignDataset(
        val_records, encoder, config,
        mean=mean, std=std, do_augment=False,
    )
    test_dataset = ISignDataset(
        test_records, encoder, config,
        mean=mean, std=std, do_augment=False,
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
    )

    logger.info(
        f"DataLoaders created: "
        f"train={len(train_dataset)}, val={len(val_dataset)}, "
        f"test={len(test_dataset)}, vocab={encoder.vocab_size} "
        f"(label_type={label_type})"
    )

    return train_loader, val_loader, test_loader, encoder


# ---------------------------------------------------------------------------
# Standalone verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = Config()
    print("=" * 60)
    print("dataset.py -- verification")
    print("=" * 60)

    # --- Simulate a mini-dataset without actual iSign files ---
    import tempfile, os

    # Create synthetic pose files
    tmp_dir = tempfile.mkdtemp()
    pose_dir = Path(tmp_dir) / "poses"
    pose_dir.mkdir()

    video_ids = [f"vid_{i:04d}" for i in range(20)]
    sentences = [
        "hello how are you",
        "i am fine thank you",
        "what is your name",
        "my name is test",
        "good morning everyone",
    ]

    # Create fake .npy files (skip vid_0019 to test missing-data filtering)
    for vid in video_ids[:-1]:  # 19 out of 20
        T = np.random.randint(30, 150)
        fake_landmarks = np.random.randn(T, cfg.landmark_dim).astype(np.float32)
        np.save(str(pose_dir / f"{vid}.npy"), fake_landmarks)

    # Create fake records (all 20, including the missing one)
    records = [
        {"video_id": vid, "text": sentences[i % len(sentences)]}
        for i, vid in enumerate(video_ids)
    ]

    # Build encoder
    encoder = LabelEncoder(cfg)
    encoder.build_vocab([r["text"] for r in records])
    print(f"  Vocab size: {encoder.vocab_size}")

    # Split
    train_recs, val_recs, test_recs = stratified_split_by_video_id(records, cfg)
    print(f"  Split: {len(train_recs)} train, {len(val_recs)} val, {len(test_recs)} test")

    # Override config for test
    test_cfg = Config(
        pose_dir=str(pose_dir),
        max_seq_length=50,
    )

    # Compute norm stats on training features
    all_feats = []
    for rec in train_recs:
        fp = pose_dir / f"{rec['video_id']}.npy"
        if not fp.exists():
            continue
        lm = np.load(str(fp))
        vel = compute_velocity(lm)
        all_feats.append(np.concatenate([lm, vel], axis=1))

    if all_feats:
        mean, std = compute_norm_stats(all_feats)
    else:
        mean = np.zeros(cfg.feature_dim, dtype=np.float32)
        std = np.ones(cfg.feature_dim, dtype=np.float32)

    # Create dataset (should filter out missing vid_0019)
    train_ds = ISignDataset(
        train_recs, encoder, test_cfg,
        mean=mean, std=std, do_augment=True,
    )
    print(f"  Dataset size after filtering: {len(train_ds)}")

    if len(train_ds) > 0:
        feat, lbl, feat_len, lbl_len = train_ds[0]
        print(f"  Sample features shape: {feat.shape}")
        print(f"  Sample labels shape:   {lbl.shape}")
        print(f"  Feature length:        {feat_len}")
        print(f"  Label length:          {lbl_len}")
        assert feat.shape == (test_cfg.max_seq_length, test_cfg.feature_dim)
        assert feat_len > 0, "Feature length should be > 0 after filtering"
        assert lbl_len > 0, "Label length should be > 0 after filtering"

    # Test DataLoader with collate_fn
    loader = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # 0 for testing on Windows
        collate_fn=collate_fn,
    )

    for batch in loader:
        if batch is None:
            print("  Skipped empty batch")
            continue
        batch_features, batch_labels, batch_feat_lens, batch_lbl_lens = batch
        print(f"  Batch features:    {batch_features.shape}")
        print(f"  Batch labels:      {batch_labels.shape}")
        print(f"  Batch feat_lens:   {batch_feat_lens}")
        print(f"  Batch label_lens:  {batch_lbl_lens}")
        # Verify all feat_lens > 0
        assert (batch_feat_lens > 0).all(), "All feat_lens must be > 0"
        assert (batch_lbl_lens > 0).all(), "All label_lens must be > 0"
        break

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir)

    print("=" * 60)
    print("[PASS] dataset.py OK")
