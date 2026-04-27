"""
Data utilities for gloss-guided structural regularization.

Expected manifest CSV formats:
    video_id,gloss
    sample_001,IX I EAT FOOD

or:
    video_id,english
    00094a2700f5-1,i need food

For iSign v1.1 CSVs with uid + pose_member, video_id is reconstructed as
``{uid}-{pose_member}``, matching the existing data_iSign/poses/*.npy files.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def read_manifest(path: str, label_column: str) -> List[Dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    records: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        field_lookup = {name.lower().strip(): name for name in fields}

        label_key = field_lookup.get(label_column.lower())
        if label_key is None:
            raise ValueError(
                f"Column '{label_column}' not found in {path}. "
                f"Available columns: {fields}"
            )

        video_key = field_lookup.get("video_id")
        uid_key = field_lookup.get("uid")
        pose_member_key = field_lookup.get("pose_member")
        if video_key is None and not (uid_key and pose_member_key):
            raise ValueError(
                "CSV must contain either video_id or uid + pose_member columns."
            )

        for row in reader:
            if video_key is not None:
                video_id = row[video_key].strip()
            else:
                video_id = f"{row[uid_key].strip()}-{row[pose_member_key].strip()}"
            text = row[label_key].strip()
            if video_id and text:
                records.append({"video_id": video_id, "text": text})

    return records


def tokenize_words(text: str) -> List[str]:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []


def tokenize_chars(text: str) -> List[str]:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return list(text) if text else []


class CTCVocabulary:
    """Tiny vocabulary wrapper with CTC blank fixed at ID 0."""

    def __init__(self, token_mode: str = "word"):
        if token_mode not in {"word", "char"}:
            raise ValueError("token_mode must be 'word' or 'char'")
        self.token_mode = token_mode
        self.blank_id = 0
        self.unk_id = 1
        self.token_to_id: Dict[str, int] = {"<blank>": 0, "<unk>": 1}
        self.id_to_token: Dict[int, str] = {0: "<blank>", 1: "<unk>"}

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    def tokenize(self, text: str) -> List[str]:
        if self.token_mode == "char":
            return tokenize_chars(text)
        return tokenize_words(text)

    def build(self, texts: Iterable[str], min_freq: int = 1) -> None:
        counts: Dict[str, int] = {}
        for text in texts:
            for token in self.tokenize(text):
                counts[token] = counts.get(token, 0) + 1
        for token in sorted(t for t, c in counts.items() if c >= min_freq):
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

    def encode(self, text: str) -> List[int]:
        return [self.token_to_id.get(t, self.unk_id) for t in self.tokenize(text)]

    def decode(self, ids: Sequence[int]) -> str:
        tokens = [
            self.id_to_token.get(int(i), "<unk>")
            for i in ids
            if int(i) != self.blank_id
        ]
        if self.token_mode == "char":
            return "".join(tokens)
        return " ".join(tokens)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "token_mode": self.token_mode,
            "token_to_id": self.token_to_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CTCVocabulary":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        vocab = cls(payload["token_mode"])
        vocab.token_to_id = {k: int(v) for k, v in payload["token_to_id"].items()}
        vocab.id_to_token = {v: k for k, v in vocab.token_to_id.items()}
        return vocab


def load_pose_features(
    pose_path: Path,
    max_frames: int,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    arr = np.load(str(pose_path)).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D pose array at {pose_path}, got {arr.shape}")

    if arr.shape[1] == 225:
        velocity = np.zeros_like(arr, dtype=np.float32)
        if len(arr) > 1:
            velocity[1:] = arr[1:] - arr[:-1]
        arr = np.concatenate([arr, velocity], axis=1)
    elif arr.shape[1] != 450:
        raise ValueError(
            f"Expected 225 raw landmarks or 450 features at {pose_path}, "
            f"got feature dimension {arr.shape[1]}"
        )

    if mean is not None and std is not None:
        arr = (arr - mean) / (std + 1e-8)

    length = min(len(arr), max_frames)
    out = np.zeros((max_frames, arr.shape[1]), dtype=np.float32)
    out[:length] = arr[:length]
    return out, length


def compute_norm_stats(records: Sequence[Dict[str, str]], pose_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    frames = []
    root = Path(pose_dir)
    for rec in records:
        path = root / f"{rec['video_id']}.npy"
        if not path.exists():
            continue
        arr = np.load(str(path)).astype(np.float32)
        if arr.shape[1] == 225:
            velocity = np.zeros_like(arr, dtype=np.float32)
            if len(arr) > 1:
                velocity[1:] = arr[1:] - arr[:-1]
            arr = np.concatenate([arr, velocity], axis=1)
        frames.append(arr)
    if not frames:
        return np.zeros(450, dtype=np.float32), np.ones(450, dtype=np.float32)
    all_frames = np.concatenate(frames, axis=0)
    return all_frames.mean(axis=0).astype(np.float32), all_frames.std(axis=0).astype(np.float32)


@dataclass
class Split:
    train: List[Dict[str, str]]
    val: List[Dict[str, str]]
    test: List[Dict[str, str]]


def split_records(
    records: Sequence[Dict[str, str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Split:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)
    n_train = int(len(indices) * train_ratio)
    n_val = int(len(indices) * val_ratio)
    train = [records[i] for i in indices[:n_train]]
    val = [records[i] for i in indices[n_train:n_train + n_val]]
    test = [records[i] for i in indices[n_train + n_val:]]
    return Split(train=train, val=val, test=test)


class PoseCTCDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, str]],
        pose_dir: str,
        vocab: CTCVocabulary,
        max_frames: int,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        self.pose_dir = Path(pose_dir)
        self.vocab = vocab
        self.max_frames = max_frames
        self.mean = mean
        self.std = std
        self.records = []
        for rec in records:
            labels = vocab.encode(rec["text"])
            path = self.pose_dir / f"{rec['video_id']}.npy"
            if path.exists() and labels:
                self.records.append(rec)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        features, feature_length = load_pose_features(
            self.pose_dir / f"{rec['video_id']}.npy",
            self.max_frames,
            self.mean,
            self.std,
        )
        labels = self.vocab.encode(rec["text"])
        if len(labels) > feature_length:
            labels = labels[:feature_length]
        return (
            torch.from_numpy(features).float(),
            torch.tensor(labels, dtype=torch.long),
            feature_length,
            len(labels),
            rec["video_id"],
            rec["text"],
        )


def ctc_collate(batch):
    batch = [x for x in batch if x[2] > 0 and x[3] > 0]
    if not batch:
        return None
    features, labels, feat_lens, label_lens, video_ids, texts = zip(*batch)
    max_label_len = max(label_lens)
    padded_labels = torch.zeros(len(batch), max_label_len, dtype=torch.long)
    for i, label in enumerate(labels):
        padded_labels[i, : len(label)] = label
    return (
        torch.stack(features),
        padded_labels,
        torch.tensor(feat_lens, dtype=torch.long),
        torch.tensor(label_lens, dtype=torch.long),
        list(video_ids),
        list(texts),
    )

