"""
inference/model_loader.py -- Reliable checkpoint, vocabulary, and stats loader.

Loads a trained Sprint 2 checkpoint and restores:
    - ISLModel with correct architecture config
    - LabelEncoder with vocabulary (id2word, word2id)
    - Normalization statistics (mean, std)
    - Label-space type (english or gloss)

All loading is CPU-safe.

Usage:
    python -m inference.model_loader
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config
from model.isl_model import ISLModel
from data.label_encoder import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loaded model bundle
# ---------------------------------------------------------------------------

class ModelBundle:
    """
    Complete loaded model bundle for inference.

    Attributes:
        model:       ISLModel in eval mode on the target device.
        encoder:     LabelEncoder with vocabulary.
        mean:        Per-feature mean (450,) or None.
        std:         Per-feature std  (450,) or None.
        label_type:  "english" or "gloss".
        vocab_size:  Total vocabulary size (including blank).
        id2word:     Token ID -> word mapping.
        word2id:     Word -> token ID mapping.
        device:      torch.device the model is on.
        config:      Model architecture config dict.
        epoch:       Training epoch of the checkpoint.
    """

    def __init__(
        self,
        model: ISLModel,
        encoder: LabelEncoder,
        mean: Optional[np.ndarray],
        std: Optional[np.ndarray],
        device: torch.device,
        checkpoint_meta: Dict,
    ):
        self.model = model
        self.encoder = encoder
        self.mean = mean
        self.std = std
        self.device = device

        self.label_type = encoder.label_type
        self.vocab_size = encoder.vocab_size
        self.id2word = dict(encoder.id2word)
        self.word2id = dict(encoder.word2id)
        self.config = checkpoint_meta.get("config", {})
        self.epoch = checkpoint_meta.get("epoch", -1)

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Apply saved normalization to features.

        Args:
            features: (T, 450) raw feature array.

        Returns:
            (T, 450) normalized feature array.
        """
        if self.mean is not None and self.std is not None:
            return (features - self.mean) / np.maximum(self.std, 1e-8)
        return features

    def predict(self, features: np.ndarray) -> torch.Tensor:
        """
        Run model forward on features.

        Args:
            features: (T, 450) numpy array (already normalized).

        Returns:
            log_probs: (T, vocab_size) tensor of log-probabilities.
        """
        x = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        x_lengths = torch.tensor([features.shape[0]], dtype=torch.int32)

        with torch.no_grad():
            log_probs, out_lengths = self.model(x, x_lengths)

        return log_probs.squeeze(0)  # (T, V)

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Model Bundle:",
            f"  Label type:    {self.label_type}",
            f"  Vocab size:    {self.vocab_size}",
            f"  Parameters:    {self.model.count_parameters():,}",
            f"  Size (MB):     {self.model.model_size_mb():.3f}",
            f"  Device:        {self.device}",
            f"  Epoch:         {self.epoch}",
            f"  Normalized:    {self.mean is not None}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------

def load_model_bundle(
    checkpoint_path: str = "./checkpoints/best_model.pth",
    vocab_path: str = "./checkpoints/vocab.json",
    norm_stats_path: str = "./data_iSign/norm_stats.npz",
    device: str = "cpu",
) -> ModelBundle:
    """
    Load a complete model bundle for inference.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        vocab_path:      Path to vocab.json.
        norm_stats_path: Path to norm_stats.npz (optional).
        device:          "cpu" or "cuda".

    Returns:
        ModelBundle ready for inference.
    """
    device = torch.device(device)

    # 1. Load checkpoint
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    logger.info(f"Loaded checkpoint from {ckpt_path} (epoch {checkpoint.get('epoch', '?')})")

    # 2. Load vocabulary
    encoder = _load_vocabulary(vocab_path, checkpoint)

    # 3. Build model
    model_cfg = checkpoint.get("config", {})
    vocab_size = checkpoint.get("vocab_size", encoder.vocab_size)

    model = ISLModel(
        input_dim=model_cfg.get("input_dim", 450),
        hidden_size=model_cfg.get("hidden_size", 128),
        vocab_size=vocab_size,
        num_layers=model_cfg.get("num_layers", 3),
        dropout=model_cfg.get("dropout", 0.1),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Model restored: {model.count_parameters():,} params, "
                f"vocab_size={vocab_size}")

    # 4. Load normalization stats
    mean, std = _load_norm_stats(norm_stats_path)

    return ModelBundle(
        model=model,
        encoder=encoder,
        mean=mean,
        std=std,
        device=device,
        checkpoint_meta=checkpoint,
    )


def _load_vocabulary(vocab_path: str, checkpoint: Dict) -> LabelEncoder:
    """Load vocabulary from JSON file or checkpoint directory."""
    vocab_file = Path(vocab_path)

    # Also try checkpoint directory
    ckpt_dir_vocab = Path(checkpoint.get("_ckpt_dir", "")) / "vocab.json"

    cfg = Config()

    if vocab_file.exists():
        encoder = LabelEncoder(cfg)
        encoder.load(str(vocab_file))
        logger.info(f"Vocabulary loaded from {vocab_file} "
                    f"({encoder.vocab_size} tokens, {encoder.label_type})")
        return encoder

    if ckpt_dir_vocab.exists():
        encoder = LabelEncoder(cfg)
        encoder.load(str(ckpt_dir_vocab))
        logger.info(f"Vocabulary loaded from {ckpt_dir_vocab}")
        return encoder

    # Fallback: build a dummy encoder with the right vocab size
    logger.warning(f"Vocabulary file not found at {vocab_path}. "
                   "Using placeholder vocabulary.")
    vocab_size = checkpoint.get("vocab_size", 50)
    encoder = LabelEncoder(cfg, label_type="english")
    # Build placeholder word2id/id2word
    encoder.word2id = {"<blank>": 0, "<unk>": 1}
    encoder.id2word = {0: "<blank>", 1: "<unk>"}
    for i in range(2, vocab_size):
        token = f"token_{i}"
        encoder.word2id[token] = i
        encoder.id2word[i] = token
    return encoder


def _load_norm_stats(
    path: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load normalization statistics (mean, std) from .npz file."""
    stats_path = Path(path)
    if not stats_path.exists():
        logger.warning(f"Normalization stats not found at {stats_path}. "
                      "Features will NOT be normalized.")
        return None, None

    try:
        data = np.load(str(stats_path))
        mean = data["mean"].astype(np.float32)
        std = data["std"].astype(np.float32)
        logger.info(f"Normalization stats loaded from {stats_path} "
                    f"(mean shape={mean.shape})")
        return mean, std
    except Exception as e:
        logger.warning(f"Failed to load norm stats: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Standalone verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("inference/model_loader.py -- verification")
    print("=" * 60)

    # Test with synthetic checkpoint
    import tempfile, os

    cfg = Config()
    vocab_size = 30

    # Create a model and save checkpoint
    model = ISLModel.from_config(cfg, vocab_size)
    checkpoint = {
        "epoch": 10,
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "config": {
            "input_dim": cfg.feature_dim,
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_gru_layers,
            "dropout": cfg.dropout,
        },
    }

    tmp_dir = tempfile.mkdtemp()
    ckpt_path = os.path.join(tmp_dir, "test_model.pth")
    torch.save(checkpoint, ckpt_path)

    # Create a vocab file
    vocab_path = os.path.join(tmp_dir, "vocab.json")
    encoder = LabelEncoder(cfg, label_type="english")
    test_labels = ["hello world", "how are you", "i am fine thank you"]
    encoder.build_vocab(test_labels)
    encoder.save(vocab_path)

    # Create norm stats
    stats_path = os.path.join(tmp_dir, "norm_stats.npz")
    np.savez(
        stats_path,
        mean=np.zeros(450, dtype=np.float32),
        std=np.ones(450, dtype=np.float32),
    )

    # Load the bundle
    bundle = load_model_bundle(
        checkpoint_path=ckpt_path,
        vocab_path=vocab_path,
        norm_stats_path=stats_path,
        device="cpu",
    )

    print(f"\n{bundle.summary()}")

    # Test predict
    features = np.random.randn(50, 450).astype(np.float32)
    normalized = bundle.normalize(features)
    log_probs = bundle.predict(normalized)
    print(f"\n  Predict output: {log_probs.shape}")
    assert log_probs.shape == (50, vocab_size)

    # Cleanup
    os.unlink(ckpt_path)
    os.unlink(vocab_path)
    os.unlink(stats_path)
    os.rmdir(tmp_dir)

    print("=" * 60)
    print("[PASS] inference/model_loader.py OK")
