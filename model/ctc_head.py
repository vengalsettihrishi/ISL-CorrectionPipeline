"""
ctc_head.py -- CTC output layer for ISL recognition.

Architecture:
    Linear(hidden_size, vocab_size)  # vocab_size already includes blank at ID 0
    → Log-softmax over vocabulary dimension

Input:  (batch, T, hidden_size) — encoder output
Output: (batch, T, vocab_size) — log probabilities over vocabulary

The blank token is always at index 0, as required by PyTorch's CTCLoss.
Log-softmax is used instead of softmax for numerical stability with CTC.

Usage:
    python -m model.ctc_head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config


class CTCHead(nn.Module):
    """
    CTC output projection + log-softmax.

    Maps encoder hidden states to log-probability distributions over
    the vocabulary (including the CTC blank token at index 0).

    Args:
        hidden_size: Dimension of encoder output.
        vocab_size:  Total vocabulary size (including blank token).
    """

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.projection = nn.Linear(hidden_size, vocab_size)

        # Initialize output projection with small weights for stable
        # CTC training start (avoids extreme log-prob values)
        nn.init.xavier_uniform_(self.projection.weight, gain=0.1)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project encoder output to log-probabilities.

        Args:
            x: (batch, T, hidden_size) encoder output.

        Returns:
            (batch, T, vocab_size) log-probabilities.
        """
        logits = self.projection(x)              # (B, T, V)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)
        return log_probs


# ---------------------------------------------------------------------------
# Standalone verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("ctc_head.py -- verification")
    print("=" * 60)

    cfg = Config()
    vocab_size = 50  # example
    head = CTCHead(cfg.hidden_size, vocab_size)
    params = sum(p.numel() for p in head.parameters())
    print(f"  Parameters:  {params:,}")
    print(f"  Size (MB):   {params * 4 / 1024**2:.3f}")

    B, T = 4, 50
    x = torch.randn(B, T, cfg.hidden_size)
    log_probs = head(x)
    print(f"  Input:       {x.shape}")
    print(f"  Output:      {log_probs.shape}")
    assert log_probs.shape == (B, T, vocab_size)

    # Verify log-softmax properties
    probs = torch.exp(log_probs)
    sums = probs.sum(dim=-1)
    print(f"  Prob sums (should be ~1.0): min={sums.min():.6f}, max={sums.max():.6f}")
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    # Verify all log_probs are <= 0
    assert (log_probs <= 0).all(), "log_probs should be <= 0"
    print("  All log_probs <= 0: PASS")

    print("=" * 60)
    print("[PASS] ctc_head.py OK")
