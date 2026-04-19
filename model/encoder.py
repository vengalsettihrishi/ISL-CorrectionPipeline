"""
encoder.py -- Feature encoder for ISL recognition.

Architecture:
    Linear(450, 128) → LayerNorm(128) → Dropout(0.1)
    → minGRU layer 1 (128 → 128) → LayerNorm → Dropout
    → minGRU layer 2 (128 → 128) → LayerNorm → Dropout
    → minGRU layer 3 (128 → 128) → LayerNorm → Dropout

Input:  (batch, T, 450) — feature vectors (landmarks + velocity)
Output: (batch, T, 128) — encoded representations

The encoder projects the 450-dim input down to 128-dim, then passes
through 3 stacked minGRU layers to capture temporal dependencies in
the sign language sequence.

Usage:
    python -m model.encoder
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config
from model.mingru import MinGRUStack


class Encoder(nn.Module):
    """
    Feature encoder: input projection → stacked minGRU.

    Args:
        input_dim:   Input feature dimension (default 450).
        hidden_size: Hidden dimension for projection and minGRU (default 128).
        num_layers:  Number of stacked minGRU layers (default 3).
        dropout:     Dropout probability (default 0.1).
    """

    def __init__(
        self,
        input_dim: int = 450,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        enable_velocity_temperature: bool = True,
        temperature_init: float = 0.5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size

        # Input projection: 450 → 128
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )

        # Stacked minGRU layers
        self.mingru_stack = MinGRUStack(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            enable_velocity_temperature=enable_velocity_temperature,
            temperature_init=temperature_init,
        )

    def forward(
        self,
        x: torch.Tensor,
        h_prev: Optional[list] = None,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, list]:
        """
        Encode a sequence of feature vectors.

        Args:
            x: (batch, T, input_dim) or (batch, input_dim) for single frame.
            h_prev: Optional list of hidden states for recurrent mode.

        Returns:
            encoded: (batch, T, hidden_size) or (batch, hidden_size).
            h_list:  List of hidden states per minGRU layer.
        """
        motion_mag = self._compute_motion_magnitude(x)

        # Project input
        projected = self.input_proj(x)  # (B, T, H) or (B, H)

        # Pass through minGRU stack
        stack_output = self.mingru_stack(
            projected,
            h_prev,
            motion_mag=motion_mag,
            return_aux=return_aux,
        )
        if return_aux:
            encoded, h_list, aux = stack_output
            aux["motion_magnitude"] = motion_mag
            return encoded, h_list, aux
        encoded, h_list = stack_output

        return encoded, h_list

    def _compute_motion_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-frame motion magnitude from the velocity half of features."""
        velocity = x[..., self.input_dim // 2:]
        return velocity.abs().mean(dim=-1)

    @classmethod
    def from_config(cls, config: Config) -> "Encoder":
        """Create encoder from pipeline config."""
        return cls(
            input_dim=config.feature_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_gru_layers,
            dropout=config.dropout,
            enable_velocity_temperature=config.enable_velocity_temperature,
            temperature_init=config.velocity_temperature_init,
        )


# ---------------------------------------------------------------------------
# Standalone verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("encoder.py -- verification")
    print("=" * 60)

    cfg = Config()
    encoder = Encoder.from_config(cfg)
    params = sum(p.numel() for p in encoder.parameters())
    print(f"  Parameters:  {params:,}")
    print(f"  Size (MB):   {params * 4 / 1024**2:.3f}")

    B, T = 4, 50
    x = torch.randn(B, T, cfg.feature_dim)

    encoder.eval()
    with torch.no_grad():
        out, h = encoder(x)
    print(f"  Input:       {x.shape}")
    print(f"  Output:      {out.shape}")
    assert out.shape == (B, T, cfg.hidden_size)

    # Single-frame (recurrent) mode
    with torch.no_grad():
        out_single, h = encoder(x[:, 0, :])
    print(f"  Single-step: {out_single.shape}")
    assert out_single.shape == (B, cfg.hidden_size)

    print("=" * 60)
    print("[PASS] encoder.py OK")
