"""
isl_model.py -- Complete ISL recognition model (minGRU + CTC).

Combines the feature encoder and CTC output head into a single module.

Methods:
    forward(x, x_lengths) → log_probs, output_lengths
    predict(x)             → decoded token sequences (greedy)
    count_parameters()     → total trainable parameter count
    model_size_mb()        → estimated model size in MB
    export_onnx(path)      → save model as ONNX for edge deployment

Architecture:
    Input (B, T, 450)
    → Encoder: Projection + 3× minGRU layers → (B, T, 128)
    → CTC Head: Linear + log-softmax → (B, T, vocab_size)

Usage:
    python -m model.isl_model
"""

from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config
from model.ctc_uncertainty import TemporalUncertaintyHead
from model.encoder import Encoder
from model.ctc_head import CTCHead


class ISLModel(nn.Module):
    """
    Complete ISL recognition model: Encoder → CTC Head.

    Args:
        input_dim:   Feature dimension per frame (450).
        hidden_size: Hidden dimension (128).
        vocab_size:  Total vocabulary size including blank (ID 0).
        num_layers:  Number of minGRU layers (3).
        dropout:     Dropout probability (0.1).
    """

    def __init__(
        self,
        input_dim: int = 450,
        hidden_size: int = 128,
        vocab_size: int = 50,
        num_layers: int = 3,
        dropout: float = 0.1,
        blank_id: int = 0,
        enable_velocity_temperature: bool = True,
        velocity_temperature_init: float = 0.5,
        enable_tup: bool = True,
        tup_blank_bias: float = 4.0,
        tup_temperature: float = 0.67,
        tup_hard_mask: bool = True,
        tup_threshold: float = 0.5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.enable_velocity_temperature = enable_velocity_temperature
        self.enable_tup = enable_tup
        self.tup_threshold = tup_threshold

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            enable_velocity_temperature=enable_velocity_temperature,
            temperature_init=velocity_temperature_init,
        )

        self.ctc_head = CTCHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )

        self.uncertainty_head = TemporalUncertaintyHead(
            hidden_size=hidden_size,
            blank_id=blank_id,
            blank_bias=tup_blank_bias,
            temperature=tup_temperature,
            hard_mask=tup_hard_mask,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass for training with CTC loss.

        Args:
            x:         (batch, T, input_dim) — padded feature sequences.
            x_lengths: (batch,) — actual (pre-padding) sequence lengths.

        Returns:
            log_probs:      (batch, T, vocab_size) — log-probabilities.
            output_lengths: (batch,) — same as x_lengths (no temporal
                            downsampling in this architecture).
        """
        # Encode
        encoded, _, encoder_aux = self.encoder(x, return_aux=True)  # (B, T, H)

        # CTC output
        logits = self.ctc_head.compute_logits(encoded)  # (B, T, V)
        if self.enable_tup:
            tup_aux = self.uncertainty_head(
                encoded,
                logits,
                lengths=x_lengths,
                threshold=self.tup_threshold,
            )
            logits = tup_aux["masked_logits"]
        else:
            device = x.device
            B, T, _ = encoded.shape
            tup_aux = {
                "masked_logits": logits,
                "pi": torch.ones(B, T, device=device, dtype=encoded.dtype),
                "z": torch.ones(B, T, device=device, dtype=encoded.dtype),
                "frame_uncertainty": torch.zeros(B, T, device=device, dtype=encoded.dtype),
                "activity_loss": torch.zeros((), device=device, dtype=encoded.dtype),
                "smooth_loss": torch.zeros((), device=device, dtype=encoded.dtype),
            }
        log_probs = torch.log_softmax(logits, dim=-1)

        # Output lengths = input lengths (no temporal pooling/striding)
        output_lengths = x_lengths.clone()

        if return_aux:
            return log_probs, output_lengths, {
                "encoder": encoder_aux,
                "tup": tup_aux,
                "logits": logits,
            }
        return log_probs, output_lengths

    def predict(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """
        Greedy CTC decoding for inference.

        Steps:
            1. Forward pass to get log-probs
            2. Argmax at each timestep
            3. Collapse consecutive identical tokens
            4. Remove blank tokens (ID 0)

        Args:
            x:         (batch, T, input_dim).
            x_lengths: (batch,) optional — if None, uses full sequence.

        Returns:
            List of decoded token ID sequences (one per batch element).
        """
        self.eval()
        with torch.no_grad():
            if x_lengths is None:
                x_lengths = torch.full(
                    (x.size(0),), x.size(1),
                    dtype=torch.int32, device=x.device,
                )

            log_probs, output_lengths = self.forward(x, x_lengths)

            # Greedy decode
            predictions = log_probs.argmax(dim=-1)  # (B, T)

            decoded = []
            for b in range(predictions.size(0)):
                seq_len = output_lengths[b].item()
                raw = predictions[b, :seq_len].tolist()

                # Collapse consecutive duplicates
                collapsed = []
                prev = None
                for token in raw:
                    if token != prev:
                        collapsed.append(token)
                    prev = token

                # Remove blank tokens
                filtered = [t for t in collapsed if t != 0]
                decoded.append(filtered)

            return decoded

    def predict_recurrent(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> List[int]:
        """
        Single-sequence recurrent prediction (for real-time inference).

        Processes one frame at a time, suitable for streaming on edge devices.

        Args:
            x: (1, T, input_dim) — single sequence.

        Returns:
            List of decoded token IDs.
        """
        self.eval()
        with torch.no_grad():
            T = x.size(1)
            h_states = None
            all_log_probs = []
            all_pi = []
            all_tau = []

            for t in range(T):
                frame = x[:, t, :]  # (1, input_dim)
                encoded, h_states, encoder_aux = self.encoder(
                    frame, h_states, return_aux=True
                )
                logits = self.ctc_head.compute_logits(encoded.unsqueeze(1))
                if self.enable_tup:
                    tup_aux = self.uncertainty_head(
                        encoded.unsqueeze(1),
                        logits,
                        lengths=torch.ones(1, dtype=torch.int32, device=x.device),
                        threshold=self.tup_threshold,
                    )
                    log_prob = torch.log_softmax(tup_aux["masked_logits"], dim=-1)
                    all_pi.append(tup_aux["pi"].squeeze(0).squeeze(0))
                else:
                    log_prob = torch.log_softmax(logits, dim=-1)
                    all_pi.append(torch.tensor(1.0, device=x.device))
                layer_taus = []
                for layer_aux in encoder_aux["layers"]:
                    tau = layer_aux.get("tau")
                    if tau is not None:
                        layer_taus.append(tau.squeeze(0).mean())
                if layer_taus:
                    all_tau.append(torch.stack(layer_taus).mean())
                all_log_probs.append(log_prob.squeeze(1))

            log_probs = torch.stack(all_log_probs, dim=1)  # (1, T, V)
            predictions = log_probs.argmax(dim=-1)  # (1, T)

            raw = predictions[0].tolist()
            collapsed = []
            prev = None
            for token in raw:
                if token != prev:
                    collapsed.append(token)
                prev = token

            decoded = [t for t in collapsed if t != 0]
            if return_aux:
                return decoded, {
                    "frame_reliability": torch.stack(all_pi).cpu(),
                    "mean_tau": torch.stack(all_tau).mean().item() if all_tau else 1.0,
                }
            return decoded

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self) -> float:
        """Estimated model size in MB (float32)."""
        return self.count_parameters() * 4 / (1024 ** 2)

    def print_model_summary(self) -> None:
        """Print a summary of model architecture and size."""
        print("=" * 60)
        print("ISL Model Summary (minGRU + CTC)")
        print("=" * 60)
        print(f"  Input dim:     {self.input_dim}")
        print(f"  Hidden size:   {self.hidden_size}")
        print(f"  Vocab size:    {self.vocab_size}")
        print(f"  Vel temp:      {'PASS' if self.enable_velocity_temperature else 'OFF'}")
        print(f"  TUP enabled:   {'PASS' if self.enable_tup else 'OFF'}")
        print(f"  Parameters:    {self.count_parameters():,}")
        print(f"  Size (MB):     {self.model_size_mb():.3f}")
        print(f"  Under 500K:    {'PASS' if self.count_parameters() < 500_000 else 'FAIL'}")
        print(f"  Under 2MB:     {'PASS' if self.model_size_mb() < 2.0 else 'FAIL'}")

        # Per-component breakdown
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        head_params = sum(p.numel() for p in self.ctc_head.parameters())
        tup_params = sum(p.numel() for p in self.uncertainty_head.parameters())
        print(f"\n  Encoder:       {enc_params:,} params ({enc_params*4/1024**2:.3f} MB)")
        print(f"  CTC Head:      {head_params:,} params ({head_params*4/1024**2:.3f} MB)")
        print(f"  TUP Head:      {tup_params:,} params ({tup_params*4/1024**2:.3f} MB)")
        print("=" * 60)

    def export_onnx(self, path: str, seq_length: int = 300) -> None:
        """
        Export model to ONNX format for edge deployment.

        Args:
            path:       Output .onnx file path.
            seq_length: Fixed sequence length for the export.
        """
        self.eval()
        dummy_x = torch.randn(1, seq_length, self.input_dim)
        dummy_lengths = torch.tensor([seq_length], dtype=torch.int32)

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            self,
            (dummy_x, dummy_lengths),
            path,
            input_names=["features", "lengths"],
            output_names=["log_probs", "output_lengths"],
            dynamic_axes={
                "features": {0: "batch", 1: "time"},
                "lengths": {0: "batch"},
                "log_probs": {0: "batch", 1: "time"},
                "output_lengths": {0: "batch"},
            },
            opset_version=17,
        )
        print(f"ONNX model exported to {path}")

    @classmethod
    def from_config(cls, config: Config, vocab_size: int) -> "ISLModel":
        """Create model from pipeline config and vocabulary size."""
        return cls(
            input_dim=config.feature_dim,
            hidden_size=config.hidden_size,
            vocab_size=vocab_size,
            num_layers=config.num_gru_layers,
            dropout=config.dropout,
            blank_id=config.ctc_blank_id,
            enable_velocity_temperature=config.enable_velocity_temperature,
            velocity_temperature_init=config.velocity_temperature_init,
            enable_tup=config.enable_tup,
            tup_blank_bias=config.tup_blank_bias,
            tup_temperature=config.tup_temperature,
            tup_hard_mask=config.tup_hard_mask,
            tup_threshold=config.tup_threshold,
        )


# ---------------------------------------------------------------------------
# Standalone verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = Config()
    vocab_size = 50  # example

    model = ISLModel.from_config(cfg, vocab_size)
    model.print_model_summary()

    B, T = 4, 50
    x = torch.randn(B, T, cfg.feature_dim)
    x_lengths = torch.full((B,), T, dtype=torch.int32)

    # Forward pass
    model.eval()
    with torch.no_grad():
        log_probs, out_lengths = model(x, x_lengths)
    print(f"\n  Forward pass:")
    print(f"    Input:          {x.shape}")
    print(f"    Log-probs:      {log_probs.shape}")
    print(f"    Output lengths: {out_lengths}")
    assert log_probs.shape == (B, T, vocab_size)

    # CTC loss test
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    # CTC expects (T, B, V)
    log_probs_ctc = log_probs.permute(1, 0, 2)  # (T, B, V)
    targets = torch.randint(1, vocab_size, (B, 5))  # random targets
    target_lengths = torch.full((B,), 5, dtype=torch.int32)
    loss = ctc_loss(log_probs_ctc, targets, out_lengths, target_lengths)
    print(f"    CTC loss:       {loss.item():.4f}")
    assert not torch.isnan(loss), "CTC loss is NaN!"
    assert not torch.isinf(loss), "CTC loss is Inf!"

    # Greedy decode
    decoded = model.predict(x)
    print(f"\n  Greedy decode:")
    for i, seq in enumerate(decoded[:2]):
        print(f"    Sample {i}: {seq[:10]}{'...' if len(seq) > 10 else ''}")

    # Recurrent predict
    single = x[:1]  # (1, T, 450)
    rec_decoded = model.predict_recurrent(single)
    print(f"    Recurrent decode: {rec_decoded[:10]}{'...' if len(rec_decoded) > 10 else ''}")

    print("\n" + "=" * 60)
    print("[PASS] isl_model.py OK")
