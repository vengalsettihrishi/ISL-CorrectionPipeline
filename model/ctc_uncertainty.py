"""
ctc_uncertainty.py -- Temporal Uncertainty Propagation (TUP) for CTC models.

Implements a lightweight frame-reliability head that predicts a keep
probability pi_t for each timestep and biases uncertain frames toward the
CTC blank symbol.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalUncertaintyHead(nn.Module):
    """
    Predict per-frame reliability and bias uncertain frames toward blank.

    The head predicts pi_t in [0, 1], where larger pi_t means the frame is
    more reliable for lexical emission. A stochastic / relaxed hard mask z_t
    is derived from pi_t and used to increase the blank logit when the frame
    is uncertain.
    """

    def __init__(
        self,
        hidden_size: int,
        blank_id: int = 0,
        blank_bias: float = 4.0,
        temperature: float = 0.67,
        hard_mask: bool = True,
    ):
        super().__init__()
        self.blank_id = blank_id
        self.blank_bias = blank_bias
        self.temperature = temperature
        self.hard_mask = hard_mask

        self.reliability = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.reliability.weight, gain=0.5)
        nn.init.zeros_(self.reliability.bias)

    def forward(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        hard_mask: Optional[bool] = None,
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden:  (B, T, H) encoder states.
            logits:  (B, T, V) pre-softmax token logits.
            lengths: (B,) valid lengths for padded sequences.
            hard_mask: override for whether to use hard thresholding / ST.
            threshold: deterministic inference threshold for z_t.

        Returns:
            Dict containing masked logits, reliability probabilities, masks,
            and unscaled regularization terms.
        """
        if hard_mask is None:
            hard_mask = self.hard_mask

        B, T, _ = hidden.shape
        valid_mask = _build_valid_mask(lengths, T, hidden.device)

        raw_pi = self.reliability(hidden).squeeze(-1)  # (B, T)
        pi = torch.sigmoid(raw_pi) * valid_mask

        if self.training:
            z = self._sample_mask(raw_pi, hard_mask=hard_mask)
        else:
            if hard_mask:
                z = (pi >= threshold).float()
            else:
                z = pi
        z = z * valid_mask

        masked_logits = logits.clone()
        blank_boost = self.blank_bias * (1.0 - z)
        masked_logits[..., self.blank_id] = masked_logits[..., self.blank_id] + blank_boost

        activity_loss = (pi * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
        if T > 1:
            smooth_mask = valid_mask[:, 1:] * valid_mask[:, :-1]
            smooth_loss = (
                (pi[:, 1:] - pi[:, :-1]).abs() * smooth_mask
            ).sum() / smooth_mask.sum().clamp_min(1.0)
        else:
            smooth_loss = torch.zeros((), device=hidden.device, dtype=hidden.dtype)

        return {
            "masked_logits": masked_logits,
            "pi": pi,
            "z": z,
            "frame_uncertainty": 1.0 - pi,
            "activity_loss": activity_loss,
            "smooth_loss": smooth_loss,
        }

    def _sample_mask(self, raw_pi: torch.Tensor, hard_mask: bool) -> torch.Tensor:
        """Binary Concrete with optional straight-through hard thresholding."""
        uniform = torch.rand_like(raw_pi).clamp_(1e-6, 1.0 - 1e-6)
        gumbel = torch.log(uniform) - torch.log1p(-uniform)
        relaxed = torch.sigmoid((raw_pi + gumbel) / self.temperature)
        if not hard_mask:
            return relaxed
        hard = (relaxed >= 0.5).float()
        return hard.detach() - relaxed.detach() + relaxed


def _build_valid_mask(
    lengths: Optional[torch.Tensor],
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    if lengths is None:
        return torch.ones(1, max_len, device=device)
    if lengths.dim() == 0:
        lengths = lengths.unsqueeze(0)
    arange = torch.arange(max_len, device=device).unsqueeze(0)
    return (arange < lengths.unsqueeze(1)).float()
