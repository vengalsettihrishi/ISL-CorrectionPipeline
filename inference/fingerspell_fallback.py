"""
inference/fingerspell_fallback.py -- Pluggable fingerspelling fallback.

The default implementation exposes the runtime interface and degrades
gracefully until a dedicated fingerspelling recognizer is trained.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class FingerspellFallbackResult:
    """Structured output from the fingerspelling fallback recognizer."""
    accepted: bool
    token: Optional[str]
    token_id: Optional[int]
    characters: List[str]
    confidence: float
    source: str
    reason: str

    def to_dict(self) -> Dict:
        return {
            "accepted": self.accepted,
            "token": self.token,
            "token_id": self.token_id,
            "characters": self.characters,
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "reason": self.reason,
        }


class FingerspellFallbackRecognizer:
    """
    Lightweight fingerspelling fallback recognizer wrapper.

    The interface is wired now so the controller can route OOV / unresolved
    spans immediately once a real character-level recognizer is trained.
    """

    def __init__(
        self,
        word2id: Optional[Dict[str, int]] = None,
        model_path: Optional[str] = None,
    ):
        self.word2id = word2id or {}
        self.model_path = model_path
        self.available = bool(model_path and Path(model_path).exists())

    def predict(
        self,
        span_features: np.ndarray,
        hint_token: Optional[str] = None,
    ) -> FingerspellFallbackResult:
        if span_features.size == 0:
            return FingerspellFallbackResult(
                accepted=False,
                token=None,
                token_id=None,
                characters=[],
                confidence=0.0,
                source="fingerspell_fallback",
                reason="empty_span",
            )

        if not self.available:
            return FingerspellFallbackResult(
                accepted=False,
                token=hint_token,
                token_id=self.word2id.get(hint_token) if hint_token else None,
                characters=[],
                confidence=0.0,
                source="fingerspell_fallback",
                reason="fingerspell_model_unavailable",
            )

        # Placeholder for future dedicated character/subword model.
        return FingerspellFallbackResult(
            accepted=False,
            token=hint_token,
            token_id=self.word2id.get(hint_token) if hint_token else None,
            characters=[],
            confidence=0.0,
            source="fingerspell_fallback",
            reason="fingerspell_model_not_implemented",
        )
