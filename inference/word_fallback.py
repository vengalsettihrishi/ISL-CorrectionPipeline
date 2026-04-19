"""
inference/word_fallback.py -- Pluggable isolated-word fallback recognizer.

The default implementation is intentionally lightweight and degrades
gracefully when no specialized word-level model has been trained yet.
It provides a stable interface for the Sprint 5 fallback controller.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass
class WordFallbackResult:
    """Structured output from the isolated-word fallback recognizer."""
    accepted: bool
    token: Optional[str]
    token_id: Optional[int]
    confidence: float
    source: str
    reason: str

    def to_dict(self) -> Dict:
        return {
            "accepted": self.accepted,
            "token": self.token,
            "token_id": self.token_id,
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "reason": self.reason,
        }


class WordFallbackRecognizer:
    """
    Lightweight wrapper for an isolated-word fallback model.

    Until a dedicated recognizer is trained, this module remains callable and
    returns deterministic "not available" results instead of raising errors.
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
    ) -> WordFallbackResult:
        """
        Predict a single lexical token for a span.

        The placeholder implementation declines when no trained recognizer is
        available. Once a specialized isolated-word recognizer is trained, its
        checkpoint can be loaded behind this same API.
        """
        if span_features.size == 0:
            return WordFallbackResult(
                accepted=False,
                token=None,
                token_id=None,
                confidence=0.0,
                source="word_fallback",
                reason="empty_span",
            )

        if not self.available:
            return WordFallbackResult(
                accepted=False,
                token=hint_token,
                token_id=self.word2id.get(hint_token) if hint_token else None,
                confidence=0.0,
                source="word_fallback",
                reason="word_model_unavailable",
            )

        # Placeholder for future specialized model integration.
        return WordFallbackResult(
            accepted=False,
            token=hint_token,
            token_id=self.word2id.get(hint_token) if hint_token else None,
            confidence=0.0,
            source="word_fallback",
            reason="word_model_not_implemented",
        )
