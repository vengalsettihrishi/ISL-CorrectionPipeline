"""
inference/fallback_controller.py -- Confidence-driven triple fallback routing.

Routes uncertain sentence-level spans to:
    1. isolated-word fallback
    2. fingerspelling fallback

The controller uses sentence confidence, span confidence, TUP uncertainty,
duration, and motion summaries to decide when to back off.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from inference.ctc_decoder import DecoderOutput, TokenSpan
from inference.fingerspell_fallback import (
    FingerspellFallbackRecognizer,
    FingerspellFallbackResult,
)
from inference.word_fallback import WordFallbackRecognizer, WordFallbackResult


@dataclass
class RoutingEvent:
    """One routing decision for a candidate uncertain span."""
    span_index: int
    original_token: str
    original_confidence: float
    original_uncertainty: float
    mean_motion: float
    action: str
    replacement_token: Optional[str] = None
    replacement_confidence: float = 0.0
    replacement_source: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "span_index": self.span_index,
            "original_token": self.original_token,
            "original_confidence": round(self.original_confidence, 4),
            "original_uncertainty": round(self.original_uncertainty, 4),
            "mean_motion": round(self.mean_motion, 4),
            "action": self.action,
            "replacement_token": self.replacement_token,
            "replacement_confidence": round(self.replacement_confidence, 4),
            "replacement_source": self.replacement_source,
            "reason": self.reason,
        }


@dataclass
class FallbackResult:
    """Output of sentence/word/fingerspelling routing."""
    decoder_output: DecoderOutput
    accepted_sentence: bool
    sentence_confidence: float
    mean_uncertainty: float
    routing_events: List[RoutingEvent] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "accepted_sentence": self.accepted_sentence,
            "sentence_confidence": round(self.sentence_confidence, 4),
            "mean_uncertainty": round(self.mean_uncertainty, 4),
            "routing_events": [event.to_dict() for event in self.routing_events],
            "decoder_output": self.decoder_output.to_dict(),
        }


class FallbackController:
    """
    Confidence-driven triple fallback controller.

    The default implementation is intentionally lightweight and conservative:
    it only replaces spans when a specialized fallback model returns a strong
    enough decision. Otherwise it keeps the sentence-level output and records
    the decline reason for analysis.
    """

    def __init__(
        self,
        word2id: Dict[str, int],
        sentence_accept_threshold: float = 0.6,
        span_uncertainty_threshold: float = 0.45,
        span_confidence_threshold: float = 0.35,
        word_accept_threshold: float = 0.6,
        spell_accept_threshold: float = 0.55,
        motion_threshold: float = 0.05,
        min_fallback_frames: int = 3,
        enable_word_fallback: bool = True,
        enable_fingerspell_fallback: bool = True,
        word_fallback: Optional[WordFallbackRecognizer] = None,
        fingerspell_fallback: Optional[FingerspellFallbackRecognizer] = None,
    ):
        self.word2id = word2id
        self.sentence_accept_threshold = sentence_accept_threshold
        self.span_uncertainty_threshold = span_uncertainty_threshold
        self.span_confidence_threshold = span_confidence_threshold
        self.word_accept_threshold = word_accept_threshold
        self.spell_accept_threshold = spell_accept_threshold
        self.motion_threshold = motion_threshold
        self.min_fallback_frames = min_fallback_frames
        self.enable_word_fallback = enable_word_fallback
        self.enable_fingerspell_fallback = enable_fingerspell_fallback
        self.word_fallback = word_fallback or WordFallbackRecognizer(word2id=word2id)
        self.fingerspell_fallback = fingerspell_fallback or FingerspellFallbackRecognizer(
            word2id=word2id
        )

    def route(
        self,
        decoder_output: DecoderOutput,
        features: Optional[np.ndarray] = None,
        velocity_magnitudes: Optional[List[float]] = None,
    ) -> FallbackResult:
        mean_uncertainty = (
            sum(decoder_output.raw_frame_uncertainties)
            / max(len(decoder_output.raw_frame_uncertainties), 1)
        ) if decoder_output.raw_frame_uncertainties else 0.0

        accepted_sentence = (
            decoder_output.sequence_confidence >= self.sentence_accept_threshold
            and mean_uncertainty <= self.span_uncertainty_threshold
        )

        spans = [self._clone_span(span) for span in decoder_output.spans]
        routing_events: List[RoutingEvent] = []

        if not accepted_sentence:
            for idx, span in enumerate(spans):
                mean_motion = self._mean_motion(span, velocity_magnitudes)
                if not self._should_route_span(span, mean_motion):
                    continue

                span_features = self._slice_features(features, span)

                if self.enable_word_fallback:
                    word_result = self.word_fallback.predict(
                        span_features,
                        hint_token=span.token,
                    )
                    if word_result.accepted and word_result.confidence >= self.word_accept_threshold:
                        spans[idx] = self._replace_span(span, word_result)
                        routing_events.append(self._accepted_event(idx, span, mean_motion, word_result))
                        continue
                    routing_events.append(self._declined_event(idx, span, mean_motion, word_result))

                if self.enable_fingerspell_fallback:
                    spell_result = self.fingerspell_fallback.predict(
                        span_features,
                        hint_token=span.token,
                    )
                    if (
                        spell_result.accepted
                        and spell_result.confidence >= self.spell_accept_threshold
                    ):
                        spans[idx] = self._replace_span(span, spell_result)
                        routing_events.append(self._accepted_event(idx, span, mean_motion, spell_result))
                        continue
                    routing_events.append(self._declined_event(idx, span, mean_motion, spell_result))

        merged_output = DecoderOutput(
            tokens=[span.token for span in spans],
            token_ids=[span.token_id for span in spans],
            sequence_confidence=_sequence_confidence([span.confidence for span in spans]),
            token_confidences=[span.confidence for span in spans],
            spans=spans,
            raw_frame_predictions=list(decoder_output.raw_frame_predictions),
            raw_frame_confidences=list(decoder_output.raw_frame_confidences),
            raw_frame_uncertainties=list(decoder_output.raw_frame_uncertainties),
        )

        return FallbackResult(
            decoder_output=merged_output,
            accepted_sentence=accepted_sentence,
            sentence_confidence=decoder_output.sequence_confidence,
            mean_uncertainty=mean_uncertainty,
            routing_events=routing_events,
        )

    def _should_route_span(self, span: TokenSpan, mean_motion: float) -> bool:
        if span.frame_count < self.min_fallback_frames:
            return False
        if span.uncertainty >= self.span_uncertainty_threshold:
            return True
        if span.confidence < self.span_confidence_threshold:
            return True
        if mean_motion > self.motion_threshold and span.confidence < self.sentence_accept_threshold:
            return True
        return False

    def _slice_features(
        self,
        features: Optional[np.ndarray],
        span: TokenSpan,
    ) -> np.ndarray:
        if features is None or features.size == 0:
            return np.zeros((0, 0), dtype=np.float32)
        start = max(0, span.start_frame)
        end = min(features.shape[0], span.end_frame)
        if start >= end:
            return np.zeros((0, features.shape[1]), dtype=features.dtype)
        return features[start:end]

    def _mean_motion(
        self,
        span: TokenSpan,
        velocity_magnitudes: Optional[List[float]],
    ) -> float:
        if not velocity_magnitudes:
            return 0.0
        start = max(0, span.start_frame)
        end = min(len(velocity_magnitudes), span.end_frame)
        if start >= end:
            return 0.0
        segment = velocity_magnitudes[start:end]
        return float(sum(segment) / max(len(segment), 1))

    def _replace_span(
        self,
        span: TokenSpan,
        replacement: object,
    ) -> TokenSpan:
        token = getattr(replacement, "token", span.token) or span.token
        token_id = getattr(replacement, "token_id", None)
        if token_id is None:
            token_id = self.word2id.get(token, self.word2id.get("<unk>", span.token_id))
        return TokenSpan(
            token=token,
            token_id=token_id,
            start_frame=span.start_frame,
            end_frame=span.end_frame,
            confidence=getattr(replacement, "confidence", span.confidence),
            uncertainty=max(0.0, span.uncertainty * 0.5),
            origin=getattr(replacement, "source", "fallback"),
        )

    def _clone_span(self, span: TokenSpan) -> TokenSpan:
        return TokenSpan(
            token=span.token,
            token_id=span.token_id,
            start_frame=span.start_frame,
            end_frame=span.end_frame,
            confidence=span.confidence,
            uncertainty=span.uncertainty,
            origin=span.origin,
        )

    def _accepted_event(
        self,
        index: int,
        span: TokenSpan,
        mean_motion: float,
        result: object,
    ) -> RoutingEvent:
        return RoutingEvent(
            span_index=index,
            original_token=span.token,
            original_confidence=span.confidence,
            original_uncertainty=span.uncertainty,
            mean_motion=mean_motion,
            action="replace",
            replacement_token=getattr(result, "token", None),
            replacement_confidence=getattr(result, "confidence", 0.0),
            replacement_source=getattr(result, "source", None),
            reason=getattr(result, "reason", ""),
        )

    def _declined_event(
        self,
        index: int,
        span: TokenSpan,
        mean_motion: float,
        result: object,
    ) -> RoutingEvent:
        return RoutingEvent(
            span_index=index,
            original_token=span.token,
            original_confidence=span.confidence,
            original_uncertainty=span.uncertainty,
            mean_motion=mean_motion,
            action="keep",
            replacement_token=getattr(result, "token", None),
            replacement_confidence=getattr(result, "confidence", 0.0),
            replacement_source=getattr(result, "source", None),
            reason=getattr(result, "reason", ""),
        )


def _sequence_confidence(confidences: List[float]) -> float:
    if not confidences:
        return 0.0
    log_confs = [math.log(max(conf, 1e-10)) for conf in confidences]
    return math.exp(sum(log_confs) / len(log_confs))
