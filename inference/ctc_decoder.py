"""
inference/ctc_decoder.py -- CTC decoding with span-aware structured output.

Provides two decoders for the English-token CTC model:

    1. GreedyDecoder:       argmax -> collapse repeats -> remove blanks
    2. PrefixBeamDecoder:   prefix beam search with configurable beam_width

Both return structured DecoderOutput with:
    - tokens, token_ids, confidences
    - per-token spans (start_frame, end_frame, confidence)
    - raw frame predictions for refinement

Confidence definition:
    - Token confidence = mean probability over frames in that token's span
    - Sequence confidence = geometric mean of token confidences (in log space)

Usage:
    python -m inference.ctc_decoder
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Structured output types
# ---------------------------------------------------------------------------

@dataclass
class TokenSpan:
    """A single decoded token with temporal span information."""
    token: str
    token_id: int
    start_frame: int
    end_frame: int
    confidence: float
    uncertainty: float = 0.0
    origin: str = "sentence"

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    def to_dict(self) -> Dict:
        return {
            "token": self.token,
            "token_id": self.token_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frame_count": self.frame_count,
            "confidence": round(self.confidence, 4),
            "uncertainty": round(self.uncertainty, 4),
            "origin": self.origin,
        }


@dataclass
class DecoderOutput:
    """Complete decoder output with tokens, spans, and diagnostics."""
    tokens: List[str]
    token_ids: List[int]
    sequence_confidence: float
    token_confidences: List[float]
    spans: List[TokenSpan]
    raw_frame_predictions: List[int]
    raw_frame_confidences: List[float]
    raw_frame_uncertainties: List[float] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Join tokens into display string."""
        return " ".join(self.tokens) if self.tokens else ""

    def to_dict(self) -> Dict:
        return {
            "tokens": self.tokens,
            "token_ids": self.token_ids,
            "text": self.text,
            "sequence_confidence": round(self.sequence_confidence, 4),
            "token_confidences": [round(c, 4) for c in self.token_confidences],
            "spans": [s.to_dict() for s in self.spans],
            "num_frames": len(self.raw_frame_predictions),
            "mean_frame_uncertainty": round(
                _mean(self.raw_frame_uncertainties), 4
            ) if self.raw_frame_uncertainties else 0.0,
        }


# ---------------------------------------------------------------------------
# Greedy Decoder
# ---------------------------------------------------------------------------

class GreedyDecoder:
    """
    Greedy CTC decoder with span-aware output.

    Algorithm:
        1. argmax at each frame -> raw prediction sequence
        2. Track contiguous runs of the same token (spans)
        3. For each span, compute mean probability as confidence
        4. Collapse consecutive identical tokens, remove blanks
        5. Return structured output with temporal span info
    """

    def __init__(self, blank_id: int = 0, id2word: Optional[Dict[int, str]] = None):
        self.blank_id = blank_id
        self.id2word = id2word or {}

    def decode(
        self,
        log_probs: torch.Tensor,
        length: Optional[int] = None,
        frame_uncertainties: Optional[torch.Tensor] = None,
    ) -> DecoderOutput:
        """
        Decode a single sequence.

        Args:
            log_probs: (T, vocab_size) -- log-probabilities from the model.
            length:    Actual sequence length (if padded). Uses full T if None.

        Returns:
            DecoderOutput with tokens, spans, and confidences.
        """
        if length is None:
            length = log_probs.size(0)

        log_probs = log_probs[:length]  # (T, V)
        probs = log_probs.exp()         # (T, V) -- actual probabilities
        if frame_uncertainties is None:
            raw_uncertainties = [0.0] * length
        else:
            raw_uncertainties = _prepare_frame_values(frame_uncertainties, length)

        # Step 1: argmax per frame
        best_ids = log_probs.argmax(dim=-1)  # (T,)
        raw_predictions = best_ids.tolist()

        # Per-frame confidence: probability of the best token
        raw_confidences = []
        for t in range(length):
            raw_confidences.append(probs[t, best_ids[t]].item())

        # Step 2: identify contiguous runs (spans)
        runs = []  # (token_id, start_frame, end_frame, [frame_probs], [frame_uncertainties])
        if length > 0:
            current_id = raw_predictions[0]
            current_start = 0
            current_probs = [raw_confidences[0]]
            current_uncs = [raw_uncertainties[0]]

            for t in range(1, length):
                if raw_predictions[t] == current_id:
                    current_probs.append(raw_confidences[t])
                    current_uncs.append(raw_uncertainties[t])
                else:
                    runs.append((current_id, current_start, t, current_probs, current_uncs))
                    current_id = raw_predictions[t]
                    current_start = t
                    current_probs = [raw_confidences[t]]
                    current_uncs = [raw_uncertainties[t]]

            runs.append((current_id, current_start, length, current_probs, current_uncs))

        # Step 3: collapse repeats and remove blanks -> build spans
        spans = []
        token_ids = []
        token_confidences = []
        tokens = []

        prev_id = None
        for token_id, start, end, frame_probs, frame_uncs in runs:
            if token_id == self.blank_id:
                prev_id = token_id
                continue
            if token_id == prev_id:
                # Consecutive repeat of same non-blank token -> merge into previous span
                if spans:
                    spans[-1] = TokenSpan(
                        token=spans[-1].token,
                        token_id=spans[-1].token_id,
                        start_frame=spans[-1].start_frame,
                        end_frame=end,
                        confidence=(spans[-1].confidence + _mean(frame_probs)) / 2.0,
                        uncertainty=(spans[-1].uncertainty + _mean(frame_uncs)) / 2.0,
                        origin=spans[-1].origin,
                    )
                    token_confidences[-1] = spans[-1].confidence
                prev_id = token_id
                continue

            # New non-blank token
            conf = _mean(frame_probs)
            word = self.id2word.get(token_id, f"<id:{token_id}>")

            spans.append(TokenSpan(
                token=word,
                token_id=token_id,
                start_frame=start,
                end_frame=end,
                confidence=conf,
                uncertainty=_mean(frame_uncs),
            ))
            token_ids.append(token_id)
            token_confidences.append(conf)
            tokens.append(word)
            prev_id = token_id

        # Sequence confidence: geometric mean in log space
        if token_confidences:
            log_confs = [math.log(max(c, 1e-10)) for c in token_confidences]
            seq_conf = math.exp(sum(log_confs) / len(log_confs))
        else:
            seq_conf = 0.0

        return DecoderOutput(
            tokens=tokens,
            token_ids=token_ids,
            sequence_confidence=seq_conf,
            token_confidences=token_confidences,
            spans=spans,
            raw_frame_predictions=raw_predictions,
            raw_frame_confidences=raw_confidences,
            raw_frame_uncertainties=raw_uncertainties,
        )

    def decode_batch(
        self,
        log_probs: torch.Tensor,
        lengths: torch.Tensor,
        frame_uncertainties: Optional[torch.Tensor] = None,
    ) -> List[DecoderOutput]:
        """
        Decode a batch of sequences.

        Args:
            log_probs: (B, T, V) -- log-probabilities.
            lengths:   (B,) -- actual sequence lengths.

        Returns:
            List of DecoderOutput, one per batch element.
        """
        results = []
        for b in range(log_probs.size(0)):
            frame_uncertainty = None
            if frame_uncertainties is not None:
                frame_uncertainty = frame_uncertainties[b]
            result = self.decode(
                log_probs[b],
                lengths[b].item(),
                frame_uncertainties=frame_uncertainty,
            )
            results.append(result)
        return results


# ---------------------------------------------------------------------------
# Prefix Beam Search Decoder
# ---------------------------------------------------------------------------

class PrefixBeamDecoder:
    """
    Prefix beam search CTC decoder.

    Maintains top-k prefix hypotheses scored by cumulative log probability.
    Handles blank and repeat transitions correctly per CTC semantics.

    Args:
        blank_id:   CTC blank token ID (default 0).
        beam_width: Number of hypotheses to maintain (default 5).
        id2word:    Vocabulary mapping (token_id -> word string).
    """

    def __init__(
        self,
        blank_id: int = 0,
        beam_width: int = 5,
        id2word: Optional[Dict[int, str]] = None,
    ):
        self.blank_id = blank_id
        self.beam_width = beam_width
        self.id2word = id2word or {}

    def decode(
        self,
        log_probs: torch.Tensor,
        length: Optional[int] = None,
        return_top_k: int = 1,
        frame_uncertainties: Optional[torch.Tensor] = None,
    ) -> DecoderOutput:
        """
        Decode a single sequence using prefix beam search.

        Args:
            log_probs:   (T, vocab_size) -- log-probabilities.
            length:      Actual sequence length.
            return_top_k: Return top-k results (default 1 = best only).

        Returns:
            DecoderOutput for the best hypothesis.
        """
        if length is None:
            length = log_probs.size(0)

        log_probs = log_probs[:length]  # (T, V)
        T, V = log_probs.shape

        # Raw frame predictions for output
        raw_preds = log_probs.argmax(dim=-1).tolist()
        raw_probs = log_probs.exp()
        raw_confs = [raw_probs[t, raw_preds[t]].item() for t in range(T)]
        raw_uncertainties = (
            _prepare_frame_values(frame_uncertainties, T)
            if frame_uncertainties is not None else [0.0] * T
        )

        # Beam state: {prefix_tuple: (log_prob_blank, log_prob_nonblank)}
        NEG_INF = float("-inf")
        beams = {(): (0.0, NEG_INF)}  # empty prefix starts with blank score 0

        for t in range(T):
            new_beams: Dict[tuple, Tuple[float, float]] = {}

            # Prune to beam_width
            scored = [(k, _logsumexp(v[0], v[1])) for k, v in beams.items()]
            scored.sort(key=lambda x: -x[1])
            active = scored[:self.beam_width]

            for prefix, _ in active:
                pb, pnb = beams[prefix]

                for c in range(V):
                    lp = log_probs[t, c].item()

                    if c == self.blank_id:
                        # Blank extends prefix without change
                        key = prefix
                        new_pb = _logsumexp(
                            new_beams.get(key, (NEG_INF, NEG_INF))[0],
                            _logsumexp(pb, pnb) + lp,
                        )
                        new_pnb = new_beams.get(key, (NEG_INF, NEG_INF))[1]
                        new_beams[key] = (new_pb, new_pnb)

                    elif prefix and c == prefix[-1]:
                        # Same char as end of prefix:
                        # Can extend without adding new char (if preceded by blank)
                        key = prefix
                        new_pb_k = new_beams.get(key, (NEG_INF, NEG_INF))[0]
                        new_pnb_k = _logsumexp(
                            new_beams.get(key, (NEG_INF, NEG_INF))[1],
                            pnb + lp,
                        )
                        new_beams[key] = (new_pb_k, new_pnb_k)

                        # Can also add new char (after blank)
                        key2 = prefix + (c,)
                        new_pb_k2 = new_beams.get(key2, (NEG_INF, NEG_INF))[0]
                        new_pnb_k2 = _logsumexp(
                            new_beams.get(key2, (NEG_INF, NEG_INF))[1],
                            pb + lp,
                        )
                        new_beams[key2] = (new_pb_k2, new_pnb_k2)

                    else:
                        # Different char -> extend prefix
                        key = prefix + (c,)
                        new_pb_k = new_beams.get(key, (NEG_INF, NEG_INF))[0]
                        new_pnb_k = _logsumexp(
                            new_beams.get(key, (NEG_INF, NEG_INF))[1],
                            _logsumexp(pb, pnb) + lp,
                        )
                        new_beams[key] = (new_pb_k, new_pnb_k)

            beams = new_beams

        # Get best hypothesis
        scored = [(k, _logsumexp(v[0], v[1])) for k, v in beams.items()]
        scored.sort(key=lambda x: -x[1])

        if not scored or not scored[0][0]:
            return DecoderOutput(
                tokens=[], token_ids=[], sequence_confidence=0.0,
                token_confidences=[], spans=[],
                raw_frame_predictions=raw_preds,
                raw_frame_confidences=raw_confs,
                raw_frame_uncertainties=raw_uncertainties,
            )

        best_prefix = list(scored[0][0])
        best_log_score = scored[0][1]

        # Build spans using greedy alignment (find where each token dominates)
        spans = _build_spans_from_alignment(
            best_prefix,
            log_probs,
            self.blank_id,
            self.id2word,
            raw_uncertainties,
        )

        tokens = [self.id2word.get(tid, f"<id:{tid}>") for tid in best_prefix]
        token_confs = [s.confidence for s in spans]
        seq_conf = math.exp(best_log_score / max(len(best_prefix), 1))

        return DecoderOutput(
            tokens=tokens,
            token_ids=best_prefix,
            sequence_confidence=min(seq_conf, 1.0),
            token_confidences=token_confs,
            spans=spans,
            raw_frame_predictions=raw_preds,
            raw_frame_confidences=raw_confs,
            raw_frame_uncertainties=raw_uncertainties,
        )

    def decode_batch(
        self,
        log_probs: torch.Tensor,
        lengths: torch.Tensor,
        frame_uncertainties: Optional[torch.Tensor] = None,
    ) -> List[DecoderOutput]:
        """Decode a batch of sequences."""
        results = []
        for b in range(log_probs.size(0)):
            frame_uncertainty = None
            if frame_uncertainties is not None:
                frame_uncertainty = frame_uncertainties[b]
            result = self.decode(
                log_probs[b],
                lengths[b].item(),
                frame_uncertainties=frame_uncertainty,
            )
            results.append(result)
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(values: List[float]) -> float:
    """Safe mean of a list of floats."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _logsumexp(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    mx = max(a, b)
    return mx + math.log(math.exp(a - mx) + math.exp(b - mx))


def _build_spans_from_alignment(
    token_ids: List[int],
    log_probs: torch.Tensor,
    blank_id: int,
    id2word: Dict[int, str],
    frame_uncertainties: Optional[List[float]] = None,
) -> List[TokenSpan]:
    """
    Build TokenSpans by finding where each decoded token is most active.

    For each token in the decoded sequence, scan the frame-level predictions
    to find the contiguous region where that token has highest probability
    among the decoded tokens.
    """
    T = log_probs.size(0)
    if not token_ids:
        return []

    probs = log_probs.exp()  # (T, V)

    # Simple approach: find the best alignment using the frame-level argmax
    frame_preds = log_probs.argmax(dim=-1).tolist()

    # Assign frames to tokens greedily
    spans = []
    token_idx = 0
    frame_idx = 0

    for i, tid in enumerate(token_ids):
        # Find start: skip blanks and previous tokens
        start = frame_idx
        while start < T and frame_preds[start] == blank_id:
            start += 1
        if start >= T:
            start = frame_idx

        # Find end: advance while this token is predicted or blank
        end = start
        while end < T:
            if frame_preds[end] == tid:
                end += 1
            elif frame_preds[end] == blank_id:
                end += 1
            else:
                break

        if end == start:
            end = min(start + 1, T)

        # Compute confidence as mean prob of this token over its span
        span_probs = [probs[t, tid].item() for t in range(start, end)]
        conf = _mean(span_probs)
        unc = _mean(frame_uncertainties[start:end]) if frame_uncertainties else 0.0
        word = id2word.get(tid, f"<id:{tid}>")

        spans.append(TokenSpan(
            token=word,
            token_id=tid,
            start_frame=start,
            end_frame=end,
            confidence=conf,
            uncertainty=unc,
        ))

        frame_idx = end

    return spans


def _prepare_frame_values(
    values: torch.Tensor,
    length: int,
) -> List[float]:
    """Convert per-frame tensor values to a Python list with the right length."""
    if values.dim() > 1:
        values = values.squeeze()
    values = values[:length]
    return values.detach().cpu().tolist()


# ---------------------------------------------------------------------------
# Standalone verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("inference/ctc_decoder.py -- verification")
    print("=" * 60)

    # Create synthetic log-probs: 50 frames, vocab=10
    T, V = 50, 10
    torch.manual_seed(42)
    logits = torch.randn(T, V)
    log_probs = F.log_softmax(logits, dim=-1)

    # Inject a clear pattern: token 3 for frames 5-15, token 7 for 20-30
    log_probs[5:15, :] = -10.0
    log_probs[5:15, 3] = -0.01
    log_probs[20:30, :] = -10.0
    log_probs[20:30, 7] = -0.01
    log_probs = F.log_softmax(log_probs, dim=-1)  # re-normalize

    id2word = {0: "<blank>", 3: "hello", 7: "world"}

    # Test greedy decoder
    greedy = GreedyDecoder(blank_id=0, id2word=id2word)
    result = greedy.decode(log_probs)
    print(f"  Greedy result: {result.text}")
    print(f"  Token IDs:     {result.token_ids}")
    print(f"  Seq confidence:{result.sequence_confidence:.4f}")
    print(f"  Spans:")
    for s in result.spans:
        print(f"    {s.token:<10} frames [{s.start_frame}-{s.end_frame}) "
              f"conf={s.confidence:.4f}")

    assert 3 in result.token_ids, "Expected token 3 (hello)"
    assert 7 in result.token_ids, "Expected token 7 (world)"

    # Test beam search decoder
    beam = PrefixBeamDecoder(blank_id=0, beam_width=5, id2word=id2word)
    beam_result = beam.decode(log_probs)
    print(f"\n  Beam result:   {beam_result.text}")
    print(f"  Token IDs:     {beam_result.token_ids}")
    print(f"  Seq confidence:{beam_result.sequence_confidence:.4f}")

    # Test batch decoding
    batch_lp = log_probs.unsqueeze(0).expand(4, -1, -1)  # (4, 50, 10)
    lengths = torch.full((4,), T, dtype=torch.int32)
    batch_results = greedy.decode_batch(batch_lp, lengths)
    assert len(batch_results) == 4
    print(f"\n  Batch decode: {len(batch_results)} results")

    print("=" * 60)
    print("[PASS] inference/ctc_decoder.py OK")
