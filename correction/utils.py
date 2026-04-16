"""
correction/utils.py -- Metric computation helpers for Sprint 4 benchmark.

Pure-Python implementations of:
    - BLEU-4 (with smoothing)
    - chrF (character F-score)
    - WER  (word error rate via edit distance)

No external dependencies (no nltk, no sacrebleu required).

Usage:
    from correction.utils import compute_bleu, compute_chrf, compute_wer
    bleu = compute_bleu("I ate food yesterday", "I ate food yesterday")
    chrf = compute_chrf("I ate food yesterday", "I ate food yesterday")
    wer  = compute_wer("I ate food yesterday", "I ate food yesterday")
"""

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple


# ============================================================================
# BLEU-4 (sentence-level with +1 smoothing)
# ============================================================================

def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """Extract n-grams from a token list."""
    return Counter(
        tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)
    )


def compute_bleu(
    hypothesis: str,
    reference: str,
    max_n: int = 4,
    smooth: bool = True,
) -> float:
    """
    Compute sentence-level BLEU score.

    Uses +1 smoothing (Lin & Och, 2004) to avoid zero scores for
    short sentences missing higher-order n-grams.

    Args:
        hypothesis: Predicted English sentence.
        reference:  Reference English sentence.
        max_n:      Maximum n-gram order (default 4 = BLEU-4).
        smooth:     Apply +1 smoothing.

    Returns:
        BLEU score in [0, 1].
    """
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()

    if not hyp_tokens or not ref_tokens:
        return 0.0

    # Brevity penalty
    bp = min(1.0, math.exp(1.0 - len(ref_tokens) / max(len(hyp_tokens), 1)))

    # N-gram precisions
    log_avg = 0.0
    for n in range(1, max_n + 1):
        hyp_ngrams = _get_ngrams(hyp_tokens, n)
        ref_ngrams = _get_ngrams(ref_tokens, n)

        # Clipped counts
        clipped = 0
        total = 0
        for ngram, count in hyp_ngrams.items():
            clipped += min(count, ref_ngrams.get(ngram, 0))
            total += count

        if smooth:
            # +1 smoothing
            precision = (clipped + 1) / (total + 1)
        else:
            if total == 0:
                return 0.0
            precision = clipped / total
            if precision == 0:
                return 0.0

        log_avg += math.log(max(precision, 1e-10)) / max_n

    return bp * math.exp(log_avg)


def compute_bleu_corpus(
    hypotheses: List[str],
    references: List[str],
    max_n: int = 4,
) -> float:
    """
    Compute corpus-level BLEU-4.

    Args:
        hypotheses: List of predicted sentences.
        references: List of reference sentences.
        max_n:      Maximum n-gram order.

    Returns:
        Corpus BLEU score in [0, 1].
    """
    if not hypotheses or not references:
        return 0.0

    total_clipped = [0] * max_n
    total_count = [0] * max_n
    total_hyp_len = 0
    total_ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_tokens = hyp.lower().split()
        ref_tokens = ref.lower().split()
        total_hyp_len += len(hyp_tokens)
        total_ref_len += len(ref_tokens)

        for n in range(1, max_n + 1):
            hyp_ngrams = _get_ngrams(hyp_tokens, n)
            ref_ngrams = _get_ngrams(ref_tokens, n)

            for ngram, count in hyp_ngrams.items():
                total_clipped[n - 1] += min(count, ref_ngrams.get(ngram, 0))
                total_count[n - 1] += count

    # Brevity penalty
    if total_hyp_len == 0:
        return 0.0
    bp = min(1.0, math.exp(1.0 - total_ref_len / total_hyp_len))

    # Geometric mean of precisions
    log_avg = 0.0
    for n in range(max_n):
        if total_count[n] == 0:
            return 0.0
        precision = (total_clipped[n] + 1) / (total_count[n] + 1)
        log_avg += math.log(precision) / max_n

    return bp * math.exp(log_avg)


# ============================================================================
# chrF (character n-gram F-score)
# ============================================================================

def _get_char_ngrams(text: str, n: int) -> Counter:
    """Extract character n-grams from a text string."""
    return Counter(text[i:i + n] for i in range(len(text) - n + 1))


def compute_chrf(
    hypothesis: str,
    reference: str,
    max_n: int = 6,
    beta: float = 2.0,
) -> float:
    """
    Compute chrF score (character n-gram F-score).

    Args:
        hypothesis: Predicted English sentence.
        reference:  Reference English sentence.
        max_n:      Maximum character n-gram order (default 6).
        beta:       F-score beta parameter (default 2.0 = recall-weighted).

    Returns:
        chrF score in [0, 1].
    """
    hyp = hypothesis.lower().strip()
    ref = reference.lower().strip()

    if not hyp or not ref:
        return 0.0

    avg_precision = 0.0
    avg_recall = 0.0
    counted = 0

    for n in range(1, max_n + 1):
        hyp_ngrams = _get_char_ngrams(hyp, n)
        ref_ngrams = _get_char_ngrams(ref, n)

        hyp_total = sum(hyp_ngrams.values())
        ref_total = sum(ref_ngrams.values())

        if hyp_total == 0 or ref_total == 0:
            continue

        # Clipped counts
        clipped = sum(
            min(hyp_ngrams[ng], ref_ngrams[ng])
            for ng in hyp_ngrams if ng in ref_ngrams
        )

        precision = clipped / hyp_total
        recall = clipped / ref_total
        avg_precision += precision
        avg_recall += recall
        counted += 1

    if counted == 0:
        return 0.0

    avg_precision /= counted
    avg_recall /= counted

    if avg_precision + avg_recall == 0:
        return 0.0

    beta_sq = beta ** 2
    chrf = (1 + beta_sq) * avg_precision * avg_recall / (
        beta_sq * avg_precision + avg_recall
    )

    return chrf


# ============================================================================
# WER (word error rate)
# ============================================================================

def _edit_distance(ref: List[str], hyp: List[str]) -> int:
    """Levenshtein edit distance between two word lists."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]


def compute_wer(
    hypothesis: str,
    reference: str,
) -> float:
    """
    Compute Word Error Rate (WER).

    WER = edit_distance(ref_words, hyp_words) / len(ref_words)

    Args:
        hypothesis: Predicted English sentence.
        reference:  Reference English sentence.

    Returns:
        WER as a float (0.0 = perfect, > 1.0 possible with insertions).
    """
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()

    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0

    dist = _edit_distance(ref_tokens, hyp_tokens)
    return dist / len(ref_tokens)


def compute_exact_match(hypothesis: str, reference: str) -> bool:
    """Check if hypothesis exactly matches reference (case-insensitive)."""
    return hypothesis.lower().strip() == reference.lower().strip()


# ============================================================================
# Aggregate metrics
# ============================================================================

def compute_aggregate_metrics(
    hypotheses: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute all metrics over a corpus.

    Args:
        hypotheses: List of predicted sentences.
        references: List of reference sentences.

    Returns:
        Dict with bleu4, chrf, wer, exact_match_rate.
    """
    if not hypotheses or not references:
        return {"bleu4": 0.0, "chrf": 0.0, "wer": 1.0, "exact_match_rate": 0.0}

    n = len(hypotheses)

    # Sentence-level metrics (averaged)
    bleu_scores = [compute_bleu(h, r) for h, r in zip(hypotheses, references)]
    chrf_scores = [compute_chrf(h, r) for h, r in zip(hypotheses, references)]
    wer_scores = [compute_wer(h, r) for h, r in zip(hypotheses, references)]
    exact_matches = [compute_exact_match(h, r) for h, r in zip(hypotheses, references)]

    # Corpus-level BLEU
    corpus_bleu = compute_bleu_corpus(hypotheses, references)

    return {
        "bleu4_sentence_avg": sum(bleu_scores) / n,
        "bleu4_corpus": corpus_bleu,
        "chrf_avg": sum(chrf_scores) / n,
        "wer_avg": sum(wer_scores) / n,
        "exact_match_rate": sum(exact_matches) / n,
        "num_samples": n,
    }


# ============================================================================
# Standalone verification
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("correction/utils.py -- verification")
    print("=" * 60)

    # Test BLEU
    h1 = "I ate food yesterday"
    r1 = "I ate food yesterday"
    bleu_perfect = compute_bleu(h1, r1)
    print(f"  BLEU (perfect):   {bleu_perfect:.4f}")
    assert bleu_perfect > 0.9, f"Expected high BLEU for identical, got {bleu_perfect}"

    h2 = "I eat food today"
    r2 = "I ate food yesterday"
    bleu_partial = compute_bleu(h2, r2)
    print(f"  BLEU (partial):   {bleu_partial:.4f}")
    assert 0.0 < bleu_partial < 1.0

    # Test chrF
    chrf_perfect = compute_chrf(h1, r1)
    print(f"  chrF (perfect):   {chrf_perfect:.4f}")
    assert chrf_perfect > 0.9

    chrf_partial = compute_chrf(h2, r2)
    print(f"  chrF (partial):   {chrf_partial:.4f}")
    assert 0.0 < chrf_partial < 1.0

    # Test WER
    wer_perfect = compute_wer(h1, r1)
    print(f"  WER  (perfect):   {wer_perfect:.4f}")
    assert wer_perfect == 0.0

    wer_partial = compute_wer(h2, r2)
    print(f"  WER  (partial):   {wer_partial:.4f}")
    assert wer_partial > 0.0

    wer_total = compute_wer("completely wrong sentence", "I ate food yesterday")
    print(f"  WER  (mismatch):  {wer_total:.4f}")

    # Test exact match
    assert compute_exact_match("Hello World", "hello world") is True
    assert compute_exact_match("Hello", "World") is False
    print("  Exact match:      PASS")

    # Test corpus BLEU
    hyps = ["I ate food yesterday", "She is happy", "What is your name"]
    refs = ["I ate food yesterday", "She is happy", "What is your name"]
    corpus_bleu = compute_bleu_corpus(hyps, refs)
    print(f"  Corpus BLEU:      {corpus_bleu:.4f}")
    assert corpus_bleu > 0.8

    # Test aggregate
    agg = compute_aggregate_metrics(hyps, refs)
    print(f"  Aggregate:        {agg}")

    print("=" * 60)
    print("[PASS] correction/utils.py OK")
