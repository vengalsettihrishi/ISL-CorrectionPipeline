"""
evaluation.py — Full Evaluation Pipeline for ISL Correction

Runs the complete evaluation protocol:
1. Takes the test sentence bank (ISL gloss → English pairs)
2. Injects noise at 10%, 20%, 30% rates
3. Runs noisy input through: no correction, rules-only, rules+LM
4. Computes BLEU-4 and chrF scores against expected English output
5. Runs ablation study (disable each rule, measure BLEU drop)
6. Outputs publication-ready results tables

USAGE:
    python evaluation.py                           # Full evaluation
    python evaluation.py --kenlm_model english.bin # With KenLM model
"""

import sys
import json
import math
import argparse
from typing import List, Tuple, Dict, Optional
from collections import Counter
from dataclasses import dataclass

# Add current directory to path for imports
sys.path.insert(0, ".")

from corrector import ISLCorrector
from noise_injector import NoiseInjector, get_test_bank
from lm_scorer import CorrectionPipeline


# ============================================================================
# Metrics (implemented from scratch — no sklearn/nltk dependency)
# ============================================================================

def compute_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """
    Compute BLEU score between a reference and hypothesis sentence.

    Implementation follows the original BLEU paper by Papineni et al. (2002).

    Args:
        reference: Expected English sentence.
        hypothesis: Generated English sentence.
        max_n: Maximum n-gram order (default 4 for BLEU-4).

    Returns:
        BLEU score between 0.0 and 1.0.
    """
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    if not hyp_tokens or not ref_tokens:
        return 0.0

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))

    # Modified n-gram precision for each n
    log_precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = _get_ngrams(ref_tokens, n)
        hyp_ngrams = _get_ngrams(hyp_tokens, n)

        if not hyp_ngrams:
            return 0.0

        # Clipped counts
        clipped = 0
        for ngram, count in hyp_ngrams.items():
            clipped += min(count, ref_ngrams.get(ngram, 0))

        total = sum(hyp_ngrams.values())
        precision = clipped / max(total, 1)

        if precision == 0:
            return 0.0

        log_precisions.append(math.log(precision))

    # Geometric mean of precisions
    avg_log_precision = sum(log_precisions) / max_n
    bleu = bp * math.exp(avg_log_precision)

    return bleu


def compute_chrf(reference: str, hypothesis: str, n: int = 6, beta: float = 2.0) -> float:
    """
    Compute chrF score (character-level F-score).

    More stable than BLEU for short sentences — recommended as a secondary metric.

    Args:
        reference: Expected English sentence.
        hypothesis: Generated English sentence.
        n: Maximum character n-gram order.
        beta: Weighting of recall vs precision (2.0 = recall weighted).

    Returns:
        chrF score between 0.0 and 1.0.
    """
    ref = reference.lower().strip()
    hyp = hypothesis.lower().strip()

    if not ref or not hyp:
        return 0.0

    precisions = []
    recalls = []

    for order in range(1, n + 1):
        ref_ngrams = _get_char_ngrams(ref, order)
        hyp_ngrams = _get_char_ngrams(hyp, order)

        if not hyp_ngrams or not ref_ngrams:
            continue

        # Precision: how many hyp n-grams appear in ref
        matched = 0
        for ng, count in hyp_ngrams.items():
            matched += min(count, ref_ngrams.get(ng, 0))
        prec = matched / max(sum(hyp_ngrams.values()), 1)

        # Recall: how many ref n-grams appear in hyp
        matched_r = 0
        for ng, count in ref_ngrams.items():
            matched_r += min(count, hyp_ngrams.get(ng, 0))
        rec = matched_r / max(sum(ref_ngrams.values()), 1)

        precisions.append(prec)
        recalls.append(rec)

    if not precisions:
        return 0.0

    avg_prec = sum(precisions) / len(precisions)
    avg_rec = sum(recalls) / len(recalls)

    if avg_prec + avg_rec == 0:
        return 0.0

    # F-score with beta weighting
    chrf = (1 + beta ** 2) * (avg_prec * avg_rec) / (beta ** 2 * avg_prec + avg_rec)
    return chrf


# Tokenization helpers

def _tokenize(sentence: str) -> List[str]:
    """Simple whitespace tokenization with punctuation handling."""
    s = sentence.lower().strip()
    # Separate punctuation
    s = s.replace(".", " ").replace("?", " ").replace("!", " ").replace(",", " ")
    return [w for w in s.split() if w]


def _get_ngrams(tokens: List[str], n: int) -> Dict[tuple, int]:
    """Extract n-grams from token list."""
    ngrams = Counter()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams[ngram] += 1
    return ngrams


def _get_char_ngrams(text: str, n: int) -> Dict[str, int]:
    """Extract character n-grams from text."""
    ngrams = Counter()
    for i in range(len(text) - n + 1):
        ngrams[text[i:i + n]] += 1
    return ngrams


# ============================================================================
# Evaluation Runner
# ============================================================================

@dataclass
class EvalResult:
    """Results for one configuration at one noise level."""
    config_name: str
    noise_rate: float
    avg_bleu: float
    avg_chrf: float
    n_samples: int


def evaluate_full(
    kenlm_model_path: Optional[str] = None,
    noise_rates: List[float] = [0.0, 0.1, 0.2, 0.3],
    seed: int = 42,
) -> List[EvalResult]:
    """
    Run the full evaluation protocol.

    Tests three configurations at each noise level:
    1. No correction (raw concatenation of noisy gloss)
    2. Rules only
    3. Rules + LM rescoring

    Returns list of EvalResult for each config × noise_rate combination.
    """
    test_bank = get_test_bank()
    injector = NoiseInjector(seed=seed)
    corrector = ISLCorrector()
    pipeline = CorrectionPipeline(corrector, kenlm_model_path)

    results = []

    for rate in noise_rates:
        bleu_no_correction = []
        bleu_rules_only = []
        bleu_rules_lm = []
        chrf_no_correction = []
        chrf_rules_only = []
        chrf_rules_lm = []

        for clean_tokens, expected_english in test_bank:
            # Apply noise (or keep clean for rate=0)
            if rate > 0:
                noisy = injector.inject(clean_tokens, rate)
            else:
                noisy = list(clean_tokens)

            # Config 1: No correction — just lowercase and join the gloss
            raw_output = " ".join(t.lower() for t in noisy) + "."

            # Config 2: Rules only
            rules_output = corrector.correct(noisy)

            # Config 3: Rules + LM rescoring
            lm_output = pipeline.correct(noisy)

            # Compute metrics
            bleu_no_correction.append(compute_bleu(expected_english, raw_output))
            bleu_rules_only.append(compute_bleu(expected_english, rules_output))
            bleu_rules_lm.append(compute_bleu(expected_english, lm_output))

            chrf_no_correction.append(compute_chrf(expected_english, raw_output))
            chrf_rules_only.append(compute_chrf(expected_english, rules_output))
            chrf_rules_lm.append(compute_chrf(expected_english, lm_output))

        n = len(test_bank)
        results.append(EvalResult("No correction", rate,
                                  sum(bleu_no_correction) / n,
                                  sum(chrf_no_correction) / n, n))
        results.append(EvalResult("Rules only", rate,
                                  sum(bleu_rules_only) / n,
                                  sum(chrf_rules_only) / n, n))
        results.append(EvalResult("Rules + LM", rate,
                                  sum(bleu_rules_lm) / n,
                                  sum(chrf_rules_lm) / n, n))

    return results


def run_ablation(
    noise_rate: float = 0.1,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Ablation study: disable each rule individually and measure BLEU drop.

    Returns dict mapping rule_id → BLEU score with that rule disabled.
    """
    test_bank = get_test_bank()
    injector = NoiseInjector(seed=seed)

    rules = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]
    rule_names = {
        "R1": "Pronoun Normalization",
        "R2": "WH-Word Fronting",
        "R3": "Negation Reordering",
        "R4": "SOV→SVO Reordering",
        "R5": "Adjective-Noun Reorder",
        "R6": "Verb Conjugation",
        "R7": "Auxiliary Insertion",
        "R8": "Capitalization & Punct",
    }

    results = {}

    # Baseline: all rules enabled
    corrector = ISLCorrector()
    bleu_scores = []
    for clean_tokens, expected in test_bank:
        noisy = injector.inject(clean_tokens, noise_rate)
        output = corrector.correct(noisy)
        bleu_scores.append(compute_bleu(expected, output))
    baseline_bleu = sum(bleu_scores) / len(bleu_scores)
    results["All rules"] = baseline_bleu

    # Disable each rule one at a time
    for rule_id in rules:
        corrector = ISLCorrector(disabled_rules={rule_id})
        injector_copy = NoiseInjector(seed=seed)  # Reset seed for consistency
        bleu_scores = []
        for clean_tokens, expected in test_bank:
            noisy = injector_copy.inject(clean_tokens, noise_rate)
            output = corrector.correct(noisy)
            bleu_scores.append(compute_bleu(expected, output))
        rule_bleu = sum(bleu_scores) / len(bleu_scores)
        results[f"Without {rule_id} ({rule_names[rule_id]})"] = rule_bleu

    return results


# ============================================================================
# Display & Reporting
# ============================================================================

def print_results_table(results: List[EvalResult]):
    """Print a publication-ready results table."""
    print("\n" + "=" * 75)
    print("TABLE: Correction Quality at Different Noise Levels")
    print("=" * 75)

    # Group by noise rate
    rates = sorted(set(r.noise_rate for r in results))
    configs = sorted(set(r.config_name for r in results),
                     key=lambda c: ["No correction", "Rules only", "Rules + LM"].index(c))

    # Header
    header = f"{'Configuration':<20}"
    for rate in rates:
        header += f" | {'BLEU@'+str(int(rate*100))+'%':>10} {'chrF@'+str(int(rate*100))+'%':>10}"
    print(header)
    print("-" * len(header))

    # Rows
    for config in configs:
        row = f"{config:<20}"
        for rate in rates:
            matching = [r for r in results if r.config_name == config and r.noise_rate == rate]
            if matching:
                r = matching[0]
                row += f" | {r.avg_bleu:>10.4f} {r.avg_chrf:>10.4f}"
        print(row)

    print("-" * len(header))
    print(f"N = {results[0].n_samples} sentence pairs per cell")


def print_ablation_table(ablation: Dict[str, float]):
    """Print ablation study results."""
    print("\n" + "=" * 60)
    print("TABLE: Ablation Study — BLEU Impact of Each Rule")
    print("=" * 60)

    baseline = ablation.get("All rules", 0)
    print(f"{'Configuration':<45} {'BLEU':>7} {'Drop':>7}")
    print("-" * 60)

    for name, bleu in sorted(ablation.items(), key=lambda x: -x[1]):
        drop = baseline - bleu
        marker = "" if name == "All rules" else f"{drop:>+7.4f}"
        print(f"{name:<45} {bleu:>7.4f} {marker}")

    print("-" * 60)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate ISL correction pipeline")
    parser.add_argument("--kenlm_model", type=str, default=None,
                        help="Path to KenLM model file (.bin or .arpa)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablation_noise", type=float, default=0.1,
                        help="Noise rate for ablation study")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    # --- Full evaluation ---
    print("\n" + "=" * 75)
    print("ISL CORRECTION PIPELINE — FULL EVALUATION")
    print("=" * 75)

    results = evaluate_full(
        kenlm_model_path=args.kenlm_model,
        noise_rates=[0.0, 0.1, 0.2, 0.3],
        seed=args.seed,
    )
    print_results_table(results)

    # --- Ablation study ---
    ablation = run_ablation(noise_rate=args.ablation_noise, seed=args.seed)
    print_ablation_table(ablation)

    # --- Sample outputs ---
    print("\n" + "=" * 75)
    print("SAMPLE OUTPUTS (at 20% noise)")
    print("=" * 75)

    test_bank = get_test_bank()[:10]
    injector = NoiseInjector(seed=args.seed)
    corrector = ISLCorrector()
    pipeline = CorrectionPipeline(corrector, args.kenlm_model)

    print(f"\n{'Clean ISL':<30} {'Noisy ISL':<30} {'Corrected':<30} {'Expected'}")
    print("-" * 120)

    for clean_tokens, expected in test_bank:
        noisy = injector.inject(clean_tokens, 0.2)
        corrected = pipeline.correct(noisy)
        print(f"{' '.join(clean_tokens):<30} {' '.join(noisy):<30} {corrected:<30} {expected}")

    # --- Save to JSON ---
    if args.output_json:
        output = {
            "results": [
                {
                    "config": r.config_name,
                    "noise_rate": r.noise_rate,
                    "bleu": round(r.avg_bleu, 4),
                    "chrf": round(r.avg_chrf, 4),
                    "n_samples": r.n_samples,
                }
                for r in results
            ],
            "ablation": {k: round(v, 4) for k, v in ablation.items()},
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
