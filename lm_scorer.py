"""
lm_scorer.py — Language Model Rescoring Layer (Layer 2)

Takes the output of the rule-based corrector and improves it by:
1. Generating alternative candidate sentences (varying aux, tense, order)
2. Scoring each candidate with a pre-trained KenLM n-gram model
3. Returning the candidate with the lowest perplexity (most natural English)

SETUP (run once):
    pip install kenlm
    # Download a pre-trained English language model:
    wget -O english.arpa.bin https://kheafield.com/code/kenlm/benchmark/

    Or build your own from any large English corpus using KenLM's lmplz tool.

USAGE:
    from lm_scorer import LMScorer
    scorer = LMScorer("english.arpa.bin")
    best = scorer.rescore("I ate food yesterday.", ["I food ate yesterday.", "I eaten food yesterday."])
    print(best)  # "I ate food yesterday."
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import re
import sys


# ============================================================================
# KenLM wrapper (graceful fallback if not installed)
# ============================================================================

try:
    import kenlm
    KENLM_AVAILABLE = True
except ImportError:
    KENLM_AVAILABLE = False


@dataclass
class ScoredCandidate:
    """A candidate sentence with its language model score."""
    sentence: str
    score: float          # Log probability (higher = more likely)
    perplexity: float     # Perplexity (lower = more natural)
    source: str           # Where this candidate came from


class LMScorer:
    """
    Language model rescoring for the ISL correction pipeline.

    Wraps a KenLM n-gram model to score English sentences.
    If KenLM is not available, falls back to a simple heuristic scorer
    so the pipeline still works (just without LM benefit).
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Path to a .arpa or .bin KenLM model file.
                       If None or KenLM not installed, uses fallback scorer.
        """
        self.model = None
        self.using_kenlm = False

        if model_path and KENLM_AVAILABLE:
            try:
                self.model = kenlm.Model(model_path)
                self.using_kenlm = True
                print(f"KenLM loaded: {model_path}")
                print(f"  Order: {self.model.order}")
            except Exception as e:
                print(f"Warning: Could not load KenLM model: {e}")
                print("  Falling back to heuristic scorer.")
        elif not KENLM_AVAILABLE:
            print("KenLM not installed. Using heuristic fallback scorer.")
            print("  Install with: pip install kenlm")

    def score_sentence(self, sentence: str) -> ScoredCandidate:
        """
        Score a single sentence.

        Returns:
            ScoredCandidate with log probability and perplexity.
        """
        clean = sentence.strip().rstrip(".?!").strip()

        if self.using_kenlm and self.model:
            log_prob = self.model.score(clean, bos=True, eos=True)
            # Perplexity = 10^(-log_prob / word_count)
            words = clean.split()
            n_words = max(len(words), 1)
            perplexity = 10 ** (-log_prob / n_words)
        else:
            # Heuristic fallback: score based on simple English patterns
            log_prob, perplexity = self._heuristic_score(clean)

        return ScoredCandidate(
            sentence=sentence,
            score=log_prob,
            perplexity=perplexity,
            source="kenlm" if self.using_kenlm else "heuristic",
        )

    def rescore(
        self,
        primary: str,
        alternatives: List[str],
        return_all: bool = False,
    ) -> str:
        """
        Score the primary candidate and alternatives, return the best.

        Args:
            primary: The main candidate from the rule-based corrector.
            alternatives: Additional candidate sentences to consider.
            return_all: If True, return list of all scored candidates.

        Returns:
            The sentence with the lowest perplexity (most natural).
        """
        all_candidates = [primary] + alternatives
        scored = [self.score_sentence(s) for s in all_candidates]
        scored.sort(key=lambda c: c.perplexity)

        if return_all:
            return scored

        return scored[0].sentence

    def score_and_compare(self, candidates: List[str]) -> List[ScoredCandidate]:
        """Score multiple candidates and return sorted by perplexity (best first)."""
        scored = [self.score_sentence(s) for s in candidates]
        scored.sort(key=lambda c: c.perplexity)
        return scored

    # -----------------------------------------------------------------------
    # Heuristic fallback scorer (used when KenLM is not installed)
    # -----------------------------------------------------------------------

    def _heuristic_score(self, sentence: str) -> Tuple[float, float]:
        """
        Simple heuristic scoring when KenLM is unavailable.

        Checks for common English patterns and penalizes anomalies.
        This is NOT a replacement for a real language model — it's a
        fallback so the pipeline doesn't crash without KenLM.

        Returns:
            (log_prob_estimate, perplexity_estimate)
        """
        words = sentence.lower().split()
        score = 0.0

        if not words:
            return -100.0, 1000.0

        # Reward: sentence starts with a common subject
        common_starts = {"i", "he", "she", "it", "we", "they", "you",
                         "the", "a", "an", "what", "where", "when",
                         "why", "who", "how", "this", "that"}
        if words[0] in common_starts:
            score += 2.0

        # Penalize: consecutive identical words
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                score -= 5.0

        # Penalize: ALL CAPS words remaining (not properly lowercased)
        for w in sentence.split():
            if w.isupper() and len(w) > 1 and w != "I":
                score -= 3.0

        # Reward: has a verb-like word
        verb_indicators = {"is", "are", "was", "were", "do", "does", "did",
                          "will", "can", "has", "have", "had"}
        if any(w in verb_indicators for w in words):
            score += 1.5

        # Reward: reasonable sentence length
        if 3 <= len(words) <= 15:
            score += 1.0

        # Penalize: "not" appearing at the very end (ISL pattern leaked through)
        if words[-1] == "not":
            score -= 4.0

        # Penalize: subject immediately followed by another subject
        subjects = {"i", "he", "she", "it", "we", "they", "you"}
        for i in range(len(words) - 1):
            if words[i] in subjects and words[i + 1] in subjects:
                score -= 5.0

        # Normalize to pseudo-perplexity
        perplexity = max(1.0, 50.0 - score)

        return score, perplexity


# ============================================================================
# Candidate Generation
# ============================================================================

class CandidateGenerator:
    """
    Generates alternative English sentences from rule-based output.

    These alternatives vary in auxiliary usage, tense, and minor phrasing
    so the language model can pick the most natural one.
    """

    def __init__(self):
        self.aux_variants = {
            "do not": ["don't"],
            "does not": ["doesn't"],
            "did not": ["didn't"],
            "will not": ["won't"],
            "can not": ["cannot", "can't"],
            "is not": ["isn't"],
            "are not": ["aren't"],
            "was not": ["wasn't"],
            "were not": ["weren't"],
        }

    def generate(self, sentence: str, isl_tokens: List[str] = None) -> List[str]:
        """
        Generate alternative phrasings from a rule-based corrected sentence.

        Args:
            sentence: The primary output from rule-based correction.
            isl_tokens: Original ISL tokens (for context).

        Returns:
            List of alternative sentences (excluding the primary).
        """
        candidates = []

        # 1. Contraction variants
        candidates.extend(self._contraction_variants(sentence))

        # 2. Auxiliary insertion variants
        candidates.extend(self._auxiliary_variants(sentence))

        # 3. Article insertion variants
        candidates.extend(self._article_variants(sentence))

        # Remove duplicates and the original
        candidates = list(set(c for c in candidates
                             if c.lower().strip() != sentence.lower().strip()))

        return candidates

    def _contraction_variants(self, sentence: str) -> List[str]:
        """Generate contracted and expanded forms."""
        variants = []
        lower = sentence.lower()

        for full, contractions in self.aux_variants.items():
            if full in lower:
                for contraction in contractions:
                    variant = sentence
                    # Case-insensitive replacement preserving original casing
                    idx = lower.find(full)
                    if idx >= 0:
                        variant = sentence[:idx] + contraction + sentence[idx + len(full):]
                        variants.append(variant)

        return variants

    def _auxiliary_variants(self, sentence: str) -> List[str]:
        """Try inserting/changing auxiliaries."""
        variants = []
        words = sentence.split()

        if len(words) < 2:
            return variants

        # Try inserting "do/does/did" before the verb if not present
        auxiliaries_present = {"do", "does", "did", "is", "am", "are",
                              "was", "were", "will", "can", "has", "have"}
        has_aux = any(w.lower() in auxiliaries_present for w in words)

        if not has_aux and len(words) >= 2:
            # Try inserting "do" after the first word (subject)
            variant = words[0] + " do " + " ".join(words[1:])
            variants.append(variant)

            # Try "does" for third person
            first_lower = words[0].lower()
            if first_lower in {"he", "she", "it"}:
                variant = words[0] + " does " + " ".join(words[1:])
                variants.append(variant)

        return variants

    def _article_variants(self, sentence: str) -> List[str]:
        """Try inserting articles before nouns."""
        variants = []
        words = sentence.split()

        # Simple heuristic: insert "the" or "a" before a lone noun
        # that follows a verb
        for i in range(1, len(words)):
            prev = words[i - 1].lower()
            curr = words[i].lower()

            # If prev is a verb-like word and curr doesn't start with article
            if (prev not in {"a", "an", "the", "my", "his", "her", "your",
                            "their", "our", "this", "that"}
                    and curr not in {"a", "an", "the", "i", "he", "she", "it",
                                    "we", "they", "you"}
                    and not curr.endswith(".")
                    and not curr.endswith("?")):

                # Try inserting "a"
                new_words = words[:i] + ["a"] + words[i:]
                variants.append(" ".join(new_words))

                # Only generate a few to keep it manageable
                if len(variants) >= 3:
                    break

        return variants


# ============================================================================
# Combined Pipeline: Rules + LM Rescoring
# ============================================================================

class CorrectionPipeline:
    """
    Complete correction pipeline: Rule-based correction → Candidate generation → LM rescoring.

    This is the main class your system should use.
    """

    def __init__(self, corrector, lm_model_path: Optional[str] = None):
        """
        Args:
            corrector: An ISLCorrector instance (from corrector.py).
            lm_model_path: Path to KenLM model file. None = heuristic fallback.
        """
        self.corrector = corrector
        self.scorer = LMScorer(lm_model_path)
        self.generator = CandidateGenerator()

    def correct(self, isl_tokens: List[str], verbose: bool = False) -> str:
        """
        Full pipeline: ISL tokens → Rule correction → Candidate generation → LM rescoring.

        Args:
            isl_tokens: ISL gloss tokens, e.g., ["I", "FOOD", "EAT", "YESTERDAY"]
            verbose: Print intermediate results.

        Returns:
            Best English sentence.
        """
        # Step 1: Rule-based correction
        primary = self.corrector.correct(isl_tokens)

        if verbose:
            print(f"  Input:   {' '.join(isl_tokens)}")
            print(f"  Rules:   {primary}")

        # Step 2: Generate alternatives
        alternatives = self.generator.generate(primary, isl_tokens)

        if verbose and alternatives:
            print(f"  Alternatives: {len(alternatives)}")
            for a in alternatives[:5]:
                print(f"    - {a}")

        # Step 3: Score and pick best
        if alternatives:
            all_scored = self.scorer.score_and_compare([primary] + alternatives)

            if verbose:
                print(f"  Scores:")
                for s in all_scored[:5]:
                    marker = " ← best" if s.sentence == all_scored[0].sentence else ""
                    print(f"    {s.perplexity:>8.1f}  {s.sentence}{marker}")

            return all_scored[0].sentence
        else:
            return primary

    def correct_rules_only(self, isl_tokens: List[str]) -> str:
        """Run only the rule-based corrector (no LM rescoring)."""
        return self.corrector.correct(isl_tokens)


# ============================================================================
# Demo
# ============================================================================

def run_demo():
    """Demonstrate the LM rescoring pipeline."""
    # Import the corrector
    sys.path.insert(0, ".")
    from corrector import ISLCorrector

    print("=" * 70)
    print("LM RESCORING DEMO")
    print("=" * 70)

    corrector = ISLCorrector()

    # Without KenLM model file, uses heuristic fallback
    pipeline = CorrectionPipeline(corrector, lm_model_path=None)

    test_cases = [
        ["I", "FOOD", "EAT", "YESTERDAY"],
        ["SHE", "HAPPY"],
        ["YOU", "NAME", "WHAT"],
        ["I", "FOOD", "WANT", "NOT"],
        ["HE", "BOOK", "BIG", "BUY"],
        ["I", "GO", "TOMORROW"],
        ["SHE", "MARKET", "GO", "YESTERDAY"],
    ]

    print("\n--- Rules Only vs Rules + LM Rescoring ---\n")
    print(f"{'ISL Input':<35} {'Rules Only':<30} {'Rules + LM':<30}")
    print("-" * 95)

    for tokens in test_cases:
        rules_only = pipeline.correct_rules_only(tokens)
        rules_plus_lm = pipeline.correct(tokens)

        input_str = " ".join(tokens)
        marker = " *" if rules_only != rules_plus_lm else ""
        print(f"{input_str:<35} {rules_only:<30} {rules_plus_lm:<30}{marker}")

    print("\n* = LM rescoring changed the output")
    print("\nNote: Using heuristic fallback. With a real KenLM model,")
    print("rescoring will be significantly more effective.")

    # Show detailed scoring for one example
    print("\n\n--- Detailed Scoring Example ---\n")
    pipeline.correct(["I", "FOOD", "WANT", "NOT"], verbose=True)


if __name__ == "__main__":
    run_demo()
