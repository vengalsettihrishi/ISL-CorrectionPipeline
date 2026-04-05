"""
noise_injector.py — Controlled Noise Injection for ISL Pipeline Evaluation

Simulates real-world classifier noise by corrupting ISL gloss sequences
at controlled rates. Three noise operations, each documented and justified.

NOISE OPERATIONS:
  1. Token Duplication — simulates the buffer receiving a repeated detection
  2. Token Deletion   — simulates a missed gesture / dropped frame
  3. Token Transposition — simulates temporal instability in gesture transitions

USAGE:
    from noise_injector import NoiseInjector
    injector = NoiseInjector(seed=42)
    noisy = injector.inject(["I", "FOOD", "EAT"], noise_rate=0.2)
    print(noisy)  # e.g., ["I", "FOOD", "FOOD", "EAT"] (duplication)
"""

import random
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class NoiseResult:
    """Result of noise injection with metadata."""
    original: List[str]
    noisy: List[str]
    noise_rate: float
    operations_applied: List[str]
    actual_corruption: float  # Measured edit distance / original length


class NoiseInjector:
    """
    Inject controlled noise into ISL gloss token sequences.

    Applies three types of noise operations at a specified rate.
    Each token has a `noise_rate` probability of being affected.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def inject(self, tokens: List[str], noise_rate: float = 0.1) -> List[str]:
        """
        Inject noise into a token sequence.

        Args:
            tokens: Clean ISL gloss tokens, e.g., ["I", "FOOD", "EAT"]
            noise_rate: Probability of each token being corrupted (0.0 to 1.0)

        Returns:
            Noisy token list (may be longer/shorter than original).
        """
        if not tokens or noise_rate <= 0:
            return list(tokens)

        result = []
        for token in tokens:
            if self.rng.random() < noise_rate:
                # Choose a noise operation
                op = self.rng.choice(["duplicate", "delete", "transpose"])

                if op == "duplicate":
                    # Token appears twice (simulates repeated detection)
                    result.append(token)
                    result.append(token)
                elif op == "delete":
                    # Token disappears (simulates missed gesture)
                    pass  # Don't add the token
                elif op == "transpose" and result:
                    # Swap with previous token (simulates temporal instability)
                    result.append(token)
                    if len(result) >= 2:
                        result[-1], result[-2] = result[-2], result[-1]
                else:
                    result.append(token)
            else:
                result.append(token)

        # Safety: never return empty sequence
        if not result and tokens:
            result = [tokens[0]]

        return result

    def inject_with_metadata(self, tokens: List[str],
                              noise_rate: float = 0.1) -> NoiseResult:
        """Inject noise and return detailed metadata."""
        noisy = self.inject(tokens, noise_rate)

        # Compute actual corruption rate (edit distance approximation)
        orig_set = list(tokens)
        noisy_set = list(noisy)
        matches = sum(1 for a, b in zip(orig_set, noisy_set) if a == b)
        max_len = max(len(orig_set), len(noisy_set), 1)
        actual_corruption = 1.0 - (matches / max_len)

        ops = []
        if len(noisy) > len(tokens):
            ops.append("duplication")
        if len(noisy) < len(tokens):
            ops.append("deletion")
        if noisy != tokens and len(noisy) == len(tokens):
            ops.append("transposition")

        return NoiseResult(
            original=tokens,
            noisy=noisy,
            noise_rate=noise_rate,
            operations_applied=ops,
            actual_corruption=actual_corruption,
        )

    def generate_noisy_dataset(
        self,
        clean_pairs: List[Tuple[List[str], str]],
        noise_rates: List[float] = [0.1, 0.2, 0.3],
    ) -> Dict[float, List[Tuple[List[str], List[str], str]]]:
        """
        Generate noisy versions of a clean ISL→English test set.

        Args:
            clean_pairs: List of (clean_isl_tokens, expected_english) pairs.
            noise_rates: List of noise rates to generate.

        Returns:
            Dict mapping noise_rate → list of (clean_tokens, noisy_tokens, expected_english).
        """
        dataset = {}
        for rate in noise_rates:
            pairs = []
            for clean_tokens, expected_english in clean_pairs:
                noisy = self.inject(clean_tokens, rate)
                pairs.append((clean_tokens, noisy, expected_english))
            dataset[rate] = pairs

        return dataset


# ============================================================================
# Built-in Test Sentence Bank
# ============================================================================

# ISL gloss → Expected English pairs
# These cover the grammar rules implemented in corrector.py
# Each pair is tagged with which rules it exercises

TEST_SENTENCE_BANK = [
    # Simple declarative (R4: SOV→SVO, R6: conjugation)
    (["I", "FOOD", "EAT"],                  "I eat food."),
    (["SHE", "BOOK", "READ"],               "She reads a book."),
    (["HE", "WATER", "DRINK"],              "He drinks water."),
    (["WE", "GAME", "PLAY"],                "We play a game."),
    (["THEY", "FOOD", "COOK"],              "They cook food."),
    (["I", "SCHOOL", "GO"],                 "I go to school."),
    (["SHE", "MUSIC", "LIKE"],              "She likes music."),
    (["HE", "WORK", "LOVE"],               "He loves work."),

    # Past tense (R6: tense from markers)
    (["I", "FOOD", "EAT", "YESTERDAY"],     "I ate food yesterday."),
    (["SHE", "CRY", "YESTERDAY"],           "She cried yesterday."),
    (["HE", "BOOK", "BUY", "LAST"],         "He bought a book last."),
    (["WE", "GAME", "PLAY", "YESTERDAY"],   "We played a game yesterday."),
    (["I", "SCHOOL", "GO", "YESTERDAY"],    "I went to school yesterday."),
    (["THEY", "FOOD", "COOK", "BEFORE"],    "They cooked food before."),

    # Future tense (R6: will insertion)
    (["I", "GO", "TOMORROW"],               "I will go tomorrow."),
    (["SHE", "COME", "TOMORROW"],           "She will come tomorrow."),
    (["HE", "FOOD", "EAT", "LATER"],        "He will eat food later."),
    (["WE", "PLAY", "TOMORROW"],            "We will play tomorrow."),

    # Questions (R2: WH-fronting)
    (["YOU", "NAME", "WHAT"],               "What is your name?"),
    (["YOU", "GO", "WHERE"],                "Where do you go?"),
    (["HE", "COME", "WHEN"],               "When does he come?"),
    (["SHE", "CRY", "WHY"],                "Why does she cry?"),
    (["YOU", "LIKE", "WHAT"],               "What do you like?"),

    # Negation (R3: negation reordering)
    (["I", "UNDERSTAND", "NOT"],            "I do not understand."),
    (["SHE", "COME", "NOT"],                "She does not come."),
    (["HE", "FOOD", "LIKE", "NOT"],         "He does not like food."),
    (["I", "FOOD", "WANT", "NOT"],          "I do not want food."),
    (["WE", "GO", "NOT"],                   "We do not go."),

    # Adjective reordering (R5)
    (["DOG", "BIG"],                        "Big dog."),
    (["CAR", "RED"],                        "Red car."),
    (["MAN", "STRONG"],                     "Strong man."),

    # Adjective + verb (R4 + R5)
    (["HE", "BOOK", "BIG", "BUY"],          "He buys a big book."),
    (["SHE", "DOG", "SMALL", "LIKE"],        "She likes a small dog."),

    # Copula (R7: auxiliary insertion)
    (["SHE", "HAPPY"],                      "She is happy."),
    (["HE", "STRONG"],                      "He is strong."),
    (["I", "TIRED"],                        "I am tired."),
    (["THEY", "BUSY"],                      "They are busy."),

    # Pronouns (R1)
    (["ME", "FOOD", "EAT"],                 "I eat food."),
    (["HIM", "HAPPY"],                      "He is happy."),

    # Combined rules
    (["SHE", "MARKET", "GO", "YESTERDAY"],  "She went to market yesterday."),
    (["I", "FOOD", "EAT", "NOT"],           "I do not eat food."),
    (["HE", "BOOK", "READ", "YESTERDAY"],   "He read a book yesterday."),
    (["YOU", "FOOD", "EAT", "WHAT"],        "What do you eat?"),
]


def get_test_bank() -> List[Tuple[List[str], str]]:
    """Return the built-in test sentence bank."""
    return TEST_SENTENCE_BANK


# ============================================================================
# Demo
# ============================================================================

def run_demo():
    print("=" * 70)
    print("NOISE INJECTION DEMO")
    print("=" * 70)

    injector = NoiseInjector(seed=42)

    sample = ["I", "FOOD", "EAT", "YESTERDAY"]
    print(f"\nOriginal: {' '.join(sample)}\n")

    for rate in [0.1, 0.2, 0.3]:
        print(f"--- Noise rate: {rate:.0%} ---")
        for trial in range(5):
            # Use different seed for each trial
            injector.rng = random.Random(42 + trial + int(rate * 100))
            noisy = injector.inject(sample, rate)
            changed = "✓ changed" if noisy != sample else "  same"
            print(f"  Trial {trial+1}: {' '.join(noisy):<40} {changed}")
        print()

    # Show dataset generation
    print("\n--- Dataset Generation ---\n")
    bank = get_test_bank()[:5]
    injector = NoiseInjector(seed=42)
    dataset = injector.generate_noisy_dataset(bank, [0.1, 0.2, 0.3])

    for rate, pairs in dataset.items():
        print(f"Noise rate {rate:.0%}:")
        for clean, noisy, expected in pairs:
            print(f"  {' '.join(clean):<30} → {' '.join(noisy)}")
        print()


if __name__ == "__main__":
    run_demo()
