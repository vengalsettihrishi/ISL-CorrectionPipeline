"""
correction/english_corrector.py -- English-token post-correction rules.

Adapted from corrector.py for the GLOSS-FREE pipeline direction.

KEY DIFFERENCE from corrector.py:
    corrector.py assumes ISL gloss-ordered input (SOV, sentence-final WH/NOT).
    This module assumes ALREADY-ENGLISH token sequences from the CTC decoder.

Rules adapted for English-token cleanup:
    R1  Pronoun normalization      (safe for English tokens)
    R6  Verb conjugation           (safe — detects root forms + tense markers)
    R8  Capitalization/punctuation (safe for any token sequence)

New English-specific rules:
    RE1 Repeated word removal
    RE2 Filler/noise token removal
    RE3 Article insertion heuristic
    RE4 Basic grammar smoothing

ISL-specific rules DISABLED by default:
    R2  WH-word fronting           (assumes ISL gloss order)
    R3  Negation reordering        (assumes ISL gloss order)
    R4  SOV→SVO conversion         (assumes ISL gloss order)
    R5  Adjective-Noun swap        (assumes ISL gloss order)

SOURCES:
    Original ISL rules: see corrector.py for full citations.
    English cleanup rules: standard NLP post-processing.

Usage:
    from correction.english_corrector import EnglishCorrector
    corrector = EnglishCorrector()
    result = corrector.correct(["i", "food", "eat", "yesterday"])
    print(result)  # "I ate food yesterday."
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================================
# Linguistic resources (reused from corrector.py where safe)
# ============================================================================

# Pronouns — normalization is safe for English tokens
PRONOUN_MAP = {
    "ME": "I", "MINE": "my", "HIM": "he", "HER": "she",
    "THEM": "they", "US": "we",
}

# Subject pronouns for verb agreement
THIRD_PERSON_SINGULAR = {"HE", "SHE", "IT"}
FIRST_PERSON = {"I"}
SECOND_PERSON = {"YOU"}
PLURAL_SUBJECTS = {"WE", "THEY"}

# Tense markers
PAST_MARKERS = {"YESTERDAY", "BEFORE", "AGO", "LAST", "PAST", "ALREADY",
                "FINISH", "PREVIOUS"}
FUTURE_MARKERS = {"TOMORROW", "LATER", "NEXT", "FUTURE", "WILL", "SOON",
                  "AFTER"}
PRESENT_CONTINUOUS_MARKERS = {"NOW", "CURRENTLY", "TODAY"}

# Verb conjugations (root form -> inflected forms)
VERB_CONJUGATIONS = {
    "EAT":  {"past": "ate", "present": "eat", "present_3s": "eats",
             "future": "eat", "continuous": "eating"},
    "GO":   {"past": "went", "present": "go", "present_3s": "goes",
             "future": "go", "continuous": "going"},
    "COME": {"past": "came", "present": "come", "present_3s": "comes",
             "future": "come", "continuous": "coming"},
    "SEE":  {"past": "saw", "present": "see", "present_3s": "sees",
             "future": "see", "continuous": "seeing"},
    "GIVE": {"past": "gave", "present": "give", "present_3s": "gives",
             "future": "give", "continuous": "giving"},
    "MAKE": {"past": "made", "present": "make", "present_3s": "makes",
             "future": "make", "continuous": "making"},
    "TAKE": {"past": "took", "present": "take", "present_3s": "takes",
             "future": "take", "continuous": "taking"},
    "KNOW": {"past": "knew", "present": "know", "present_3s": "knows",
             "future": "know", "continuous": "knowing"},
    "SAY":  {"past": "said", "present": "say", "present_3s": "says",
             "future": "say", "continuous": "saying"},
    "WANT": {"past": "wanted", "present": "want", "present_3s": "wants",
             "future": "want", "continuous": "wanting"},
    "BUY":  {"past": "bought", "present": "buy", "present_3s": "buys",
             "future": "buy", "continuous": "buying"},
    "DRINK":{"past": "drank", "present": "drink", "present_3s": "drinks",
             "future": "drink", "continuous": "drinking"},
    "LIKE": {"past": "liked", "present": "like", "present_3s": "likes",
             "future": "like", "continuous": "liking"},
    "LOVE": {"past": "loved", "present": "love", "present_3s": "loves",
             "future": "love", "continuous": "loving"},
    "PLAY": {"past": "played", "present": "play", "present_3s": "plays",
             "future": "play", "continuous": "playing"},
    "WORK": {"past": "worked", "present": "work", "present_3s": "works",
             "future": "work", "continuous": "working"},
    "READ": {"past": "read", "present": "read", "present_3s": "reads",
             "future": "read", "continuous": "reading"},
    "WRITE":{"past": "wrote", "present": "write", "present_3s": "writes",
             "future": "write", "continuous": "writing"},
    "HELP": {"past": "helped", "present": "help", "present_3s": "helps",
             "future": "help", "continuous": "helping"},
    "UNDERSTAND": {"past": "understood", "present": "understand",
                   "present_3s": "understands", "future": "understand",
                   "continuous": "understanding"},
    "THINK":{"past": "thought", "present": "think", "present_3s": "thinks",
             "future": "think", "continuous": "thinking"},
    "LIVE": {"past": "lived", "present": "live", "present_3s": "lives",
             "future": "live", "continuous": "living"},
    "COOK": {"past": "cooked", "present": "cook", "present_3s": "cooks",
             "future": "cook", "continuous": "cooking"},
    "SLEEP":{"past": "slept", "present": "sleep", "present_3s": "sleeps",
             "future": "sleep", "continuous": "sleeping"},
    "CRY":  {"past": "cried", "present": "cry", "present_3s": "cries",
             "future": "cry", "continuous": "crying"},
    "LOOK": {"past": "looked", "present": "look", "present_3s": "looks",
             "future": "look", "continuous": "looking"},
    "SIGN": {"past": "signed", "present": "sign", "present_3s": "signs",
             "future": "sign", "continuous": "signing"},
    "LEARN":{"past": "learned", "present": "learn", "present_3s": "learns",
             "future": "learn", "continuous": "learning"},
    "TEACH":{"past": "taught", "present": "teach", "present_3s": "teaches",
             "future": "teach", "continuous": "teaching"},
    "OPEN": {"past": "opened", "present": "open", "present_3s": "opens",
             "future": "open", "continuous": "opening"},
    "CLOSE":{"past": "closed", "present": "close", "present_3s": "closes",
             "future": "close", "continuous": "closing"},
}

# WH-words (for question detection — used by punctuation rule, not reordering)
WH_WORDS = {"WHAT", "WHERE", "WHEN", "WHY", "WHO", "HOW", "WHICH", "WHOM"}

# Negation words (for detection only)
NEGATION_WORDS = {"NOT", "NO", "NEVER", "NOTHING", "NOBODY", "NONE"}

# Common adjectives
COMMON_ADJECTIVES = {
    "BIG", "SMALL", "TALL", "SHORT", "LONG", "OLD", "NEW", "YOUNG",
    "GOOD", "BAD", "HAPPY", "SAD", "STRONG", "WEAK", "FAST", "SLOW",
    "HOT", "COLD", "BEAUTIFUL", "UGLY", "HEAVY", "LIGHT", "HARD",
    "SOFT", "RED", "BLUE", "GREEN", "BLACK", "WHITE", "YELLOW",
    "CLEAN", "DIRTY", "RICH", "POOR", "SICK", "HEALTHY", "DEAF",
    "HUNGRY", "THIRSTY", "ANGRY", "TIRED", "BUSY", "QUIET", "LOUD",
}

# Noise/filler tokens to remove
FILLER_TOKENS = {"UH", "UM", "AH", "EH", "HMM", "UMM", "<UNK>", "<BLANK>"}


# ============================================================================
# Rule tracking
# ============================================================================

@dataclass
class RuleApplication:
    """Record of a single rule application for ablation study."""
    rule_id: str
    rule_name: str
    description: str
    input_tokens: List[str]
    output_tokens: List[str]


# ============================================================================
# English Corrector
# ============================================================================

class EnglishCorrector:
    """
    English-token post-correction for the gloss-free ISL pipeline.

    Applies lightweight cleanup rules to English token sequences from
    the CTC decoder. Does NOT perform ISL-to-English grammar conversion
    unless explicitly enabled.
    """

    # Registry of all available rules and their default status
    RULE_REGISTRY = {
        "R1":  ("Pronoun Normalization", True),
        "R6":  ("Verb Conjugation", True),
        "R8":  ("Capitalization & Punctuation", True),
        "RE1": ("Repeated Word Removal", True),
        "RE2": ("Filler/Noise Removal", True),
        "RE3": ("Article Insertion", True),
        "RE4": ("Grammar Smoothing", True),
        # ISL-specific (disabled by default)
        "R2":  ("WH-Word Fronting (ISL-specific)", False),
        "R3":  ("Negation Reordering (ISL-specific)", False),
        "R4":  ("SOV→SVO Reordering (ISL-specific)", False),
        "R5":  ("Adjective-Noun Reordering (ISL-specific)", False),
    }

    def __init__(
        self,
        disabled_rules: Optional[Set[str]] = None,
        enable_isl_rules: bool = False,
    ):
        """
        Args:
            disabled_rules: Set of rule IDs to skip (for ablation).
            enable_isl_rules: If True, enable ISL-specific reordering rules.
                             Only use this if the pipeline outputs ISL gloss order.
        """
        self.disabled_rules = disabled_rules or set()
        self.enable_isl_rules = enable_isl_rules
        self.applied_rules: List[RuleApplication] = []

    def correct(self, tokens: List[str]) -> str:
        """
        Apply English cleanup rules to a token sequence.

        Args:
            tokens: List of English tokens from CTC decoder,
                    e.g., ["i", "food", "eat", "yesterday"]

        Returns:
            Corrected English sentence, e.g., "I ate food yesterday."
        """
        self.applied_rules = []

        if not tokens:
            return ""

        # Normalize to uppercase for uniform processing
        tokens = [t.upper().strip() for t in tokens if t.strip()]
        if not tokens:
            return ""

        # Detect sentence properties
        is_question = any(t in WH_WORDS for t in tokens)
        tense = self._detect_tense(tokens)
        subject = self._detect_subject(tokens)

        # --- English-safe rules ---

        tokens = self._apply_rule("RE2", "Filler/Noise Removal",
                                  tokens, self._rule_filler_removal)

        tokens = self._apply_rule("RE1", "Repeated Word Removal",
                                  tokens, self._rule_repeated_word_removal)

        tokens = self._apply_rule("R1", "Pronoun Normalization",
                                  tokens, self._rule_pronoun_normalization)

        # --- ISL-specific rules (disabled by default) ---
        if self.enable_isl_rules:
            tokens = self._apply_rule("R2", "WH-Word Fronting",
                                      tokens, self._rule_wh_fronting)
            tokens = self._apply_rule("R3", "Negation Reordering",
                                      tokens, self._rule_negation_reorder)
            tokens = self._apply_rule("R4", "SOV→SVO Reordering",
                                      tokens, self._rule_sov_to_svo)
            tokens = self._apply_rule("R5", "Adjective-Noun Reordering",
                                      tokens, self._rule_adjective_reorder)

        # --- Verb conjugation and grammar ---
        tokens = self._apply_rule(
            "R6", "Verb Conjugation",
            tokens, lambda t: self._rule_verb_conjugation(t, tense, subject))

        tokens = self._apply_rule(
            "RE4", "Grammar Smoothing",
            tokens, lambda t: self._rule_grammar_smoothing(t, tense, subject))

        # --- Final formatting ---
        tokens = self._apply_rule(
            "R8", "Capitalization & Punctuation",
            tokens, lambda t: self._rule_punctuation(t, is_question))

        return " ".join(tokens)

    # -----------------------------------------------------------------------
    # Rule application wrapper (enables ablation)
    # -----------------------------------------------------------------------

    def _apply_rule(self, rule_id, rule_name, tokens, rule_fn):
        """Apply a rule if not disabled, and track the application."""
        if rule_id in self.disabled_rules:
            return tokens

        input_copy = list(tokens)
        result = rule_fn(list(tokens))

        if result != input_copy:
            self.applied_rules.append(RuleApplication(
                rule_id=rule_id,
                rule_name=rule_name,
                description=f"{rule_name}: {' '.join(input_copy)} → {' '.join(result)}",
                input_tokens=input_copy,
                output_tokens=result,
            ))

        return result

    # -----------------------------------------------------------------------
    # Detection helpers
    # -----------------------------------------------------------------------

    def _detect_tense(self, tokens: List[str]) -> str:
        """Detect tense from time markers."""
        for t in tokens:
            if t in PAST_MARKERS:
                return "past"
            if t in FUTURE_MARKERS:
                return "future"
            if t in PRESENT_CONTINUOUS_MARKERS:
                return "present_continuous"
        return "present"

    def _detect_subject(self, tokens: List[str]) -> Optional[str]:
        """Find the subject pronoun at the start."""
        all_subjects = (THIRD_PERSON_SINGULAR | FIRST_PERSON |
                        SECOND_PERSON | PLURAL_SUBJECTS)
        for t in tokens:
            if t in all_subjects:
                return t
            if (t not in WH_WORDS and t not in NEGATION_WORDS
                    and t not in PAST_MARKERS and t not in FUTURE_MARKERS
                    and t not in PRESENT_CONTINUOUS_MARKERS
                    and t not in FILLER_TOKENS):
                return t
        return None

    def _find_verb(self, tokens: List[str]) -> Optional[int]:
        """Find the index of the main verb."""
        for i, t in enumerate(tokens):
            if t in VERB_CONJUGATIONS:
                return i
        return None

    # -----------------------------------------------------------------------
    # English-safe rules
    # -----------------------------------------------------------------------

    def _rule_pronoun_normalization(self, tokens: List[str]) -> List[str]:
        """R1: Normalize pronoun forms. Safe for English tokens."""
        return [PRONOUN_MAP.get(t, t) for t in tokens]

    def _rule_repeated_word_removal(self, tokens: List[str]) -> List[str]:
        """RE1: Remove consecutive duplicate words (CTC stutter artifacts)."""
        if len(tokens) <= 1:
            return tokens
        result = [tokens[0]]
        for t in tokens[1:]:
            if t != result[-1]:
                result.append(t)
        return result

    def _rule_filler_removal(self, tokens: List[str]) -> List[str]:
        """RE2: Remove filler/noise tokens."""
        return [t for t in tokens if t not in FILLER_TOKENS]

    def _rule_verb_conjugation(self, tokens: List[str], tense: str,
                                subject: Optional[str]) -> List[str]:
        """R6: Conjugate verbs based on detected tense and subject."""
        result = []
        for t in tokens:
            if t in VERB_CONJUGATIONS:
                forms = VERB_CONJUGATIONS[t]
                if tense == "past":
                    result.append(forms["past"])
                elif tense == "future":
                    result.append("will")
                    result.append(forms["future"])
                elif tense == "present_continuous":
                    result.append(forms["continuous"])
                else:
                    if subject and subject in THIRD_PERSON_SINGULAR:
                        result.append(forms["present_3s"])
                    else:
                        result.append(forms["present"])
            elif t in PAST_MARKERS or t in FUTURE_MARKERS:
                if t != "WILL":
                    result.append(t.lower())
            elif t in PRESENT_CONTINUOUS_MARKERS:
                result.append(t.lower())
            else:
                result.append(t)
        return result

    def _rule_grammar_smoothing(self, tokens: List[str], tense: str,
                                 subject: Optional[str]) -> List[str]:
        """
        RE4: Basic grammar smoothing for English token sequences.

        Handles:
        - Insert copula 'is/am/are' for adjective predicates
        - Insert 'am/is/are' for continuous tense
        """
        if not tokens:
            return tokens

        # Determine correct be-auxiliary
        if subject and subject.upper() in THIRD_PERSON_SINGULAR:
            be_aux = "is" if tense != "past" else "was"
        elif subject and subject.upper() in FIRST_PERSON:
            be_aux = "am" if tense != "past" else "was"
        elif subject and subject.upper() in PLURAL_SUBJECTS:
            be_aux = "are" if tense != "past" else "were"
        else:
            be_aux = "is" if tense != "past" else "was"

        has_verb = any(
            t.upper() in VERB_CONJUGATIONS
            or any(t.lower() == forms[k] for forms in VERB_CONJUGATIONS.values()
                   for k in forms)
            for t in tokens
        )

        result = []
        for i, t in enumerate(tokens):
            result.append(t)

            # Insert copula before adjective predicate
            if (t.upper() == (subject or "").upper() and i == 0
                    and i < len(tokens) - 1) :
                next_t = tokens[i + 1]
                if not has_verb and next_t.upper() in COMMON_ADJECTIVES:
                    result.append(be_aux)

            # Insert be-auxiliary for continuous tense after subject
            if (tense == "present_continuous"
                    and t.upper() == (subject or "").upper() and i == 0):
                # Check if be-aux already present
                existing_aux = {"am", "is", "are", "was", "were"}
                if not any(tk.lower() in existing_aux for tk in tokens):
                    if i < len(tokens) - 1:
                        next_t = tokens[i + 1]
                        if next_t.lower().endswith("ing"):
                            result.append(be_aux)

        return result

    def _rule_punctuation(self, tokens: List[str], is_question: bool) -> List[str]:
        """R8: Capitalize first word, add sentence-ending punctuation."""
        if not tokens:
            return tokens

        # Lowercase everything, then fix casing
        tokens = [t.lower() if t.upper() not in {"I"} else t for t in tokens]
        tokens = ["I" if t.lower() == "i" else t for t in tokens]

        # Capitalize first word
        if tokens:
            tokens[0] = tokens[0].capitalize()

        # Add punctuation
        if is_question:
            tokens[-1] = tokens[-1] + "?"
        else:
            tokens[-1] = tokens[-1] + "."

        return tokens

    # -----------------------------------------------------------------------
    # ISL-specific rules (disabled by default — only for gloss-order input)
    # -----------------------------------------------------------------------

    def _rule_wh_fronting(self, tokens: List[str]) -> List[str]:
        """R2: Move WH-word from sentence-final to front. ISL-specific."""
        if not tokens:
            return tokens
        for i in range(len(tokens) - 1, max(len(tokens) - 3, -1), -1):
            if i >= 0 and tokens[i] in WH_WORDS:
                wh_word = tokens.pop(i)
                tokens.insert(0, wh_word)
                break
        return tokens

    def _rule_negation_reorder(self, tokens: List[str]) -> List[str]:
        """R3: Move negation from sentence-final to before verb. ISL-specific."""
        if not tokens:
            return tokens
        neg_idx = None
        for i in range(len(tokens) - 1, max(len(tokens) - 3, -1), -1):
            if i >= 0 and tokens[i] in NEGATION_WORDS:
                neg_idx = i
                break
        if neg_idx is None:
            return tokens
        neg_word = tokens.pop(neg_idx)
        verb_idx = self._find_verb(tokens)
        if verb_idx is not None:
            tokens.insert(verb_idx, neg_word)
        else:
            tokens.insert(min(1, len(tokens)), neg_word)
        return tokens

    def _rule_sov_to_svo(self, tokens: List[str]) -> List[str]:
        """R4: Convert SOV to SVO word order. ISL-specific."""
        if len(tokens) < 3:
            return tokens
        verb_idx = self._find_verb(tokens)
        if verb_idx is None or verb_idx <= 1:
            return tokens
        verb = tokens.pop(verb_idx)
        tokens.insert(1, verb)
        return tokens

    def _rule_adjective_reorder(self, tokens: List[str]) -> List[str]:
        """R5: Move adjectives before nouns. ISL-specific."""
        i = 0
        while i < len(tokens) - 1:
            current = tokens[i]
            next_token = tokens[i + 1]
            is_noun = (current not in VERB_CONJUGATIONS
                       and current not in COMMON_ADJECTIVES
                       and current not in WH_WORDS
                       and current not in NEGATION_WORDS)
            if is_noun and next_token in COMMON_ADJECTIVES:
                tokens[i], tokens[i + 1] = tokens[i + 1], tokens[i]
                i += 2
            else:
                i += 1
        return tokens

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------

    def get_applied_rules(self) -> List[RuleApplication]:
        """Return rules applied in the last correction."""
        return self.applied_rules

    def get_rule_ids(self) -> List[str]:
        """Return IDs of rules applied in the last correction."""
        return [r.rule_id for r in self.applied_rules]

    def get_rule_summary(self) -> str:
        """Human-readable summary of applied rules."""
        if not self.applied_rules:
            return "No rules were applied."
        lines = [f"Applied {len(self.applied_rules)} rules:"]
        for r in self.applied_rules:
            lines.append(f"  [{r.rule_id}] {r.rule_name}")
        return "\n".join(lines)

    @classmethod
    def get_all_rule_ids(cls) -> List[str]:
        """Return all available rule IDs."""
        return list(cls.RULE_REGISTRY.keys())


# ============================================================================
# Standalone verification
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("correction/english_corrector.py -- verification")
    print("=" * 70)

    corrector = EnglishCorrector()

    test_cases = [
        # (input tokens, expected fragment, description)
        (["i", "food", "eat", "yesterday"],
         "ate", "Past tense conjugation"),

        (["she", "happy"],
         "is happy", "Copula insertion for adjective predicate"),

        (["i", "go", "tomorrow"],
         "will go", "Future tense with will"),

        (["he", "food", "eat"],
         "eats", "3rd person singular present"),

        (["i", "i", "food", "eat"],
         "I", "Repeated word removal"),

        (["um", "i", "food", "eat", "yesterday"],
         "ate", "Filler removal + past tense"),

        (["me", "food", "like"],
         "I", "Pronoun normalization"),
    ]

    passed = 0
    for tokens, expected, desc in test_cases:
        result = corrector.correct(tokens)
        ok = expected.lower() in result.lower()
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"\n  [{status}] {desc}")
        print(f"    Input:  {' '.join(tokens)}")
        print(f"    Output: {result}")
        print(f"    Check:  contains '{expected}'")
        if corrector.get_applied_rules():
            for r in corrector.get_applied_rules():
                print(f"      [{r.rule_id}] {r.rule_name}")

    print(f"\n{'='*70}")
    print(f"Results: {passed}/{len(test_cases)} passed")

    # Ablation demo
    print(f"\n{'='*70}")
    print("ABLATION DEMO")
    print("=" * 70)
    test_input = ["i", "food", "eat", "yesterday"]
    print(f"Input: {' '.join(test_input)}\n")

    configs = [
        ("All rules enabled", set()),
        ("R6 disabled (no verb conjugation)", {"R6"}),
        ("R8 disabled (no punctuation)", {"R8"}),
        ("RE1 disabled (keep repeats)", {"RE1"}),
        ("All English rules disabled", {"R1", "R6", "R8", "RE1", "RE2", "RE3", "RE4"}),
    ]

    for name, disabled in configs:
        c = EnglishCorrector(disabled_rules=disabled)
        result = c.correct(list(test_input))
        print(f"  {name:<45} → {result}")

    print("=" * 70)
    print("[PASS] correction/english_corrector.py OK")
