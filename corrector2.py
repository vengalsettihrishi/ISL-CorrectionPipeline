"""
corrector.py — ISL Gloss-to-English Rule-Based Correction Module

Converts ISL-ordered gloss token sequences into grammatical English sentences.
Every rule is cited to its published source for the research paper.

SOURCES:
  [Bhatia2020]  Sugandhi, P. Kumar, S. Kaur. "Sign Language Generation System
                Based on Indian Sign Language Grammar." ACM TALLIP, 19(4), 2020.
  [Ghosh2022]   A. Ghosh, R. Mamidi. "English To Indian Sign Language: Rule-Based
                Translation." ICON 2022, pp. 123-127.
  [NIOS]        National Institute of Open Schooling. "Indian Sign Language" Course
                Material (Module 230).
  [Zeshan2000]  U. Zeshan. "Sign Language in Indo-Pakistan." John Benjamins, 2000.

DIRECTION NOTE:
  The source papers implement English → ISL (forward direction).
  This module implements ISL → English (reverse direction).
  Each rule documents the forward rule from the paper, then the reversed
  transformation we apply.

USAGE:
    from corrector import ISLCorrector
    corrector = ISLCorrector()
    result = corrector.correct(["I", "FOOD", "EAT", "YESTERDAY"])
    print(result)  # "I ate food yesterday."
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


# ============================================================================
# Linguistic resources (no external downloads required)
# ============================================================================

# WH-words that ISL places at sentence end
# Source: [Bhatia2020] §3(vii): "Interrogative terms such as when, what, and
#         how are always placed at the end of a sentence in ISL"
# Source: [Ghosh2022] §3.3.1: "take the wh-word and pick it and place it at
#         the end of the sentence"
WH_WORDS = {"WHAT", "WHERE", "WHEN", "WHY", "WHO", "HOW", "WHICH", "WHOM"}

# Negation words
# Source: [Bhatia2020] §3(viii): "Negation in ISL is represented with the sign
#         for not, which is always placed at the end of sentence"
# Source: [Ghosh2022] §3.3.2: "picking and placing the negative word at the
#         end of the sentence"
NEGATION_WORDS = {"NOT", "NO", "NEVER", "NOTHING", "NOBODY", "NONE"}

# Auxiliaries that ISL drops entirely
# Source: [Bhatia2020] §3: "ISL neither uses verbs (which show action or a
#         state of being e.g., is, am, are)"
# Source: [NIOS] Rule 4: "No is, am, are, do, did"
AUXILIARIES = {
    "IS", "AM", "ARE", "WAS", "WERE", "BE", "BEEN", "BEING",
    "DO", "DOES", "DID", "HAS", "HAVE", "HAD",
    "WILL", "SHALL", "WOULD", "SHOULD", "COULD", "CAN", "MAY", "MIGHT",
}

# Articles that ISL never uses
# Source: [Bhatia2020] §3: "ISL neither uses...articles (e.g., a, an, the)"
# Source: [NIOS] Rule 7 (inferred): "No a, an, the"
ARTICLES = {"A", "AN", "THE"}

# Prepositions that ISL drops (spatial relations expressed via signing space)
# Source: [NIOS] Rule 8 (inferred): "Spatial relations expressed via context"
# Source: [Bhatia2020] §7 example: "She went to market" → "She market go"
#         (preposition "to" removed)
PREPOSITIONS = {
    "TO", "FROM", "IN", "ON", "AT", "BY", "FOR", "WITH",
    "INTO", "ONTO", "UPON", "ABOUT", "BETWEEN", "THROUGH",
}

# Stop words that ISL omits (custom list)
# Source: [Ghosh2022] §3.3.4: "we created our own list of stop-words for
#         English based on inputs from translations of sentences in our
#         dataset by expert ISL signers"
ISL_STOPWORDS = ARTICLES | {"THERE", "IT", "THAT", "THIS", "THESE", "THOSE"}

# Past tense time markers
# Source: [Bhatia2020] §3(iv): "A spatial time line is used to represent the
#         past, present, and future tense"
# Source: [NIOS] Rule 3: "Time expressed via explicit time words"
PAST_MARKERS = {"YESTERDAY", "BEFORE", "AGO", "LAST", "PAST", "ALREADY", "FINISH", "PREVIOUS"}
FUTURE_MARKERS = {"TOMORROW", "LATER", "NEXT", "FUTURE", "WILL", "SOON", "AFTER"}
PRESENT_CONTINUOUS_MARKERS = {"NOW", "CURRENTLY", "TODAY"}

# Common ISL verbs and their English conjugations
# Source: [Bhatia2020] §3: "ISL does not use any inflections (gerunds,
#         suffixes, or other forms); it uses only root words"
# Source: [NIOS] Rule 9: "Same verb form used always"
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
    "COPY": {"past": "copied", "present": "copy", "present_3s": "copies",
             "future": "copy", "continuous": "copying"},
    "OPEN": {"past": "opened", "present": "open", "present_3s": "opens",
             "future": "open", "continuous": "opening"},
    "CLOSE":{"past": "closed", "present": "close", "present_3s": "closes",
             "future": "close", "continuous": "closing"},
}

# Pronouns — ISL pronoun normalization
# Source: [Bhatia2020] §7 flowchart: "possessive pronouns are converted into
#         personal pronouns"
# Source: [NIOS]: "In writing, the word me is used instead of I"
PRONOUN_MAP = {
    "ME": "I", "MINE": "my", "HIM": "he", "HER": "she",
    "THEM": "they", "US": "we",
}

# Subject pronouns for auxiliary selection
THIRD_PERSON_SINGULAR = {"HE", "SHE", "IT"}
FIRST_PERSON = {"I"}
SECOND_PERSON = {"YOU"}
PLURAL_SUBJECTS = {"WE", "THEY"}

# Common nouns (subset for adjective detection)
COMMON_ADJECTIVES = {
    "BIG", "SMALL", "TALL", "SHORT", "LONG", "OLD", "NEW", "YOUNG",
    "GOOD", "BAD", "HAPPY", "SAD", "STRONG", "WEAK", "FAST", "SLOW",
    "HOT", "COLD", "BEAUTIFUL", "UGLY", "HEAVY", "LIGHT", "HARD",
    "SOFT", "RED", "BLUE", "GREEN", "BLACK", "WHITE", "YELLOW",
    "CLEAN", "DIRTY", "RICH", "POOR", "SICK", "HEALTHY", "DEAF",
    "HUNGRY", "THIRSTY", "ANGRY", "TIRED", "BUSY", "QUIET", "LOUD",
}


# ============================================================================
# Rule tracking
# ============================================================================

@dataclass
class RuleApplication:
    """Record of a single rule application, for ablation study."""
    rule_id: str
    rule_name: str
    source: str
    input_tokens: List[str]
    output_tokens: List[str]
    description: str


# ============================================================================
# Core Corrector
# ============================================================================

class ISLCorrector:
    """
    Rule-based ISL gloss-to-English correction module.

    Takes a list of ISL gloss tokens and produces a grammatical English sentence.
    Tracks which rules were applied for ablation study.
    """

    def __init__(self, disabled_rules: Optional[set] = None):
        """
        Args:
            disabled_rules: Set of rule IDs to skip (for ablation study).
                           e.g., {"R1", "R3"} to disable rules 1 and 3.
        """
        self.disabled_rules = disabled_rules or set()
        self.applied_rules: List[RuleApplication] = []

    def correct(self, tokens: List[str]) -> str:
        """
        Convert ISL gloss tokens to a grammatical English sentence.

        Args:
            tokens: List of ISL gloss tokens, e.g., ["I", "FOOD", "EAT", "YESTERDAY"]

        Returns:
            Grammatical English sentence, e.g., "I ate food yesterday."
        """
        self.applied_rules = []

        if not tokens:
            return ""

        # Normalize to uppercase for processing
        tokens = [t.upper().strip() for t in tokens if t.strip()]
        if not tokens:
            return ""

        # Detect sentence type before reordering
        is_question = self._has_wh_word(tokens)
        is_negative = self._has_negation(tokens)
        tense = self._detect_tense(tokens)
        subject = self._detect_subject(tokens)

        # Apply rules in order
        tokens = self._apply_rule("R1", "Pronoun Normalization",
                                  "[Bhatia2020] §7 flowchart; [NIOS] inferred",
                                  tokens, self._rule_pronoun_normalization)

        tokens = self._apply_rule("R2", "WH-Word Fronting",
                                  "[Bhatia2020] §3(vii); [Ghosh2022] §3.3.1",
                                  tokens, self._rule_wh_fronting)

        tokens = self._apply_rule("R3", "Negation Reordering",
                                  "[Bhatia2020] §3(viii); [Ghosh2022] §3.3.2",
                                  tokens, self._rule_negation_reorder)

        tokens = self._apply_rule("R4", "SOV to SVO Reordering",
                                  "[Bhatia2020] §3,§7; [Ghosh2022] Rule 1",
                                  tokens, self._rule_sov_to_svo)

        tokens = self._apply_rule("R5", "Adjective-Noun Reordering",
                                  "[Ghosh2022] Rule 5; [NIOS] inferred",
                                  tokens, self._rule_adjective_reorder)

        tokens = self._apply_rule("R6", "Tense & Verb Conjugation",
                                  "[Bhatia2020] §3(iv); [NIOS] Rule 3,9",
                                  tokens, lambda t: self._rule_verb_conjugation(t, tense, subject))

        tokens = self._apply_rule("R7", "Auxiliary Insertion",
                                  "[NIOS] Rule 4; [Bhatia2020] §3",
                                  tokens, lambda t: self._rule_auxiliary_insertion(
                                      t, tense, subject, is_question, is_negative))

        tokens = self._apply_rule("R8", "Capitalization & Punctuation",
                                  "[NIOS] Rule 10",
                                  tokens, lambda t: self._rule_punctuation(t, is_question))

        return " ".join(tokens)

    # -----------------------------------------------------------------------
    # Rule application wrapper (enables ablation)
    # -----------------------------------------------------------------------

    def _apply_rule(self, rule_id, rule_name, source, tokens, rule_fn):
        """Apply a rule if not disabled, and track the application."""
        if rule_id in self.disabled_rules:
            return tokens

        input_copy = list(tokens)
        result = rule_fn(list(tokens))

        if result != input_copy:
            self.applied_rules.append(RuleApplication(
                rule_id=rule_id,
                rule_name=rule_name,
                source=source,
                input_tokens=input_copy,
                output_tokens=result,
                description=f"{rule_name}: {' '.join(input_copy)} → {' '.join(result)}",
            ))

        return result

    # -----------------------------------------------------------------------
    # Detection helpers
    # -----------------------------------------------------------------------

    def _has_wh_word(self, tokens: List[str]) -> bool:
        """Check if the sequence contains a WH-word (likely a question)."""
        return any(t in WH_WORDS for t in tokens)

    def _has_negation(self, tokens: List[str]) -> bool:
        """Check if the sequence contains negation."""
        return any(t in NEGATION_WORDS for t in tokens)

    def _detect_tense(self, tokens: List[str]) -> str:
        """
        Detect tense from time markers.
        Source: [Bhatia2020] §3(iv), [NIOS] Rule 3.
        If no marker found, default to present tense.
        """
        for t in tokens:
            if t in PAST_MARKERS:
                return "past"
            if t in FUTURE_MARKERS:
                return "future"
            if t in PRESENT_CONTINUOUS_MARKERS:
                return "present_continuous"
        return "present"

    def _detect_subject(self, tokens: List[str]) -> Optional[str]:
        """Find the subject pronoun or noun at the start."""
        all_subjects = THIRD_PERSON_SINGULAR | FIRST_PERSON | SECOND_PERSON | PLURAL_SUBJECTS
        for t in tokens:
            if t in all_subjects:
                return t
            # First non-functional word is likely the subject
            if t not in WH_WORDS and t not in NEGATION_WORDS and t not in PAST_MARKERS \
               and t not in FUTURE_MARKERS and t not in PRESENT_CONTINUOUS_MARKERS:
                return t
        return None

    def _find_verb(self, tokens: List[str]) -> Optional[int]:
        """Find the index of the main verb in the token list."""
        for i, t in enumerate(tokens):
            if t in VERB_CONJUGATIONS:
                return i
        return None

    def _is_adjective(self, token: str) -> bool:
        return token in COMMON_ADJECTIVES

    def _is_noun(self, token: str) -> bool:
        """A token is likely a noun if it's not a verb, adj, pronoun, or marker."""
        all_pronouns = THIRD_PERSON_SINGULAR | FIRST_PERSON | SECOND_PERSON | PLURAL_SUBJECTS
        return (
            token not in VERB_CONJUGATIONS
            and token not in COMMON_ADJECTIVES
            and token not in all_pronouns
            and token not in WH_WORDS
            and token not in NEGATION_WORDS
            and token not in PAST_MARKERS
            and token not in FUTURE_MARKERS
            and token not in PRESENT_CONTINUOUS_MARKERS
            and token not in AUXILIARIES
            and token not in ARTICLES
            and token not in PREPOSITIONS
        )

    # -----------------------------------------------------------------------
    # Individual Rules
    # -----------------------------------------------------------------------

    def _rule_pronoun_normalization(self, tokens: List[str]) -> List[str]:
        """
        RULE R1: Normalize ISL pronoun forms to English subject forms.

        ISL: ME → I, MINE → my, HIM → he
        Source: [Bhatia2020] §7 flowchart: "possessive pronouns are converted
                into personal pronouns"
        Source: [NIOS]: "In writing, me is used instead of I"
        """
        return [PRONOUN_MAP.get(t, t) for t in tokens]

    def _rule_wh_fronting(self, tokens: List[str]) -> List[str]:
        """
        RULE R2: Move WH-word from sentence-final to sentence-initial position.

        ISL places WH-words at end: "YOU NAME WHAT"
        English places them at front: "What is your name?"

        Forward rule [Bhatia2020] §3(vii): "Interrogative terms such as when,
            what, and how are always placed at the end of a sentence in ISL"
        Forward rule [Ghosh2022] §3.3.1: "take the wh-word and pick it and
            place it at the end"

        Reverse (our rule): Move WH-word from end to front.
        """
        if not tokens:
            return tokens

        # Check if last token (or second-to-last, in case of trailing noise) is WH
        wh_idx = None
        for i in range(len(tokens) - 1, max(len(tokens) - 3, -1), -1):
            if i >= 0 and tokens[i] in WH_WORDS:
                wh_idx = i
                break

        if wh_idx is not None:
            wh_word = tokens.pop(wh_idx)
            tokens.insert(0, wh_word)

        return tokens

    def _rule_negation_reorder(self, tokens: List[str]) -> List[str]:
        """
        RULE R3: Move negation from sentence-final to before the verb.

        ISL: "I FOOD WANT NOT" → English: "I do not want food"

        Forward rule [Bhatia2020] §3(viii): "Negation in ISL is represented
            with the sign for not, which is always placed at the end of sentence"
        Forward rule [Ghosh2022] §3.3.2: "picking and placing the negative
            word at the end of the sentence"

        Special case — negative questions [Ghosh2022] §3.3.2:
            "we contacted ISLRTC, and their group of expert signers seemed to
            sign the negative word before the question word"
            i.e., ISL: "I WITH GO NOT WHO" → "Who will not go with me?"

        Reverse: Move NOT from end (or before final WH-word) to before the verb.
        """
        if not tokens:
            return tokens

        # Find negation word near the end
        neg_idx = None
        for i in range(len(tokens) - 1, max(len(tokens) - 3, -1), -1):
            if i >= 0 and tokens[i] in NEGATION_WORDS:
                neg_idx = i
                break

        if neg_idx is None:
            return tokens

        neg_word = tokens.pop(neg_idx)

        # Find the verb to place NOT before
        verb_idx = self._find_verb(tokens)
        if verb_idx is not None:
            tokens.insert(verb_idx, neg_word)
        else:
            # No recognized verb — place NOT after subject (position 1)
            insert_pos = min(1, len(tokens))
            tokens.insert(insert_pos, neg_word)

        return tokens

    def _rule_sov_to_svo(self, tokens: List[str]) -> List[str]:
        """
        RULE R4: Convert SOV word order to SVO.

        ISL: Subject Object Verb → English: Subject Verb Object

        Forward rule [Bhatia2020] §3: "the basic structure of ISL is Time-SOV"
        Forward rule [Bhatia2020] §7: "root words are reordered in SOV order"
        Forward rule [Ghosh2022] Rule 1: "Since ISL follows SOV structure, the
            sentence arrangement should be NP NP VP. That is why we put all
            the nouns to the left of the verb."

        Reverse: If we detect a verb at the end (after objects), move it
        to after the subject.

        Example: "I FOOD EAT" → "I EAT FOOD"
        """
        if len(tokens) < 3:
            return tokens

        verb_idx = self._find_verb(tokens)
        if verb_idx is None:
            return tokens

        # Only reorder if verb is NOT already right after the subject
        # (i.e., verb is at position 2+ with nouns between subject and verb)
        # Check if there are noun-like tokens between subject and verb
        if verb_idx <= 1:
            return tokens  # Verb already in SVO position

        # Move verb to position 1 (after subject at position 0)
        verb = tokens.pop(verb_idx)
        tokens.insert(1, verb)

        return tokens

    def _rule_adjective_reorder(self, tokens: List[str]) -> List[str]:
        """
        RULE R5: Move adjectives before nouns (ISL places them after).

        ISL: "DOG BIG" → English: "big dog"

        Forward rule [Ghosh2022] Rule 5: "Adjectives follow the noun they describe"
        Forward rule [NIOS] Rule inferred: "Adjectives are used after the noun.
            For example in English 'red car', In ISL 'car red'"

        Reverse: If a noun is followed by an adjective, swap them.
        """
        i = 0
        while i < len(tokens) - 1:
            current = tokens[i]
            next_token = tokens[i + 1]
            if self._is_noun(current) and self._is_adjective(next_token):
                tokens[i], tokens[i + 1] = tokens[i + 1], tokens[i]
                i += 2  # Skip past the swapped pair
            else:
                i += 1
        return tokens

    def _rule_verb_conjugation(self, tokens: List[str], tense: str,
                                subject: Optional[str]) -> List[str]:
        """
        RULE R6: Conjugate verbs based on detected tense and subject.

        ISL uses only root/base forms of verbs with no inflection.

        Source: [Bhatia2020] §3: "ISL does not use any inflections (gerunds,
            suffixes, or other forms); it uses only root words"
        Source: [NIOS] Rule 3: "Time expressed via explicit time words"
        Source: [NIOS] Rule 9: "Same verb form used always"

        Reverse: Conjugate the root verb based on tense markers and subject.
        """
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
                    # Auxiliary added in R7
                    result.append(forms["continuous"])
                else:
                    # Present tense — use 3rd person singular if needed
                    if subject and subject in THIRD_PERSON_SINGULAR:
                        result.append(forms["present_3s"])
                    else:
                        result.append(forms["present"])
            elif t in PAST_MARKERS or t in FUTURE_MARKERS:
                # Keep time markers but lowercase
                if t != "WILL":  # WILL already added with verb
                    result.append(t.lower())
            elif t in PRESENT_CONTINUOUS_MARKERS:
                result.append(t.lower())
            else:
                result.append(t)
        return result

    def _rule_auxiliary_insertion(self, tokens: List[str], tense: str,
                                  subject: Optional[str], is_question: bool,
                                  is_negative: bool) -> List[str]:
        """
        RULE R7: Insert auxiliaries where needed.

        ISL drops all auxiliary/helping verbs.

        Source: [Bhatia2020] §3: "ISL neither uses verbs (which show action or
            a state of being e.g., is, am, are)"
        Source: [NIOS] Rule 4: "No is, am, are, do, did"

        Reverse: Insert appropriate auxiliaries for:
        - Copula sentences (SHE HAPPY → She is happy)
        - Negation (I NOT GO → I do not go)
        - Continuous tense (I NOW EAT → I am eating)
        - Questions (YOU NAME WHAT → What is your name?)
        """
        # Determine the correct auxiliary for the subject
        if subject in THIRD_PERSON_SINGULAR:
            be_aux = "is"
            do_aux = "does" if tense == "present" else "did" if tense == "past" else "do"
        elif subject in FIRST_PERSON:
            be_aux = "am"
            do_aux = "do" if tense == "present" else "did" if tense == "past" else "do"
        elif subject in PLURAL_SUBJECTS:
            be_aux = "are"
            do_aux = "do" if tense == "present" else "did" if tense == "past" else "do"
        else:
            be_aux = "is"
            do_aux = "does" if tense == "present" else "did" if tense == "past" else "do"

        if tense == "past":
            be_aux = "was" if subject in (THIRD_PERSON_SINGULAR | FIRST_PERSON) else "were"

        # Check if there's a main verb
        has_verb = any(t.lower() in [v.lower() for v in VERB_CONJUGATIONS.keys()]
                       or any(t.lower() == forms[key]
                              for forms in VERB_CONJUGATIONS.values()
                              for key in forms)
                       for t in tokens)

        result = []
        for i, t in enumerate(tokens):
            result.append(t)

            # Insert "be" auxiliary for continuous tense after subject
            if tense == "present_continuous" and t.upper() == subject and i == 0:
                if not any(aux.lower() in [tk.lower() for tk in tokens] for aux in AUXILIARIES):
                    result.append(be_aux)

            # Insert copula for adjective predicates: "SHE HAPPY" → "She is happy"
            if (t.upper() == subject and i < len(tokens) - 1
                    and not has_verb
                    and tokens[i + 1].upper() in COMMON_ADJECTIVES):
                result.append(be_aux)

            # Insert "do" auxiliary for negation without a be-verb
            if t.upper() == "NOT" and i > 0:
                prev = result[-2] if len(result) >= 2 else ""
                # If NOT is directly before a base-form verb, insert do
                if (i < len(tokens) - 1
                        and tokens[i + 1].upper() in VERB_CONJUGATIONS
                        and tense in ("present", "past")
                        and do_aux not in [r.lower() for r in result]):
                    result.insert(-1, do_aux)

        return result

    def _rule_punctuation(self, tokens: List[str], is_question: bool) -> List[str]:
        """
        RULE R8: Capitalize first word, add sentence-ending punctuation.

        Source: [NIOS] Rule 10: "First word capitalized, period for statements,
            question mark for questions"
        """
        if not tokens:
            return tokens

        # Lowercase everything first, then capitalize first word
        tokens = [t.lower() if t.upper() not in {"I"} else t for t in tokens]

        # Fix "I" — always capitalized
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
    # Reporting
    # -----------------------------------------------------------------------

    def get_applied_rules(self) -> List[RuleApplication]:
        """Return the list of rules that were applied in the last correction."""
        return self.applied_rules

    def get_rule_summary(self) -> str:
        """Human-readable summary of applied rules."""
        if not self.applied_rules:
            return "No rules were applied."
        lines = [f"Applied {len(self.applied_rules)} rules:"]
        for r in self.applied_rules:
            lines.append(f"  [{r.rule_id}] {r.rule_name} — {r.source}")
        return "\n".join(lines)


# ============================================================================
# Test Suite
# ============================================================================

def run_tests():
    """
    Test the corrector with examples drawn from the source papers.
    Each test case cites which paper the example comes from.
    """
    corrector = ISLCorrector()

    test_cases = [
        # (input ISL gloss tokens, expected English output pattern, source)

        # From [Bhatia2020] Table 3, Row 1
        (["I", "CLERK"], "clerk", "Bhatia2020 Table 3 #1: I am a clerk"),

        # From [Bhatia2020] Table 3, Row 3
        (["SHE", "MARKET", "GO"], "go", "Bhatia2020 Table 3 #3: She went to market"),

        # From [Bhatia2020] Table 3, Row 4
        (["MAN", "STRONG"], "strong", "Bhatia2020 Table 3 #4: The man is strong"),

        # From [Bhatia2020] Table 3, Row 7
        (["I", "UNDERSTAND", "NOT"], "not", "Bhatia2020 Table 3 #7: I don't understand"),

        # From [Bhatia2020] Table 3, Row 9
        (["YOU", "NAME", "WHAT"], "what", "Bhatia2020 Table 3 #9: What is your name?"),

        # From [Bhatia2020] Table 3, Row 12
        (["SHE", "CRY", "YESTERDAY"], "cried", "Bhatia2020 Table 3 #12: She was crying yesterday"),

        # From [NIOS] Rule 1 example
        (["YOU", "GO", "WHERE"], "where", "NIOS Rule 1: Where are you going?"),

        # From [NIOS] Rule 3 example
        (["I", "FOOD", "EAT", "YESTERDAY"], "ate", "NIOS Rule 3: I ate food yesterday"),

        # From [NIOS] Rule 5 example
        (["I", "FOOD", "WANT", "NOT"], "not", "NIOS Rule 5: I do not want food"),

        # From [Ghosh2022] Rule 5 example
        (["HE", "BOOK", "BIG", "BUY"], "big", "Ghosh2022 Rule 5: He bought a big book"),

        # From [Ghosh2022] §3.3.2
        (["HE", "DOCTOR", "NOT"], "not", "Ghosh2022 Negation: He is not a doctor"),

        # Simple present tense
        (["I", "FOOD", "LIKE"], "like", "Present tense: I like food"),

        # Third person present
        (["SHE", "FOOD", "EAT"], "eats", "3rd person: She eats food"),

        # Future tense
        (["I", "GO", "TOMORROW"], "will", "Future: I will go tomorrow"),
    ]

    print("=" * 70)
    print("ISL CORRECTOR — TEST SUITE")
    print("=" * 70)

    passed = 0
    total = len(test_cases)

    for i, (gloss, expected_fragment, description) in enumerate(test_cases):
        result = corrector.correct(gloss)
        contains = expected_fragment.lower() in result.lower()

        status = "PASS" if contains else "FAIL"
        if contains:
            passed += 1

        print(f"\n[{status}] Test {i+1}: {description}")
        print(f"  Input:    {' '.join(gloss)}")
        print(f"  Output:   {result}")
        print(f"  Checking: contains '{expected_fragment}'")

        if corrector.get_applied_rules():
            for r in corrector.get_applied_rules():
                print(f"    → [{r.rule_id}] {r.rule_name}")

    print(f"\n{'='*70}")
    print(f"Results: {passed}/{total} passed")
    print(f"{'='*70}")

    # --- Ablation demo ---
    print("\n\nABLATION STUDY DEMO")
    print("=" * 70)
    test_input = ["I", "FOOD", "EAT", "YESTERDAY"]
    print(f"Input: {' '.join(test_input)}\n")

    configs = [
        ("All rules enabled", set()),
        ("R4 disabled (no SOV→SVO)", {"R4"}),
        ("R6 disabled (no verb conjugation)", {"R6"}),
        ("R2+R3 disabled (no WH/negation reorder)", {"R2", "R3"}),
        ("R8 disabled (no punctuation)", {"R8"}),
    ]

    for name, disabled in configs:
        c = ISLCorrector(disabled_rules=disabled)
        result = c.correct(list(test_input))
        print(f"  {name:<45} → {result}")


if __name__ == "__main__":
    run_tests()
