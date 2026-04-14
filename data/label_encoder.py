"""
label_encoder.py -- Tokenize iSign labels and build vocabulary.

The iSign dataset (Exploration-Lab/iSign) contains video-sentence pairs.
This module supports TWO label modes:

  1. GLOSS mode: If the CSV has a gloss column (e.g. "gloss", "glosses",
     "isl_gloss"), those are used as labels. This produces a true ISL
     gloss recognizer for the pipeline: video -> gloss -> grammar correction.

  2. ENGLISH mode: If only English translations are available, those are
     used as word-level labels. This is English-word CTC, NOT gloss
     recognition. The grammar correction stage would be redundant in
     this mode. The code documents this clearly.

Pipeline:
    1. Parse CSV to detect label column (prefer gloss over english).
    2. Tokenize labels into word-level tokens (lowercased, cleaned).
    3. Build vocabulary: unique token -> integer ID.
    4. Reserve ID 0 for CTC blank token.
    5. Save vocabulary as JSON.

Usage:
    python -m data.label_encoder
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean and normalize a text string for tokenization.

    Steps:
        1. Lowercase.
        2. Remove punctuation except apostrophes (e.g. "don't" stays).
        3. Collapse multiple spaces.
        4. Strip leading/trailing whitespace.

    Args:
        text: Raw text string (English sentence or gloss sequence).

    Returns:
        Cleaned text string.
    """
    text = text.lower().strip()
    # Keep apostrophes for contractions, remove other punctuation
    text = re.sub(r"[^\w\s']", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """
    Tokenize a text string into word-level tokens.

    Args:
        text: Raw text string.

    Returns:
        List of word tokens.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return cleaned.split()


# ---------------------------------------------------------------------------
# Vocabulary building
# ---------------------------------------------------------------------------

class LabelEncoder:
    """
    Encode text labels into integer sequences using a word-level vocabulary.

    The vocabulary maps each unique word/gloss to an integer ID, with
    ID 0 reserved for the CTC blank token.

    Attributes:
        word2id: Dict mapping word -> integer ID.
        id2word: Dict mapping integer ID -> word.
        blank_id: The CTC blank token ID (always 0).
        label_type: "gloss" or "english" -- what kind of labels we encode.
    """

    def __init__(
        self,
        config: Config,
        word2id: Optional[Dict[str, int]] = None,
        label_type: str = "english",
    ):
        """
        Args:
            config: Pipeline configuration.
            word2id: Pre-built vocabulary. If None, must call build_vocab()
                     or load() before encoding.
            label_type: "gloss" or "english" -- the semantic type of labels.
        """
        self.config = config
        self.blank_id = config.ctc_blank_id
        self.word2id: Dict[str, int] = word2id or {}
        self.id2word: Dict[int, str] = {v: k for k, v in self.word2id.items()}
        self.label_type: str = label_type

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including the blank token."""
        return len(self.word2id)

    def build_vocab(
        self,
        sentences: List[str],
        min_freq: int = 1,
    ) -> None:
        """
        Build vocabulary from a list of label strings.

        ID 0 is always reserved for the CTC blank token.
        ID 1 is reserved for <unk> (unseen tokens at inference).
        Word IDs start at 2.

        Args:
            sentences: List of raw label strings (glosses or English).
            min_freq: Minimum frequency for a token to be included.
        """
        # Count token frequencies
        freq: Dict[str, int] = {}
        for sentence in sentences:
            for word in tokenize(sentence):
                freq[word] = freq.get(word, 0) + 1

        # Filter by min_freq and sort alphabetically for determinism
        words = sorted(
            [w for w, c in freq.items() if c >= min_freq]
        )

        # Build mapping: 0 = blank, 1 = <unk>, 2+ = tokens
        self.word2id = {"<blank>": self.blank_id, "<unk>": 1}
        for i, word in enumerate(words, start=2):
            self.word2id[word] = i

        self.id2word = {v: k for k, v in self.word2id.items()}
        logger.info(
            f"Vocabulary built: {self.vocab_size} tokens "
            f"({self.label_type} mode, including <blank> and <unk>)"
        )

    def encode(self, text: str) -> List[int]:
        """
        Encode a label string into a list of integer token IDs.

        Unknown tokens are mapped to <unk> (ID 1).

        Args:
            text: Raw label string.

        Returns:
            List of integer IDs.
        """
        unk_id = self.word2id.get("<unk>", 1)
        tokens = tokenize(text)
        return [self.word2id.get(w, unk_id) for w in tokens]

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of integer IDs back into a string.

        Blank tokens (ID 0) are skipped.

        Args:
            ids: List of integer token IDs.

        Returns:
            Decoded string.
        """
        words = []
        for token_id in ids:
            if token_id == self.blank_id:
                continue
            word = self.id2word.get(token_id, "<unk>")
            words.append(word)
        return " ".join(words)

    def save(self, path: Optional[str] = None) -> None:
        """
        Save vocabulary to a JSON file.

        Also saves label_type so loaders know what the vocab represents.

        Args:
            path: File path. Defaults to config.vocab_path.
        """
        path = path or self.config.vocab_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "_label_type": self.label_type,
            "_vocab_size": self.vocab_size,
            "word2id": self.word2id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"Vocabulary saved to {path} ({self.vocab_size} tokens, {self.label_type})")

    def load(self, path: Optional[str] = None) -> None:
        """
        Load vocabulary from a JSON file.

        Args:
            path: File path. Defaults to config.vocab_path.

        Raises:
            FileNotFoundError: If the vocabulary file does not exist.
        """
        path = path or self.config.vocab_path
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Vocabulary file not found at {path}. "
                "Build the vocabulary first with build_vocab()."
            )
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # Handle both old format (flat dict) and new format (nested)
        if "word2id" in payload:
            self.word2id = payload["word2id"]
            self.label_type = payload.get("_label_type", "english")
        else:
            # Legacy flat format
            self.word2id = payload
            self.label_type = "english"

        self.id2word = {v: k for k, v in self.word2id.items()}
        logger.info(
            f"Vocabulary loaded from {path} "
            f"({self.vocab_size} tokens, {self.label_type})"
        )


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

# Column name candidates in priority order (gloss preferred over english)
_GLOSS_COLUMNS = ["gloss", "glosses", "isl_gloss", "isl", "sign", "signs"]
_ENGLISH_COLUMNS = ["english", "text", "translation", "sentence"]
_VIDEO_ID_COLUMNS = ["video_id", "id", "vid", "name", "filename"]


def parse_isign_csv(
    csv_path: str,
) -> Tuple[List[Dict[str, str]], str]:
    """
    Parse the iSign CSV file to extract video IDs and labels.

    Column detection priority:
        1. Look for a GLOSS column first (gloss, glosses, isl_gloss, ...).
           If found, labels are ISL glosses -> label_type = "gloss".
        2. Fall back to an ENGLISH column (english, text, translation, ...).
           Labels are English words -> label_type = "english".

    When label_type is "english", the system is doing English-word CTC,
    NOT ISL gloss recognition. The downstream grammar correction stage
    would be redundant in this mode.

    Args:
        csv_path: Path to the iSign CSV file.

    Returns:
        records:    List of dicts with keys 'video_id' and 'text'.
        label_type: "gloss" if gloss column found, "english" otherwise.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If neither gloss nor english columns are found.
    """
    import csv

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    records = []
    label_type = "english"  # default

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # Detect columns
        vid_col = _find_column(fieldnames, _VIDEO_ID_COLUMNS)

        # Prefer gloss over english
        gloss_col = _find_column(fieldnames, _GLOSS_COLUMNS)
        english_col = _find_column(fieldnames, _ENGLISH_COLUMNS)

        if gloss_col is not None:
            label_col = gloss_col
            label_type = "gloss"
            logger.info(
                f"Found gloss column '{gloss_col}' -- using ISL gloss labels"
            )
        elif english_col is not None:
            label_col = english_col
            label_type = "english"
            logger.warning(
                f"No gloss column found in CSV. "
                f"Using English column '{english_col}' as labels. "
                f"This is English-word CTC, NOT gloss recognition. "
                f"Available columns: {fieldnames}"
            )
        else:
            raise ValueError(
                f"CSV must have a gloss or english column. "
                f"Found: {fieldnames}"
            )

        if vid_col is None:
            raise ValueError(
                f"CSV must have a video_id column. "
                f"Found: {fieldnames}"
            )

        for row in reader:
            video_id = row[vid_col].strip()
            text = row[label_col].strip()
            if video_id and text:
                records.append({"video_id": video_id, "text": text})

    logger.info(
        f"Parsed {len(records)} records from {csv_path} "
        f"(label_type={label_type})"
    )
    return records, label_type


def _find_column(
    fieldnames: List[str],
    candidates: List[str],
) -> Optional[str]:
    """Find the first matching column name (case-insensitive)."""
    fieldnames_lower = {f.lower().strip(): f for f in fieldnames}
    for candidate in candidates:
        if candidate in fieldnames_lower:
            return fieldnames_lower[candidate]
    return None


# ---------------------------------------------------------------------------
# Convenience: build everything from CSV
# ---------------------------------------------------------------------------

def build_label_encoder_from_csv(
    csv_path: str,
    config: Config,
    min_freq: int = 1,
) -> Tuple[LabelEncoder, List[Dict[str, str]]]:
    """
    Build a LabelEncoder from the iSign CSV file.

    Args:
        csv_path: Path to the iSign CSV.
        config:   Pipeline configuration.
        min_freq: Minimum word frequency to include in vocab.

    Returns:
        encoder: Trained LabelEncoder.
        records: List of {'video_id': ..., 'text': ...} dicts.
    """
    records, label_type = parse_isign_csv(csv_path)
    sentences = [r["text"] for r in records]

    encoder = LabelEncoder(config, label_type=label_type)
    encoder.build_vocab(sentences, min_freq=min_freq)
    encoder.save()

    return encoder, records


# ---------------------------------------------------------------------------
# Standalone verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = Config()
    print("=" * 60)
    print("label_encoder.py -- verification")
    print("=" * 60)

    # Build vocabulary from sample sentences
    sample_sentences = [
        "Hello how are you",
        "I am fine thank you",
        "What is your name",
        "My name is Hrishi",
        "Please sit down",
        "Thank you very much",
        "How are you doing today",
        "I don't understand",
        "Can you repeat that",
        "Good morning everyone",
    ]

    encoder = LabelEncoder(cfg, label_type="english")
    encoder.build_vocab(sample_sentences)

    print(f"  Vocab size:   {encoder.vocab_size}")
    print(f"  Blank ID:     {encoder.blank_id}")
    print(f"  Label type:   {encoder.label_type}")
    print(f"  Sample words: {list(encoder.word2id.items())[:8]}")

    # Encode / decode roundtrip
    test_sentence = "Hello how are you"
    encoded = encoder.encode(test_sentence)
    decoded = encoder.decode(encoded)
    print(f"  Original:  '{test_sentence}'")
    print(f"  Encoded:   {encoded}")
    print(f"  Decoded:   '{decoded}'")

    # Test unknown word handling
    unk_encoded = encoder.encode("Hello xyz unknown")
    print(f"  Unknown test: {unk_encoded}")
    assert unk_encoded[1] == 1, "Unknown words should map to <unk> (ID 1)"
    assert unk_encoded[2] == 1, "Unknown words should map to <unk> (ID 1)"

    # Save / load roundtrip
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    tmp_path = tmp.name
    tmp.close()
    encoder.save(tmp_path)
    encoder2 = LabelEncoder(cfg)
    encoder2.load(tmp_path)
    os.unlink(tmp_path)

    assert encoder.word2id == encoder2.word2id, "Vocab mismatch after save/load"
    assert encoder2.label_type == "english", "Label type not preserved"
    print("  Save/load roundtrip: PASS")

    # Test tokenization
    assert tokenize("Hello, World!") == ["hello", "world"]
    assert tokenize("  spaces   galore  ") == ["spaces", "galore"]
    assert tokenize("don't stop") == ["don't", "stop"]
    print("  Tokenization tests:  PASS")

    print("=" * 60)
    print("[PASS] label_encoder.py OK")
