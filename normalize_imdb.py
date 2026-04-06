#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import re
import unicodedata
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("imdb_dataset.csv")
DEFAULT_OUTPUT = Path("imdb_dataset_normalized.csv")

# Set to False if you want to keep stopwords in `text_ml`.
ML_REMOVE_STOPWORDS = True

SMART_CHAR_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
        "\u200b": " ",
        "\ufeff": " ",
    }
)

HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
ML_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
NON_ALPHA_RE = re.compile(r"[^a-z]")
EXCLAMATION_RUN_RE = re.compile(r"!{4,}")
QUESTION_RUN_RE = re.compile(r"\?{4,}")
DOT_RUN_RE = re.compile(r"\.{4,}")
SPECIAL_NEGATION_PATTERNS = (
    (re.compile(r"\bcan't\b"), "can not"),
    (re.compile(r"\bcannot\b"), "can not"),
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bshan't\b"), "shall not"),
    (re.compile(r"\bain't\b"), "is not"),
)
GENERIC_NT_RE = re.compile(r"\b([a-z]+)n't\b")

NEGATION_WORDS = {"not", "no", "never"}
FALLBACK_STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for local dataset normalization."""
    parser = argparse.ArgumentParser(
        description="Normalize an IMDb sentiment dataset for ML and deep-learning model families."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the input CSV file. Default: imdb_dataset.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the output CSV file. Default: imdb_dataset_normalized.csv",
    )
    return parser.parse_args()


def load_stopwords() -> set[str]:
    """
    Load English stopwords with a safe fallback.

    Negation words are intentionally excluded so they are preserved in `text_ml`.
    """
    try:
        import nltk
        from nltk.corpus import stopwords

        nltk.data.find("corpora/stopwords")
        words = set(stopwords.words("english"))
    except Exception:
        words = set(FALLBACK_STOPWORDS)

    return words - NEGATION_WORDS


def build_lemmatizer():
    """
    Build a WordNet lemmatizer when NLTK and the WordNet corpus are available.

    If resources are missing, the script falls back to no lemmatization.
    """
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer

        nltk.data.find("corpora/wordnet")
        return WordNetLemmatizer()
    except Exception:
        return None


STOPWORDS = load_stopwords()
LEMMATIZER = build_lemmatizer()


def clean_common(text: object) -> str:
    """
    Apply shared cleanup used by both `text_ml` and `text_dl`.

    Steps:
    - replace missing values with an empty string
    - unescape HTML entities
    - normalize common Unicode punctuation/noise
    - remove HTML tags
    - normalize whitespace
    - strip edges
    - lowercase
    """
    if pd.isna(text):
        return ""

    cleaned = str(text)
    cleaned = html.unescape(cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = cleaned.translate(SMART_CHAR_TRANSLATION)
    cleaned = unicodedata.normalize("NFKC", cleaned)
    cleaned = HTML_TAG_RE.sub(" ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = HTML_TAG_RE.sub(" ", cleaned)
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip().lower()


def expand_negation_contractions(text: str) -> str:
    """Expand common negation contractions so negation survives ML token cleanup."""
    expanded = text
    for pattern, replacement in SPECIAL_NEGATION_PATTERNS:
        expanded = pattern.sub(replacement, expanded)
    expanded = GENERIC_NT_RE.sub(r"\1 not", expanded)
    return expanded


def lemmatize_token(token: str) -> str:
    """
    Lemmatize a token when a lemmatizer is available.

    The pass order is a simple heuristic that improves normalization without
    requiring a full POS tagger.
    """
    if not token or LEMMATIZER is None:
        return token

    try:
        token = LEMMATIZER.lemmatize(token, pos="n")
        token = LEMMATIZER.lemmatize(token, pos="v")
        token = LEMMATIZER.lemmatize(token, pos="a")
        token = LEMMATIZER.lemmatize(token, pos="r")
        return token
    except Exception:
        return token


def normalize_ml_from_common(text: str) -> str:
    """
    Create a compact representation for CountVectorizer / TF-IDF style models.

    This path removes punctuation/special characters, keeps alphabetic tokens,
    optionally removes stopwords, applies lemmatization when available, and joins
    short negation patterns such as `not good` -> `not_good`.
    """
    expanded = expand_negation_contractions(text)
    raw_tokens = ML_TOKEN_RE.findall(expanded)
    normalized_tokens: list[str] = []
    index = 0

    while index < len(raw_tokens):
        token = NON_ALPHA_RE.sub("", raw_tokens[index])
        if not token:
            index += 1
            continue

        if token in NEGATION_WORDS and index + 1 < len(raw_tokens):
            next_token = NON_ALPHA_RE.sub("", raw_tokens[index + 1])
            if next_token and next_token not in STOPWORDS and next_token not in NEGATION_WORDS:
                next_token = lemmatize_token(next_token)
                normalized_tokens.append(f"{token}_{next_token}")
                index += 2
                continue

        token = lemmatize_token(token)
        if ML_REMOVE_STOPWORDS and token in STOPWORDS:
            index += 1
            continue

        normalized_tokens.append(token)
        index += 1

    return " ".join(normalized_tokens)


def normalize_dl_from_common(text: str) -> str:
    """
    Create a light-cleaned representation for sequence models.

    Sentence structure, stopwords, punctuation, and natural negation wording are
    intentionally preserved as much as possible.
    """
    normalized = EXCLAMATION_RUN_RE.sub("!!!", text)
    normalized = QUESTION_RUN_RE.sub("???", normalized)
    normalized = DOT_RUN_RE.sub("...", normalized)
    normalized = WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()


def normalize_ml(text: object) -> str:
    """Public ML normalization entry point."""
    return normalize_ml_from_common(clean_common(text))


def normalize_dl(text: object) -> str:
    """Public deep-learning normalization entry point."""
    return normalize_dl_from_common(clean_common(text))


def validate_columns(df: pd.DataFrame) -> None:
    """Ensure the input dataset has the required columns."""
    required_columns = {"text", "label"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing_str}")


def normalize_dataset(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Read the source CSV, add normalized outputs, and save the final dataset."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    validate_columns(df)

    # Preserve the original text column while safely replacing missing values.
    df = df[["text", "label"]].copy()
    df["text"] = df["text"].fillna("").astype(str)

    # Compute the common-cleaned text once, then derive the ML and DL variants.
    common_text = df["text"].map(clean_common)
    df["text_ml"] = common_text.map(normalize_ml_from_common)
    df["text_dl"] = common_text.map(normalize_dl_from_common)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        output_path,
        index=False,
        columns=["text", "label", "text_ml", "text_dl"],
        encoding="utf-8",
    )
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print the required processing summary and one example row."""
    print(f"Total rows processed: {len(df):,}")

    print("Example:")
    if df.empty:
        print("original text: ")
        print("text_ml: ")
        print("text_dl: ")
        return

    sample = df.iloc[0]
    print(f"original text: {sample['text']}")
    print(f"text_ml: {sample['text_ml']}")
    print(f"text_dl: {sample['text_dl']}")


def main() -> None:
    """CLI entry point for local execution."""
    args = parse_args()
    normalized_df = normalize_dataset(input_path=args.input, output_path=args.output)
    print_summary(normalized_df)


if __name__ == "__main__":
    main()
