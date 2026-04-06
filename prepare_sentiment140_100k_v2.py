#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import re
import unicodedata
from pathlib import Path
from typing import Callable

import pandas as pd


SEED = 42
DEFAULT_INPUT = Path("sentiment140_100k_clean_balanced.csv")
DEFAULT_OUTPUT = Path("sentiment140_100k_clean_balanced_v2.csv")

# Core normalization patterns.
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+\b")
MENTION_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
HASHTAG_RE = re.compile(r"(?<!\w)#([A-Za-z][A-Za-z0-9_]*)")
WHITESPACE_RE = re.compile(r"\s+")
EXCESS_REPEAT_RE = re.compile(r"([a-z])\1{3,}", flags=re.IGNORECASE)
EXCESS_EXCLAMATION_RE = re.compile(r"!{3,}")
EXCESS_QUESTION_RE = re.compile(r"\?{3,}")
EXCESS_DOT_RE = re.compile(r"\.{4,}")
BROKEN_UNICODE_RE = re.compile(r"[�]+")

# Light quality checks after cleaning.
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]",
    flags=re.UNICODE,
)
EMOTICON_RE = re.compile(
    r"(?x)"
    r"(?:[:;=8][\-o\*']?[\)\]\(\[dDpP/\:\}\{@\|\\])"
    r"|(?:<3)"
)
PLACEHOLDER_ONLY_RE = re.compile(r"^(?:@user|[\W_])+$", flags=re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Further normalize sentiment140_100k_clean_balanced.csv while preserving "
            "sentiment meaning and class balance."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mode",
        choices=("transformer", "classical"),
        default="transformer",
        help=(
            "Cleaning profile to apply. 'transformer' is the default gentle pipeline. "
            "'classical' is a slightly more aggressive alternative for TF-IDF style models."
        ),
    )
    return parser.parse_args()


def base_unicode_cleanup(text: str) -> str:
    """Normalize Unicode and decode lightweight HTML noise."""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", " ").replace("\ufeff", " ").replace("\xa0", " ")
    return text


def normalize_repeated_characters(text: str, max_repeats: int = 3) -> str:
    """Reduce exaggerated character runs while preserving emphasis."""
    return EXCESS_REPEAT_RE.sub(lambda match: match.group(1) * max_repeats, text)


def normalize_punctuation_runs(text: str) -> str:
    """Keep expressive punctuation, but avoid overly noisy repetitions."""
    text = EXCESS_EXCLAMATION_RE.sub("!!", text)
    text = EXCESS_QUESTION_RE.sub("??", text)
    text = EXCESS_DOT_RE.sub("...", text)
    return text


def normalize_hashtags(text: str) -> str:
    """Keep hashtag meaning while dropping the hash symbol itself."""
    return HASHTAG_RE.sub(lambda match: match.group(1), text)


def normalize_mentions(text: str, placeholder: str = "@user") -> str:
    """Replace raw usernames with a stable placeholder."""
    return MENTION_RE.sub(placeholder, text)


def minimal_noise_cleanup(text: str) -> str:
    """Remove obvious broken characters without stripping useful symbols."""
    return BROKEN_UNICODE_RE.sub(" ", text)


def final_spacing_cleanup(text: str) -> str:
    """Normalize whitespace and trim the cleaned text."""
    return WHITESPACE_RE.sub(" ", text).strip()


def clean_text_for_transformer(raw_text: object) -> str:
    """
    Main cleaning pipeline for transformer/BERT-style models.

    This preserves punctuation, emojis, emoticons, negation, and most informal language,
    while still normalizing the noisiest social-media artifacts.
    """
    text = "" if pd.isna(raw_text) else str(raw_text)
    text = base_unicode_cleanup(text)
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = normalize_mentions(text, placeholder="@user")
    text = normalize_hashtags(text)
    text = normalize_repeated_characters(text, max_repeats=3)
    text = normalize_punctuation_runs(text)
    text = minimal_noise_cleanup(text)
    text = final_spacing_cleanup(text)
    return text


def clean_text_for_classical(raw_text: object) -> str:
    """
    Optional slightly more aggressive pipeline for classical ML models.

    It is not used by default. Compared with the transformer pipeline, it removes the
    mention placeholder and smooths punctuation further to reduce sparse token noise.
    """
    text = clean_text_for_transformer(raw_text)
    text = text.replace("@user", " ")
    text = re.sub(r"([!?]){2,}", r"\1", text)
    text = final_spacing_cleanup(text)
    return text


def is_usable_text(text: str) -> bool:
    """Drop rows that become empty or mostly meaningless after cleaning."""
    if not text:
        return False
    if PLACEHOLDER_ONLY_RE.fullmatch(text):
        return False

    words = WORD_RE.findall(text)
    has_word = bool(words)
    has_emoji = bool(EMOJI_RE.search(text))
    has_emoticon = bool(EMOTICON_RE.search(text))

    # Remove rows that collapse to placeholders, punctuation, or tiny fragments.
    if not has_word and not has_emoji and not has_emoticon:
        return False

    visible = re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)
    if not visible:
        return False

    placeholder_stripped = re.sub(r"@user", " ", text)
    placeholder_stripped = re.sub(r"[!?.,:;'\-_/()\[\]{}]+", " ", placeholder_stripped)
    placeholder_stripped = final_spacing_cleanup(placeholder_stripped)
    stripped_words = WORD_RE.findall(placeholder_stripped)

    if not stripped_words and not has_emoji and not has_emoticon:
        return False

    if len(stripped_words) == 1 and len(stripped_words[0]) == 1 and not (has_emoji or has_emoticon):
        return False

    return True


def choose_cleaner(mode: str) -> Callable[[object], str]:
    if mode == "classical":
        return clean_text_for_classical
    return clean_text_for_transformer


def load_source_dataframe(path: Path) -> pd.DataFrame:
    """Load and validate the expected 2-column balanced source dataset."""
    df = pd.read_csv(path, usecols=["text", "label"], on_bad_lines="skip")
    df = df[["text", "label"]].copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    return df


def remove_conflicting_and_duplicate_texts(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """
    Remove exact duplicate cleaned texts.

    If a cleaned text appears under conflicting labels, remove all copies of that text
    because it becomes ambiguous supervision after normalization.
    """
    label_nunique = df.groupby("text")["label"].nunique()
    conflicting_texts = set(label_nunique[label_nunique > 1].index)
    conflicting_removed = int(df["text"].isin(conflicting_texts).sum())
    if conflicting_texts:
        df = df[~df["text"].isin(conflicting_texts)].copy()

    before_drop = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first").copy()
    duplicate_removed = before_drop - len(df)
    return df, duplicate_removed, conflicting_removed


def rebalance_dataset(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Downsample the larger class so the final dataset remains exactly balanced."""
    class_counts = df["label"].value_counts().to_dict()
    if not class_counts:
        return df.iloc[0:0].copy()

    target_per_class = min(class_counts.values())
    balanced_parts = []
    for label in sorted(class_counts):
        class_df = df[df["label"] == label].copy()
        if len(class_df) > target_per_class:
            class_df = class_df.sample(n=target_per_class, random_state=seed)
        balanced_parts.append(class_df)

    balanced = pd.concat(balanced_parts, ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return balanced[["text", "label"]]


def run_cleaning(input_path: Path, output_path: Path, mode: str) -> None:
    """Execute the cleaning pipeline and print a concise audit summary."""
    cleaner = choose_cleaner(mode)
    df = load_source_dataframe(input_path)

    input_row_count = len(df)
    class_counts_before = df["label"].value_counts().sort_index().to_dict()

    # Apply the selected text normalization pipeline.
    df["text"] = df["text"].map(cleaner)

    # Drop rows that become empty or unusable after cleaning.
    unusable_mask = ~df["text"].map(is_usable_text)
    unusable_removed = int(unusable_mask.sum())
    df = df.loc[~unusable_mask, ["text", "label"]].copy()

    # Remove exact duplicates and ambiguous conflicting texts created by cleaning.
    df, duplicate_removed, conflicting_removed = remove_conflicting_and_duplicate_texts(df)

    # Downsample the larger class if needed so final labels remain balanced.
    df = rebalance_dataset(df, seed=SEED)

    output_row_count = len(df)
    class_counts_after = df["label"].value_counts().sort_index().to_dict()
    rows_removed_total = input_row_count - output_row_count

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, columns=["text", "label"], encoding="utf-8")

    print(f"Cleaning mode: {mode}")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print("")
    print("Summary")
    print(f"- Input row count: {input_row_count:,}")
    print(f"- Output row count: {output_row_count:,}")
    print(f"- Rows removed: {rows_removed_total:,}")
    print(f"- Duplicate count removed: {duplicate_removed:,}")
    print(f"- Conflicting duplicate rows removed: {conflicting_removed:,}")
    print(f"- Rows removed as unusable after cleaning: {unusable_removed:,}")
    print(f"- Class counts before: {class_counts_before}")
    print(f"- Class counts after: {class_counts_after}")
    print("- 10 example cleaned rows:")
    print(df.head(10).to_string(index=False))


def main() -> None:
    args = parse_args()
    run_cleaning(input_path=args.input, output_path=args.output, mode=args.mode)


if __name__ == "__main__":
    main()
