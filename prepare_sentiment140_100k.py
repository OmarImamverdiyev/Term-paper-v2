#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import html
import random
import re
import unicodedata
from pathlib import Path

import pandas as pd


# Reproducible sampling seed.
SEED = 42

# Final balanced dataset target.
TARGET_PER_CLASS = 50_000
FINAL_TOTAL_ROWS = TARGET_PER_CLASS * 2

# Stream the large CSV in chunks to keep memory usage reasonable.
CHUNK_SIZE = 100_000

# Default file names for this project.
INPUT_CANDIDATES = ("sentiment140.csv", "Sentiment140.csv")
OUTPUT_FILE = "sentiment140_100k_clean_balanced.csv"


# Text-cleaning and quality-filtering regexes.
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+\b")
MENTION_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
HASHTAG_RE = re.compile(r"(?<!\w)#([A-Za-z][A-Za-z0-9_]*)")
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
RT_PREFIX_RE = re.compile(r"^\s*rt\b[:\s-]*", flags=re.IGNORECASE)
RT_ONLY_RE = re.compile(r"^\s*rt(?:\s+|[:\-])*?$", flags=re.IGNORECASE)
MULTISPACE_RE = re.compile(r"\s+")
LETTER_REPEAT_RE = re.compile(r"([A-Za-z])\1{3,}")
PUNCT_REPEAT_RE = re.compile(r"([!?.,])\1{3,}")
REPEATED_WORD_RE = re.compile(r"\b([A-Za-z]+)(?:\s+\1){2,}\b", flags=re.IGNORECASE)
NOISE_RE = re.compile(r"[ÃÂ�]|â(?:€™|€|œ)|ðŸ")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+",
    flags=re.UNICODE,
)


class ReservoirSampler:
    """Keep a reproducible random sample of a fixed size from a stream."""

    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.items: list[dict[str, object]] = []
        self.seen = 0
        self.rng = random.Random(seed)

    def add(self, item: dict[str, object]) -> None:
        self.seen += 1
        if len(self.items) < self.size:
            self.items.append(item)
            return

        slot = self.rng.randrange(self.seen)
        if slot < self.size:
            self.items[slot] = item


def resolve_input_file() -> Path:
    """Resolve the expected dataset file while tolerating case differences."""
    for candidate in INPUT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find any of these input files in {Path.cwd()}: {', '.join(INPUT_CANDIDATES)}"
    )


def inspect_source_layout(path: Path) -> dict[str, object]:
    """
    Inspect the actual file structure first.

    Sentiment140 is often stored without a header and with six columns:
    sentiment, id, date, query, user, text.
    """
    raw_preview = pd.read_csv(
        path,
        header=None,
        nrows=5,
        encoding="latin-1",
        on_bad_lines="skip",
    )

    if raw_preview.shape[1] >= 6:
        first_col = pd.to_numeric(raw_preview.iloc[:, 0], errors="coerce")
        if first_col.notna().all():
            print("Detected source layout: original 6-column Sentiment140 without header.")
            print("Mapped columns: label -> column 0, text -> column 5")
            return {
                "header": None,
                "names": ["raw_label", "id", "date", "query", "user", "text"],
                "usecols": [0, 5],
                "label_col": "raw_label",
                "text_col": "text",
            }

    header_preview = pd.read_csv(
        path,
        nrows=5,
        encoding="latin-1",
        on_bad_lines="skip",
    )
    normalized = {str(col).strip().lower(): col for col in header_preview.columns}

    label_name = next(
        (normalized[name] for name in ("label", "sentiment", "polarity", "target") if name in normalized),
        None,
    )
    text_name = next(
        (normalized[name] for name in ("text", "tweet", "content", "sentence") if name in normalized),
        None,
    )

    if label_name is None or text_name is None:
        raise RuntimeError(
            "Could not determine the label/text columns automatically. "
            f"Observed columns: {list(header_preview.columns)}"
        )

    print("Detected source layout: header-based CSV.")
    print(f"Mapped columns: label -> {label_name!r}, text -> {text_name!r}")
    return {
        "header": 0,
        "names": None,
        "usecols": [label_name, text_name],
        "label_col": label_name,
        "text_col": text_name,
    }


def normalize_repeated_patterns(text: str) -> str:
    """Tame exaggerated repeated letters/punctuation without destroying meaning."""
    text = LETTER_REPEAT_RE.sub(lambda match: match.group(1) * 2, text)
    text = PUNCT_REPEAT_RE.sub(lambda match: match.group(1) * 3, text)
    return text


def remove_social_only_content(raw_text: str) -> str:
    """Remove social-media markers to check whether any real sentence content remains."""
    text = URL_RE.sub(" ", raw_text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)
    text = re.sub(r"[^A-Za-z\s']", " ", text)
    return MULTISPACE_RE.sub(" ", text).strip()


def clean_tweet_text(raw_text: str) -> str:
    """Clean the tweet while preserving sentiment-bearing words as much as possible."""
    text = html.unescape(str(raw_text))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", " ").replace("\ufeff", " ")
    text = RT_PREFIX_RE.sub("", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)
    text = EMOJI_RE.sub(" ", text)
    text = normalize_repeated_patterns(text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip(" \t\r\n-_:;,.!?")


def looks_high_quality(raw_text: str, cleaned_text: str) -> bool:
    """Apply robust training-oriented filters to remove low-value rows."""
    if not cleaned_text:
        return False
    if RT_ONLY_RE.fullmatch(cleaned_text):
        return False

    raw_without_social = remove_social_only_content(raw_text)
    if not raw_without_social:
        return False

    words = WORD_RE.findall(cleaned_text)
    meaningful_words = [word for word in words if len(word) > 1 or word.lower() in {"i", "a"}]
    if len(meaningful_words) < 3:
        return False

    token_count = len(cleaned_text.split())
    if token_count == 0:
        return False

    alpha_chars = sum(char.isalpha() for char in cleaned_text)
    visible_chars = sum((not char.isspace()) for char in cleaned_text)
    ascii_alpha_chars = sum(("a" <= char.lower() <= "z") for char in cleaned_text if char.isalpha())

    if alpha_chars == 0 or visible_chars == 0:
        return False
    if alpha_chars / visible_chars < 0.45:
        return False
    if len(meaningful_words) / token_count < 0.45:
        return False
    if alpha_chars and ascii_alpha_chars / alpha_chars < 0.85:
        return False

    raw_social_tokens = (
        len(URL_RE.findall(raw_text))
        + len(MENTION_RE.findall(raw_text))
        + len(HASHTAG_RE.findall(raw_text))
    )
    raw_token_count = max(len(str(raw_text).split()), 1)
    if raw_social_tokens / raw_token_count > 0.7 and len(meaningful_words) < 4:
        return False

    if REPEATED_WORD_RE.search(cleaned_text):
        return False
    if len(LETTER_REPEAT_RE.findall(raw_text)) >= 2 and len(meaningful_words) < 5:
        return False
    if NOISE_RE.search(cleaned_text) and alpha_chars / visible_chars < 0.7:
        return False

    return True


def text_dedup_key(cleaned_text: str) -> bytes:
    """Build a compact dedupe key for exact-duplicate detection after cleaning."""
    normalized = MULTISPACE_RE.sub(" ", cleaned_text).strip().casefold()
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).digest()


def iter_source_chunks(path: Path, layout: dict[str, object]):
    """Yield only the source label and text columns as pandas chunks."""
    yield from pd.read_csv(
        path,
        header=layout["header"],
        names=layout["names"],
        usecols=layout["usecols"],
        encoding="latin-1",
        chunksize=CHUNK_SIZE,
        on_bad_lines="skip",
    )


def prepare_dataset(input_path: Path, output_path: Path) -> None:
    """Create the final balanced 100k dataset and print a concise summary."""
    layout = inspect_source_layout(input_path)

    class_samplers = {
        0: ReservoirSampler(size=TARGET_PER_CLASS, seed=SEED),
        1: ReservoirSampler(size=TARGET_PER_CLASS, seed=SEED + 1),
    }
    usable_counts = {0: 0, 1: 0}
    seen_hashes: set[bytes] = set()
    duplicates_removed = 0
    original_rows = 0

    for chunk_number, chunk in enumerate(iter_source_chunks(input_path, layout), start=1):
        label_col = layout["label_col"]
        text_col = layout["text_col"]

        original_rows += len(chunk)
        chunk = chunk.rename(columns={label_col: "raw_label", text_col: "raw_text"})
        chunk = chunk[["raw_label", "raw_text"]].copy()
        chunk["raw_label"] = pd.to_numeric(chunk["raw_label"], errors="coerce")
        chunk = chunk[chunk["raw_label"].isin([0, 4])]

        if chunk.empty:
            continue

        for row in chunk.itertuples(index=False):
            raw_label = int(row.raw_label)
            raw_text = "" if pd.isna(row.raw_text) else str(row.raw_text)

            cleaned_text = clean_tweet_text(raw_text)
            if not looks_high_quality(raw_text, cleaned_text):
                continue

            dedupe_hash = text_dedup_key(cleaned_text)
            if dedupe_hash in seen_hashes:
                duplicates_removed += 1
                continue
            seen_hashes.add(dedupe_hash)

            mapped_label = 0 if raw_label == 0 else 1
            usable_counts[mapped_label] += 1
            class_samplers[mapped_label].add({"text": cleaned_text, "label": mapped_label})

        if chunk_number % 5 == 0:
            print(
                f"Processed {original_rows:,} rows | usable negatives: {usable_counts[0]:,} | "
                f"usable positives: {usable_counts[1]:,}"
            )

    if usable_counts[0] < TARGET_PER_CLASS or usable_counts[1] < TARGET_PER_CLASS:
        raise RuntimeError(
            "Not enough usable rows remained after filtering to build an exact balanced 100k subset. "
            f"Usable negatives={usable_counts[0]:,}, usable positives={usable_counts[1]:,}"
        )

    final_df = pd.DataFrame(class_samplers[0].items + class_samplers[1].items, columns=["text", "label"])
    final_df = final_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    if len(final_df) != FINAL_TOTAL_ROWS:
        raise RuntimeError(f"Expected {FINAL_TOTAL_ROWS:,} final rows, found {len(final_df):,}.")

    final_counts = final_df["label"].value_counts().sort_index().to_dict()
    if final_counts.get(0, 0) != TARGET_PER_CLASS or final_counts.get(1, 0) != TARGET_PER_CLASS:
        raise RuntimeError(f"Final class balance is incorrect: {final_counts}")

    final_df.to_csv(output_path, index=False, columns=["text", "label"], encoding="utf-8")

    usable_total = usable_counts[0] + usable_counts[1]
    print("\nSummary")
    print(f"- Original row count: {original_rows:,}")
    print(f"- Remaining usable rows after filtering: {usable_total:,}")
    print(f"- Final class counts: {final_counts}")
    print(f"- Duplicates removed: {duplicates_removed:,}")
    print("- Sample rows:")
    print(final_df.head(5).to_string(index=False))
    print(f"\nSaved cleaned dataset to: {output_path}")


def main() -> None:
    input_path = resolve_input_file()
    output_path = Path(OUTPUT_FILE)
    prepare_dataset(input_path=input_path, output_path=output_path)


if __name__ == "__main__":
    main()
