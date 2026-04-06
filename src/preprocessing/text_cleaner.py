from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from src.utils.filesystem import resolve_path


SMART_PUNCT_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "`": "'",
    }
)
URL_PATTERN = re.compile(r"(?i)(?:https?://|www\.)\S+")
MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
NUMBER_PATTERN = re.compile(r"\d+(?:[.,:/-]\d+)*")
WORD_PATTERN = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*", flags=re.UNICODE)
PUNCT_RUN_PATTERN = re.compile(r"[!?]+|\.{2,}")
EMOTICON_PATTERN = re.compile(
    r"""(?ix)
    (?:
        (?:[:;=8][\-o\*']?[\)\]\(\[dDpP/\:\}\{@\|\\])
        |
        (?:[\)\]\(\[dDpP/\:\}\{@\|\\][\-o\*']?[:;=8])
        |
        <3
    )
    """
)
TOKEN_PATTERN = re.compile(
    r"""(?ix)
    (?:https?://|www\.)\S+
    |@[A-Za-z0-9_]+
    |\#[A-Za-z0-9_]+
    |<3
    |(?:(?:[:;=8][\-o\*']?[\)\]\(\[dDpP/\:\}\{@\|\\])|(?:[\)\]\(\[dDpP/\:\}\{@\|\\][\-o\*']?[:;=8]))
    |[^\W\d_]+(?:'[^\W\d_]+)*
    |\d+(?:[.,:/-]\d+)*
    |[!?]+
    |\.{2,}
    |[\u2600-\u26FF\u2700-\u27BF\U0001F300-\U0001FAFF]
    |[^\w\s]
    """
)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]",
    flags=re.UNICODE,
)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_text(text: str) -> str:
    return _normalize_whitespace(str(text).translate(SMART_PUNCT_TRANSLATION))


def _is_emoji(token: str) -> bool:
    return bool(EMOJI_PATTERN.search(token))


def _is_emoticon(token: str) -> bool:
    return bool(EMOTICON_PATTERN.fullmatch(token))


def _compress_punctuation_run(token: str) -> str:
    if token.startswith("."):
        return "..."
    if set(token) == {"!"}:
        return "!" if len(token) == 1 else "!!" if len(token) == 2 else "!!!"
    if set(token) == {"?"}:
        return "?" if len(token) == 1 else "??" if len(token) == 2 else "???"
    if set(token).issubset({"!", "?"}):
        return "!?"
    return token


def _load_stopwords(stopword_cfg: dict[str, Any], config_dir: Path | None) -> set[str]:
    if not stopword_cfg.get("enabled", False):
        return set()

    source = str(stopword_cfg.get("source", "list")).lower()
    if source == "list":
        words = stopword_cfg.get("words", [])
        return {str(word).strip().lower() for word in words if str(word).strip()}

    if source == "file":
        if not stopword_cfg.get("path"):
            raise ValueError("Stopword source 'file' requires a 'path'.")
        if config_dir is None:
            raise ValueError("Relative stopword paths require a config directory.")
        stopword_path = resolve_path(stopword_cfg["path"], config_dir=config_dir)
        with stopword_path.open("r", encoding="utf-8") as handle:
            return {line.strip().lower() for line in handle if line.strip()}

    if source == "nltk":
        try:
            from nltk.corpus import stopwords
        except ImportError as exc:
            raise ImportError("NLTK is required for stopword source 'nltk'.") from exc

        language = stopword_cfg.get("language")
        if not language:
            raise ValueError("Stopword source 'nltk' requires a language.")
        return {word.strip().lower() for word in stopwords.words(language)}

    raise ValueError(f"Unsupported stopword source: {source}")


def _build_normalizer(normalization_cfg: dict[str, Any]) -> Callable[[str], str]:
    if not normalization_cfg.get("enabled", False):
        return lambda token: token

    normalization_type = str(normalization_cfg.get("type", "none")).lower()
    language = normalization_cfg.get("language")

    if normalization_type == "porter":
        try:
            from nltk.stem import PorterStemmer
        except ImportError as exc:
            raise ImportError("NLTK is required for Porter stemming.") from exc
        stemmer = PorterStemmer()
        return stemmer.stem

    if normalization_type == "snowball":
        try:
            from nltk.stem import SnowballStemmer
        except ImportError as exc:
            raise ImportError("NLTK is required for Snowball stemming.") from exc
        if not language:
            raise ValueError("Snowball stemming requires a language.")
        stemmer = SnowballStemmer(language)
        return stemmer.stem

    if normalization_type == "wordnet":
        try:
            from nltk.stem import WordNetLemmatizer
        except ImportError as exc:
            raise ImportError("NLTK is required for WordNet lemmatization.") from exc
        lemmatizer = WordNetLemmatizer()
        return lemmatizer.lemmatize

    if normalization_type == "none":
        return lambda token: token

    raise ValueError(f"Unsupported normalization type: {normalization_type}")


@dataclass
class TextPreprocessor:
    config: dict[str, Any]
    config_dir: Path | None = None

    def __post_init__(self) -> None:
        self.stopwords = _load_stopwords(self.config.get("stopwords", {}), self.config_dir)
        self.normalizer = _build_normalizer(self.config.get("normalization", {}))

    def _transform_special_token(self, token: str) -> str | None:
        url_handling = str(self.config.get("url_handling", "remove")).lower()
        mention_handling = str(self.config.get("mention_handling", "remove")).lower()
        hashtag_handling = str(self.config.get("hashtag_handling", "strip_hash")).lower()
        number_handling = str(self.config.get("number_handling", "mask")).lower()
        punctuation_handling = str(self.config.get("punctuation_handling", "keep_sentiment")).lower()
        preserve_emojis = bool(self.config.get("preserve_emojis", True))
        compress_punctuation = bool(self.config.get("compress_punctuation", True))

        if URL_PATTERN.fullmatch(token):
            if url_handling == "remove":
                return None
            if url_handling == "mask":
                return "<url>"
            return token

        if MENTION_PATTERN.fullmatch(token):
            if mention_handling == "remove":
                return None
            if mention_handling == "mask":
                return "<user>"
            return token

        if token.startswith("#"):
            if hashtag_handling == "remove":
                return None
            if hashtag_handling == "strip_hash":
                token = token[1:]
                if not token:
                    return None
                return token
            return token

        if NUMBER_PATTERN.fullmatch(token):
            if number_handling == "remove":
                return None
            if number_handling == "mask":
                return "<num>"
            return token

        if _is_emoji(token) or _is_emoticon(token):
            return token if preserve_emojis else None

        if PUNCT_RUN_PATTERN.fullmatch(token):
            token = _compress_punctuation_run(token) if compress_punctuation else token
            if punctuation_handling == "remove":
                return None
            if punctuation_handling == "keep_sentiment":
                return token
            return token

        if not WORD_PATTERN.fullmatch(token):
            if punctuation_handling == "remove":
                return None
            if punctuation_handling == "keep_sentiment":
                return None
            return token

        return token

    def tokenize(self, text: str) -> list[str]:
        working = _normalize_text(text)
        if self.config.get("lowercase", True):
            working = working.lower()

        tokens: list[str] = []
        for match in TOKEN_PATTERN.finditer(working):
            token = match.group(0)
            token = self._transform_special_token(token)
            if token is None:
                continue
            token = token.strip()
            if not token:
                continue
            if WORD_PATTERN.fullmatch(token) and token in self.stopwords:
                continue
            if WORD_PATTERN.fullmatch(token):
                token = self.normalizer(token)
                if token in self.stopwords or not token:
                    continue
            tokens.append(token)
        return tokens

    def preprocess_text(self, text: str) -> str:
        tokens = self.tokenize(text)
        processed = " ".join(tokens)
        if self.config.get("whitespace_cleanup", True):
            return _normalize_whitespace(processed)
        return processed

    def preprocess_many(self, texts: Iterable[str]) -> tuple[list[str], list[list[str]]]:
        processed_texts: list[str] = []
        token_lists: list[list[str]] = []
        for text in texts:
            tokens = self.tokenize(text)
            processed = " ".join(tokens)
            if self.config.get("whitespace_cleanup", True):
                processed = _normalize_whitespace(processed)
            processed_texts.append(processed)
            token_lists.append(tokens)
        return processed_texts, token_lists
