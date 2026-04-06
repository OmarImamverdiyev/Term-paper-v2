from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
URL_TOKEN = "<url>"
USER_TOKEN = "@user"
NUMBER_TOKEN = "<num>"

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
    """
)
csv.field_size_limit(2**31 - 1)

DEFAULT_RANDOM_SEED = 42
DEFAULT_SAMPLE_SIZE = 0
DEFAULT_BOW_FEATURES = 200
DEFAULT_EMBED_MAX_LEN = 80
DEFAULT_EMBED_OOV_MIN_FREQ = 2
DEFAULT_EMBED_MAX_EXTRA_VOCAB = 30000
DEFAULT_EMBED_INIT_STD = 0.05
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_MODEL_CACHE_DIR = "model_cache"
DEFAULT_CACHE_MANIFEST_FILE = "model_cache_manifest.json"
PIPELINE_VERSION = "sentiment140_supervised_tweet_aware_v3"
SENTIMENT_LABEL_NAMES = {
    0: "negative",
    2: "neutral",
    4: "positive",
}
SEQUENCE_MODEL_SETTINGS = [
    ("RNN", "rnn", False),
    ("Bidirectional RNN", "rnn", True),
    ("LSTM", "lstm", False),
]
DENSE_MODEL_SETTINGS = [
    ("Linear", "linear", False),
    ("MLP", "mlp", False),
    ("Deep MLP", "deep_mlp", False),
]


@dataclass
class FeatureSet:
    train_x: np.ndarray
    val_x: np.ndarray
    test_x: np.ndarray
    train_len: np.ndarray
    val_len: np.ndarray
    test_len: np.ndarray
    embedding_matrix: np.ndarray | None
    is_token_feature: bool


@dataclass
class SentimentDataset:
    docs: list[str]
    labels: np.ndarray
    label_to_index: dict[int, int]
    class_distribution: dict[int, int]
    available_rows: int


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    hidden_size: int


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, lengths: np.ndarray, y: np.ndarray, is_token_feature: bool) -> None:
        self.y = torch.from_numpy(y.astype(np.int64))
        self.lengths = torch.from_numpy(lengths.astype(np.int64))

        if is_token_feature:
            self.x = torch.from_numpy(x.astype(np.int64))
        else:
            self.x = torch.from_numpy(x.astype(np.float32))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.lengths[idx], self.y[idx]


class RecurrentClassifier(nn.Module):
    def __init__(
        self,
        architecture: str,
        num_classes: int,
        hidden_size: int,
        bidirectional: bool,
        embedding_matrix: np.ndarray | None,
    ) -> None:
        super().__init__()
        self.embedding: nn.Embedding | None = None
        self.is_lstm = architecture == "lstm"
        self.bidirectional = bidirectional

        if embedding_matrix is not None:
            embeddings = torch.tensor(embedding_matrix, dtype=torch.float32)
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=0)
            input_size = embeddings.shape[1]
        else:
            input_size = 1

        if architecture == "rnn":
            self.encoder: nn.Module = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=bidirectional,
                nonlinearity="tanh",
            )
        elif architecture == "lstm":
            self.encoder = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        out_size = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(out_size, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if self.embedding is None:
            raise RuntimeError("RecurrentClassifier requires token-based features with an embedding matrix.")

        sequence = self.embedding(x)
        packed_sequence = pack_padded_sequence(sequence, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _encoded, hidden = self.encoder(packed_sequence)

        hidden_state = hidden[0] if self.is_lstm else hidden
        if self.bidirectional:
            final_hidden = torch.cat((hidden_state[-2], hidden_state[-1]), dim=1)
        else:
            final_hidden = hidden_state[-1]

        logits = self.classifier(self.dropout(final_hidden))
        return logits


class DenseClassifier(nn.Module):
    def __init__(
        self,
        architecture: str,
        input_size: int,
        num_classes: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        if architecture == "linear":
            self.network = nn.Linear(input_size, num_classes)
        elif architecture == "mlp":
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, num_classes),
            )
        elif architecture == "deep_mlp":
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, num_classes),
            )
        else:
            raise ValueError(f"Unsupported dense architecture: {architecture}")

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        del lengths
        return self.network(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dataset_file(task_dir: Path) -> Path:
    candidates = [
        task_dir / "Sentiment140_v2.csv",
        task_dir / "sentiment140_v2.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for candidate in task_dir.rglob("*"):
        if candidate.is_file() and candidate.name.lower() == "sentiment140_v2.csv":
            return candidate

    raise FileNotFoundError("Could not find Task5/Sentiment140_v2.csv.")


def resolve_vectors_file(project_root: Path, task_name: str) -> Path:
    candidates = [
        project_root / task_name / "output" / "vectors.txt",
        project_root / task_name.lower() / "output" / "vectors.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {task_name}/output/vectors.txt")


def file_signature(path: Path) -> dict[str, str | int]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def build_cache_signature(
    args: argparse.Namespace,
    dataset_file: Path,
    word2vec_file: Path,
    glove_file: Path,
) -> str:
    payload = {
        "dataset_file": file_signature(dataset_file),
        "word2vec_file": file_signature(word2vec_file),
        "glove_file": file_signature(glove_file),
        "pipeline_version": PIPELINE_VERSION,
        "sample_size": int(args.sample_size),
        "bow_features": int(args.bow_features),
        "max_len": int(args.max_len),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "hidden_size": int(args.hidden_size),
        "seed": int(args.seed),
        "embed_oov_min_freq": DEFAULT_EMBED_OOV_MIN_FREQ,
        "embed_max_extra_vocab": DEFAULT_EMBED_MAX_EXTRA_VOCAB,
        "embed_init_std": DEFAULT_EMBED_INIT_STD,
        "embeddings_trainable": True,
    }
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def cache_slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower())
    return slug.strip("_")


def checkpoint_file_name(feature_name: str, model_name: str) -> str:
    return f"{cache_slug(feature_name)}__{cache_slug(model_name)}.pt"


def load_cache_manifest(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return payload
    return None


def save_cache_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True)


def normalize_text(text: str) -> str:
    normalized = text.translate(SMART_PUNCT_TRANSLATION)
    return re.sub(r"\s+", " ", normalized).strip()


def compress_punctuation_run(token: str) -> str:
    if token.startswith("."):
        return "..."
    if set(token) == {"!"}:
        return "!" if len(token) == 1 else "!!" if len(token) == 2 else "!!!"
    if set(token) == {"?"}:
        return "?" if len(token) == 1 else "??" if len(token) == 2 else "???"
    if set(token).issubset({"!", "?"}):
        return "!?"
    return token


def normalize_token(token: str) -> str | None:
    stripped = token.strip()
    if not stripped:
        return None

    lowered = stripped.lower()
    if URL_PATTERN.fullmatch(lowered):
        return URL_TOKEN
    if MENTION_PATTERN.fullmatch(lowered):
        return USER_TOKEN
    if NUMBER_PATTERN.fullmatch(lowered):
        return NUMBER_TOKEN
    if PUNCT_RUN_PATTERN.fullmatch(lowered):
        return compress_punctuation_run(lowered)
    return lowered


def tokenize(text: str) -> list[str]:
    normalized_text = normalize_text(text)
    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(normalized_text):
        token = normalize_token(match.group(0))
        if token:
            tokens.append(token)
    return tokens


def random_embedding_vector(dim: int) -> np.ndarray:
    return np.random.normal(loc=0.0, scale=DEFAULT_EMBED_INIT_STD, size=dim).astype(np.float32)


def token_counts(docs: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for doc in docs:
        counts.update(tokenize(doc))
    return counts


def squash_character_repetitions(text: str, max_repeat: int) -> str:
    result: list[str] = []
    previous = ""
    repeat_count = 0
    for char in text:
        if char == previous:
            repeat_count += 1
        else:
            previous = char
            repeat_count = 1

        if repeat_count <= max_repeat:
            result.append(char)

    return "".join(result)


def embedding_seed_candidates(token: str) -> list[str]:
    candidates: list[str] = []

    def add(candidate: str) -> None:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    lowered = token.lower()
    add(lowered)

    if lowered == URL_TOKEN:
        add("url")
        add("link")
    elif lowered == USER_TOKEN:
        add("user")
        add("person")
    elif lowered == NUMBER_TOKEN:
        add("number")
        add("num")

    base_word = lowered[1:] if lowered.startswith("#") else lowered
    if lowered.startswith("#"):
        add(base_word)

    if lowered in {"!!", "!!!"}:
        add("!")
    elif lowered in {"??", "???"}:
        add("?")
    elif lowered == "!?":
        add("!")
        add("?")

    if WORD_PATTERN.fullmatch(base_word):
        if "'" in base_word:
            add(base_word.replace("'", ""))
        add(squash_character_repetitions(base_word, max_repeat=2))
        add(squash_character_repetitions(base_word, max_repeat=1))

    return candidates


def seed_embedding_vector(token: str, token_to_id: dict[str, int], embedding_matrix: np.ndarray) -> np.ndarray:
    dim = int(embedding_matrix.shape[1])
    for candidate in embedding_seed_candidates(token):
        candidate_id = token_to_id.get(candidate)
        if candidate_id is not None:
            return embedding_matrix[candidate_id].copy()
    return random_embedding_vector(dim)


def extend_embedding_vocabulary(
    train_docs: list[str],
    token_to_id: dict[str, int],
    embedding_matrix: np.ndarray,
    min_freq: int = DEFAULT_EMBED_OOV_MIN_FREQ,
    max_extra_vocab: int = DEFAULT_EMBED_MAX_EXTRA_VOCAB,
) -> tuple[dict[str, int], np.ndarray]:
    counts = token_counts(train_docs)
    additional_tokens: list[str] = []
    for token, freq in counts.most_common():
        if freq < min_freq:
            break
        if token in token_to_id:
            continue
        additional_tokens.append(token)
        if len(additional_tokens) >= max_extra_vocab:
            break

    if not additional_tokens:
        return token_to_id, embedding_matrix

    expanded_matrix = np.zeros((embedding_matrix.shape[0] + len(additional_tokens), embedding_matrix.shape[1]), dtype=np.float32)
    expanded_matrix[: embedding_matrix.shape[0]] = embedding_matrix
    expanded_token_to_id = dict(token_to_id)

    for token in additional_tokens:
        token_id = len(expanded_token_to_id)
        expanded_token_to_id[token] = token_id
        expanded_matrix[token_id] = seed_embedding_vector(token, token_to_id, embedding_matrix)

    return expanded_token_to_id, expanded_matrix


def resolve_token_id(token: str, token_to_id: dict[str, int]) -> int:
    token_id = token_to_id.get(token)
    if token_id is not None:
        return token_id

    for candidate in embedding_seed_candidates(token):
        candidate_id = token_to_id.get(candidate)
        if candidate_id is not None:
            return candidate_id

    return token_to_id[UNK_TOKEN]


def sentiment_label_name(label: int) -> str:
    return SENTIMENT_LABEL_NAMES.get(label, f"class {label}")


def format_class_distribution(class_distribution: dict[int, int]) -> str:
    return ", ".join(
        f"{label} ({sentiment_label_name(label)}): {count}"
        for label, count in sorted(class_distribution.items())
    )


def load_sentiment_dataset(dataset_file: Path, sample_size: int | None, random_seed: int) -> SentimentDataset:
    df = pd.read_csv(dataset_file, usecols=["polarity", "text"], encoding="utf-8")
    df["polarity"] = pd.to_numeric(df["polarity"], errors="coerce")
    df = df.dropna(subset=["polarity"]).copy()
    df["polarity"] = df["polarity"].astype(int)
    df["text"] = df["text"].fillna("").astype(str).map(normalize_text)

    available_rows = int(len(df))
    if available_rows < 100:
        raise RuntimeError("Too few labeled rows available in Sentiment140_v2.csv.")

    if sample_size is not None and sample_size > 0 and available_rows > sample_size:
        unique_labels = int(df["polarity"].nunique())
        if sample_size < unique_labels:
            raise ValueError(
                f"--sample-size must be at least the number of classes ({unique_labels}) for stratified sampling."
            )
        df, _unused = train_test_split(
            df,
            train_size=sample_size,
            random_state=random_seed,
            stratify=df["polarity"],
        )

    df = df.reset_index(drop=True)
    label_values = sorted(int(value) for value in df["polarity"].unique().tolist())
    if len(label_values) < 2:
        raise RuntimeError("Sentiment140_v2.csv must contain at least two sentiment classes.")

    label_to_index = {label: idx for idx, label in enumerate(label_values)}
    labels = df["polarity"].map(label_to_index).to_numpy(dtype=np.int64)
    class_distribution = {label: int((df["polarity"] == label).sum()) for label in label_values}
    docs = df["text"].tolist()

    return SentimentDataset(
        docs=docs,
        labels=labels,
        label_to_index=label_to_index,
        class_distribution=class_distribution,
        available_rows=available_rows,
    )


def split_data(
    docs: list[str],
    labels: np.ndarray,
    random_seed: int,
) -> tuple[list[str], list[str], list[str], np.ndarray, np.ndarray, np.ndarray]:
    train_docs, temp_docs, y_train, y_temp = train_test_split(
        docs,
        labels,
        test_size=0.30,
        random_state=random_seed,
        stratify=labels,
    )
    val_docs, test_docs, y_val, y_test = train_test_split(
        temp_docs,
        y_temp,
        test_size=0.50,
        random_state=random_seed,
        stratify=y_temp,
    )
    return train_docs, val_docs, test_docs, y_train, y_val, y_test


def compute_term_strength_from_pmi(train_count_matrix: sparse.csr_matrix) -> np.ndarray:
    x_binary = train_count_matrix.copy().tocsr().astype(np.float64)
    x_binary.data = np.ones_like(x_binary.data)

    cooccurrence = (x_binary.T @ x_binary).toarray()
    np.fill_diagonal(cooccurrence, 0.0)

    term_document_freq = np.asarray(x_binary.sum(axis=0)).ravel()
    total_cooccurrence = float(cooccurrence.sum())

    eps = 1e-12
    denom = np.outer(term_document_freq, term_document_freq) + eps
    pmi = np.log((cooccurrence * total_cooccurrence + eps) / denom)
    pmi[pmi < 0] = 0.0

    term_strength = pmi.mean(axis=1)
    max_value = float(np.max(term_strength)) if term_strength.size else 0.0
    if max_value > 0.0:
        term_strength = term_strength / max_value

    term_strength = np.nan_to_num(term_strength, copy=False).astype(np.float32)
    return term_strength


def build_bow_features(
    train_docs: list[str],
    val_docs: list[str],
    test_docs: list[str],
    max_features: int,
) -> dict[str, FeatureSet]:
    count_vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=3,
        max_df=0.95,
        lowercase=False,
        preprocessor=None,
        token_pattern=None,
        tokenizer=tokenize,
    )
    x_train_count = count_vectorizer.fit_transform(train_docs).astype(np.float32)
    x_val_count = count_vectorizer.transform(val_docs).astype(np.float32)
    x_test_count = count_vectorizer.transform(test_docs).astype(np.float32)

    tfidf_vectorizer = TfidfVectorizer(
        vocabulary=count_vectorizer.vocabulary_,
        min_df=3,
        max_df=0.95,
        lowercase=False,
        preprocessor=None,
        token_pattern=None,
        tokenizer=tokenize,
    )
    x_train_tfidf = tfidf_vectorizer.fit_transform(train_docs).astype(np.float32)
    x_val_tfidf = tfidf_vectorizer.transform(val_docs).astype(np.float32)
    x_test_tfidf = tfidf_vectorizer.transform(test_docs).astype(np.float32)

    term_strength = compute_term_strength_from_pmi(x_train_count.tocsr())
    x_train_pmi = x_train_count.multiply(term_strength).astype(np.float32)
    x_val_pmi = x_val_count.multiply(term_strength).astype(np.float32)
    x_test_pmi = x_test_count.multiply(term_strength).astype(np.float32)

    seq_len = int(x_train_count.shape[1])
    if seq_len <= 0:
        raise RuntimeError("Vocabulary is empty; cannot build BOW feature sets.")

    def to_features(matrix_train: sparse.csr_matrix, matrix_val: sparse.csr_matrix, matrix_test: sparse.csr_matrix) -> FeatureSet:
        train_x = matrix_train.toarray().astype(np.float32)
        val_x = matrix_val.toarray().astype(np.float32)
        test_x = matrix_test.toarray().astype(np.float32)
        return FeatureSet(
            train_x=train_x,
            val_x=val_x,
            test_x=test_x,
            train_len=np.full(train_x.shape[0], seq_len, dtype=np.int64),
            val_len=np.full(val_x.shape[0], seq_len, dtype=np.int64),
            test_len=np.full(test_x.shape[0], seq_len, dtype=np.int64),
            embedding_matrix=None,
            is_token_feature=False,
        )

    return {
        "Count Vectorizer": to_features(x_train_count, x_val_count, x_test_count),
        "TF-IDF": to_features(x_train_tfidf, x_val_tfidf, x_test_tfidf),
        "PMI": to_features(x_train_pmi, x_val_pmi, x_test_pmi),
    }


def load_pretrained_embeddings(vectors_file: Path) -> tuple[dict[str, int], np.ndarray]:
    words: list[str] = []
    vectors: list[np.ndarray] = []
    dim: int | None = None

    with open(vectors_file, "r", encoding="utf-8", errors="ignore") as handle:
        first_line = handle.readline().strip().split()
        has_header = len(first_line) == 2 and first_line[0].isdigit() and first_line[1].isdigit()
        if has_header:
            dim = int(first_line[1])
        else:
            if len(first_line) > 2:
                dim = len(first_line) - 1
                word = first_line[0]
                values = np.asarray(first_line[1:], dtype=np.float32)
                if values.size == dim:
                    norm = float(np.linalg.norm(values))
                    if norm > 0.0:
                        values = values / norm
                    words.append(word)
                    vectors.append(values)

        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            if dim is None:
                dim = len(parts) - 1
            if len(parts) != dim + 1:
                continue

            word = parts[0]
            values = np.asarray(parts[1:], dtype=np.float32)
            if values.size != dim:
                continue

            norm = float(np.linalg.norm(values))
            if norm > 0.0:
                values = values / norm

            words.append(word)
            vectors.append(values)

    if dim is None or not vectors:
        raise RuntimeError(f"No vectors could be loaded from {vectors_file}")

    embedding_matrix = np.zeros((len(vectors) + 2, dim), dtype=np.float32)
    embedding_matrix[1] = random_embedding_vector(dim)
    token_to_id: dict[str, int] = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }
    for idx, (word, vector) in enumerate(zip(words, vectors), start=2):
        embedding_matrix[idx] = vector
        token_to_id[word] = idx

    return token_to_id, embedding_matrix


def docs_to_token_sequences(
    docs: list[str],
    token_to_id: dict[str, int],
    max_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    sequences = np.zeros((len(docs), max_len), dtype=np.int64)
    lengths = np.zeros(len(docs), dtype=np.int64)

    for doc_index, doc in enumerate(docs):
        token_ids = [resolve_token_id(token, token_to_id) for token in tokenize(doc)]
        if not token_ids:
            continue

        token_ids = token_ids[:max_len]
        length = len(token_ids)
        sequences[doc_index, :length] = token_ids
        lengths[doc_index] = length

    lengths = np.maximum(lengths, 1)
    return sequences, lengths


def build_embedding_feature_set(
    train_docs: list[str],
    val_docs: list[str],
    test_docs: list[str],
    vectors_file: Path,
    max_len: int,
) -> FeatureSet:
    token_to_id, embedding_matrix = load_pretrained_embeddings(vectors_file)
    token_to_id, embedding_matrix = extend_embedding_vocabulary(
        train_docs=train_docs,
        token_to_id=token_to_id,
        embedding_matrix=embedding_matrix,
    )

    train_x, train_len = docs_to_token_sequences(train_docs, token_to_id, max_len=max_len)
    val_x, val_len = docs_to_token_sequences(val_docs, token_to_id, max_len=max_len)
    test_x, test_len = docs_to_token_sequences(test_docs, token_to_id, max_len=max_len)

    return FeatureSet(
        train_x=train_x,
        val_x=val_x,
        test_x=test_x,
        train_len=train_len,
        val_len=val_len,
        test_len=test_len,
        embedding_matrix=embedding_matrix,
        is_token_feature=True,
    )


def model_settings_for_feature(feature_name: str) -> list[tuple[str, str, bool]]:
    if feature_name in {"Word2Vec", "GloVe"}:
        return SEQUENCE_MODEL_SETTINGS
    return DENSE_MODEL_SETTINGS


def build_classifier(
    architecture: str,
    bidirectional: bool,
    feature_set: FeatureSet,
    num_classes: int,
    hidden_size: int,
) -> nn.Module:
    if feature_set.is_token_feature:
        return RecurrentClassifier(
            architecture=architecture,
            num_classes=num_classes,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            embedding_matrix=feature_set.embedding_matrix,
        )

    if bidirectional:
        raise ValueError("Bidirectional models are only supported for token-based sequence features.")

    input_size = int(feature_set.train_x.shape[1])
    if input_size <= 0:
        raise RuntimeError("Dense feature set is empty; cannot build classifier.")

    return DenseClassifier(
        architecture=architecture,
        input_size=input_size,
        num_classes=num_classes,
        hidden_size=hidden_size,
    )


def make_dataloaders(
    feature_set: FeatureSet,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = SequenceDataset(feature_set.train_x, feature_set.train_len, y_train, feature_set.is_token_feature)
    val_dataset = SequenceDataset(feature_set.val_x, feature_set.val_len, y_val, feature_set.is_token_feature)
    test_dataset = SequenceDataset(feature_set.test_x, feature_set.test_len, y_test, feature_set.is_token_feature)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for x_batch, lengths_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            lengths_batch = lengths_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch, lengths_batch)
            loss = criterion(logits, y_batch)
            total_loss += float(loss.item()) * y_batch.size(0)
            total_count += y_batch.size(0)

    return total_loss / max(total_count, 1)


def train_and_score(
    architecture: str,
    bidirectional: bool,
    feature_set: FeatureSet,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    config: TrainConfig,
    device: torch.device,
    checkpoint_file: Path | None = None,
    cache_signature: str | None = None,
    feature_name: str = "",
    model_name: str = "",
) -> tuple[dict[str, float], float, bool]:
    metric_keys = ("accuracy", "precision", "recall", "f1")

    if checkpoint_file is not None and cache_signature and checkpoint_file.exists():
        try:
            checkpoint = torch.load(checkpoint_file, map_location="cpu")
            if isinstance(checkpoint, dict) and checkpoint.get("cache_signature") == cache_signature:
                cached_metrics = checkpoint.get("metrics")
                if isinstance(cached_metrics, dict) and all(key in cached_metrics for key in metric_keys):
                    metrics = {key: float(cached_metrics[key]) for key in metric_keys}
                    train_seconds = float(checkpoint.get("train_seconds", 0.0))
                    return metrics, train_seconds, True
        except Exception:
            # Corrupted or incompatible cache; retrain and overwrite.
            pass

    train_loader, val_loader, test_loader = make_dataloaders(
        feature_set=feature_set,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        batch_size=config.batch_size,
    )

    model = build_classifier(
        architecture=architecture,
        bidirectional=bidirectional,
        feature_set=feature_set,
        num_classes=num_classes,
        hidden_size=config.hidden_size,
    ).to(device)

    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / max(float(class_weights.mean()), 1e-8)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    start_time = time.perf_counter()

    for _epoch in range(config.epochs):
        model.train()
        for x_batch, lengths_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            lengths_batch = lengths_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch, lengths_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        val_loss = evaluate_loss(model, val_loader, device=device, criterion=criterion)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    train_seconds = time.perf_counter() - start_time

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for x_batch, lengths_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            lengths_batch = lengths_batch.to(device)
            logits = model(x_batch, lengths_batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            y_pred.extend(preds.tolist())
            y_true.extend(y_batch.numpy().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    if checkpoint_file is not None and cache_signature:
        state_dict_to_save = best_state
        if state_dict_to_save is None:
            state_dict_to_save = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cache_signature": cache_signature,
                "feature": feature_name,
                "model": model_name,
                "architecture": architecture,
                "bidirectional": bidirectional,
                "hidden_size": config.hidden_size,
                "num_classes": int(num_classes),
                "metrics": metrics,
                "train_seconds": float(train_seconds),
                "state_dict": state_dict_to_save,
            },
            checkpoint_file,
        )

    return metrics, float(train_seconds), False


def format_table(df: pd.DataFrame) -> str:
    lines = [
        "| Feature | Model | Accuracy | Precision | Recall | F1 | Train Time (s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.feature} | {row.model} | {row.accuracy:.4f} | {row.precision:.4f} | "
            f"{row.recall:.4f} | {row.f1:.4f} | {row.train_seconds:.2f} |"
        )
    return "\n".join(lines)


def write_report(
    report_file: Path,
    dataset_file: Path,
    dataset: SentimentDataset,
    sample_size: int,
    bow_features: int,
    max_len: int,
    train_config: TrainConfig,
    split_sizes: tuple[int, int, int],
    results_df: pd.DataFrame,
) -> None:
    best_idx = int(results_df["f1"].idxmax())
    best_row = results_df.loc[best_idx]
    sampling_mode = (
        "full dataset"
        if sample_size <= 0 or sample_size >= dataset.available_rows
        else f"stratified sample capped at {sample_size} rows"
    )
    label_map = ", ".join(
        f"{label} ({sentiment_label_name(label)}) -> {index}"
        for label, index in sorted(dataset.label_to_index.items())
    )
    train_size, val_size, test_size = split_sizes

    lines = [
        "# Task5 - Sentiment Classification with Dense and Recurrent Models",
        "",
        "## Setup",
        "",
        f"- Dataset: `{dataset_file}`",
        "- Source dataset: `Task5/Sentiment140_v2.csv`",
        f"- Samples used in this run: **{len(dataset.docs)}**",
        f"- Sampling mode: **{sampling_mode}**",
        "- Gold labels are read directly from the `polarity` column and remapped internally to contiguous class IDs for PyTorch training.",
        f"- Label mapping: **{label_map}**",
        f"- Class distribution: **{format_class_distribution(dataset.class_distribution)}**",
        f"- Train/validation/test split: **{train_size} / {val_size} / {test_size}**",
        "- A tweet-aware tokenizer is used across all feature sets, preserving hashtags, mentions, contractions, emoticons, emojis, elongated words, and repeated punctuation.",
        "- Non-sequential feature sets (`Count Vectorizer`, `TF-IDF`, `PMI`) use dense classifiers so vocabulary dimensions are not treated as a fake time axis.",
        "- Sequence feature sets (`Word2Vec`, `GloVe`) extend the pretrained vocabularies with frequent tweet tokens and map any remaining unseen items to `<unk>` instead of silently dropping them.",
        "- Pretrained sequence embeddings are fine-tuned during sentiment training (`freeze=False`) so the model can adapt them to Sentiment140.",
        "- Sequence feature sets use packed recurrent models and pool the true forward/backward hidden states.",
        "",
        "## Training Configuration",
        "",
        f"- BOW vocabulary size (`Count`, `TF-IDF`, `PMI`): **{bow_features}**",
        f"- Max token length (`Word2Vec`, `GloVe`): **{max_len}**",
        f"- Extra tweet-token embeddings added per pretrained vocabulary (max): **{DEFAULT_EMBED_MAX_EXTRA_VOCAB}**",
        f"- Minimum train frequency before adding a new tweet-token embedding: **{DEFAULT_EMBED_OOV_MIN_FREQ}**",
        f"- Batch size: **{train_config.batch_size}**",
        f"- Epochs per model: **{train_config.epochs}**",
        f"- Hidden size: **{train_config.hidden_size}**",
        f"- Learning rate: **{train_config.learning_rate}**",
        "",
        "## Results Table",
        "",
        format_table(results_df),
        "",
        "## Best Combination",
        "",
        f"- Best by macro F1: **{best_row['model']} + {best_row['feature']}**",
        f"- Accuracy: **{best_row['accuracy']:.4f}**",
        f"- Precision: **{best_row['precision']:.4f}**",
        f"- Recall: **{best_row['recall']:.4f}**",
        f"- F1: **{best_row['f1']:.4f}**",
        "",
        "## Output Files",
        "",
        "- `output/task5_results.csv`",
        "- `output/task5_report.md`",
        "- `output/model_cache/*.pt` (saved model checkpoints for each feature/model combination)",
        "- `output/model_cache_manifest.json` (cache signature and checkpoint index)",
    ]

    with open(report_file, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task5: sentiment classification with dense and recurrent model comparisons")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Maximum number of rows to use from Sentiment140_v2.csv; set to 0 to use the full dataset.",
    )
    parser.add_argument("--bow-features", type=int, default=DEFAULT_BOW_FEATURES)
    parser.add_argument("--max-len", type=int, default=DEFAULT_EMBED_MAX_LEN)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    task_dir = Path(__file__).resolve().parent
    project_root = task_dir.parent
    output_dir = task_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_cache_dir = output_dir / DEFAULT_MODEL_CACHE_DIR
    cache_manifest_file = output_dir / DEFAULT_CACHE_MANIFEST_FILE

    csv_file = output_dir / "task5_results.csv"
    report_file = output_dir / "task5_report.md"

    dataset_file = resolve_dataset_file(task_dir)
    word2vec_file = resolve_vectors_file(project_root, "Task2")
    glove_file = resolve_vectors_file(project_root, "Task3")
    cache_signature = build_cache_signature(args=args, dataset_file=dataset_file, word2vec_file=word2vec_file, glove_file=glove_file)

    feature_order = ["Count Vectorizer", "TF-IDF", "PMI", "Word2Vec", "GloVe"]
    feature_model_settings = {feature_name: model_settings_for_feature(feature_name) for feature_name in feature_order}
    expected_checkpoint_files = [
        model_cache_dir / checkpoint_file_name(feature_name, model_name)
        for feature_name in feature_order
        for model_name, _architecture, _bidirectional in feature_model_settings[feature_name]
    ]

    cache_manifest = load_cache_manifest(cache_manifest_file)
    if (
        cache_manifest is not None
        and cache_manifest.get("cache_signature") == cache_signature
        and csv_file.exists()
        and report_file.exists()
        and all(path.exists() for path in expected_checkpoint_files)
    ):
        print("Using cached Task5 models and previous metrics.")
        print(f"Cache signature: {cache_signature}")
        print(f"Results CSV: {csv_file}")
        print(f"Report: {report_file}")
        print("")
        print(format_table(pd.read_csv(csv_file)))
        return

    print("Loading labeled tweets...")
    dataset = load_sentiment_dataset(dataset_file=dataset_file, sample_size=args.sample_size, random_seed=args.seed)
    print(f"Samples selected: {len(dataset.docs)}")
    print(f"Class distribution: {format_class_distribution(dataset.class_distribution)}")

    print("Splitting train/validation/test...")
    train_docs, val_docs, test_docs, y_train, y_val, y_test = split_data(
        docs=dataset.docs,
        labels=dataset.labels,
        random_seed=args.seed,
    )

    print("Building Count/TF-IDF/PMI feature sets...")
    feature_sets = build_bow_features(
        train_docs=train_docs,
        val_docs=val_docs,
        test_docs=test_docs,
        max_features=args.bow_features,
    )

    print("Building Word2Vec feature set...")
    feature_sets["Word2Vec"] = build_embedding_feature_set(
        train_docs=train_docs,
        val_docs=val_docs,
        test_docs=test_docs,
        vectors_file=word2vec_file,
        max_len=args.max_len,
    )

    print("Building GloVe feature set...")
    feature_sets["GloVe"] = build_embedding_feature_set(
        train_docs=train_docs,
        val_docs=val_docs,
        test_docs=test_docs,
        vectors_file=glove_file,
        max_len=args.max_len,
    )

    train_config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    num_classes = int(np.unique(dataset.labels).size)
    results: list[dict[str, float | str]] = []
    cache_hit_count = 0

    for feature_name in feature_order:
        feature_set = feature_sets[feature_name]
        for model_name, architecture, bidirectional in feature_model_settings[feature_name]:
            checkpoint_file = model_cache_dir / checkpoint_file_name(feature_name, model_name)
            print(f"Training {model_name} on {feature_name}...")
            metrics, seconds, from_cache = train_and_score(
                architecture=architecture,
                bidirectional=bidirectional,
                feature_set=feature_set,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                num_classes=num_classes,
                config=train_config,
                device=device,
                checkpoint_file=checkpoint_file,
                cache_signature=cache_signature,
                feature_name=feature_name,
                model_name=model_name,
            )
            if from_cache:
                cache_hit_count += 1
            results.append(
                {
                    "feature": feature_name,
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "train_seconds": seconds,
                    "source": "cache" if from_cache else "trained",
                }
            )

    results_df = pd.DataFrame(results)
    model_order = {
        **{("Count Vectorizer", model_name): idx for idx, (model_name, _architecture, _bidirectional) in enumerate(DENSE_MODEL_SETTINGS)},
        **{("TF-IDF", model_name): idx for idx, (model_name, _architecture, _bidirectional) in enumerate(DENSE_MODEL_SETTINGS)},
        **{("PMI", model_name): idx for idx, (model_name, _architecture, _bidirectional) in enumerate(DENSE_MODEL_SETTINGS)},
        **{("Word2Vec", model_name): idx for idx, (model_name, _architecture, _bidirectional) in enumerate(SEQUENCE_MODEL_SETTINGS)},
        **{("GloVe", model_name): idx for idx, (model_name, _architecture, _bidirectional) in enumerate(SEQUENCE_MODEL_SETTINGS)},
    }
    results_df["feature_order"] = results_df["feature"].map({name: idx for idx, name in enumerate(feature_order)})
    results_df["feature_model_key"] = list(zip(results_df["feature"], results_df["model"]))
    results_df["model_order"] = results_df["feature_model_key"].map(model_order)
    results_df = results_df.sort_values(["feature_order", "model_order"]).drop(
        columns=["feature_order", "feature_model_key", "model_order"]
    )

    results_df.to_csv(csv_file, index=False)
    write_report(
        report_file=report_file,
        dataset_file=dataset_file,
        dataset=dataset,
        sample_size=args.sample_size,
        bow_features=args.bow_features,
        max_len=args.max_len,
        train_config=train_config,
        split_sizes=(len(train_docs), len(val_docs), len(test_docs)),
        results_df=results_df,
    )
    save_cache_manifest(
        cache_manifest_file,
        {
            "cache_signature": cache_signature,
            "generated_at_epoch": int(time.time()),
            "pipeline_version": PIPELINE_VERSION,
            "dataset_file": file_signature(dataset_file),
            "word2vec_file": file_signature(word2vec_file),
            "glove_file": file_signature(glove_file),
            "results_csv": str(csv_file),
            "report_file": str(report_file),
            "checkpoint_files": [str(path) for path in expected_checkpoint_files],
            "args": {
                "sample_size": int(args.sample_size),
                "bow_features": int(args.bow_features),
                "max_len": int(args.max_len),
                "batch_size": int(args.batch_size),
                "epochs": int(args.epochs),
                "learning_rate": float(args.learning_rate),
                "hidden_size": int(args.hidden_size),
                "seed": int(args.seed),
            },
            "class_distribution": dataset.class_distribution,
            "label_to_index": dataset.label_to_index,
        },
    )

    print("Task5 completed.")
    print(f"Results CSV: {csv_file}")
    print(f"Report: {report_file}")
    print(f"Cached combinations reused in this run: {cache_hit_count}/{len(results)}")
    print("")
    print(format_table(results_df))


if __name__ == "__main__":
    main()
