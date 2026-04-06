from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.features.base import FeatureArtifact, FeatureSet
from src.utils.filesystem import save_json


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def _require_gensim() -> Any:
    try:
        from gensim.models import Word2Vec
    except ImportError as exc:
        raise ImportError(
            "gensim is required for Word2Vec features. Install dependencies from requirements.txt."
        ) from exc
    return Word2Vec


def _default_worker_count() -> int:
    return max(1, min(4, os.cpu_count() or 1))


def _sorted_vocabulary(token_counts: Counter[str], min_freq: int, max_size: int | None) -> list[tuple[str, int]]:
    if min_freq < 1:
        raise ValueError("Sequence vocabulary min frequency must be at least 1.")

    items = [(token, freq) for token, freq in token_counts.items() if freq >= min_freq]
    items.sort(key=lambda item: (-item[1], item[0]))
    if max_size is not None:
        items = items[:max_size]
    return items


def _truncate_tokens(tokens: list[str], max_length: int, strategy: str) -> tuple[list[str], bool]:
    if max_length < 1:
        raise ValueError("max_sequence_length must be at least 1.")
    if len(tokens) <= max_length:
        return tokens, False

    if strategy == "head":
        return tokens[:max_length], True
    if strategy == "tail":
        return tokens[-max_length:], True
    if strategy == "head_tail":
        head_len = max_length // 2
        tail_len = max_length - head_len
        return tokens[:head_len] + tokens[-tail_len:], True

    raise ValueError(f"Unsupported truncation strategy: {strategy}")


def _encode_sequences(
    token_lists: list[list[str]],
    token_to_id: dict[str, int],
    max_length: int,
    truncation_strategy: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    pad_id = token_to_id[PAD_TOKEN]
    unk_id = token_to_id[UNK_TOKEN]

    sequences = np.full((len(token_lists), max_length), pad_id, dtype=np.int64)
    lengths = np.zeros(len(token_lists), dtype=np.int64)

    truncated_rows = 0
    unknown_tokens = 0
    total_tokens = 0

    for row_index, original_tokens in enumerate(token_lists):
        tokens = original_tokens if original_tokens else [UNK_TOKEN]
        tokens, was_truncated = _truncate_tokens(tokens, max_length=max_length, strategy=truncation_strategy)
        truncated_rows += int(was_truncated)

        token_ids: list[int] = []
        for token in tokens:
            token_id = token_to_id.get(token, unk_id)
            token_ids.append(token_id)
            total_tokens += 1
            unknown_tokens += int(token_id == unk_id)

        lengths[row_index] = len(token_ids)
        sequences[row_index, : len(token_ids)] = np.asarray(token_ids, dtype=np.int64)

    return sequences, lengths, {
        "rows": float(len(token_lists)),
        "max_sequence_length": float(max_length),
        "truncated_rows": float(truncated_rows),
        "truncation_rate": float(truncated_rows / max(len(token_lists), 1)),
        "total_tokens_after_truncation": float(total_tokens),
        "unknown_tokens_after_truncation": float(unknown_tokens),
        "unknown_token_rate": float(unknown_tokens / max(total_tokens, 1)),
    }


def _maybe_normalize_embeddings(embedding_matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    normalized = embedding_matrix / norms
    normalized[0] = 0.0
    return normalized.astype(np.float32)


def _save_word2vec_artifacts(
    artifact_dir: Path,
    artifact_prefix: str,
    model: Any,
    token_to_id: dict[str, int],
    embedding_matrix: np.ndarray,
    training_config: dict[str, Any],
    vocabulary_stats: dict[str, Any],
) -> dict[str, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    keyed_vectors_path = artifact_dir / f"{artifact_prefix}_keyed_vectors.kv"
    full_model_path = artifact_dir / f"{artifact_prefix}_model.gensim"
    training_config_path = artifact_dir / f"{artifact_prefix}_training_config.json"
    vocabulary_stats_path = artifact_dir / f"{artifact_prefix}_vocabulary_stats.json"
    vocab_mapping_path = artifact_dir / f"{artifact_prefix}_sequence_vocab.joblib"
    embedding_matrix_path = artifact_dir / f"{artifact_prefix}_embedding_matrix.npy"

    model.wv.save(str(keyed_vectors_path))
    model.save(str(full_model_path))
    save_json(training_config_path, training_config)
    save_json(vocabulary_stats_path, vocabulary_stats)
    joblib.dump(
        {
            "token_to_id": token_to_id,
            "id_to_token": [token for token, _token_id in sorted(token_to_id.items(), key=lambda item: item[1])],
        },
        vocab_mapping_path,
    )
    np.save(embedding_matrix_path, embedding_matrix)

    return {
        "keyed_vectors": str(keyed_vectors_path),
        "full_model": str(full_model_path),
        "training_config": str(training_config_path),
        "vocabulary_stats": str(vocabulary_stats_path),
        "sequence_vocab": str(vocab_mapping_path),
        "embedding_matrix": str(embedding_matrix_path),
    }


def build_word2vec_features(
    feature_name: str,
    config: dict[str, Any],
    train_tokens: list[list[str]],
    val_tokens: list[list[str]],
    test_tokens: list[list[str]],
    seed: int,
    artifact_dir: Path | None = None,
    artifact_prefix: str | None = None,
) -> FeatureSet:
    Word2Vec = _require_gensim()

    vector_size = int(config.get("vector_size", 200))
    window = int(config.get("window", 5))
    min_count = int(config.get("min_count", 2))
    sg = int(config.get("sg", 1))
    negative = int(config.get("negative", 10))
    epochs = int(config.get("epochs", 15))
    workers = int(config.get("workers", _default_worker_count()))
    sample = float(config.get("sample", 1.0e-5))
    ns_exponent = float(config.get("ns_exponent", 0.75))
    alpha = float(config.get("alpha", 0.025))
    min_alpha = float(config.get("min_alpha", 0.0007))
    hs = int(config.get("hs", 0))
    shrink_windows = bool(config.get("shrink_windows", True))

    max_sequence_length = int(config.get("max_sequence_length", config.get("max_seq_len", 56)))
    truncation_strategy = str(config.get("truncation_strategy", "head")).lower()
    sequence_min_freq = int(config.get("sequence_min_freq", min_count))
    sequence_max_vocab_size = config.get("sequence_max_vocab_size")
    if sequence_max_vocab_size is not None:
        sequence_max_vocab_size = int(sequence_max_vocab_size)
    embedding_init_std = float(config.get("embedding_init_std", 0.05))
    normalize_embeddings = bool(config.get("normalize_embeddings", False))

    train_sentences = [tokens if tokens else [UNK_TOKEN] for tokens in train_tokens]
    w2v_model = Word2Vec(
        sentences=train_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        negative=negative,
        epochs=epochs,
        workers=workers,
        seed=seed,
        sample=sample,
        ns_exponent=ns_exponent,
        alpha=alpha,
        min_alpha=min_alpha,
        hs=hs,
        shrink_windows=shrink_windows,
    )

    token_counts: Counter[str] = Counter()
    for tokens in train_tokens:
        token_counts.update(tokens)

    sequence_vocabulary = _sorted_vocabulary(
        token_counts=token_counts,
        min_freq=sequence_min_freq,
        max_size=sequence_max_vocab_size,
    )

    token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, _frequency in sequence_vocabulary:
        if token in token_to_id:
            continue
        token_to_id[token] = len(token_to_id)

    rng = np.random.default_rng(seed)
    embedding_matrix = np.zeros((len(token_to_id), vector_size), dtype=np.float32)
    embedding_matrix[token_to_id[UNK_TOKEN]] = rng.normal(0.0, embedding_init_std, size=vector_size).astype(np.float32)

    pretrained_tokens = 0
    for token, token_id in token_to_id.items():
        if token in {PAD_TOKEN, UNK_TOKEN}:
            continue
        if token in w2v_model.wv:
            embedding_matrix[token_id] = np.asarray(w2v_model.wv[token], dtype=np.float32)
            pretrained_tokens += 1
        else:
            embedding_matrix[token_id] = rng.normal(0.0, embedding_init_std, size=vector_size).astype(np.float32)

    if normalize_embeddings:
        embedding_matrix = _maybe_normalize_embeddings(embedding_matrix)

    train_x, train_lengths, train_sequence_stats = _encode_sequences(
        train_tokens,
        token_to_id=token_to_id,
        max_length=max_sequence_length,
        truncation_strategy=truncation_strategy,
    )
    val_x, val_lengths, val_sequence_stats = _encode_sequences(
        val_tokens,
        token_to_id=token_to_id,
        max_length=max_sequence_length,
        truncation_strategy=truncation_strategy,
    )
    test_x, test_lengths, test_sequence_stats = _encode_sequences(
        test_tokens,
        token_to_id=token_to_id,
        max_length=max_sequence_length,
        truncation_strategy=truncation_strategy,
    )

    training_config = {
        "seed": seed,
        "word2vec": {
            "vector_size": vector_size,
            "window": window,
            "min_count": min_count,
            "sg": sg,
            "negative": negative,
            "epochs": epochs,
            "workers": workers,
            "sample": sample,
            "ns_exponent": ns_exponent,
            "alpha": alpha,
            "min_alpha": min_alpha,
            "hs": hs,
            "shrink_windows": shrink_windows,
        },
        "sequence": {
            "max_sequence_length": max_sequence_length,
            "truncation_strategy": truncation_strategy,
            "sequence_min_freq": sequence_min_freq,
            "sequence_max_vocab_size": sequence_max_vocab_size,
            "embedding_init_std": embedding_init_std,
            "normalize_embeddings": normalize_embeddings,
            "pad_token": PAD_TOKEN,
            "unk_token": UNK_TOKEN,
        },
    }

    total_train_tokens = int(sum(len(tokens) for tokens in train_tokens))
    vocabulary_stats: dict[str, Any] = {
        "feature_name": feature_name,
        "train_corpus": {
            "documents": len(train_tokens),
            "tokens": total_train_tokens,
            "unique_tokens": len(token_counts),
        },
        "word2vec_vocabulary_size": len(w2v_model.wv),
        "sequence_vocabulary_size": len(token_to_id),
        "sequence_vocabulary_coverage_with_word2vec": float(
            pretrained_tokens / max(len(token_to_id) - 2, 1)
        ),
        "pretrained_sequence_tokens": pretrained_tokens,
        "special_tokens": {
            "pad": PAD_TOKEN,
            "unk": UNK_TOKEN,
            "pad_id": token_to_id[PAD_TOKEN],
            "unk_id": token_to_id[UNK_TOKEN],
        },
        "sequence_stats": {
            "train": train_sequence_stats,
            "validation": val_sequence_stats,
            "test": test_sequence_stats,
        },
    }

    artifact_paths: dict[str, str] = {}
    if artifact_dir is not None:
        artifact_paths = _save_word2vec_artifacts(
            artifact_dir=artifact_dir,
            artifact_prefix=artifact_prefix or feature_name,
            model=w2v_model,
            token_to_id=token_to_id,
            embedding_matrix=embedding_matrix,
            training_config=training_config,
            vocabulary_stats=vocabulary_stats,
        )

    diagnostics: list[str] = []
    if vector_size < 100:
        diagnostics.append("Word2Vec vector_size is below 100; this is often weak for sentiment classification.")
    if len(token_to_id) < 5000:
        diagnostics.append("Sequence vocabulary is unexpectedly small; check preprocessing and min frequency settings.")
    if train_sequence_stats["truncation_rate"] > 0.05:
        diagnostics.append("More than 5% of training rows are truncated; consider increasing max_sequence_length.")
    if vocabulary_stats["sequence_vocabulary_coverage_with_word2vec"] < 0.85:
        diagnostics.append("Sequence vocabulary coverage by Word2Vec is below 85%; consider lowering min_count.")

    artifact = FeatureArtifact(
        name=feature_name,
        feature_type="word2vec",
        transformer=None,
        metadata={
            "token_to_id": token_to_id,
            "embedding_matrix": embedding_matrix,
            "vector_size": vector_size,
            "embedding_dim": vector_size,
            "vocabulary_size": len(token_to_id),
            "word2vec_vocabulary_size": len(w2v_model.wv),
            "max_sequence_length": max_sequence_length,
            "sequence_min_freq": sequence_min_freq,
            "truncation_strategy": truncation_strategy,
            "config": config,
            "training_config": training_config,
            "vocabulary_stats": vocabulary_stats,
            "artifact_paths": artifact_paths,
            "diagnostics": diagnostics,
        },
    )

    return FeatureSet(
        train_x=train_x,
        val_x=val_x,
        test_x=test_x,
        artifact=artifact,
        is_sequence=True,
        train_lengths=train_lengths,
        val_lengths=val_lengths,
        test_lengths=test_lengths,
    )
