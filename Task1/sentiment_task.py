from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from core.ml import (
    BernoulliNB,
    LogisticBinary,
    MultinomialNB,
    classification_metrics,
    mcnemar_exact_p,
)
from core.paths import SEED
from core.text_utils import tokenize_words

try:
    from scipy import sparse
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.naive_bayes import BernoulliNB as SkBernoulliNB
    from sklearn.naive_bayes import MultinomialNB as SkMultinomialNB

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


random.seed(SEED)
np.random.seed(SEED)

TASK3_CUSTOM_MAX_SAMPLES_DEFAULT = 5000
TASK3_FEATURE_SETS = ("bow", "lexicon", "bow_lexicon")


AZ_POSITIVE = {
    "yaxşı",
    "yaxsi",
    "gözəl",
    "gozel",
    "əla",
    "ela",
    "super",
    "superdir",
    "təşəkkür",
    "tesekkur",
    "sağol",
    "sagol",
    "saqol",
    "afərin",
    "afarin",
    "mükəmməl",
    "mukemmel",
    "qəşəng",
    "qeseng",
    "best",
    "sevirəm",
    "sevirem",
    "uğurlar",
    "ugurlar",
    "halal",
    "bravo",
    "çox",
    "cok",
    "cox",
    "möhtəşəm",
    "mohtesem",
    "bəyəndim",
    "beyendim",
    "razıyam",
    "raziyam",
    "müsbət",
    "musbet",
    "maraqlı",
    "maraqli",
}
AZ_NEGATIVE = {
    "pis",
    "bərbad",
    "berbad",
    "biabır",
    "biyabir",
    "rəzil",
    "rezil",
    "nifrət",
    "nefret",
    "zəif",
    "zeif",
    "səhv",
    "sehv",
    "yalan",
    "kötü",
    "kotu",
    "problem",
    "qəzəb",
    "qezeb",
    "utanc",
    "biabırçılıq",
    "biyabirciliq",
    "bezdim",
    "faciə",
    "facie",
    "iyrənc",
    "iyrenc",
    "bəyənmədim",
    "beyenmedim",
    "narazı",
    "narazi",
    "uğursuz",
    "ugursuz",
    "dəhşət",
    "dehset",
}
NEGATION_TOKENS = {
    "deyil",
    "deyilem",
    "deyilsen",
    "yox",
    "yoxdur",
    "olmur",
    "heç",
    "hec",
    "nə",
    "ne",
    "never",
    "none",
    "not",
    "no",
}


NEGATION_SCOPE_TOKENS = {
    "deyil",
    "deyilem",
    "deyilsen",
    "yox",
    "yoxdur",
    "not",
    "no",
    "none",
    "never",
    "olmur",
}
POST_NEGATION_TOKENS = {
    "deyil",
    "deyilem",
    "deyilsen",
    "yox",
    "yoxdur",
}
NEGATION_SCOPE_BREAKERS = {
    "amma",
    "ancaq",
    "lakin",
    "fakat",
    "fəqət",
    "but",
    "ve",
    "və",
}
NEGATION_PRE_WINDOW = 3
NEGATION_POST_WINDOW = 2
TASK1_DATASET_CANDIDATES = (
    "sentiment140_100k_clean_balanced_v2.csv",
    "sentiment140_100k_clean_balanced.csv",
    "sentiment_dataset/dataset_v1.csv",
    "sentiment_dataset/dataset.csv",
)


def sentiment_dataset_path_from_root(root: Path) -> Path:
    for candidate in TASK1_DATASET_CANDIDATES:
        candidate_path = root / candidate
        if candidate_path.exists():
            return candidate_path
    return root / TASK1_DATASET_CANDIDATES[0]


def _parse_binary_sentiment_label(raw_value: str) -> int | None:
    value = raw_value.strip().lower()
    if not value:
        return None
    if value in {"1", "positive", "pos", "true", "yes"}:
        return 1
    if value in {"0", "-1", "negative", "neg", "false", "no"}:
        return 0
    try:
        return 1 if float(value) > 0 else 0
    except ValueError:
        return None


def _parse_ternary_sentiment_label(raw_value: str) -> int | None:
    value = raw_value.strip().lower()
    if not value:
        return None
    if value in {"1", "+1", "positive", "pos", "true", "yes"}:
        return 1
    if value in {"0", "neutral", "neu", "mixed"}:
        return 0
    if value in {"-1", "negative", "neg", "false", "no"}:
        return -1
    try:
        num = float(value)
        if num > 0:
            return 1
        if num < 0:
            return -1
        return 0
    except ValueError:
        return None


def _normalize_label_scheme(label_scheme: str) -> str:
    scheme = (label_scheme or "binary").strip().lower()
    if scheme in {"binary", "binary_01", "0_1", "01"}:
        return "binary"
    if scheme in {"ternary", "ternary_-1_0_1", "-1_0_1", "tri", "3class"}:
        return "ternary"
    raise ValueError(
        f"Unsupported sentiment label scheme: {label_scheme}. "
        "Use 'binary' or 'ternary'."
    )


def parse_sentiment_label(raw_value: str, label_scheme: str = "binary") -> int | None:
    scheme = _normalize_label_scheme(label_scheme)
    if scheme == "ternary":
        return _parse_ternary_sentiment_label(raw_value)
    return _parse_binary_sentiment_label(raw_value)


def load_sentiment_dataset(
    dataset_path: Path,
    label_scheme: str = "binary",
) -> Tuple[List[str], List[int], str]:
    texts: List[str] = []
    labels: List[int] = []
    scheme = _normalize_label_scheme(label_scheme)

    if not dataset_path.exists():
        return texts, labels, f"missing:{dataset_path}"

    text_keys = ("text", "comment_text", "content", "review", "sentence")
    label_keys = ("label", "sentiment", "polarity", "target", "class")

    try:
        with dataset_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return [], [], f"invalid_csv_no_header:{dataset_path}"

            cols = {c.strip().lower(): c for c in reader.fieldnames}
            text_col = next((cols[k] for k in text_keys if k in cols), None)
            label_col = next((cols[k] for k in label_keys if k in cols), None)
            if not text_col or not label_col:
                return [], [], f"invalid_columns:{dataset_path}"

            for row in reader:
                text = (row.get(text_col) or "").strip()
                if not text:
                    continue
                label = parse_sentiment_label(row.get(label_col) or "", label_scheme=scheme)
                if label is None:
                    continue
                texts.append(text)
                labels.append(label)
    except Exception:
        return [], [], f"read_error:{dataset_path}"

    source = f"sentiment_dataset:{dataset_path}"
    if scheme != "binary":
        source = f"{source}|label_scheme:{scheme}"
    return texts, labels, source


def build_vocab_for_classification(
    texts: Sequence[str],
    min_freq: int = 2,
    max_vocab: int = 30000,
) -> Dict[str, int]:
    freq = Counter()
    for t in texts:
        freq.update(tokenize_words(t))
    items = [(w, c) for w, c in freq.items() if c >= min_freq]
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:max_vocab]
    return {w: i for i, (w, _c) in enumerate(items)}


def vectorize_bow_counts(texts: Sequence[str], vocab: Dict[str, int]) -> np.ndarray:
    x = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, t in enumerate(texts):
        counts = Counter(tokenize_words(t))
        for w, c in counts.items():
            j = vocab.get(w)
            if j is not None:
                x[i, j] = c
    return x


def vectorize_bow_binary(texts: Sequence[str], vocab: Dict[str, int]) -> np.ndarray:
    x = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, t in enumerate(texts):
        seen = set(tokenize_words(t))
        for w in seen:
            j = vocab.get(w)
            if j is not None:
                x[i, j] = 1.0
    return x


def _negation_count_around(tokens: Sequence[str], idx: int) -> int:
    count = 0

    left = max(0, idx - NEGATION_PRE_WINDOW)
    for j in range(idx - 1, left - 1, -1):
        tok = tokens[j]
        if tok in NEGATION_SCOPE_BREAKERS:
            break
        if tok in NEGATION_SCOPE_TOKENS:
            count += 1

    right = min(len(tokens), idx + 1 + NEGATION_POST_WINDOW)
    for j in range(idx + 1, right):
        tok = tokens[j]
        if tok in NEGATION_SCOPE_BREAKERS:
            break
        if tok in POST_NEGATION_TOKENS:
            count += 1

    return count


def _negation_aware_polarity_counts(tokens: Sequence[str]) -> Tuple[int, int, int]:
    pos = 0
    neg = 0
    negated_hits = 0

    for idx, tok in enumerate(tokens):
        if tok in AZ_POSITIVE:
            polarity = 1
        elif tok in AZ_NEGATIVE:
            polarity = -1
        else:
            continue

        negation_count = _negation_count_around(tokens, idx)
        if negation_count % 2 == 1:
            polarity *= -1
            negated_hits += 1

        if polarity > 0:
            pos += 1
        else:
            neg += 1

    return pos, neg, negated_hits


def sentiment_lexicon_features(texts: Sequence[str]) -> np.ndarray:
    feats = np.zeros((len(texts), 6), dtype=np.float32)
    for i, t in enumerate(texts):
        toks = tokenize_words(t)
        n = max(len(toks), 1)
        pos, neg, negated_hits = _negation_aware_polarity_counts(toks)
        sentiment_hits = max(pos + neg, 1)
        has_negation = any(w in NEGATION_TOKENS for w in toks)
        feats[i, 0] = pos
        feats[i, 1] = neg
        feats[i, 2] = pos - neg
        feats[i, 3] = (pos - neg) / n
        feats[i, 4] = 1.0 if "!" in t else 0.0
        feats[i, 5] = max(1.0 if has_negation else 0.0, float(negated_hits) / sentiment_hits)
    return feats


def sentiment_lexicon_nonnegative_features(texts: Sequence[str]) -> np.ndarray:
    signed = sentiment_lexicon_features(texts)
    nonneg = np.zeros_like(signed, dtype=np.float32)
    nonneg[:, 0] = signed[:, 0]
    nonneg[:, 1] = signed[:, 1]
    nonneg[:, 2] = np.maximum(signed[:, 2], 0.0)
    nonneg[:, 3] = np.maximum(-signed[:, 2], 0.0)
    nonneg[:, 4] = signed[:, 4]
    nonneg[:, 5] = signed[:, 5]
    return nonneg


def sentiment_lexicon_binary_features(texts: Sequence[str]) -> np.ndarray:
    dense = sentiment_lexicon_features(texts)
    binary = np.zeros_like(dense, dtype=np.float32)
    binary[:, 0] = (dense[:, 0] > 0).astype(np.float32)
    binary[:, 1] = (dense[:, 1] > 0).astype(np.float32)
    binary[:, 2] = (dense[:, 2] > 0).astype(np.float32)
    binary[:, 3] = (dense[:, 2] < 0).astype(np.float32)
    binary[:, 4] = dense[:, 4]
    binary[:, 5] = dense[:, 5]
    return binary


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1_vals: List[float] = []
    for cls in (0, 1):
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2.0 * prec * rec / max(prec + rec, 1e-12)
        f1_vals.append(f1)
    return float(sum(f1_vals) / len(f1_vals))


def _metrics_with_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    out = classification_metrics(y_true, y_pred)
    out["macro_f1"] = _macro_f1(y_true, y_pred)
    return out


def _cv_splitter(y: np.ndarray) -> "StratifiedKFold | None":
    if not SKLEARN_AVAILABLE:
        return None
    counts = np.bincount(y.astype(np.int64), minlength=2)
    min_class = int(counts.min())
    n_splits = min(5, min_class)
    if n_splits < 2:
        return None
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def _cv_mean_f1(model: object, x: object, y: np.ndarray, cv: "StratifiedKFold | None") -> float:
    if cv is None:
        return float("nan")
    scores = cross_val_score(model, x, y, scoring="f1_macro", cv=cv)
    return float(np.mean(scores))


def _select_best(
    scores: Dict[str, float],
    accuracies: Dict[str, float],
) -> str:
    names = list(scores.keys())
    names.sort(key=lambda n: (scores[n], accuracies[n]), reverse=True)
    return names[0]


def _significance_of_best(
    best_name: str,
    accuracies: Dict[str, float],
    pvals: Dict[Tuple[str, str], float],
    alpha: float = 0.05,
) -> float:
    others = [n for n in accuracies if n != best_name]
    for other in others:
        key = tuple(sorted((best_name, other)))
        p = pvals.get(key, 1.0)
        if not (accuracies[best_name] >= accuracies[other] and p < alpha):
            return 0.0
    return 1.0


def _result_sort_key(row: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    return (
        float(row["dev_macro_f1"]),
        float(row["dev_accuracy"]),
        float(row["test_macro_f1"]),
        float(row["test_accuracy"]),
    )


def _best_feature_row(rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return max(rows.values(), key=_result_sort_key)


def _attach_feature_rows(
    out: Dict[str, object],
    prefix: str,
    rows: Dict[str, Dict[str, Any]],
    param_keys: Sequence[str],
) -> None:
    for feature_set in TASK3_FEATURE_SETS:
        row = rows[feature_set]
        out[f"{prefix}_{feature_set}_dev_accuracy"] = float(row["dev_accuracy"])
        out[f"{prefix}_{feature_set}_dev_f1"] = float(row["dev_f1"])
        out[f"{prefix}_{feature_set}_dev_macro_f1"] = float(row["dev_macro_f1"])
        out[f"{prefix}_{feature_set}_accuracy"] = float(row["test_accuracy"])
        out[f"{prefix}_{feature_set}_f1"] = float(row["test_f1"])
        out[f"{prefix}_{feature_set}_macro_f1"] = float(row["test_macro_f1"])
        for param_key in param_keys:
            if param_key in row:
                value = row[param_key]
                out[f"{prefix}_{feature_set}_{param_key}"] = (
                    float(value) if isinstance(value, (int, float)) else value
                )


def _stratified_split_indices(y: np.ndarray, test_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    train_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_test = max(1, int(round(len(cls_idx) * test_ratio)))
        test_parts.append(cls_idx[:n_test])
        train_parts.append(cls_idx[n_test:])

    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def _normalize_task3_max_samples(max_samples: int | None) -> int | None:
    if max_samples is None:
        if SKLEARN_AVAILABLE:
            return None
        return TASK3_CUSTOM_MAX_SAMPLES_DEFAULT
    if max_samples <= 0:
        return None
    return int(max_samples)


def _stratified_sample_indices(y: np.ndarray, max_samples: int) -> np.ndarray:
    total = int(len(y))
    if max_samples >= total:
        return np.arange(total, dtype=np.int64)

    rng = np.random.default_rng(SEED)
    classes, counts = np.unique(y, return_counts=True)
    raw_targets = (counts.astype(np.float64) / float(total)) * float(max_samples)
    targets = np.floor(raw_targets).astype(np.int64)
    targets = np.minimum(targets, counts)

    if max_samples >= len(classes):
        targets = np.maximum(targets, 1)
        targets = np.minimum(targets, counts)

    while int(targets.sum()) < max_samples:
        deficits = counts - targets
        candidates = np.where(deficits > 0)[0]
        if len(candidates) == 0:
            break
        ranked = sorted(
            candidates.tolist(),
            key=lambda i: (raw_targets[i] - targets[i], deficits[i]),
            reverse=True,
        )
        grew = False
        for idx in ranked:
            if targets[idx] < counts[idx]:
                targets[idx] += 1
                grew = True
                if int(targets.sum()) >= max_samples:
                    break
        if not grew:
            break

    while int(targets.sum()) > max_samples:
        if max_samples >= len(classes):
            reducible = np.where(targets > 1)[0]
        else:
            reducible = np.where(targets > 0)[0]
        if len(reducible) == 0:
            break
        idx = int(reducible[np.argmax(targets[reducible])])
        targets[idx] -= 1

    sampled_parts: List[np.ndarray] = []
    for cls, take in zip(classes.tolist(), targets.tolist()):
        if take <= 0:
            continue
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        sampled_parts.append(cls_idx[:take])

    if not sampled_parts:
        return np.arange(total, dtype=np.int64)

    sampled_idx = np.concatenate(sampled_parts).astype(np.int64, copy=False)
    rng.shuffle(sampled_idx)

    if len(sampled_idx) > max_samples:
        sampled_idx = sampled_idx[:max_samples]
    elif len(sampled_idx) < max_samples:
        chosen = np.zeros(total, dtype=bool)
        chosen[sampled_idx] = True
        remaining = np.where(~chosen)[0]
        rng.shuffle(remaining)
        need = max_samples - len(sampled_idx)
        sampled_idx = np.concatenate([sampled_idx, remaining[:need]])

    return sampled_idx


def _best_alpha_custom(
    xtr: np.ndarray,
    ytr: np.ndarray,
    model_kind: str,
) -> float:
    train_idx, dev_idx = _stratified_split_indices(ytr, test_ratio=0.2)
    x_train, x_dev = xtr[train_idx], xtr[dev_idx]
    y_train, y_dev = ytr[train_idx], ytr[dev_idx]

    best_alpha = 1.0
    best_score = -1.0
    for alpha in (0.05, 0.1, 0.3, 0.5, 1.0, 2.0):
        if model_kind == "mnb":
            model = MultinomialNB(alpha=alpha).fit(x_train, y_train)
        else:
            model = BernoulliNB(alpha=alpha).fit(x_train, y_train)
        pred = model.predict(x_dev)
        score = _macro_f1(y_dev, pred)
        if score > best_score:
            best_score = score
            best_alpha = alpha
    return float(best_alpha)


def _best_lr_custom(xtr: np.ndarray, ytr: np.ndarray) -> Tuple[float, float]:
    train_idx, dev_idx = _stratified_split_indices(ytr, test_ratio=0.2)
    x_train, x_dev = xtr[train_idx], xtr[dev_idx]
    y_train, y_dev = ytr[train_idx], ytr[dev_idx]

    best_lr = 0.2
    best_reg = 1e-4
    best_score = -1.0
    for lr in (0.05, 0.1, 0.2):
        for reg in (1e-5, 1e-4, 1e-3):
            model = LogisticBinary(
                lr=lr,
                epochs=35,
                reg_type="l2",
                reg_strength=reg,
            ).fit(x_train, y_train)
            pred = model.predict(x_dev)
            score = _macro_f1(y_dev, pred)
            if score > best_score:
                best_score = score
                best_lr = lr
                best_reg = reg
    return float(best_lr), float(best_reg)


def _run_task3_sklearn(
    texts: Sequence[str],
    y: np.ndarray,
    data_source: str,
    test_ratio: float = 0.2,
    dev_ratio_within_train: float = 0.2,
) -> Dict[str, object]:
    x_pool_text, x_test_text, y_pool, y_test = train_test_split(
        list(texts),
        y,
        test_size=test_ratio,
        random_state=SEED,
        stratify=y,
    )
    x_train_text, x_dev_text, y_train, y_dev = train_test_split(
        x_pool_text,
        y_pool,
        test_size=dev_ratio_within_train,
        random_state=SEED + 1,
        stratify=y_pool,
    )

    vectorizer = CountVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
    )
    xtr_counts = vectorizer.fit_transform(x_train_text)
    xdv_counts = vectorizer.transform(x_dev_text)
    xte_counts = vectorizer.transform(x_test_text)

    xtr_lex_mnb = sentiment_lexicon_nonnegative_features(x_train_text)
    xdv_lex_mnb = sentiment_lexicon_nonnegative_features(x_dev_text)
    xte_lex_mnb = sentiment_lexicon_nonnegative_features(x_test_text)
    xtr_lex_lr = sentiment_lexicon_features(x_train_text)
    xdv_lex_lr = sentiment_lexicon_features(x_dev_text)
    xte_lex_lr = sentiment_lexicon_features(x_test_text)
    xtr_lex_bin = sentiment_lexicon_binary_features(x_train_text)
    xdv_lex_bin = sentiment_lexicon_binary_features(x_dev_text)
    xte_lex_bin = sentiment_lexicon_binary_features(x_test_text)

    xtr_bow_bin = (xtr_counts > 0).astype(np.float32)
    xdv_bow_bin = (xdv_counts > 0).astype(np.float32)
    xte_bow_bin = (xte_counts > 0).astype(np.float32)
    mnb_inputs = {
        "bow": (xtr_counts, xdv_counts, xte_counts),
        "lexicon": (
            sparse.csr_matrix(xtr_lex_mnb),
            sparse.csr_matrix(xdv_lex_mnb),
            sparse.csr_matrix(xte_lex_mnb),
        ),
        "bow_lexicon": (
            sparse.hstack([xtr_counts, sparse.csr_matrix(xtr_lex_mnb)], format="csr"),
            sparse.hstack([xdv_counts, sparse.csr_matrix(xdv_lex_mnb)], format="csr"),
            sparse.hstack([xte_counts, sparse.csr_matrix(xte_lex_mnb)], format="csr"),
        ),
    }
    bnb_inputs = {
        "bow": (xtr_bow_bin, xdv_bow_bin, xte_bow_bin),
        "lexicon": (
            sparse.csr_matrix(xtr_lex_bin),
            sparse.csr_matrix(xdv_lex_bin),
            sparse.csr_matrix(xte_lex_bin),
        ),
        "bow_lexicon": (
            sparse.hstack([xtr_bow_bin, sparse.csr_matrix(xtr_lex_bin)], format="csr"),
            sparse.hstack([xdv_bow_bin, sparse.csr_matrix(xdv_lex_bin)], format="csr"),
            sparse.hstack([xte_bow_bin, sparse.csr_matrix(xte_lex_bin)], format="csr"),
        ),
    }
    lr_inputs = {
        "bow": (xtr_counts, xdv_counts, xte_counts),
        "lexicon": (
            sparse.csr_matrix(xtr_lex_lr),
            sparse.csr_matrix(xdv_lex_lr),
            sparse.csr_matrix(xte_lex_lr),
        ),
        "bow_lexicon": (
            sparse.hstack([xtr_counts, sparse.csr_matrix(xtr_lex_lr)], format="csr"),
            sparse.hstack([xdv_counts, sparse.csr_matrix(xdv_lex_lr)], format="csr"),
            sparse.hstack([xte_counts, sparse.csr_matrix(xte_lex_lr)], format="csr"),
        ),
    }

    mnb_rows: Dict[str, Dict[str, Any]] = {}
    for feature_set, (xtr, xdv, xte) in mnb_inputs.items():
        best_alpha = 1.0
        best_score = float("-inf")
        best_acc = float("-inf")
        for alpha in (0.05, 0.1, 0.3, 0.5, 1.0, 2.0):
            model = SkMultinomialNB(alpha=alpha).fit(xtr, y_train)
            pred_dev = model.predict(xdv).astype(np.int64)
            score = _macro_f1(y_dev, pred_dev)
            acc = float((pred_dev == y_dev).mean())
            if score > best_score or (score == best_score and acc > best_acc):
                best_score = score
                best_acc = acc
                best_alpha = alpha
        model = SkMultinomialNB(alpha=best_alpha).fit(xtr, y_train)
        pred_dev = model.predict(xdv).astype(np.int64)
        pred_test = model.predict(xte).astype(np.int64)
        dev_metrics = _metrics_with_macro_f1(y_dev, pred_dev)
        test_metrics = _metrics_with_macro_f1(y_test, pred_test)
        mnb_rows[feature_set] = {
            "feature_set": feature_set,
            "alpha": float(best_alpha),
            "dev_accuracy": float(dev_metrics["accuracy"]),
            "dev_f1": float(dev_metrics["f1"]),
            "dev_macro_f1": float(dev_metrics["macro_f1"]),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_f1": float(test_metrics["f1"]),
            "test_macro_f1": float(test_metrics["macro_f1"]),
            "pred_dev": pred_dev,
            "pred_test": pred_test,
        }

    bnb_rows: Dict[str, Dict[str, Any]] = {}
    for feature_set, (xtr, xdv, xte) in bnb_inputs.items():
        best_alpha = 1.0
        best_score = float("-inf")
        best_acc = float("-inf")
        for alpha in (0.05, 0.1, 0.3, 0.5, 1.0, 2.0):
            model = SkBernoulliNB(alpha=alpha, binarize=0.0).fit(xtr, y_train)
            pred_dev = model.predict(xdv).astype(np.int64)
            score = _macro_f1(y_dev, pred_dev)
            acc = float((pred_dev == y_dev).mean())
            if score > best_score or (score == best_score and acc > best_acc):
                best_score = score
                best_acc = acc
                best_alpha = alpha
        model = SkBernoulliNB(alpha=best_alpha, binarize=0.0).fit(xtr, y_train)
        pred_dev = model.predict(xdv).astype(np.int64)
        pred_test = model.predict(xte).astype(np.int64)
        dev_metrics = _metrics_with_macro_f1(y_dev, pred_dev)
        test_metrics = _metrics_with_macro_f1(y_test, pred_test)
        bnb_rows[feature_set] = {
            "feature_set": feature_set,
            "alpha": float(best_alpha),
            "dev_accuracy": float(dev_metrics["accuracy"]),
            "dev_f1": float(dev_metrics["f1"]),
            "dev_macro_f1": float(dev_metrics["macro_f1"]),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_f1": float(test_metrics["f1"]),
            "test_macro_f1": float(test_metrics["macro_f1"]),
            "pred_dev": pred_dev,
            "pred_test": pred_test,
        }

    lr_rows: Dict[str, Dict[str, Any]] = {}
    for feature_set, (xtr, xdv, xte) in lr_inputs.items():
        best_c = 1.0
        best_weight = "none"
        best_score = float("-inf")
        best_acc = float("-inf")
        for c in (0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0):
            for class_weight in (None, "balanced"):
                model = LogisticRegression(
                    C=c,
                    solver="liblinear",
                    max_iter=3000,
                    random_state=SEED,
                    class_weight=class_weight,
                ).fit(xtr, y_train)
                pred_dev = model.predict(xdv).astype(np.int64)
                score = _macro_f1(y_dev, pred_dev)
                acc = float((pred_dev == y_dev).mean())
                if score > best_score or (score == best_score and acc > best_acc):
                    best_score = score
                    best_acc = acc
                    best_c = c
                    best_weight = "balanced" if class_weight == "balanced" else "none"
        model = LogisticRegression(
            C=best_c,
            solver="liblinear",
            max_iter=3000,
            random_state=SEED,
            class_weight=None if best_weight == "none" else "balanced",
        ).fit(xtr, y_train)
        pred_dev = model.predict(xdv).astype(np.int64)
        pred_test = model.predict(xte).astype(np.int64)
        dev_metrics = _metrics_with_macro_f1(y_dev, pred_dev)
        test_metrics = _metrics_with_macro_f1(y_test, pred_test)
        lr_rows[feature_set] = {
            "feature_set": feature_set,
            "c": float(best_c),
            "class_weight": best_weight,
            "dev_accuracy": float(dev_metrics["accuracy"]),
            "dev_f1": float(dev_metrics["f1"]),
            "dev_macro_f1": float(dev_metrics["macro_f1"]),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_f1": float(test_metrics["f1"]),
            "test_macro_f1": float(test_metrics["macro_f1"]),
            "pred_dev": pred_dev,
            "pred_test": pred_test,
        }

    best_mnb = _best_feature_row(mnb_rows)
    best_bnb = _best_feature_row(bnb_rows)
    best_lr = _best_feature_row(lr_rows)
    model_rows = {
        "multinomial_nb": best_mnb,
        "bernoulli_nb": best_bnb,
        "logistic_regression": best_lr,
    }

    pred_mnb = np.asarray(best_mnb["pred_test"], dtype=np.int64)
    pred_bnb = np.asarray(best_bnb["pred_test"], dtype=np.int64)
    pred_lr = np.asarray(best_lr["pred_test"], dtype=np.int64)
    p_lr_vs_mnb = mcnemar_exact_p(y_test, pred_lr, pred_mnb)
    p_lr_vs_bnb = mcnemar_exact_p(y_test, pred_lr, pred_bnb)
    p_mnb_vs_bnb = mcnemar_exact_p(y_test, pred_mnb, pred_bnb)

    macro_scores = {
        "multinomial_nb": float(best_mnb["test_macro_f1"]),
        "bernoulli_nb": float(best_bnb["test_macro_f1"]),
        "logistic_regression": float(best_lr["test_macro_f1"]),
    }
    accuracies = {
        "multinomial_nb": float(best_mnb["test_accuracy"]),
        "bernoulli_nb": float(best_bnb["test_accuracy"]),
        "logistic_regression": float(best_lr["test_accuracy"]),
    }
    best_model = _select_best(macro_scores, accuracies)
    pvals = {
        ("logistic_regression", "multinomial_nb"): p_lr_vs_mnb,
        ("bernoulli_nb", "logistic_regression"): p_lr_vs_bnb,
        ("bernoulli_nb", "multinomial_nb"): p_mnb_vs_bnb,
    }
    pvals_norm = {tuple(sorted(k)): v for k, v in pvals.items()}
    best_significant = _significance_of_best(best_model, accuracies, pvals_norm, alpha=0.05)
    best_feature_set = str(model_rows[best_model]["feature_set"])

    out: Dict[str, object] = {
        "num_samples": float(len(texts)),
        "positive_ratio": float(y.mean()),
        "train_examples": float(len(y_train)),
        "dev_examples": float(len(y_dev)),
        "test_examples": float(len(y_test)),
        "num_features_bow": float(len(vectorizer.vocabulary_)),
        "num_features_lexicon": 6.0,
        "feature_sets_compared_count": float(len(TASK3_FEATURE_SETS)),
        "data_source_code": 0.0,
        "data_source": data_source,
        "uses_sklearn_models": 1.0,
        "mnb_best_alpha": float(best_mnb["alpha"]),
        "bnb_best_alpha": float(best_bnb["alpha"]),
        "lr_best_c": float(best_lr["c"]),
        "lr_class_weight_balanced": 1.0 if best_lr["class_weight"] == "balanced" else 0.0,
        "mnb_best_feature_set": str(best_mnb["feature_set"]),
        "bnb_best_feature_set": str(best_bnb["feature_set"]),
        "lr_best_feature_set": str(best_lr["feature_set"]),
        "mnb_cv_macro_f1": float(best_mnb["dev_macro_f1"]),
        "bnb_cv_macro_f1": float(best_bnb["dev_macro_f1"]),
        "lr_cv_macro_f1": float(best_lr["dev_macro_f1"]),
        "mnb_dev_accuracy": float(best_mnb["dev_accuracy"]),
        "mnb_dev_f1": float(best_mnb["dev_f1"]),
        "mnb_dev_macro_f1": float(best_mnb["dev_macro_f1"]),
        "bnb_dev_accuracy": float(best_bnb["dev_accuracy"]),
        "bnb_dev_f1": float(best_bnb["dev_f1"]),
        "bnb_dev_macro_f1": float(best_bnb["dev_macro_f1"]),
        "lr_dev_accuracy": float(best_lr["dev_accuracy"]),
        "lr_dev_f1": float(best_lr["dev_f1"]),
        "lr_dev_macro_f1": float(best_lr["dev_macro_f1"]),
        "mnb_accuracy": float(best_mnb["test_accuracy"]),
        "mnb_f1": float(best_mnb["test_f1"]),
        "mnb_macro_f1": float(best_mnb["test_macro_f1"]),
        "bnb_accuracy": float(best_bnb["test_accuracy"]),
        "bnb_f1": float(best_bnb["test_f1"]),
        "bnb_macro_f1": float(best_bnb["test_macro_f1"]),
        "lr_accuracy": float(best_lr["test_accuracy"]),
        "lr_f1": float(best_lr["test_f1"]),
        "lr_macro_f1": float(best_lr["test_macro_f1"]),
        "p_lr_vs_mnb": p_lr_vs_mnb,
        "p_lr_vs_bnb": p_lr_vs_bnb,
        "p_mnb_vs_bnb": p_mnb_vs_bnb,
        "best_classifier": best_model,
        "best_classifier_feature_set": best_feature_set,
        "best_classifier_with_features": f"{best_model}:{best_feature_set}",
        "best_significant_vs_others_alpha0_05": best_significant,
    }
    _attach_feature_rows(out, "mnb", mnb_rows, ("alpha",))
    _attach_feature_rows(out, "bnb", bnb_rows, ("alpha",))
    _attach_feature_rows(out, "lr", lr_rows, ("c", "class_weight"))
    return out


def _run_task3_custom(
    texts: Sequence[str],
    y: np.ndarray,
    data_source: str,
    test_ratio: float = 0.2,
    dev_ratio_within_train: float = 0.2,
) -> Dict[str, object]:
    train_pool_idx, test_idx = _stratified_split_indices(y, test_ratio=test_ratio)
    rel_train_idx, rel_dev_idx = _stratified_split_indices(
        y[train_pool_idx],
        test_ratio=dev_ratio_within_train,
    )
    train_idx = train_pool_idx[rel_train_idx]
    dev_idx = train_pool_idx[rel_dev_idx]

    x_train_text = [texts[i] for i in train_idx]
    x_dev_text = [texts[i] for i in dev_idx]
    x_test_text = [texts[i] for i in test_idx]
    y_train = y[train_idx]
    y_dev = y[dev_idx]
    y_test = y[test_idx]

    vocab = build_vocab_for_classification(x_train_text, min_freq=2, max_vocab=20000)
    xtr_counts = vectorize_bow_counts(x_train_text, vocab)
    xdv_counts = vectorize_bow_counts(x_dev_text, vocab)
    xte_counts = vectorize_bow_counts(x_test_text, vocab)
    xtr_binary = vectorize_bow_binary(x_train_text, vocab)
    xdv_binary = vectorize_bow_binary(x_dev_text, vocab)
    xte_binary = vectorize_bow_binary(x_test_text, vocab)

    xtr_lex_lr = sentiment_lexicon_features(x_train_text)
    xdv_lex_lr = sentiment_lexicon_features(x_dev_text)
    xte_lex_lr = sentiment_lexicon_features(x_test_text)
    xtr_lex_mnb = sentiment_lexicon_nonnegative_features(x_train_text)
    xdv_lex_mnb = sentiment_lexicon_nonnegative_features(x_dev_text)
    xte_lex_mnb = sentiment_lexicon_nonnegative_features(x_test_text)
    xtr_lex_bin = sentiment_lexicon_binary_features(x_train_text)
    xdv_lex_bin = sentiment_lexicon_binary_features(x_dev_text)
    xte_lex_bin = sentiment_lexicon_binary_features(x_test_text)

    mnb_inputs = {
        "bow": (xtr_counts, xdv_counts, xte_counts),
        "lexicon": (xtr_lex_mnb, xdv_lex_mnb, xte_lex_mnb),
        "bow_lexicon": (
            np.hstack([xtr_counts, xtr_lex_mnb]),
            np.hstack([xdv_counts, xdv_lex_mnb]),
            np.hstack([xte_counts, xte_lex_mnb]),
        ),
    }
    bnb_inputs = {
        "bow": (xtr_binary, xdv_binary, xte_binary),
        "lexicon": (xtr_lex_bin, xdv_lex_bin, xte_lex_bin),
        "bow_lexicon": (
            np.hstack([xtr_binary, xtr_lex_bin]),
            np.hstack([xdv_binary, xdv_lex_bin]),
            np.hstack([xte_binary, xte_lex_bin]),
        ),
    }
    lr_inputs = {
        "bow": (xtr_counts, xdv_counts, xte_counts),
        "lexicon": (xtr_lex_lr, xdv_lex_lr, xte_lex_lr),
        "bow_lexicon": (
            np.hstack([xtr_counts, xtr_lex_lr]),
            np.hstack([xdv_counts, xdv_lex_lr]),
            np.hstack([xte_counts, xte_lex_lr]),
        ),
    }

    mnb_rows: Dict[str, Dict[str, Any]] = {}
    for feature_set, (xtr, xdv, xte) in mnb_inputs.items():
        best_alpha = 1.0
        best_score = float("-inf")
        best_acc = float("-inf")
        for alpha in (0.05, 0.1, 0.3, 0.5, 1.0, 2.0):
            model = MultinomialNB(alpha=alpha).fit(xtr, y_train)
            pred_dev = model.predict(xdv)
            score = _macro_f1(y_dev, pred_dev)
            acc = float((pred_dev == y_dev).mean())
            if score > best_score or (score == best_score and acc > best_acc):
                best_score = score
                best_acc = acc
                best_alpha = alpha
        model = MultinomialNB(alpha=best_alpha).fit(xtr, y_train)
        pred_dev = model.predict(xdv).astype(np.int64)
        pred_test = model.predict(xte).astype(np.int64)
        dev_metrics = _metrics_with_macro_f1(y_dev, pred_dev)
        test_metrics = _metrics_with_macro_f1(y_test, pred_test)
        mnb_rows[feature_set] = {
            "feature_set": feature_set,
            "alpha": float(best_alpha),
            "dev_accuracy": float(dev_metrics["accuracy"]),
            "dev_f1": float(dev_metrics["f1"]),
            "dev_macro_f1": float(dev_metrics["macro_f1"]),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_f1": float(test_metrics["f1"]),
            "test_macro_f1": float(test_metrics["macro_f1"]),
            "pred_dev": pred_dev,
            "pred_test": pred_test,
        }

    bnb_rows: Dict[str, Dict[str, Any]] = {}
    for feature_set, (xtr, xdv, xte) in bnb_inputs.items():
        best_alpha = 1.0
        best_score = float("-inf")
        best_acc = float("-inf")
        for alpha in (0.05, 0.1, 0.3, 0.5, 1.0, 2.0):
            model = BernoulliNB(alpha=alpha).fit(xtr, y_train)
            pred_dev = model.predict(xdv)
            score = _macro_f1(y_dev, pred_dev)
            acc = float((pred_dev == y_dev).mean())
            if score > best_score or (score == best_score and acc > best_acc):
                best_score = score
                best_acc = acc
                best_alpha = alpha
        model = BernoulliNB(alpha=best_alpha).fit(xtr, y_train)
        pred_dev = model.predict(xdv).astype(np.int64)
        pred_test = model.predict(xte).astype(np.int64)
        dev_metrics = _metrics_with_macro_f1(y_dev, pred_dev)
        test_metrics = _metrics_with_macro_f1(y_test, pred_test)
        bnb_rows[feature_set] = {
            "feature_set": feature_set,
            "alpha": float(best_alpha),
            "dev_accuracy": float(dev_metrics["accuracy"]),
            "dev_f1": float(dev_metrics["f1"]),
            "dev_macro_f1": float(dev_metrics["macro_f1"]),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_f1": float(test_metrics["f1"]),
            "test_macro_f1": float(test_metrics["macro_f1"]),
            "pred_dev": pred_dev,
            "pred_test": pred_test,
        }

    lr_rows: Dict[str, Dict[str, Any]] = {}
    for feature_set, (xtr, xdv, xte) in lr_inputs.items():
        best_lr = 0.2
        best_reg = 1e-4
        best_score = float("-inf")
        best_acc = float("-inf")
        for lr in (0.05, 0.1, 0.2):
            for reg in (1e-5, 1e-4, 1e-3):
                model = LogisticBinary(
                    lr=lr,
                    epochs=35,
                    reg_type="l2",
                    reg_strength=reg,
                ).fit(xtr, y_train)
                pred_dev = model.predict(xdv)
                score = _macro_f1(y_dev, pred_dev)
                acc = float((pred_dev == y_dev).mean())
                if score > best_score or (score == best_score and acc > best_acc):
                    best_score = score
                    best_acc = acc
                    best_lr = lr
                    best_reg = reg
        model = LogisticBinary(
            lr=best_lr,
            epochs=35,
            reg_type="l2",
            reg_strength=best_reg,
        ).fit(xtr, y_train)
        pred_dev = model.predict(xdv).astype(np.int64)
        pred_test = model.predict(xte).astype(np.int64)
        dev_metrics = _metrics_with_macro_f1(y_dev, pred_dev)
        test_metrics = _metrics_with_macro_f1(y_test, pred_test)
        lr_rows[feature_set] = {
            "feature_set": feature_set,
            "lr": float(best_lr),
            "reg_strength": float(best_reg),
            "dev_accuracy": float(dev_metrics["accuracy"]),
            "dev_f1": float(dev_metrics["f1"]),
            "dev_macro_f1": float(dev_metrics["macro_f1"]),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_f1": float(test_metrics["f1"]),
            "test_macro_f1": float(test_metrics["macro_f1"]),
            "pred_dev": pred_dev,
            "pred_test": pred_test,
        }

    best_mnb = _best_feature_row(mnb_rows)
    best_bnb = _best_feature_row(bnb_rows)
    best_lr = _best_feature_row(lr_rows)
    model_rows = {
        "multinomial_nb": best_mnb,
        "bernoulli_nb": best_bnb,
        "logistic_regression": best_lr,
    }

    pred_mnb = np.asarray(best_mnb["pred_test"], dtype=np.int64)
    pred_bnb = np.asarray(best_bnb["pred_test"], dtype=np.int64)
    pred_lr = np.asarray(best_lr["pred_test"], dtype=np.int64)
    p_lr_vs_mnb = mcnemar_exact_p(y_test, pred_lr, pred_mnb)
    p_lr_vs_bnb = mcnemar_exact_p(y_test, pred_lr, pred_bnb)
    p_mnb_vs_bnb = mcnemar_exact_p(y_test, pred_mnb, pred_bnb)

    macro_scores = {
        "multinomial_nb": float(best_mnb["test_macro_f1"]),
        "bernoulli_nb": float(best_bnb["test_macro_f1"]),
        "logistic_regression": float(best_lr["test_macro_f1"]),
    }
    accuracies = {
        "multinomial_nb": float(best_mnb["test_accuracy"]),
        "bernoulli_nb": float(best_bnb["test_accuracy"]),
        "logistic_regression": float(best_lr["test_accuracy"]),
    }
    best_model = _select_best(macro_scores, accuracies)
    pvals = {
        ("logistic_regression", "multinomial_nb"): p_lr_vs_mnb,
        ("bernoulli_nb", "logistic_regression"): p_lr_vs_bnb,
        ("bernoulli_nb", "multinomial_nb"): p_mnb_vs_bnb,
    }
    pvals_norm = {tuple(sorted(k)): v for k, v in pvals.items()}
    best_significant = _significance_of_best(best_model, accuracies, pvals_norm, alpha=0.05)
    best_feature_set = str(model_rows[best_model]["feature_set"])

    out: Dict[str, object] = {
        "num_samples": float(len(texts)),
        "positive_ratio": float(y.mean()),
        "train_examples": float(len(y_train)),
        "dev_examples": float(len(y_dev)),
        "test_examples": float(len(y_test)),
        "num_features_bow": float(len(vocab)),
        "num_features_lexicon": 6.0,
        "feature_sets_compared_count": float(len(TASK3_FEATURE_SETS)),
        "data_source_code": 0.0,
        "data_source": data_source,
        "uses_sklearn_models": 0.0,
        "mnb_best_alpha": float(best_mnb["alpha"]),
        "bnb_best_alpha": float(best_bnb["alpha"]),
        "lr_best_lr": float(best_lr["lr"]),
        "lr_best_reg_strength": float(best_lr["reg_strength"]),
        "mnb_best_feature_set": str(best_mnb["feature_set"]),
        "bnb_best_feature_set": str(best_bnb["feature_set"]),
        "lr_best_feature_set": str(best_lr["feature_set"]),
        "mnb_cv_macro_f1": float(best_mnb["dev_macro_f1"]),
        "bnb_cv_macro_f1": float(best_bnb["dev_macro_f1"]),
        "lr_cv_macro_f1": float(best_lr["dev_macro_f1"]),
        "mnb_dev_accuracy": float(best_mnb["dev_accuracy"]),
        "mnb_dev_f1": float(best_mnb["dev_f1"]),
        "mnb_dev_macro_f1": float(best_mnb["dev_macro_f1"]),
        "bnb_dev_accuracy": float(best_bnb["dev_accuracy"]),
        "bnb_dev_f1": float(best_bnb["dev_f1"]),
        "bnb_dev_macro_f1": float(best_bnb["dev_macro_f1"]),
        "lr_dev_accuracy": float(best_lr["dev_accuracy"]),
        "lr_dev_f1": float(best_lr["dev_f1"]),
        "lr_dev_macro_f1": float(best_lr["dev_macro_f1"]),
        "mnb_accuracy": float(best_mnb["test_accuracy"]),
        "mnb_f1": float(best_mnb["test_f1"]),
        "mnb_macro_f1": float(best_mnb["test_macro_f1"]),
        "bnb_accuracy": float(best_bnb["test_accuracy"]),
        "bnb_f1": float(best_bnb["test_f1"]),
        "bnb_macro_f1": float(best_bnb["test_macro_f1"]),
        "lr_accuracy": float(best_lr["test_accuracy"]),
        "lr_f1": float(best_lr["test_f1"]),
        "lr_macro_f1": float(best_lr["test_macro_f1"]),
        "p_lr_vs_mnb": p_lr_vs_mnb,
        "p_lr_vs_bnb": p_lr_vs_bnb,
        "p_mnb_vs_bnb": p_mnb_vs_bnb,
        "best_classifier": best_model,
        "best_classifier_feature_set": best_feature_set,
        "best_classifier_with_features": f"{best_model}:{best_feature_set}",
        "best_significant_vs_others_alpha0_05": best_significant,
    }
    _attach_feature_rows(out, "mnb", mnb_rows, ("alpha",))
    _attach_feature_rows(out, "bnb", bnb_rows, ("alpha",))
    _attach_feature_rows(out, "lr", lr_rows, ("lr", "reg_strength"))
    return out


def run_task3(
    root: Path,
    max_samples: int | None = None,
    dataset_path: Path | None = None,
    test_ratio: float = 0.2,
    dev_ratio_within_train: float = 0.2,
) -> Dict[str, object]:
    if dataset_path is None:
        dataset_path = sentiment_dataset_path_from_root(root)
    texts, labels, data_source = load_sentiment_dataset(dataset_path)
    original_num_samples = len(texts)
    effective_max_samples = _normalize_task3_max_samples(max_samples)
    sampled_for_memory = 0.0

    if effective_max_samples is not None and len(texts) > effective_max_samples:
        y_all = np.array(labels, dtype=np.int64)
        sample_idx = _stratified_sample_indices(y_all, effective_max_samples)
        texts = [texts[int(i)] for i in sample_idx.tolist()]
        labels = [int(y_all[int(i)]) for i in sample_idx.tolist()]
        sampled_for_memory = 1.0

    if len(texts) < 500:
        return {
            "error": 1.0,
            "num_samples": float(len(texts)),
            "num_samples_original": float(original_num_samples),
            "sampled_for_memory": sampled_for_memory,
            "task3_max_samples": (
                float(effective_max_samples) if effective_max_samples is not None else -1.0
            ),
            "data_source": data_source,
            "dataset_path": str(dataset_path),
        }

    y = np.array(labels, dtype=np.int64)
    if len(np.unique(y)) < 2:
        return {
            "error": 1.0,
            "num_samples": float(len(texts)),
            "num_samples_original": float(original_num_samples),
            "sampled_for_memory": sampled_for_memory,
            "task3_max_samples": (
                float(effective_max_samples) if effective_max_samples is not None else -1.0
            ),
            "data_source": data_source,
            "dataset_path": str(dataset_path),
        }

    if SKLEARN_AVAILABLE:
        metrics = _run_task3_sklearn(
            texts,
            y,
            data_source,
            test_ratio=test_ratio,
            dev_ratio_within_train=dev_ratio_within_train,
        )
    else:
        metrics = _run_task3_custom(
            texts,
            y,
            data_source,
            test_ratio=test_ratio,
            dev_ratio_within_train=dev_ratio_within_train,
        )
    metrics["uses_only_sentiment_dataset"] = 1.0
    metrics["num_samples_original"] = float(original_num_samples)
    metrics["sampled_for_memory"] = sampled_for_memory
    metrics["task3_max_samples"] = (
        float(effective_max_samples) if effective_max_samples is not None else -1.0
    )
    metrics["dataset_path"] = str(dataset_path)
    return metrics
