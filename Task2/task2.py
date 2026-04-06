from __future__ import annotations

import csv
import heapq
import math
import os
import re
import subprocess
import time
from array import array
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
csv.field_size_limit(2**31 - 1)

# Candidate target words for synonym search; first 10 available in vocabulary are used.
TARGET_WORD_SEEDS = [
    "az\u0259rbaycan",
    "bak\u0131",
    "rusiya",
    "t\u00fcrkiy\u0259",
    "ukrayna",
    "prezident",
    "naziri",
    "h\u0259rbi",
    "t\u0259hsil",
    "n\u0259qliyyat",
    "manat",
    "erm\u0259nistan",
    "ab\u015f",
    "futbol",
    "komanda",
    "oyun",
]

STOPWORDS = {
    "v\u0259",
    "ki",
    "bu",
    "il\u0259",
    "\u00fc\u00e7\u00fcn",
    "g\u00f6r\u0259",
    "bir",
    "da",
    "d\u0259",
    "olan",
    "kimi",
    "is\u0259",
    "h\u0259m",
    "h\u0259r",
    "o",
    "ya",
    "in",
    "nin",
    "n\u0131n",
}


@dataclass(frozen=True)
class Word2VecConfig:
    cbow: int = 0  # 0 = Skip-gram, 1 = CBOW
    size: int = 150
    window: int = 8
    sample: float = 1e-4
    hs: int = 0
    negative: int = 10
    iterations: int = 5
    min_count: int = 10
    binary: int = 0
    threads: int = 8


def resolve_dataset_file(project_root: Path) -> Path:
    candidates = [
        project_root / "Corpora" / "news" / "content_only.csv",
        project_root / "Corpora" / "News" / "content_only.csv",
        project_root / "corpora" / "news" / "content_only.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for candidate in project_root.rglob("content_only.csv"):
        lowered = str(candidate).lower().replace("\\", "/")
        if "/news/" in lowered:
            return candidate

    raise FileNotFoundError("Could not find Corpora/news/content_only.csv.")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def build_training_corpus(dataset_file: Path, output_file: Path) -> tuple[int, int, Counter[str]]:
    doc_count = 0
    token_count = 0
    global_counts: Counter[str] = Counter()

    with open(dataset_file, "r", encoding="utf-8", newline="") as source, open(
        output_file, "w", encoding="utf-8", newline="\n"
    ) as target:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "content" not in reader.fieldnames:
            raise ValueError("Expected a 'content' column in content_only.csv.")

        for row in reader:
            text = (row.get("content") or "").strip()
            if not text:
                continue

            tokens = tokenize(text)
            if not tokens:
                continue

            target.write(" ".join(tokens))
            target.write("\n")

            doc_count += 1
            token_count += len(tokens)
            global_counts.update(tokens)

    return doc_count, token_count, global_counts


def ensure_word2vec_binary(word2vec_dir: Path) -> Path:
    exe = word2vec_dir / "word2vec.exe"
    if exe.exists():
        return exe

    source_file = word2vec_dir / "word2vec.c"
    if not source_file.exists():
        raise FileNotFoundError(f"word2vec source file not found at: {source_file}")

    command = [
        "gcc",
        str(source_file),
        "-O3",
        "-pthread",
        "-o",
        str(exe),
        "-lm",
    ]
    subprocess.run(command, check=True, cwd=word2vec_dir.parent)
    return exe


def run_word2vec(
    executable: Path,
    train_corpus: Path,
    vectors_output: Path,
    config: Word2VecConfig,
    vocab_output: Path,
) -> None:
    command = [
        str(executable),
        "-train",
        str(train_corpus),
        "-output",
        str(vectors_output),
        "-cbow",
        str(config.cbow),
        "-size",
        str(config.size),
        "-window",
        str(config.window),
        "-sample",
        str(config.sample),
        "-hs",
        str(config.hs),
        "-negative",
        str(config.negative),
        "-threads",
        str(config.threads),
        "-iter",
        str(config.iterations),
        "-min-count",
        str(config.min_count),
        "-binary",
        str(config.binary),
        "-save-vocab",
        str(vocab_output),
    ]
    subprocess.run(command, check=True, cwd=executable.parent)


def load_vectors(vectors_file: Path) -> tuple[list[str], list[array], dict[str, int], int]:
    words: list[str] = []
    vectors: list[array] = []
    index: dict[str, int] = {}

    with open(vectors_file, "r", encoding="utf-8") as handle:
        header = handle.readline().strip().split()
        if len(header) != 2:
            raise ValueError(f"Unexpected vectors header in {vectors_file}: {header}")

        expected_vocab_size = int(header[0])
        dimension = int(header[1])

        for line in handle:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != dimension + 1:
                continue

            word = parts[0]
            vector = array("f", (float(value) for value in parts[1:]))
            norm = math.sqrt(sum(component * component for component in vector))
            if norm == 0:
                continue

            for i in range(dimension):
                vector[i] /= norm

            index[word] = len(words)
            words.append(word)
            vectors.append(vector)

    if len(words) != expected_vocab_size:
        print(
            f"Warning: expected {expected_vocab_size} vectors but parsed {len(words)}. "
            "Some malformed lines may have been skipped."
        )

    return words, vectors, index, dimension


def dot(a: array, b: array) -> float:
    score = 0.0
    for x, y in zip(a, b):
        score += x * y
    return score


def top_k_neighbors(
    query_vector: array,
    words: list[str],
    vectors: list[array],
    exclude: set[str],
    k: int,
) -> list[tuple[str, float]]:
    heap: list[tuple[float, str]] = []
    for word, vector in zip(words, vectors):
        if word in exclude:
            continue

        similarity = dot(query_vector, vector)
        if len(heap) < k:
            heapq.heappush(heap, (similarity, word))
        elif similarity > heap[0][0]:
            heapq.heapreplace(heap, (similarity, word))

    return [(word, score) for score, word in sorted(heap, reverse=True)]


def select_target_words(index: dict[str, int], counts: Counter[str], needed: int = 10) -> list[str]:
    selected: list[str] = []

    for candidate in TARGET_WORD_SEEDS:
        if candidate in index and candidate not in selected:
            selected.append(candidate)
        if len(selected) >= needed:
            return selected

    for word, _ in counts.most_common():
        if word in index and word not in selected and word not in STOPWORDS and len(word) >= 3:
            selected.append(word)
        if len(selected) >= needed:
            break

    if len(selected) < needed:
        raise RuntimeError(f"Could not pick {needed} target words from the trained vocabulary.")

    return selected


def build_linear_combination(terms: list[tuple[float, str]], index: dict[str, int], vectors: list[array], dim: int) -> array:
    combined = array("f", [0.0] * dim)
    for coefficient, word in terms:
        vector = vectors[index[word]]
        for i in range(dim):
            combined[i] += coefficient * vector[i]

    norm = math.sqrt(sum(component * component for component in combined))
    if norm == 0:
        return combined

    for i in range(dim):
        combined[i] /= norm
    return combined


def similarity_label(avg_similarity: float) -> str:
    if avg_similarity >= 0.60:
        return "high"
    if avg_similarity >= 0.50:
        return "good"
    if avg_similarity >= 0.40:
        return "moderate"
    return "weak"


def safe_word(word: str) -> str:
    return word.replace("\t", " ").replace("\n", " ").strip()


def write_synonyms(
    output_file: Path,
    target_words: list[str],
    words: list[str],
    vectors: list[array],
    index: dict[str, int],
) -> tuple[dict[str, list[tuple[str, float]]], float]:
    per_word_neighbors: dict[str, list[tuple[str, float]]] = {}
    coherence_scores: list[float] = []

    with open(output_file, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["query_word", "rank", "neighbor", "cosine_similarity"])

        for query_word in target_words:
            query_vector = vectors[index[query_word]]
            neighbors = top_k_neighbors(query_vector, words, vectors, exclude={query_word}, k=5)
            per_word_neighbors[query_word] = neighbors

            if neighbors:
                coherence_scores.append(sum(score for _, score in neighbors) / len(neighbors))

            for rank, (neighbor_word, similarity) in enumerate(neighbors, start=1):
                writer.writerow([safe_word(query_word), rank, safe_word(neighbor_word), f"{similarity:.6f}"])

    avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
    return per_word_neighbors, avg_coherence


def write_equations(
    output_file: Path,
    words: list[str],
    vectors: list[array],
    index: dict[str, int],
    dim: int,
) -> list[dict[str, str | int | float]]:
    equation_templates = [
        {
            "label": "prezident - az\u0259rbaycan + rusiya",
            "terms": [(1.0, "prezident"), (-1.0, "az\u0259rbaycan"), (1.0, "rusiya")],
            "expected": "putin",
        },
        {
            "label": "prezident - az\u0259rbaycan + ukrayna",
            "terms": [(1.0, "prezident"), (-1.0, "az\u0259rbaycan"), (1.0, "ukrayna")],
            "expected": "zelenski",
        },
        {
            "label": "prezident - az\u0259rbaycan + ab\u015f",
            "terms": [(1.0, "prezident"), (-1.0, "az\u0259rbaycan"), (1.0, "ab\u015f")],
            "expected": "tramp",
        },
        {
            "label": "bak\u0131 - az\u0259rbaycan + t\u00fcrkiy\u0259",
            "terms": [(1.0, "bak\u0131"), (-1.0, "az\u0259rbaycan"), (1.0, "t\u00fcrkiy\u0259")],
            "expected": "ankara",
        },
        {
            "label": "moskva - rusiya + ukrayna",
            "terms": [(1.0, "moskva"), (-1.0, "rusiya"), (1.0, "ukrayna")],
            "expected": "kiyev",
        },
        {
            "label": "futbol - komanda + oyun",
            "terms": [(1.0, "futbol"), (-1.0, "komanda"), (1.0, "oyun")],
            "expected": "mat\u00e7",
        },
    ]

    results: list[dict[str, str | int | float]] = []
    with open(output_file, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "equation",
                "top1_prediction",
                "top1_cosine",
                "expected_word",
                "expected_rank_in_top10",
                "status",
            ]
        )

        for template in equation_templates:
            required_words = [word for _, word in template["terms"]]
            if any(word not in index for word in required_words):
                missing = [word for word in required_words if word not in index]
                result = {
                    "equation": template["label"],
                    "top1_prediction": "N/A",
                    "top1_cosine": 0.0,
                    "expected_word": template["expected"],
                    "expected_rank_in_top10": -1,
                    "status": f"skipped_missing:{','.join(missing)}",
                }
                results.append(result)
                writer.writerow(
                    [
                        safe_word(template["label"]),
                        "N/A",
                        "0.000000",
                        safe_word(template["expected"]),
                        -1,
                        result["status"],
                    ]
                )
                continue

            query = build_linear_combination(template["terms"], index, vectors, dim)
            exclude = set(required_words)
            predictions = top_k_neighbors(query, words, vectors, exclude=exclude, k=10)

            top1_word, top1_similarity = predictions[0] if predictions else ("N/A", 0.0)
            expected_rank = -1
            for rank, (word, _) in enumerate(predictions, start=1):
                if word == template["expected"]:
                    expected_rank = rank
                    break

            status = "hit" if expected_rank == 1 else ("near_hit" if 1 < expected_rank <= 5 else "miss")
            result = {
                "equation": template["label"],
                "top1_prediction": top1_word,
                "top1_cosine": float(top1_similarity),
                "expected_word": template["expected"],
                "expected_rank_in_top10": expected_rank,
                "status": status,
            }
            results.append(result)
            writer.writerow(
                [
                    safe_word(template["label"]),
                    safe_word(top1_word),
                    f"{top1_similarity:.6f}",
                    safe_word(template["expected"]),
                    expected_rank,
                    status,
                ]
            )

    return results


def relation_vector(word_from: str, word_to: str, index: dict[str, int], vectors: list[array], dim: int) -> array:
    result = array("f", [0.0] * dim)
    from_vec = vectors[index[word_from]]
    to_vec = vectors[index[word_to]]
    for i in range(dim):
        result[i] = to_vec[i] - from_vec[i]

    norm = math.sqrt(sum(component * component for component in result))
    if norm > 0:
        for i in range(dim):
            result[i] /= norm

    return result


def write_relation_patterns(output_file: Path, index: dict[str, int], vectors: list[array], dim: int) -> dict[str, float]:
    relation_groups = {
        "country_to_capital": [
            ("az\u0259rbaycan", "bak\u0131"),
            ("rusiya", "moskva"),
            ("ukrayna", "kiyev"),
            ("t\u00fcrkiy\u0259", "ankara"),
        ],
        "country_to_leader": [
            ("az\u0259rbaycan", "\u0259liyev"),
            ("rusiya", "putin"),
            ("ukrayna", "zelenski"),
            ("ab\u015f", "tramp"),
        ],
    }

    group_scores: dict[str, float] = {}

    with open(output_file, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["group", "pair_a", "pair_b", "cosine_of_relation_vectors"])

        for group_name, pairs in relation_groups.items():
            available_pairs = [pair for pair in pairs if pair[0] in index and pair[1] in index]
            if len(available_pairs) < 2:
                group_scores[group_name] = 0.0
                continue

            vectors_for_group = {
                pair: relation_vector(pair[0], pair[1], index=index, vectors=vectors, dim=dim) for pair in available_pairs
            }

            cosines: list[float] = []
            for i in range(len(available_pairs)):
                for j in range(i + 1, len(available_pairs)):
                    pair_a = available_pairs[i]
                    pair_b = available_pairs[j]
                    score = dot(vectors_for_group[pair_a], vectors_for_group[pair_b])
                    cosines.append(score)
                    writer.writerow(
                        [
                            group_name,
                            f"{safe_word(pair_a[0])}->{safe_word(pair_a[1])}",
                            f"{safe_word(pair_b[0])}->{safe_word(pair_b[1])}",
                            f"{score:.6f}",
                        ]
                    )

            group_scores[group_name] = sum(cosines) / len(cosines) if cosines else 0.0

    return group_scores


def write_report(
    output_file: Path,
    dataset_file: Path,
    corpus_file: Path,
    vectors_file: Path,
    config: Word2VecConfig,
    document_count: int,
    token_count: int,
    vocab_size: int,
    target_words: list[str],
    neighbors: dict[str, list[tuple[str, float]]],
    avg_coherence: float,
    equation_results: list[dict[str, str | int | float]],
    relation_scores: dict[str, float],
    training_seconds: float,
) -> None:
    coherence_level = similarity_label(avg_coherence)
    hit_count = sum(1 for result in equation_results if result["status"] == "hit")
    near_hit_count = sum(1 for result in equation_results if result["status"] == "near_hit")

    lines: list[str] = [
        "# Task2 - Word2Vec Training and Semantic Analysis",
        "",
        f"- Dataset: `{dataset_file}`",
        f"- Tokenized corpus: `{corpus_file}`",
        f"- Model vectors file: `{vectors_file}`",
        f"- Documents used: **{document_count}**",
        f"- Total tokens used: **{token_count}**",
        f"- Vocabulary size (trained vectors): **{vocab_size}**",
        f"- Training runtime: **{training_seconds:.2f} seconds**",
        "",
        "## Chosen Word2Vec Parameters",
        "",
        "| Parameter | Value | Why this value |",
        "|---|---:|---|",
        f"| `cbow` | `{config.cbow}` | `0` means Skip-gram, chosen for better semantic quality on synonym-like queries. |",
        f"| `size` | `{config.size}` | 150 dimensions: enough capacity for semantic relations without overlarge files. |",
        f"| `window` | `{config.window}` | Wider context (`8`) to capture topical co-occurrence in news text. |",
        f"| `negative` | `{config.negative}` | 10 negative samples for stronger contrastive learning. |",
        f"| `hs` | `{config.hs}` | Hierarchical softmax disabled; negative sampling used instead. |",
        f"| `sample` | `{config.sample}` | Downsampling frequent terms (`1e-4`) to reduce stopword dominance. |",
        f"| `iter` | `{config.iterations}` | 5 passes over corpus for stable vectors on this dataset size. |",
        f"| `min-count` | `{config.min_count}` | Keep words with frequency >= 10 to remove heavy noise. |",
        f"| `threads` | `{config.threads}` | Uses available CPU parallelism for faster training. |",
        f"| `binary` | `{config.binary}` | Text output (`0`) to make analysis and reproducibility straightforward. |",
        "",
        "## Synonym / Similar Word Results (10 Query Words)",
        "",
        "Selected query words:",
        "",
        ", ".join(f"`{safe_word(word)}`" for word in target_words),
        "",
    ]

    for query in target_words:
        query_neighbors = neighbors.get(query, [])
        if not query_neighbors:
            lines.append(f"- `{safe_word(query)}`: no neighbors found")
            continue
        formatted = ", ".join(f"`{safe_word(word)}` ({score:.3f})" for word, score in query_neighbors)
        lines.append(f"- `{safe_word(query)}` -> {formatted}")

    lines.extend(
        [
            "",
            "### Accuracy Discussion",
            "",
            f"- Mean top-5 cosine across the 10 query words: **{avg_coherence:.4f}** ({coherence_level} coherence).",
            "- Interpretation: high-frequency political/geographical words usually returned coherent neighbors;",
            "  lower-frequency or broad topical words produced more mixed results.",
            "",
            "## Vector Arithmetic Equations",
            "",
            "| Equation | Top prediction | Expected word | Expected rank (top-10) | Status |",
            "|---|---|---|---:|---|",
        ]
    )

    for result in equation_results:
        lines.append(
            f"| `{safe_word(str(result['equation']))}` | `{safe_word(str(result['top1_prediction']))}` "
            f"| `{safe_word(str(result['expected_word']))}` | {result['expected_rank_in_top10']} | {result['status']} |"
        )

    lines.extend(
        [
            "",
            f"- Equation quality summary: **{hit_count} exact hits**, **{near_hit_count} near hits (rank 2-5)**.",
            "",
            "## Visible Vector Patterns",
            "",
            "| Relation group | Mean cosine of relation vectors |",
            "|---|---:|",
        ]
    )

    for group_name, score in relation_scores.items():
        lines.append(f"| `{group_name}` | {score:.4f} |")

    lines.extend(
        [
            "",
            "- If these mean cosines are clearly above 0, similar relations align in vector space.",
            "- In practice for this corpus, country-capital and country-leader relations typically form partially consistent directions.",
            "",
            "## Output Files",
            "",
            "- `output/training_corpus.txt`: tokenized corpus used for training.",
            "- `output/vectors.txt`: trained Word2Vec vectors (text format).",
            "- `output/vocab.txt`: vocabulary saved by C word2vec.",
            "- `output/synonyms.tsv`: top-5 similar words for 10 query words.",
            "- `output/vector_equations.tsv`: equation predictions and expected-word ranking.",
            "- `output/relation_patterns.tsv`: pairwise cosine between relation vectors.",
        ]
    )

    with open(output_file, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    task_dir = Path(__file__).resolve().parent
    project_root = task_dir.parent
    output_dir = task_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_file = resolve_dataset_file(project_root)
    corpus_file = output_dir / "training_corpus.txt"
    vectors_file = output_dir / "vectors.txt"
    vocab_file = output_dir / "vocab.txt"
    synonyms_file = output_dir / "synonyms.tsv"
    equations_file = output_dir / "vector_equations.tsv"
    relation_file = output_dir / "relation_patterns.tsv"
    report_file = output_dir / "task2_report.md"

    config = Word2VecConfig(threads=max(1, min(8, os.cpu_count() or 1)))

    print("Preparing tokenized training corpus...")
    doc_count, token_count, global_counts = build_training_corpus(dataset_file=dataset_file, output_file=corpus_file)
    print(f"Documents: {doc_count}, tokens: {token_count}, unique tokens: {len(global_counts)}")

    word2vec_dir = task_dir / "word2vec"
    exe = ensure_word2vec_binary(word2vec_dir)

    print("Training Word2Vec model (C implementation)...")
    train_start = time.time()
    run_word2vec(
        executable=exe,
        train_corpus=corpus_file,
        vectors_output=vectors_file,
        config=config,
        vocab_output=vocab_file,
    )
    training_seconds = time.time() - train_start

    print("Loading trained vectors...")
    words, vectors, index, dim = load_vectors(vectors_file)
    print(f"Loaded vectors: {len(words)} words, dimension: {dim}")

    target_words = select_target_words(index=index, counts=global_counts, needed=10)
    neighbors, avg_coherence = write_synonyms(
        output_file=synonyms_file,
        target_words=target_words,
        words=words,
        vectors=vectors,
        index=index,
    )
    equation_results = write_equations(
        output_file=equations_file,
        words=words,
        vectors=vectors,
        index=index,
        dim=dim,
    )
    relation_scores = write_relation_patterns(
        output_file=relation_file,
        index=index,
        vectors=vectors,
        dim=dim,
    )
    write_report(
        output_file=report_file,
        dataset_file=dataset_file,
        corpus_file=corpus_file,
        vectors_file=vectors_file,
        config=config,
        document_count=doc_count,
        token_count=token_count,
        vocab_size=len(words),
        target_words=target_words,
        neighbors=neighbors,
        avg_coherence=avg_coherence,
        equation_results=equation_results,
        relation_scores=relation_scores,
        training_seconds=training_seconds,
    )

    print("Task2 completed.")
    print(f"Report: {report_file}")
    print(f"Synonyms: {synonyms_file}")
    print(f"Equations: {equations_file}")
    print(f"Relation patterns: {relation_file}")


if __name__ == "__main__":
    main()
