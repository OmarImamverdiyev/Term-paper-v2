# Task2

This folder contains Task2 for training and analyzing Word2Vec on the main dataset in `Corpora/news/`.

The implementation uses the original C code from:

- https://github.com/tmikolov/word2vec

Local clone location used by this task:

- `Task2/word2vec`

## Run

From project root:

```powershell
./venv/Scripts/python.exe Task2/task2.py
```

## Outputs (`Task2/output/`)

- `task2_report.md`: parameter choices, synonym quality discussion, vector arithmetic patterns.
- `synonyms.tsv`: top-5 similar words for 10 query words.
- `vector_equations.tsv`: results of vector equations and expected-word rank.
- `relation_patterns.tsv`: cosine similarity between relation vectors.
- `training_corpus.txt`: tokenized corpus generated from `content_only.csv`.
- `vectors.txt`: trained word vectors (text format).
- `vocab.txt`: vocabulary dumped by C word2vec.
