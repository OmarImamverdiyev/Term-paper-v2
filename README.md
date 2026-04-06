# Modular Sentiment Research Pipeline

DATASET: https://drive.google.com/drive/folders/1PjAxfQfD__dYNrEGyyZQYpyiE2bL2HDP?usp=sharing

This repository now includes a reusable experiment framework for sentiment classification research in `src/` with config-driven execution from `scripts/run_experiments.py`.

The design goal is to make English experiments and future Azerbaijani translated experiments directly comparable by reusing the exact same saved split artifacts from `splits/`.

## Included combinations

The framework supports these feature-model combinations:

- CountVectorizer + Multinomial Naive Bayes
- TfidfVectorizer + Multinomial Naive Bayes
- CountVectorizer + linear classifier
- TfidfVectorizer + linear classifier
- CountVectorizer + MLP
- TfidfVectorizer + MLP
- PMI features + linear classifier
- Word2Vec + Bidirectional RNN
- Word2Vec + LSTM

The linear classifier is configurable as Logistic Regression or Linear SVM.

## Project structure

```text
data/
src/
  preprocessing/
  features/
  models/
  training/
  evaluation/
  utils/
configs/
scripts/
results/
logs/
models/
splits/
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you enable NLTK stopwords or WordNet lemmatization in a config, download the corresponding resources:

```bash
python -m nltk.downloader stopwords wordnet omw-1.4
```

## Config overview

Each YAML config defines:

- `datasets`: CSV path, text column, label column, optional ID column, task type, and split artifact settings
- `preprocessing`: reusable language-agnostic text cleanup profiles
- `features`: count, tf-idf, PMI, and Word2Vec settings
- `models`: Naive Bayes, linear, MLP, BiRNN, and LSTM settings
- `experiments`: batch runs that reference one dataset + one preprocessing profile + one feature + one model

## Run experiments

Main full-dataset run on `sentiment140_100k_clean_balanced_v2.csv`:

```bash
python main.py
```

Equivalent explicit command:

```bash
python scripts/run_experiments.py
```

Example smaller sample pipeline on the bundled Sentiment140 CSV:

```bash
python scripts/run_experiments.py --config configs/sample_sentiment140_pipeline.yaml
```

Run only selected experiments from the same config:

```bash
python scripts/run_experiments.py --config configs/sample_sentiment140_pipeline.yaml --feature word2vec_default --model birnn_default
```

Run only the dedicated Logistic Regression and Naive Bayes tuning pipeline:

```bash
python scripts/run_traditional_ml_tuning.py --config configs/main_sentiment140_full.yaml
```

Every run saves:

- `results/<run_id>/summary.csv`
- `results/<run_id>/summary.json`
- `results/<run_id>/experiment_details/*.json`
- `models/<run_id>/feature_artifacts/*`
- `models/<run_id>/model_artifacts/*`
- `logs/<run_id>.log`
- `splits/<artifact_name>.json`

## Reusing splits for English vs Azerbaijani comparison

1. Run the English dataset with `split.mode: create_or_reuse` and a stable `artifact_name`.
2. Put the translated Azerbaijani CSV in a second dataset entry.
3. Point the translated dataset to the same `artifact_name`.
4. Set the translated dataset split mode to `reuse`.
5. Keep row IDs aligned across the source and translated CSVs. If you do not have an explicit ID column, keep row order identical.

This is the key mechanism that preserves identical train/validation/test membership for machine-translation impact analysis later.

## Notes on preprocessing

The preprocessing module is configurable for:

- lowercase
- URL handling
- mention handling
- hashtag handling
- punctuation handling
- number handling
- whitespace cleanup
- emoji preservation
- optional stopword removal
- optional stemming or lemmatization

Nothing is hardcoded for English only. Language-specific resources are only used when you explicitly enable them in config.

## Results aggregation

Combine past run summaries into one table:

```bash
python scripts/aggregate_results.py
```

This writes:

- `results/aggregated_results.csv`
- `results/aggregated_results.json`
- `results/best_by_dataset.csv`
