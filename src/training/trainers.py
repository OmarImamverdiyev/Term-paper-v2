from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_classification_metrics
from src.features.base import FeatureSet
from src.models.neural import DenseFeatureDataset, SequenceFeatureDataset, build_torch_model
from src.models.sklearn_models import build_sklearn_model


@dataclass
class ExperimentOutcome:
    train_metrics: dict[str, Any]
    val_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    train_seconds: float
    inference_seconds: float
    model_path: Path
    extra_metadata: dict[str, Any]


def validate_experiment_compatibility(feature_type: str, model_type: str) -> None:
    sequence_feature = feature_type == "word2vec"
    if model_type == "multinomial_nb" and feature_type not in {"count", "tfidf"}:
        raise ValueError("MultinomialNB is only supported for count/tfidf features in this framework.")
    if model_type == "linear" and sequence_feature:
        raise ValueError("Linear classifiers require vector features, not sequence features.")
    if model_type == "mlp" and sequence_feature:
        raise ValueError("MLP is configured for vector features, not Word2Vec sequences.")
    if model_type in {"birnn", "lstm"} and not sequence_feature:
        raise ValueError(f"{model_type} requires Word2Vec sequence features.")


def _build_dataloaders(
    feature_set: FeatureSet,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int,
    eval_batch_size: int,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    if feature_set.is_sequence:
        train_dataset = SequenceFeatureDataset(feature_set.train_x, feature_set.train_lengths, train_labels)
        val_dataset = SequenceFeatureDataset(feature_set.val_x, feature_set.val_lengths, val_labels)
        test_dataset = SequenceFeatureDataset(feature_set.test_x, feature_set.test_lengths, test_labels)
    else:
        train_dataset = DenseFeatureDataset(feature_set.train_x, train_labels)
        val_dataset = DenseFeatureDataset(feature_set.val_x, val_labels)
        test_dataset = DenseFeatureDataset(feature_set.test_x, test_labels)

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    train_eval_loader = DataLoader(train_dataset, batch_size=eval_batch_size, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, **loader_kwargs)
    return train_loader, train_eval_loader, val_loader, test_loader


def _clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _resolve_class_weights(
    class_weight_config: Any,
    train_labels: np.ndarray,
    label_ids: list[int],
    device: torch.device,
) -> torch.Tensor | None:
    if class_weight_config is None or class_weight_config is False:
        return None
    if isinstance(class_weight_config, str) and class_weight_config.lower() == "none":
        return None

    if isinstance(class_weight_config, (list, tuple)):
        weights = torch.tensor(class_weight_config, dtype=torch.float32, device=device)
        if weights.numel() != len(label_ids):
            raise ValueError("Manual class weights must match the number of labels.")
        return weights

    if str(class_weight_config).lower() != "balanced":
        raise ValueError("class_weights must be 'none', 'balanced', or an explicit list.")

    counts = np.bincount(train_labels, minlength=len(label_ids)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / max(float(weights.mean()), 1.0e-8)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _build_optimizer(model: nn.Module, model_config: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_name = str(model_config.get("optimizer", "adam")).lower()
    learning_rate = float(model_config.get("learning_rate", 1.0e-3))
    weight_decay = float(model_config.get("weight_decay", 0.0))

    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=float(model_config.get("momentum", 0.9)),
            nesterov=bool(model_config.get("nesterov", False)),
            weight_decay=weight_decay,
        )
    if optimizer_name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    model_config: dict[str, Any],
) -> tuple[torch.optim.lr_scheduler.LRScheduler | None, str | None]:
    scheduler_config = model_config.get("scheduler")
    if scheduler_config is None or scheduler_config is False:
        scheduler_config = {"type": "reduce_on_plateau"}
    elif isinstance(scheduler_config, str) and scheduler_config.lower() == "none":
        scheduler_config = {"type": "reduce_on_plateau"}

    if isinstance(scheduler_config, str):
        scheduler_config = {"type": scheduler_config}
    if not isinstance(scheduler_config, dict):
        raise ValueError("scheduler must be omitted, a string, or a mapping.")

    scheduler_type = str(scheduler_config.get("type", "reduce_on_plateau")).lower()
    if scheduler_type == "none":
        return None, None

    if scheduler_type == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(scheduler_config.get("mode", "max")).lower(),
            factor=float(scheduler_config.get("factor", 0.5)),
            patience=int(scheduler_config.get("patience", 1)),
            threshold=float(scheduler_config.get("threshold", 1.0e-4)),
            min_lr=float(scheduler_config.get("min_lr", 1.0e-5)),
        )
        return scheduler, scheduler_type

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_config.get("t_max", model_config.get("epochs", 10))),
            eta_min=float(scheduler_config.get("min_lr", 1.0e-6)),
        )
        return scheduler, scheduler_type

    raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def _evaluate_torch_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    label_ids: list[int],
    positive_class_id: int | None,
    measure_time: bool = False,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    predictions: list[int] = []
    labels_list: list[int] = []

    start_time = time.perf_counter() if measure_time else None
    with torch.no_grad():
        for inputs, lengths, labels in loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            lengths = lengths.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")

            logits = model(inputs, lengths)
            loss = criterion(logits, labels)

            batch_predictions = torch.argmax(logits, dim=1)
            predictions.extend(batch_predictions.cpu().numpy().astype(np.int64).tolist())
            labels_list.extend(labels.cpu().numpy().astype(np.int64).tolist())

            total_loss += float(loss.item()) * labels.size(0)
            total_count += labels.size(0)

    elapsed = time.perf_counter() - start_time if start_time is not None else 0.0
    y_true = np.asarray(labels_list, dtype=np.int64)
    y_pred = np.asarray(predictions, dtype=np.int64)

    return {
        "loss": float(total_loss / max(total_count, 1)),
        "metrics": compute_classification_metrics(y_true, y_pred, label_ids, positive_class_id),
        "predictions": y_pred,
        "inference_seconds": float(elapsed),
    }


def train_sklearn_experiment(
    model_config: dict[str, Any],
    feature_set: FeatureSet,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    label_ids: list[int],
    positive_class_id: int | None,
    seed: int,
    model_output_path: Path,
) -> ExperimentOutcome:
    model = build_sklearn_model(model_config=model_config, seed=seed)

    start_train = time.perf_counter()
    model.fit(feature_set.train_x, train_labels)
    train_seconds = time.perf_counter() - start_train

    train_predictions = model.predict(feature_set.train_x)
    val_predictions = model.predict(feature_set.val_x)

    start_inference = time.perf_counter()
    test_predictions = model.predict(feature_set.test_x)
    inference_seconds = time.perf_counter() - start_inference

    train_metrics = compute_classification_metrics(train_labels, train_predictions, label_ids, positive_class_id)
    val_metrics = compute_classification_metrics(val_labels, val_predictions, label_ids, positive_class_id)
    test_metrics = compute_classification_metrics(test_labels, test_predictions, label_ids, positive_class_id)

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "model_config": model_config,
            "label_ids": label_ids,
            "positive_class_id": positive_class_id,
        },
        model_output_path,
    )

    return ExperimentOutcome(
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        train_seconds=float(train_seconds),
        inference_seconds=float(inference_seconds),
        model_path=model_output_path,
        extra_metadata={
            "test_predictions": test_predictions.astype(int).tolist(),
        },
    )


def train_torch_experiment(
    model_config: dict[str, Any],
    feature_set: FeatureSet,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    label_ids: list[int],
    positive_class_id: int | None,
    seed: int,
    device: torch.device,
    model_output_path: Path,
    logger: logging.Logger | None = None,
) -> ExperimentOutcome:
    input_dim = None if feature_set.is_sequence else int(feature_set.train_x.shape[1])
    embedding_matrix = feature_set.artifact.metadata.get("embedding_matrix")
    model = build_torch_model(
        model_config=model_config,
        input_dim=input_dim,
        output_dim=len(label_ids),
        embedding_matrix=embedding_matrix,
    ).to(device)

    batch_size = int(model_config.get("batch_size", 128))
    eval_batch_size = int(model_config.get("eval_batch_size", batch_size))
    epochs = int(model_config.get("epochs", 15))
    early_stopping_patience = int(model_config.get("patience", model_config.get("early_stopping_patience", 3)))
    early_stopping_min_delta = float(model_config.get("early_stopping_min_delta", 0.0))
    gradient_clip_norm = model_config.get("gradient_clip_norm", 1.0)
    num_workers = int(model_config.get("num_workers", 0))
    checkpoint_metric = str(model_config.get("checkpoint_metric", "macro_f1")).lower()

    train_loader, train_eval_loader, val_loader, test_loader = _build_dataloaders(
        feature_set=feature_set,
        train_labels=np.asarray(train_labels, dtype=np.int64),
        val_labels=np.asarray(val_labels, dtype=np.int64),
        test_labels=np.asarray(test_labels, dtype=np.int64),
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        seed=seed,
        num_workers=num_workers,
        device=device,
    )

    class_weights = _resolve_class_weights(
        class_weight_config=model_config.get("class_weights", "none"),
        train_labels=np.asarray(train_labels, dtype=np.int64),
        label_ids=label_ids,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = _build_optimizer(model=model, model_config=model_config)
    scheduler, scheduler_type = _build_scheduler(optimizer=optimizer, model_config=model_config)

    model_output_path.parent.mkdir(parents=True, exist_ok=True)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_val_score = float("-inf")
    stale_epochs = 0
    epochs_ran = 0
    history: list[dict[str, Any]] = []

    start_train = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_examples = 0

        for inputs, lengths, labels in train_loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            lengths = lengths.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
            optimizer.step()

            total_train_loss += float(loss.item()) * labels.size(0)
            total_train_examples += labels.size(0)

        train_loss = float(total_train_loss / max(total_train_examples, 1))
        val_eval = _evaluate_torch_model(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
            label_ids=label_ids,
            positive_class_id=positive_class_id,
        )
        val_loss = float(val_eval["loss"])
        val_metrics = val_eval["metrics"]
        current_lr = float(optimizer.param_groups[0]["lr"])

        if checkpoint_metric == "loss":
            current_score = -val_loss
        else:
            if checkpoint_metric not in val_metrics:
                raise ValueError(f"Unsupported checkpoint metric: {checkpoint_metric}")
            current_score = float(val_metrics[checkpoint_metric])

        history_row = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_f1": float(val_metrics["f1"]),
        }
        history.append(history_row)

        if logger is not None:
            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_accuracy=%.4f | val_macro_f1=%.4f | lr=%.6f",
                epoch,
                epochs,
                train_loss,
                val_loss,
                val_metrics["accuracy"],
                val_metrics["macro_f1"],
                current_lr,
            )

        improved = current_score > (best_val_score + early_stopping_min_delta)
        if improved:
            best_val_score = current_score
            best_epoch = epoch
            stale_epochs = 0
            best_state = _clone_state_dict(model.state_dict())
            torch.save(
                {
                    "state_dict": best_state,
                    "model_config": model_config,
                    "label_ids": label_ids,
                    "positive_class_id": positive_class_id,
                    "input_dim": input_dim,
                    "embedding_matrix": embedding_matrix,
                    "best_epoch": best_epoch,
                    "best_val_score": best_val_score,
                    "checkpoint_metric": checkpoint_metric,
                    "history": history,
                },
                model_output_path,
            )
        else:
            stale_epochs += 1

        if scheduler is not None:
            if scheduler_type == "reduce_on_plateau":
                metric_for_scheduler = current_score if getattr(scheduler, "mode", "max") == "max" else val_loss
                scheduler.step(metric_for_scheduler)
            else:
                scheduler.step()

        epochs_ran = epoch
        if stale_epochs >= early_stopping_patience:
            if logger is not None:
                logger.info(
                    "Early stopping triggered after epoch %d (best_epoch=%d, checkpoint_metric=%s, best_score=%.4f)",
                    epoch,
                    best_epoch,
                    checkpoint_metric,
                    best_val_score,
                )
            break

    train_seconds = time.perf_counter() - start_train

    if best_state is None:
        best_state = _clone_state_dict(model.state_dict())
        torch.save(
            {
                "state_dict": best_state,
                "model_config": model_config,
                "label_ids": label_ids,
                "positive_class_id": positive_class_id,
                "input_dim": input_dim,
                "embedding_matrix": embedding_matrix,
                "best_epoch": epochs_ran,
                "best_val_score": best_val_score,
                "checkpoint_metric": checkpoint_metric,
                "history": history,
            },
            model_output_path,
        )

    checkpoint = torch.load(model_output_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    train_eval = _evaluate_torch_model(
        model=model,
        loader=train_eval_loader,
        device=device,
        criterion=criterion,
        label_ids=label_ids,
        positive_class_id=positive_class_id,
    )
    val_eval = _evaluate_torch_model(
        model=model,
        loader=val_loader,
        device=device,
        criterion=criterion,
        label_ids=label_ids,
        positive_class_id=positive_class_id,
    )
    test_eval = _evaluate_torch_model(
        model=model,
        loader=test_loader,
        device=device,
        criterion=criterion,
        label_ids=label_ids,
        positive_class_id=positive_class_id,
        measure_time=True,
    )

    return ExperimentOutcome(
        train_metrics=train_eval["metrics"],
        val_metrics=val_eval["metrics"],
        test_metrics=test_eval["metrics"],
        train_seconds=float(train_seconds),
        inference_seconds=float(test_eval["inference_seconds"]),
        model_path=model_output_path,
        extra_metadata={
            "best_epoch": int(checkpoint.get("best_epoch", best_epoch or epochs_ran)),
            "best_val_score": float(checkpoint.get("best_val_score", best_val_score)),
            "checkpoint_metric": checkpoint_metric,
            "epochs_ran": int(epochs_ran),
            "history": history,
            "train_loss": float(train_eval["loss"]),
            "val_loss": float(val_eval["loss"]),
            "test_loss": float(test_eval["loss"]),
            "test_predictions": test_eval["predictions"].astype(int).tolist(),
        },
    )
