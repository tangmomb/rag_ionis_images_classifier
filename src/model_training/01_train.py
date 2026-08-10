from __future__ import annotations

import argparse
import contextlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embeddings import DEFAULT_CLIP_MODEL, DEFAULT_DINO_MODEL, EmbeddingConfig, FrozenBackboneEmbedder, embed_image_paths
from mlp_classifier import TorchMLPClassifier
from utils import choose_device, ensure_dir, list_images


CLASS_NAMES = ["footage", "graphic", "mixture"]
LABEL_MAPPING = {name: index for index, name in enumerate(CLASS_NAMES)}


class TeeWriter:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def collect_training_data(data_dir: str | Path) -> tuple[list[Path], np.ndarray]:
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Training dataset not found: {root}")

    paths: list[Path] = []
    labels: list[int] = []
    counts: dict[str, int] = {}
    for class_name in CLASS_NAMES:
        class_dir = root / class_name
        class_paths = list_images(class_dir)
        counts[class_name] = len(class_paths)
        paths.extend(class_paths)
        labels.extend([LABEL_MAPPING[class_name]] * len(class_paths))

    missing = [class_name for class_name, count in counts.items() if count == 0]
    if missing:
        details = ", ".join(f"{class_name}={count}" for class_name, count in counts.items())
        raise FileNotFoundError(
            f"Expected non-empty footage/, graphic/ and mixture/ directories in {root}; found {details}"
        )
    return paths, np.asarray(labels, dtype=np.int64)


def split_dataset(
    labels: np.ndarray,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    if int(counts.min()) < 10:
        raise ValueError("Each class must contain at least 10 images to create stratified train, validation and test sets.")

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size / (1.0 - test_size),
        random_state=seed,
        stratify=labels[train_val_idx],
    )
    return train_idx, val_idx, test_idx


def print_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = np.argmax(probabilities, axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_macro": float(precision_score(y_true, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=list(range(len(CLASS_NAMES))),
        ).tolist(),
    }
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Precision macro: {metrics['precision_macro']:.4f}")
    print(f"Recall macro:    {metrics['recall_macro']:.4f}")
    print(f"F1 macro:        {metrics['f1_macro']:.4f}")
    print(f"Confusion matrix rows=true cols=pred, classes={CLASS_NAMES}:")
    print(np.asarray(metrics["confusion_matrix"]))
    print(
        classification_report(
            y_true,
            predictions,
            target_names=CLASS_NAMES,
            labels=list(range(len(CLASS_NAMES))),
            zero_division=0,
        )
    )
    return metrics


def label_counts(labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(labels.astype(np.int64), minlength=len(CLASS_NAMES))
    return {class_name: int(counts[index]) for index, class_name in enumerate(CLASS_NAMES)}


def resolve_model_out(model_out: str | None) -> Path:
    if model_out:
        return Path(model_out)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("models") / f"image_classifier_{timestamp}.joblib"


def train(args: argparse.Namespace) -> None:
    paths, labels = collect_training_data(args.data)
    if args.val_size <= 0 or args.test_size <= 0:
        raise ValueError("--val-size and --test-size must both be > 0")
    if args.val_size + args.test_size >= 1.0:
        raise ValueError("--val-size + --test-size must be < 1")

    train_idx, val_idx, test_idx = split_dataset(
        labels,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    print(f"Dataset: {Path(args.data).resolve()}")
    print(f"Class counts: {label_counts(labels)}")
    print(f"Split: train={len(train_idx)} | validation={len(val_idx)} | test={len(test_idx)}")

    config = EmbeddingConfig(
        dino_model=args.dino_model,
        clip_model=args.clip_model,
        crop_bottom=args.crop_bottom,
    )
    device = choose_device(args.device)
    print(f"Using device: {device}")
    embedder = FrozenBackboneEmbedder(config=config, device=device)
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.data) / ".embedding_cache"
    embeddings = embed_image_paths(paths, embedder, batch_size=args.batch_size, cache_dir=cache_dir)

    x_train, x_val, x_test = embeddings[train_idx], embeddings[val_idx], embeddings[test_idx]
    y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]

    scaler = StandardScaler()
    if args.classifier == "logreg":
        classifier = LogisticRegression(
            max_iter=args.max_iter,
            class_weight="balanced",
            C=args.c,
            random_state=args.seed,
        )
    else:
        classifier = TorchMLPClassifier(
            hidden_dim=args.hidden_dim,
            num_classes=len(CLASS_NAMES),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            random_state=args.seed,
        )
    model = Pipeline([("scaler", scaler), ("classifier", classifier)])
    model.fit(x_train, y_train)

    print("\nValidation metrics")
    val_metrics = print_metrics(y_val, model.predict_proba(x_val))
    print("\nTest metrics")
    test_metrics = print_metrics(y_test, model.predict_proba(x_test))

    model_out = resolve_model_out(args.model_out)
    payload = {
        "classifier": model,
        "crop_bottom": args.crop_bottom,
        "backbones": {"dino": args.dino_model, "clip": args.clip_model},
        "embedding_config": config.to_dict(),
        "label_mapping": LABEL_MAPPING,
        "class_names": CLASS_NAMES,
        "metrics": {"validation": val_metrics, "test": test_metrics},
        "split": {
            "train_size": len(train_idx),
            "validation_size": len(val_idx),
            "test_size": len(test_idx),
            "seed": args.seed,
        },
        "classifier_type": args.classifier,
    }
    ensure_dir(model_out.parent)
    joblib.dump(payload, model_out)
    print(f"Saved model to {model_out}")


def ask_classifier(args: argparse.Namespace) -> None:
    if args.classifier is not None:
        return
    while True:
        value = input("Classifier to train (`logreg` or `mlp`) [logreg]: ").strip().lower()
        if not value:
            args.classifier = "logreg"
            return
        if value in {"logreg", "mlp"}:
            args.classifier = value
            return
        print("Enter `logreg` or `mlp`.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a three-class image classifier.")
    parser.add_argument(
        "--data",
        default="data/frames/dataset",
        help="Directory containing non-empty footage/, graphic/ and mixture/ folders.",
    )
    parser.add_argument(
        "--model-out",
        default=None,
        help="Output model path. Default: models/image_classifier_YYYY-MM-DD_HH-MM-SS.joblib",
    )
    parser.add_argument("--crop-bottom", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None, help="cuda, cpu, or auto if omitted.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--dino-model", default=DEFAULT_DINO_MODEL)
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--classifier", choices=["logreg", "mlp"], default=None)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ask_classifier(args)
    model_out = resolve_model_out(args.model_out)
    args.model_out = str(model_out)
    log_path = model_out.with_suffix(".log")
    ensure_dir(log_path.parent)

    with log_path.open("w", encoding="utf-8") as log_handle:
        stdout_tee = TeeWriter(sys.stdout, log_handle)
        stderr_tee = TeeWriter(sys.stderr, log_handle)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Training log: {log_path}")
            train(args)

    print(f"Saved training log to {log_path}")


if __name__ == "__main__":
    main()
