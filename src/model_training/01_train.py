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
FILTER_CLASS_IDS = [LABEL_MAPPING["graphic"], LABEL_MAPPING["mixture"]]


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


def video_group_from_path(path: Path) -> str:
    stem = path.stem
    if "__" in stem:
        return stem.split("__", 1)[0]
    if "_t" in stem:
        return stem.split("_t", 1)[0]
    return path.parent.name


def collect_training_data(data_dir: str | Path) -> tuple[list[Path], np.ndarray]:
    root = Path(data_dir)
    paths: list[Path] = []
    labels: list[int] = []
    counts: dict[str, int] = {}
    for class_name in CLASS_NAMES:
        class_paths = list_images(root / class_name)
        counts[class_name] = len(class_paths)
        paths.extend(class_paths)
        labels.extend([LABEL_MAPPING[class_name]] * len(class_paths))

    missing = [class_name for class_name, count in counts.items() if count == 0]
    if missing:
        details = ", ".join(f"{class_name}={count}" for class_name, count in counts.items())
        raise FileNotFoundError(
            f"Expected labeled images in {', '.join(str(root / name) for name in CLASS_NAMES)}; found {details}"
        )
    return paths, np.asarray(labels, dtype=np.int64)


def maybe_stratify(labels: np.ndarray) -> np.ndarray | None:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    return labels if len(np.unique(labels)) == len(CLASS_NAMES) and int(counts.min()) >= 2 else None


def split_by_image(
    labels: np.ndarray,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    stratify = maybe_stratify(labels)
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    train_val_labels = labels[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size / (1.0 - test_size),
        random_state=seed,
        stratify=maybe_stratify(train_val_labels),
    )
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def split_by_video(
    paths: list[Path],
    labels: np.ndarray,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    groups = np.asarray([video_group_from_path(path) for path in paths])
    unique_groups = np.asarray(sorted(set(groups.tolist())))
    if len(unique_groups) < 3:
        raise ValueError("Need at least 3 video groups for --split-mode video.")

    group_labels: list[int] = []
    for group in unique_groups:
        group_indices = np.flatnonzero(groups == group)
        counts = np.bincount(labels[group_indices], minlength=len(CLASS_NAMES))
        group_labels.append(int(np.argmax(counts)))

    group_labels_array = np.asarray(group_labels, dtype=np.int64)
    stratify = maybe_stratify(group_labels_array)
    train_val_groups, test_groups = train_test_split(
        unique_groups,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    train_val_group_labels = np.asarray(
        [group_labels_array[np.flatnonzero(unique_groups == group)[0]] for group in train_val_groups],
        dtype=np.int64,
    )
    train_groups, val_groups = train_test_split(
        train_val_groups,
        test_size=val_size / (1.0 - test_size),
        random_state=seed,
        stratify=maybe_stratify(train_val_group_labels),
    )
    train_group_set = set(train_groups.tolist())
    val_group_set = set(val_groups.tolist())
    test_group_set = set(test_groups.tolist())
    train_idx = np.asarray([index for index, group in enumerate(groups) if group in train_group_set], dtype=np.int64)
    val_idx = np.asarray([index for index, group in enumerate(groups) if group in val_group_set], dtype=np.int64)
    test_idx = np.asarray([index for index, group in enumerate(groups) if group in test_group_set], dtype=np.int64)
    if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
        raise ValueError("Video split produced an empty train, validation, or test set.")

    metadata = {
        "train_groups": sorted(train_group_set),
        "val_groups": sorted(val_group_set),
        "test_groups": sorted(test_group_set),
        "n_train_groups": len(train_group_set),
        "n_val_groups": len(val_group_set),
        "n_test_groups": len(test_group_set),
        "n_total_groups": len(unique_groups),
    }
    return train_idx, val_idx, test_idx, metadata


def make_split(paths: list[Path], labels: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if args.split_mode == "image":
        train_idx, val_idx, test_idx = split_by_image(labels, val_size=args.val_size, test_size=args.test_size, seed=args.seed)
        return train_idx, val_idx, test_idx, {"mode": "image"}

    try:
        train_idx, val_idx, test_idx, metadata = split_by_video(
            paths,
            labels,
            val_size=args.val_size,
            test_size=args.test_size,
            seed=args.seed,
        )
        metadata["mode"] = "video"
        return train_idx, val_idx, test_idx, metadata
    except ValueError:
        if not args.fallback_to_image_split:
            raise
        print("[warn] Video split impossible; falling back to image split.")
        train_idx, val_idx, test_idx = split_by_image(labels, val_size=args.val_size, test_size=args.test_size, seed=args.seed)
        return train_idx, val_idx, test_idx, {"mode": "image", "fallback_from": "video"}


def select_threshold_for_recall(
    y_true: np.ndarray,
    proba_filter: np.ndarray,
    min_precision: float = 0.60,
    fallback: float = 0.5,
) -> float:
    y_binary = np.isin(y_true, FILTER_CLASS_IDS).astype(int)
    best_threshold = fallback
    best_recall = -1.0
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = (proba_filter >= threshold).astype(int)
        precision = precision_score(y_binary, pred, zero_division=0)
        recall = recall_score(y_binary, pred, zero_division=0)
        f1 = f1_score(y_binary, pred, zero_division=0)
        if precision >= min_precision and (recall > best_recall or (recall == best_recall and f1 > best_f1)):
            best_threshold = float(threshold)
            best_recall = float(recall)
            best_f1 = float(f1)
    return best_threshold


def print_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = np.argmax(proba, axis=1)
    y_filter_true = np.isin(y_true, FILTER_CLASS_IDS).astype(int)
    proba_filter = proba[:, FILTER_CLASS_IDS].sum(axis=1)
    y_filter_pred = (proba_filter >= threshold).astype(int)
    metrics = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_filter": float(precision_score(y_filter_true, y_filter_pred, zero_division=0)),
        "recall_filter": float(recall_score(y_filter_true, y_filter_pred, zero_division=0)),
        "f1_filter": float(f1_score(y_filter_true, y_filter_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES)))).tolist(),
        "filter_confusion_matrix": confusion_matrix(y_filter_true, y_filter_pred, labels=[0, 1]).tolist(),
    }
    print(f"Threshold: {threshold:.3f}")
    print(f"Accuracy:         {metrics['accuracy']:.4f}")
    print(f"Precision macro:  {metrics['precision_macro']:.4f}")
    print(f"Recall macro:     {metrics['recall_macro']:.4f}")
    print(f"F1 macro:         {metrics['f1_macro']:.4f}")
    print(f"Precision filter: {metrics['precision_filter']:.4f}")
    print(f"Recall filter:    {metrics['recall_filter']:.4f}")
    print(f"F1 filter:        {metrics['f1_filter']:.4f}")
    print(f"Confusion matrix rows=true cols=pred, classes={CLASS_NAMES}:")
    print(np.asarray(metrics["confusion_matrix"]))
    print("Filter confusion matrix rows=true cols=pred, classes=[keep, filter]:")
    print(np.asarray(metrics["filter_confusion_matrix"]))
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, labels=list(range(len(CLASS_NAMES))), zero_division=0))
    return metrics


def label_counts(labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(labels.astype(np.int64), minlength=len(CLASS_NAMES))
    return {class_name: int(counts[index]) for index, class_name in enumerate(CLASS_NAMES)}


def print_split_paths(split_name: str, paths: list[Path], indices: np.ndarray) -> None:
    print(f"{split_name} selected images ({len(indices)}):")
    for index in indices.tolist():
        print(f"  {paths[int(index)]}")


def resolve_model_out(model_out: str | None) -> Path:
    if model_out:
        return Path(model_out)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("models") / f"frame_filter_{timestamp}.joblib"


def train(args: argparse.Namespace) -> None:
    paths, labels = collect_training_data(args.data)
    if args.val_size <= 0 or args.test_size <= 0:
        raise ValueError("--val-size and --test-size must both be > 0")
    if args.val_size + args.test_size >= 1.0:
        raise ValueError("--val-size + --test-size must be < 1")
    config = EmbeddingConfig(
        dino_model=args.dino_model,
        clip_model=args.clip_model,
        crop_bottom=args.crop_bottom,
    )
    device = choose_device(args.device)
    print(f"Using device: {device}")
    embedder = FrozenBackboneEmbedder(config=config, device=device)
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.data) / ".embedding_cache"
    x = embed_image_paths(paths, embedder, batch_size=args.batch_size, cache_dir=cache_dir)

    train_idx, val_idx, test_idx, split_metadata = make_split(paths, labels, args)
    x_train, x_val, x_test = x[train_idx], x[val_idx], x[test_idx]
    y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]
    print(
        f"Split mode: {split_metadata['mode']} | "
        f"train images={len(train_idx)} | val images={len(val_idx)} | test images={len(test_idx)}"
    )
    print(f"Train class counts: {label_counts(y_train)}")
    print(f"Val class counts:   {label_counts(y_val)}")
    print(f"Test class counts:  {label_counts(y_test)}")
    print_split_paths("Train", paths, train_idx)
    print_split_paths("Val", paths, val_idx)
    print_split_paths("Test", paths, test_idx)
    if split_metadata["mode"] == "video":
        print(
            f"Train videos={split_metadata['n_train_groups']} | "
            f"val videos={split_metadata['n_val_groups']} | "
            f"test videos={split_metadata['n_test_groups']}"
        )

    scaler = StandardScaler()
    if args.classifier == "logreg":
        classifier = LogisticRegression(
            max_iter=args.max_iter,
            class_weight={
                LABEL_MAPPING["footage"]: 1.0,
                LABEL_MAPPING["graphic"]: args.graphic_class_weight,
                LABEL_MAPPING["mixture"]: args.mixture_class_weight,
            },
            C=args.c,
            random_state=args.seed,
        )
        model = Pipeline([("scaler", scaler), ("classifier", classifier)])
    else:
        classifier = TorchMLPClassifier(
            hidden_dim=args.hidden_dim,
            num_classes=len(CLASS_NAMES),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            threshold=args.threshold,
            device=device,
            random_state=args.seed,
        )
        model = Pipeline([("scaler", scaler), ("classifier", classifier)])

    model.fit(x_train, y_train)
    val_proba = model.predict_proba(x_val)
    val_proba_filter = val_proba[:, FILTER_CLASS_IDS].sum(axis=1)
    recommended_threshold = args.threshold
    if args.auto_threshold:
        recommended_threshold = select_threshold_for_recall(
            y_val,
            val_proba_filter,
            min_precision=args.min_precision_for_auto_threshold,
            fallback=args.threshold,
        )
    print("\nValidation metrics")
    val_metrics = print_metrics(y_val, val_proba, threshold=recommended_threshold)
    test_proba = model.predict_proba(x_test)
    print("\nTest metrics")
    test_metrics = print_metrics(y_test, test_proba, threshold=recommended_threshold)

    model_out = resolve_model_out(args.model_out)

    payload = {
        "classifier": model,
        "crop_bottom": args.crop_bottom,
        "backbones": {
            "dino": args.dino_model,
            "clip": args.clip_model,
        },
        "embedding_config": config.to_dict(),
        "scaler": model.named_steps.get("scaler"),
        "threshold_recommended": recommended_threshold,
        "label_mapping": LABEL_MAPPING,
        "class_names": CLASS_NAMES,
        "filter_class_names": ["graphic", "mixture"],
        "metrics": {
            "val": val_metrics,
            "test": test_metrics,
        },
        "split": split_metadata,
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
    parser = argparse.ArgumentParser(description="Train frame filter classifier.")
    parser.add_argument(
        "--data",
        default="data/frames/03_corrected_sort",
        help="Directory containing footage/, graphic/ and mixture/ image folders.",
    )
    parser.add_argument("--model-out", default=None, help="Output model path. Default: models/frame_filter_YYYY-MM-DD_HH-MM-SS.joblib")
    parser.add_argument("--crop-bottom", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None, help="cuda, cpu, or auto if omitted.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--dino-model", default=DEFAULT_DINO_MODEL)
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--classifier", choices=["logreg", "mlp"], default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--auto-threshold", action="store_true", help="Choose a threshold on validation data favoring graphic+mixture recall.")
    parser.add_argument("--min-precision-for-auto-threshold", type=float, default=0.60)
    parser.add_argument("--graphic-class-weight", "--filter-class-weight", dest="graphic_class_weight", type=float, default=2.0)
    parser.add_argument("--mixture-class-weight", type=float, default=1.5)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--split-mode", choices=["video", "image"], default="video", help="Split by video id by default; use image only for quick debugging.")
    parser.add_argument("--fallback-to-image-split", action="store_true", help="Fallback to image split if video split cannot be built.")
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

    with log_path.open("w", encoding="utf-8") as log_handle:
        stdout_tee = TeeWriter(sys.stdout, log_handle)
        stderr_tee = TeeWriter(sys.stderr, log_handle)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Training log: {log_path}")
            train(args)

    print(f"Saved training log to {log_path}")


if __name__ == "__main__":
    main()
