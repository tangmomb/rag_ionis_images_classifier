from __future__ import annotations

import argparse
import sys
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


def split_by_image(labels: np.ndarray, val_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    stratify = labels if len(np.unique(labels)) == len(CLASS_NAMES) and int(counts.min()) >= 2 else None
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        random_state=seed,
        stratify=stratify,
    )
    return np.asarray(train_idx), np.asarray(val_idx)


def split_by_video(paths: list[Path], labels: np.ndarray, val_size: float, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    groups = np.asarray([video_group_from_path(path) for path in paths])
    unique_groups = np.asarray(sorted(set(groups.tolist())))
    if len(unique_groups) < 2:
        raise ValueError("Need at least 2 video groups for --split-mode video.")

    group_labels: list[int] = []
    for group in unique_groups:
        group_indices = np.flatnonzero(groups == group)
        counts = np.bincount(labels[group_indices], minlength=len(CLASS_NAMES))
        group_labels.append(int(np.argmax(counts)))

    group_labels_array = np.asarray(group_labels, dtype=np.int64)
    group_counts = np.bincount(group_labels_array, minlength=len(CLASS_NAMES))
    stratify = group_labels_array if len(np.unique(group_labels_array)) > 1 and int(group_counts[group_counts > 0].min()) >= 2 else None

    train_groups, val_groups = train_test_split(
        unique_groups,
        test_size=val_size,
        random_state=seed,
        stratify=stratify,
    )
    train_group_set = set(train_groups.tolist())
    val_group_set = set(val_groups.tolist())
    train_idx = np.asarray([index for index, group in enumerate(groups) if group in train_group_set], dtype=np.int64)
    val_idx = np.asarray([index for index, group in enumerate(groups) if group in val_group_set], dtype=np.int64)
    if train_idx.size == 0 or val_idx.size == 0:
        raise ValueError("Video split produced an empty train or validation set.")

    metadata = {
        "train_groups": sorted(train_group_set),
        "val_groups": sorted(val_group_set),
        "n_train_groups": len(train_group_set),
        "n_val_groups": len(val_group_set),
        "n_total_groups": len(unique_groups),
    }
    return train_idx, val_idx, metadata


def make_split(paths: list[Path], labels: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if args.split_mode == "image":
        train_idx, val_idx = split_by_image(labels, val_size=args.val_size, seed=args.seed)
        return train_idx, val_idx, {"mode": "image"}

    try:
        train_idx, val_idx, metadata = split_by_video(paths, labels, val_size=args.val_size, seed=args.seed)
        metadata["mode"] = "video"
        return train_idx, val_idx, metadata
    except ValueError:
        if not args.fallback_to_image_split:
            raise
        print("[warn] Video split impossible; falling back to image split.")
        train_idx, val_idx = split_by_image(labels, val_size=args.val_size, seed=args.seed)
        return train_idx, val_idx, {"mode": "image", "fallback_from": "video"}


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


def train(args: argparse.Namespace) -> None:
    paths, labels = collect_training_data(args.data)
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

    train_idx, val_idx, split_metadata = make_split(paths, labels, args)
    x_train, x_val = x[train_idx], x[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]
    print(
        f"Split mode: {split_metadata['mode']} | "
        f"train images={len(train_idx)} | val images={len(val_idx)}"
    )
    if split_metadata["mode"] == "video":
        print(
            f"Train videos={split_metadata['n_train_groups']} | "
            f"val videos={split_metadata['n_val_groups']}"
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
    metrics = print_metrics(y_val, val_proba, threshold=recommended_threshold)

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
        "metrics": metrics,
        "split": split_metadata,
        "classifier_type": args.classifier,
    }
    ensure_dir(Path(args.model_out).parent)
    joblib.dump(payload, args.model_out)
    print(f"Saved model to {args.model_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frame filter classifier.")
    parser.add_argument(
        "--data",
        default="data/frames/corrected_sort",
        help="Directory containing footage/, graphic/ and mixture/ image folders.",
    )
    parser.add_argument("--model-out", default="models/frame_filter.joblib")
    parser.add_argument("--crop-bottom", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None, help="cuda, cpu, or auto if omitted.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--dino-model", default=DEFAULT_DINO_MODEL)
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--classifier", choices=["logreg", "mlp"], default="logreg")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--auto-threshold", action="store_true", help="Choose a threshold on validation data favoring graphic+mixture recall.")
    parser.add_argument("--min-precision-for-auto-threshold", type=float, default=0.60)
    parser.add_argument("--graphic-class-weight", "--filter-class-weight", dest="graphic_class_weight", type=float, default=2.0)
    parser.add_argument("--mixture-class-weight", type=float, default=1.5)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--split-mode", choices=["video", "image"], default="video", help="Validation split by video id by default; use image only for quick debugging.")
    parser.add_argument("--fallback-to-image-split", action="store_true", help="Fallback to image split if video split cannot be built.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
