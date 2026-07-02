from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def collect_training_data(data_dir: str | Path) -> tuple[list[Path], np.ndarray]:
    root = Path(data_dir)
    footage = list_images(root / "footage")
    graphic = list_images(root / "graphic")
    if not footage or not graphic:
        raise FileNotFoundError(
            f"Expected labeled images in {root / 'footage'} and {root / 'graphic'}; found {len(footage)} footage, {len(graphic)} graphic"
        )
    paths = footage + graphic
    labels = np.asarray([0] * len(footage) + [1] * len(graphic), dtype=np.int64)
    return paths, labels


def select_threshold_for_recall(
    y_true: np.ndarray,
    proba_filter: np.ndarray,
    min_precision: float = 0.60,
    fallback: float = 0.5,
) -> float:
    best_threshold = fallback
    best_recall = -1.0
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = (proba_filter >= threshold).astype(int)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        if precision >= min_precision and (recall > best_recall or (recall == best_recall and f1 > best_f1)):
            best_threshold = float(threshold)
            best_recall = float(recall)
            best_f1 = float(f1)
    return best_threshold


def print_metrics(y_true: np.ndarray, proba_filter: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (proba_filter >= threshold).astype(int)
    metrics = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_graphic": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_graphic": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_graphic": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    print(f"Threshold: {threshold:.3f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision graphic: {metrics['precision_graphic']:.4f}")
    print(f"Recall graphic:    {metrics['recall_graphic']:.4f}")
    print(f"F1 graphic:        {metrics['f1_graphic']:.4f}")
    print("Confusion matrix [[footage->footage, footage->graphic], [graphic->footage, graphic->graphic]]:")
    print(np.asarray(metrics["confusion_matrix"]))
    print(classification_report(y_true, y_pred, target_names=["footage", "graphic"], zero_division=0))
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

    stratify = labels if len(np.unique(labels)) == 2 and min(np.bincount(labels)) >= 2 else None
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        labels,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=stratify,
    )

    scaler = StandardScaler()
    if args.classifier == "logreg":
        classifier = LogisticRegression(
            max_iter=args.max_iter,
            class_weight={0: 1.0, 1: args.graphic_class_weight},
            C=args.c,
            random_state=args.seed,
        )
        model = Pipeline([("scaler", scaler), ("classifier", classifier)])
    else:
        classifier = TorchMLPClassifier(
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            threshold=args.threshold,
            device=device,
            random_state=args.seed,
        )
        model = Pipeline([("scaler", scaler), ("classifier", classifier)])

    model.fit(x_train, y_train)
    val_proba = model.predict_proba(x_val)[:, 1]
    recommended_threshold = args.threshold
    if args.auto_threshold:
        recommended_threshold = select_threshold_for_recall(
            y_val,
            val_proba,
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
        "label_mapping": {"footage": 0, "graphic": 1},
        "metrics": metrics,
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
        help="Directory containing footage/ and graphic/ image folders.",
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
    parser.add_argument("--auto-threshold", action="store_true", help="Choose a threshold on validation data favoring graphic recall.")
    parser.add_argument("--min-precision-for-auto-threshold", type=float, default=0.60)
    parser.add_argument("--graphic-class-weight", "--filter-class-weight", dest="graphic_class_weight", type=float, default=2.0)
    parser.add_argument("--val-size", type=float, default=0.20)
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
