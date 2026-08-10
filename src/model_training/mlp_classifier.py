from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        input_dim: int | None = None,
        hidden_dim: int = 256,
        num_classes: int = 2,
        epochs: int = 25,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str = "cpu",
        random_state: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.random_state = random_state
        self.model: torch.nn.Module | None = None
        self.classes_: np.ndarray | None = None

    def _build(self, input_dim: int) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.20),
            torch.nn.Linear(self.hidden_dim, self.num_classes),
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchMLPClassifier":
        torch.manual_seed(self.random_state)
        self.input_dim = int(x.shape[1])
        self.classes_ = np.asarray(sorted(set(int(value) for value in y.tolist())), dtype=np.int64)
        self.num_classes = int(max(int(self.classes_.max()) + 1, len(self.classes_)))
        self.model = self._build(self.input_dim).to(self.device)

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        counts = np.bincount(y.astype(np.int64), minlength=self.num_classes).astype(np.float32)
        counts = np.maximum(counts, 1.0)
        weights = counts.sum() / (counts * self.num_classes)
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=self.device))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
        self.model.eval()
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("MLP classifier is not fitted")
        with torch.inference_mode():
            xb = torch.tensor(x, dtype=torch.float32, device=self.device)
            logits = self.model(xb).detach().cpu()
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1).astype(int)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        if self.model is not None:
            state["model_state_dict"] = {key: value.cpu() for key, value in self.model.state_dict().items()}
        state["model"] = None
        state["device"] = "cpu"
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        model_state = state.pop("model_state_dict", None)
        self.__dict__.update(state)
        if self.input_dim is not None:
            self.model = self._build(self.input_dim)
            if model_state is not None:
                self.model.load_state_dict(model_state)
            self.model.eval()
