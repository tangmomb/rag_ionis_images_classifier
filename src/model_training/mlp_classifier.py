from __future__ import annotations

from typing import Any

import numpy as np
import torch


class TorchMLPClassifier:
    def __init__(
        self,
        input_dim: int | None = None,
        hidden_dim: int = 256,
        epochs: int = 25,
        batch_size: int = 64,
        lr: float = 1e-3,
        threshold: float = 0.5,
        device: str = "cpu",
        random_state: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.threshold = threshold
        self.device = device
        self.random_state = random_state
        self.model: torch.nn.Module | None = None

    def _build(self, input_dim: int) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.20),
            torch.nn.Linear(self.hidden_dim, 1),
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchMLPClassifier":
        torch.manual_seed(self.random_state)
        self.input_dim = int(x.shape[1])
        self.model = self._build(self.input_dim).to(self.device)

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
        positives = max(float(np.sum(y == 1)), 1.0)
        negatives = max(float(np.sum(y == 0)), 1.0)
        pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=self.device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
            logits = self.model(xb).detach().cpu().numpy().reshape(-1)
            probs_filter = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - probs_filter, probs_filter])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= self.threshold).astype(int)

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
