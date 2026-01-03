from typing import Any

import numpy as np
import torch


class PlantPredictor:
    def __init__(
        self, model: torch.nn.Module, class_names: list[str], device: str = "cpu"
    ) -> None:
        self.model = model
        self.class_names = class_names
        self.device = device
        self.model.to(device)
        self.model.eval()

    def preprocess_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Preprocess batch of images for inference."""
        return batch.to(self.device)

    def predict_proba(self, batch: torch.Tensor) -> np.ndarray:
        """Get class probabilities for batch."""
        with torch.no_grad():
            batch = self.preprocess_batch(batch)
            outputs = self.model(batch)
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()

    def predict_classes(self, batch: torch.Tensor) -> np.ndarray:
        """Get predicted classes for batch."""
        probabilities = self.predict_proba(batch)
        return np.argmax(probabilities, axis=1)

    def format_predictions(
        self,
        class_indices: np.ndarray,
        probabilities: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Format predictions for API response."""
        predictions = []
        for idx, prob in zip(class_indices, probabilities, strict=True):
            predictions.append(
                {
                    "class_index": int(idx),
                    "class_name": self.class_names[idx],
                    "confidence": float(prob[idx]),
                    "probabilities": {
                        self.class_names[i]: float(p) for i, p in enumerate(prob)
                    },
                },
            )
        return predictions

    def predict(self, batch: torch.Tensor) -> list[dict[str, Any]]:
        """Complete prediction pipeline."""
        probabilities = self.predict_proba(batch)
        class_indices = self.predict_classes(batch)
        return self.format_predictions(class_indices, probabilities)
