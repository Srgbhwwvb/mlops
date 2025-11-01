from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from typing import List, Tuple, Dict


def calculate_classification_metrics(
    predictions: List[int], targets: List[int]
) -> Dict[str, float]:
    """Calculate classification metrics."""
    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(targets, predictions, average="macro")

    return {"accuracy": accuracy, "macro_f1": macro_f1}


def softmax_to_predictions(probabilities: np.ndarray) -> np.ndarray:
    """Convert softmax probabilities to class predictions."""
    return np.argmax(probabilities, axis=1)


def validate_predictions(predictions: np.ndarray, num_classes: int) -> bool:
    """Validate that predictions are within expected range."""
    return (predictions >= 0).all() and (predictions < num_classes).all()
