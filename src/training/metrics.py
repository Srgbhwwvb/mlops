import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def calculate_classification_metrics(
    predictions: list[int],
    targets: list[int],
) -> dict[str, float]:
    """Calculate classification metrics."""
    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(targets, predictions, average="macro")

    return {"accuracy": accuracy, "macro_f1": macro_f1}


def softmax_to_predictions(probabilities: np.ndarray) -> np.ndarray:
    """Convert softmax probabilities to class predictions."""
    return np.argmax(probabilities, axis=1)


def validate_predictions(predictions: np.ndarray, num_classes: int) -> bool:
    """Validate that predictions are within expected range."""
    return bool((predictions >= 0).all()) and bool((predictions < num_classes).all())
