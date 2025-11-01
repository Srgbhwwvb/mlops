import pytest
import numpy as np

from src.training.metrics import (
    calculate_classification_metrics,
    softmax_to_predictions,
    validate_predictions,
)


class TestMetrics:
    def test_calculate_classification_metrics(self):
        """Test classification metrics calculation."""
        predictions = [0, 1, 0, 1]
        targets = [0, 1, 1, 1]

        metrics = calculate_classification_metrics(predictions, targets)

        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert isinstance(metrics["accuracy"], float)
        assert isinstance(metrics["macro_f1"], float)

        # Test perfect predictions
        perfect_pred = [0, 1, 2]
        perfect_targ = [0, 1, 2]
        metrics = calculate_classification_metrics(perfect_pred, perfect_targ)
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0

    def test_softmax_to_predictions(self):
        """Test softmax to predictions conversion."""
        probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]])

        predictions = softmax_to_predictions(probabilities)
        expected = np.array([0, 1, 2])

        np.testing.assert_array_equal(predictions, expected)

    def test_validate_predictions(self):
        """Test predictions validation."""
        valid_predictions = np.array([0, 1, 2])
        assert validate_predictions(valid_predictions, num_classes=3) == True

        invalid_predictions = np.array([0, 1, 5])
        assert validate_predictions(invalid_predictions, num_classes=3) == False

        negative_predictions = np.array([-1, 0, 1])
        assert validate_predictions(negative_predictions, num_classes=3) == False
