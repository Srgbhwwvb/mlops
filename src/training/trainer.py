import logging
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from config import Config

from .metrics import calculate_classification_metrics


class PlantTrainer:
    model: torch.nn.Module
    device: torch.device
    config: Config
    class_names: list[str]

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        config: Config,
        class_names: list[str],
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.class_names = class_names

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        self.best_val_f1 = 0
        self.patience_counter = 0
        self.metrics = {
            "train_accuracy": [],
            "train_f1": [],
            "train_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "val_loss": [],
        }

    def _create_optimizer(self) -> Adam:
        """Create optimizer with validated parameters."""
        lr = self.config.training_config.learning_rate
        weight_decay = self.config.training_config.weight_decay
        return Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def _create_scheduler(self) -> ReduceLROnPlateau:
        """Create learning rate scheduler."""
        scheduler_config = self.config.training_config.scheduler_config
        return ReduceLROnPlateau(
            self.optimizer,
            mode=scheduler_config["mode"],
            factor=scheduler_config["factor"],
            patience=scheduler_config["patience"],
        )

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Train for one epoch and return metrics."""
        self.model.train()
        total_loss: float = 0.0
        all_predictions = []
        all_targets = []

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch}",
            unit="batch",
        )

        for batch_idx, (data, target) in pbar:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()

            pred = output.argmax(dim=1)
            total_loss += loss.item()

            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            avg_loss_so_far = total_loss / (batch_idx + 1)
            pbar.set_postfix({"Loss": f"{avg_loss_so_far:.4f}"})

        metrics = calculate_classification_metrics(all_predictions, all_targets)
        metrics["loss"] = total_loss / len(self.train_loader)

        logging.info(
            f"Train Epoch {epoch} - Accuracy: {metrics['accuracy']:.4f}, "
            f"Macro-F1: {metrics['macro_f1']:.4f}, Loss: {metrics['loss']:.4f}",
        )

        return metrics

    def validate(self) -> dict[str, float]:
        """Validate model and return metrics."""
        self.model.eval()
        val_loss: float = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += F.cross_entropy(output, target, reduction="sum").item()
                pred = output.argmax(dim=1)

                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        metrics = calculate_classification_metrics(all_predictions, all_targets)
        metrics["loss"] = val_loss / len(self.val_loader.dataset)

        logging.info(
            f"Validation - Accuracy: {metrics['accuracy']:.4f}, "
            f"Macro-F1: {metrics['macro_f1']:.4f}, Loss: {metrics['loss']:.4f}",
        )

        return metrics

    def train(self) -> dict[str, Any]:
        """Full training loop."""
        logging.info("Starting training...")

        epochs = self.config.training_config.epochs
        patience = self.config.training_config.patience

        logging.info(f"Training for {epochs} epochs")
        logging.info(f"Early stopping patience: {patience}")

        for epoch in range(1, epochs + 1):
            logging.info(f"\n{'=' * 50}")
            logging.info(f"Epoch {epoch}/{epochs}")
            logging.info(f"{'=' * 50}")

            # Training
            train_metrics = self.train_epoch(epoch)
            self.metrics["train_accuracy"].append(train_metrics["accuracy"])
            self.metrics["train_f1"].append(train_metrics["macro_f1"])
            self.metrics["train_loss"].append(train_metrics["loss"])

            # Validation
            val_metrics = self.validate()
            self.metrics["val_accuracy"].append(val_metrics["accuracy"])
            self.metrics["val_f1"].append(val_metrics["macro_f1"])
            self.metrics["val_loss"].append(val_metrics["loss"])

            # Update learning rate
            self.scheduler.step(val_metrics["loss"])

            # Save best model
            if val_metrics["macro_f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["macro_f1"]
                self.patience_counter = 0

                # Save in Hugging Face format
                save_path = self.config.output_config.model_dir / "best_model"

                save_path.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(save_path)
                logging.info(
                    f"Saved best model with Macro-F1: {val_metrics['macro_f1']:.4f}",
                )
            else:
                self.patience_counter += 1
                logging.info(
                    f" No improvement. Patience counter: {self.patience_counter}/{patience}",
                )
                if self.patience_counter >= patience:
                    logging.info(f" Early stopping at epoch {epoch}")
                    break

        # Save final model
        final_path = self.config.output_config.model_dir / "final_model"
        final_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(final_path)
        logging.info("Saved final model")

        # Save metrics
        self._save_metrics()
        logging.info("Training completed!")

        return self.get_training_summary()

    def _save_metrics(self):
        """Save training metrics to CSV."""
        if not self.metrics["train_accuracy"]:
            logging.warning("No metrics to save!")
            return

        metrics_df = pd.DataFrame(
            {
                "epoch": list(range(1, len(self.metrics["train_accuracy"]) + 1)),
                "train_accuracy": self.metrics["train_accuracy"],
                "train_macro_f1": self.metrics["train_f1"],
                "train_loss": self.metrics["train_loss"],
                "val_accuracy": self.metrics["val_accuracy"],
                "val_macro_f1": self.metrics["val_f1"],
                "val_loss": self.metrics["val_loss"],
            },
        )

        self.config.output_config.model_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = self.config.output_config.model_dir / "training_metrics.csv"

        metrics_df.to_csv(metrics_path, index=False)
        logging.info(f"Metrics saved to {metrics_path}")

    def get_training_summary(self) -> dict[str, Any]:
        """Get summary of training results."""
        if not self.metrics["train_accuracy"]:
            return {}

        return {
            "best_val_f1": self.best_val_f1,
            "final_train_accuracy": self.metrics["train_accuracy"][-1],
            "final_val_accuracy": self.metrics["val_accuracy"][-1],
            "final_train_loss": self.metrics["train_loss"][-1],
            "final_val_loss": self.metrics["val_loss"][-1],
            "total_epochs_trained": len(self.metrics["train_accuracy"]),
        }

    def mock_training_step(self, batch_size: int = 2) -> dict[str, float]:
        """Mock training step for testing without real data."""
        # Create mock data
        mock_data = torch.randn(batch_size, 3, 224, 224).to(self.device)
        mock_target = torch.randint(0, len(self.class_names), (batch_size,)).to(
            self.device,
        )

        # Training step
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(mock_data)
        loss = F.cross_entropy(output, mock_target)
        loss.backward()
        self.optimizer.step()

        # Calculate metrics
        pred = output.argmax(dim=1)
        predictions = pred.cpu().numpy()
        targets = mock_target.cpu().numpy().tolist()

        metrics = calculate_classification_metrics(predictions, targets)
        metrics["loss"] = loss.item()

        return metrics
