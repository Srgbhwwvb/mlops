import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from tqdm import tqdm
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional, Any

from .metrics import calculate_classification_metrics


class PlantTrainer:
    def __init__(self, model, train_loader, val_loader, device, config, class_names):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = self._validate_and_convert_config(config)
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

    def _validate_and_convert_config(self, config: Dict) -> Dict:
        """Validate and convert config values to proper types."""
        converted_config = config.copy()

        # Convert training parameters
        training_config = converted_config["training"]
        training_config["learning_rate"] = float(training_config["learning_rate"])
        training_config["weight_decay"] = float(training_config["weight_decay"])
        training_config["batch_size"] = int(training_config["batch_size"])
        training_config["epochs"] = int(training_config["epochs"])
        training_config["patience"] = int(training_config["patience"])

        # Convert scheduler parameters
        scheduler_config = training_config["scheduler_config"]
        scheduler_config["factor"] = float(scheduler_config["factor"])
        scheduler_config["patience"] = int(scheduler_config["patience"])

        # Convert data parameters
        data_config = converted_config["data"]
        data_config["val_size"] = float(data_config["val_size"])
        data_config["random_seed"] = int(data_config["random_seed"])

        # Convert transform parameters
        transforms_config = converted_config["transforms"]
        transforms_config["image_size"] = int(transforms_config["image_size"])

        train_transforms = transforms_config["train"]
        train_transforms["RandomHorizontalFlip"] = float(
            train_transforms["RandomHorizontalFlip"]
        )
        train_transforms["RandomVerticalFlip"] = float(
            train_transforms["RandomVerticalFlip"]
        )
        train_transforms["RandomRotation"] = int(train_transforms["RandomRotation"])

        # Convert output parameters
        output_config = converted_config["output"]
        output_config["save_frequency"] = int(output_config.get("save_frequency", 1))

        logging.info("Config validation and conversion completed successfully")
        return converted_config

    def _create_optimizer(self) -> Adam:
        """Create optimizer with validated parameters."""
        lr = self.config["training"]["learning_rate"]
        weight_decay = self.config["training"]["weight_decay"]
        return Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def _create_scheduler(self) -> ReduceLROnPlateau:
        """Create learning rate scheduler."""
        scheduler_config = self.config["training"]["scheduler_config"]
        return ReduceLROnPlateau(
            self.optimizer,
            mode=scheduler_config["mode"],
            factor=scheduler_config["factor"],
            patience=scheduler_config["patience"],
            # verbose=True,
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
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
            f'Train Epoch {epoch} - Accuracy: {metrics["accuracy"]:.4f}, '
            f'Macro-F1: {metrics["macro_f1"]:.4f}, Loss: {metrics["loss"]:.4f}'
        )

        return metrics

    def validate(self) -> Dict[str, float]:
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
            f'Validation - Accuracy: {metrics["accuracy"]:.4f}, '
            f'Macro-F1: {metrics["macro_f1"]:.4f}, Loss: {metrics["loss"]:.4f}'
        )

        return metrics

    def train(self) -> Dict[str, Any]:
        """Full training loop."""
        logging.info("Starting training...")

        epochs = self.config["training"]["epochs"]
        patience = self.config["training"]["patience"]

        logging.info(f"Training for {epochs} epochs")
        logging.info(f"Early stopping patience: {patience}")

        for epoch in range(1, epochs + 1):
            logging.info(f"\n{'='*50}")
            logging.info(f"Epoch {epoch}/{epochs}")
            logging.info(f"{'='*50}")

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
                save_path = os.path.join(
                    self.config["output"]["model_dir"], "best_model"
                )
                os.makedirs(save_path, exist_ok=True)
                self.model.save_pretrained(save_path)
                logging.info(
                    f"✅ Saved best model with Macro-F1: {val_metrics['macro_f1']:.4f}"
                )
            else:
                self.patience_counter += 1
                logging.info(
                    f"⏳ No improvement. Patience counter: {self.patience_counter}/{patience}"
                )
                if self.patience_counter >= patience:
                    logging.info(f"🛑 Early stopping at epoch {epoch}")
                    break

        # Save final model
        final_path = os.path.join(self.config["output"]["model_dir"], "final_model")
        os.makedirs(final_path, exist_ok=True)
        self.model.save_pretrained(final_path)
        logging.info("✅ Saved final model")

        # Save metrics
        self._save_metrics()
        logging.info("🎉 Training completed!")

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
            }
        )

        os.makedirs(self.config["output"]["model_dir"], exist_ok=True)
        metrics_path = os.path.join(
            self.config["output"]["model_dir"], "training_metrics.csv"
        )
        metrics_df.to_csv(metrics_path, index=False)
        logging.info(f"📊 Metrics saved to {metrics_path}")

    def get_training_summary(self) -> Dict[str, Any]:
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

    def mock_training_step(self, batch_size: int = 2) -> Dict[str, float]:
        """Mock training step for testing without real data."""
        # Create mock data
        mock_data = torch.randn(batch_size, 3, 224, 224).to(self.device)
        mock_target = torch.randint(0, len(self.class_names), (batch_size,)).to(
            self.device
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
