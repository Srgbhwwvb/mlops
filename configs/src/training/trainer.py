import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score
import logging
from tqdm import tqdm
import pandas as pd
import os
from typing import Dict, List, Tuple

class PlantTrainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        self.optimizer = Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode=config['training']['scheduler_config']['mode'],
            factor=config['training']['scheduler_config']['factor'],
            patience=config['training']['scheduler_config']['patience']
        )
        
        self.best_val_f1 = 0
        self.patience_counter = 0
        
        self.metrics = {
            'train_accuracy': [], 'train_f1': [], 'train_loss': [],
            'val_accuracy': [], 'val_f1': [], 'val_loss': []
        }

    def calculate_metrics(self, all_predictions: List[int], all_targets: List[int]) -> Tuple[float, float]:
        """Calculate accuracy and macro F1-score."""
        accuracy = accuracy_score(all_targets, all_predictions)
        macro_f1 = f1_score(all_targets, all_predictions, average='macro')
        return accuracy, macro_f1

    def train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        """Train for one epoch."""
        self.model.train()
        correct = 0
        total_loss = 0
        
        all_predictions = []
        all_targets = []

        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), 
                    desc=f"Epoch {epoch}", unit="batch")
        
        for batch_idx, (data, target) in pbar:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total_loss += loss.item()
            
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            current_acc = correct / ((batch_idx + 1) * self.train_loader.batch_size)
            pbar.set_postfix({
                "Loss": total_loss / (batch_idx + 1), 
                "Acc": current_acc
            })

        accuracy, macro_f1 = self.calculate_metrics(all_predictions, all_targets)
        avg_loss = total_loss / len(self.train_loader)
        
        logging.info(f'Train Epoch {epoch} - Accuracy: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}, Loss: {avg_loss:.4f}')
        return accuracy, macro_f1, avg_loss

    def validate(self) -> Tuple[float, float, float]:
        """Validate the model."""
        self.model.eval()
        val_loss = 0
        correct = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        val_loss /= len(self.val_loader.dataset)
        accuracy, macro_f1 = self.calculate_metrics(all_predictions, all_targets)
        
        logging.info(f'Validation - Accuracy: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}, Loss: {val_loss:.4f}')
        return accuracy, macro_f1, val_loss

    def train(self):
        """Full training loop."""
        logging.info("Starting training...")
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            logging.info(f"\n{'='*50}")
            logging.info(f"Epoch {epoch}/{self.config['training']['epochs']}")
            logging.info(f"{'='*50}")
            
            # Training
            train_acc, train_f1, train_loss = self.train_epoch(epoch)
            self.metrics['train_accuracy'].append(train_acc)
            self.metrics['train_f1'].append(train_f1)
            self.metrics['train_loss'].append(train_loss)
            
            # Validation
            val_acc, val_f1, val_loss = self.validate()
            self.metrics['val_accuracy'].append(val_acc)
            self.metrics['val_f1'].append(val_f1)
            self.metrics['val_loss'].append(val_loss)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.patience_counter = 0
                
                # Save in Hugging Face format
                save_path = os.path.join(self.config['output']['model_dir'], "best_model")
                self.model.save_pretrained(save_path)
                logging.info(f"Saved best model with Macro-F1: {val_f1:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config['training']['patience']:
                    logging.info(f"Early stopping at epoch {epoch}")
                    break

        # Save final model
        final_path = os.path.join(self.config['output']['model_dir'], "final_model")
        self.model.save_pretrained(final_path)
        
        # Save metrics
        self._save_metrics()
        logging.info("Training completed!")

    def _save_metrics(self):
        """Save training metrics to CSV."""
        metrics_df = pd.DataFrame({
            'epoch': list(range(1, len(self.metrics['train_accuracy']) + 1)),
            'train_accuracy': self.metrics['train_accuracy'],
            'train_macro_f1': self.metrics['train_f1'],
            'train_loss': self.metrics['train_loss'],
            'val_accuracy': self.metrics['val_accuracy'],
            'val_macro_f1': self.metrics['val_f1'],
            'val_loss': self.metrics['val_loss']
        })
        
        os.makedirs(self.config['output']['model_dir'], exist_ok=True)
        metrics_df.to_csv(
            os.path.join(self.config['output']['model_dir'], 'training_metrics.csv'), 
            index=False
        )