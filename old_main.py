import os
import glob
import torch
import torchvision
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

# Исправление SSL проблем
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Оптимизации для CPU
torch.set_num_threads(os.cpu_count() or 4)


class CustomImageDataset(Dataset):
    """Датасет для загрузки изображений и их меток."""
    
    LABELS = [
        'Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 
        'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize', 
        'Scentless Mayweed', 'Shepherds Purse', 
        'Small-flowered Cranesbill', 'Sugar beet'
    ]

    def __init__(self, img_dir, transform=None):
        self.img_paths = glob.glob(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label_name = img_path.split(os.sep)[-2]
        label = self.LABELS.index(label_name)
        return image, label


class TestDataset(Dataset):
    """Датасет для тестовых изображений."""
    
    def __init__(self, img_dir, transform=None):
        self.img_paths = glob.glob(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image_name = os.path.basename(img_path)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image_name, image


class ResNet50(nn.Module):
    """Модель на основе ResNet50."""
    
    def __init__(self, num_classes=12):
        super().__init__()
        # Используем новый API с weights вместо pretrained
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.model(x)


def calculate_metrics(all_predictions, all_targets):
    """Вычисление Accuracy и Macro-averaged F1-Score."""
    accuracy = accuracy_score(all_targets, all_predictions)
    macro_f1 = f1_score(all_targets, all_predictions, average='macro')
    return accuracy, macro_f1


def train(model, device, train_loader, optimizer, epoch):
    """Функция обучения для одной эпохи."""
    model.train()
    correct = 0
    total_loss = 0
    total_samples = len(train_loader.dataset)
    
    all_predictions = []
    all_targets = []

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                desc=f"Epoch {epoch}", unit="batch")
    
    for batch_idx, (data, target) in pbar:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total_loss += loss.item()
        
        all_predictions.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        
        current_acc = correct / ((batch_idx + 1) * train_loader.batch_size)
        pbar.set_postfix({
            "Loss": total_loss / (batch_idx + 1), 
            "Acc": current_acc
        })

    # Вычисляем метрики
    accuracy, macro_f1 = calculate_metrics(all_predictions, all_targets)
    avg_loss = total_loss / len(train_loader)
    
    print(f'\nTrain - Accuracy: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}, Loss: {avg_loss:.4f}')
    return accuracy, macro_f1, avg_loss


def validate(model, device, val_loader):
    """Функция валидации с расчетом метрик."""
    model.eval()
    val_loss = 0
    correct = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    
    # Вычисляем метрики
    accuracy, macro_f1 = calculate_metrics(all_predictions, all_targets)
    
    print(f'Validation - Accuracy: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}, Loss: {val_loss:.4f}')
    return accuracy, macro_f1, val_loss


def plot_metrics(metrics_dict, filename_prefix):
    """Построение графиков метрик."""
    epochs = list(range(1, len(metrics_dict['train_accuracy']) + 1))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, metrics_dict['train_accuracy'], label='Train Accuracy')
    plt.plot(epochs, metrics_dict['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.title('Macro-averaged F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.plot(epochs, metrics_dict['train_f1'], label='Train F1')
    plt.plot(epochs, metrics_dict['val_f1'], label='Val F1')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_metrics.png')
    plt.close()
    
    plt.figure(figsize=(8, 4))
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, metrics_dict['train_loss'], label='Train Loss')
    plt.plot(epochs, metrics_dict['val_loss'], label='Val Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{filename_prefix}_loss.png')
    plt.close()


def main():
    # Конфигурация путей
    BASE_DIR = r"C:\Users\Alex\Downloads\plant-seedlings-classification"
    TRAIN_PATH = os.path.join(BASE_DIR, "train", "*", "*.png")
    TEST_PATH = os.path.join(BASE_DIR, "test", "*.png")
    
    # Проверка существования путей
    if not os.path.exists(BASE_DIR):
        print(f"Ошибка: Директория {BASE_DIR} не существует!")
        return
    
    # Определение устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CPU cores: {os.cpu_count()}")

    # Трансформации для данных
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.3),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Загрузка данных
    try:
        full_dataset = CustomImageDataset(TRAIN_PATH, transform=train_transform)
        print(f"Loaded {len(full_dataset)} training images")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Замена трансформаций для валидации
    val_dataset.dataset.transform = val_transform

    # Оптимизированные DataLoader
    num_workers = min(4, os.cpu_count() or 4)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16,  # Уменьшенный batch size для ResNet50 на CPU
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=16, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=False
    )

    # Инициализация модели ResNet50
    print("Using ResNet50...")
    model = ResNet50(num_classes=12).to(device)
    
    # Оптимизатор и планировщик
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Словарь для хранения метрик
    metrics = {
        'train_accuracy': [],
        'train_f1': [],
        'train_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_loss': []
    }

    # Ранняя остановка
    best_val_f1 = 0
    patience = 5
    patience_counter = 0

    # Обучение
    for epoch in range(1, 5):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/4")
        print(f"{'='*50}")
        
        # Обучение
        train_acc, train_f1, train_loss = train(model, device, train_loader, optimizer, epoch)
        metrics['train_accuracy'].append(train_acc)
        metrics['train_f1'].append(train_f1)
        metrics['train_loss'].append(train_loss)
        
        # Валидация
        val_acc, val_f1, val_loss = validate(model, device, val_loader)
        metrics['val_accuracy'].append(val_acc)
        metrics['val_f1'].append(val_f1)
        metrics['val_loss'].append(val_loss)
        
        # Обновление learning rate
        scheduler.step(val_loss)
        
        # Ранняя остановка по Macro-F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_resnet50_model.pth')
            print(f"Saved best model with Macro-F1: {val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Сохранение графиков и модели
    plot_metrics(metrics, 'resnet50')
    
    # Сохранение метрик в CSV
    metrics_df = pd.DataFrame({
        'epoch': list(range(1, len(metrics['train_accuracy']) + 1)),
        'train_accuracy': metrics['train_accuracy'],
        'train_macro_f1': metrics['train_f1'],
        'train_loss': metrics['train_loss'],
        'val_accuracy': metrics['val_accuracy'],
        'val_macro_f1': metrics['val_f1'],
        'val_loss': metrics['val_loss']
    })
    metrics_df.to_csv('resnet50_training_metrics.csv', index=False)
    print("Training completed and metrics saved!")


def predict():
    """Функция для предсказания на тестовых данных."""
    BASE_DIR = r"C:\Users\Alex\Downloads\plant-seedlings-classification"
    TEST_PATH = os.path.join(BASE_DIR, "test", "*.png")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = TestDataset(TEST_PATH, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=8,
        shuffle=False, 
        num_workers=min(2, os.cpu_count() or 2),
        pin_memory=False
    )

    model = ResNet50(num_classes=12)
    model.load_state_dict(torch.load('best_resnet50_model.pth', map_location=device))
    model.to(device)
    model.eval()

    image_names = []
    predictions = []
    
    with torch.no_grad():
        for image_name, data in tqdm(test_loader, desc="Processing test images"):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            for i in range(len(image_name)):
                predictions.append(CustomImageDataset.LABELS[pred[i].item()])
                image_names.append(image_name[i])

    # Сохранение результатов
    results = pd.DataFrame({'file': image_names, 'species': predictions})
    results.to_csv('submission.csv', index=False)
    print("Submission file saved successfully!")


if __name__ == "__main__":
    main()
    predict()