import os

def check_all_files():
    files_to_check = {
        'src/utils/config.py': ['load_config', 'save_config'],
        'src/data/dataset.py': ['PlantDataset', 'create_data_loaders'],
        'src/models/resnet.py': ['ResNet50', 'ResNetConfig'],
        'src/training/trainer.py': ['PlantTrainer'],
        'configs/train_config.yaml': None
    }
    
    for file_path, required_functions in files_to_check.items():
        if os.path.exists(file_path):
            print(f"✅ {file_path} существует")
            if required_functions:
                try:
                    # Динамически импортируем модуль для проверки функций
                    module_name = file_path.replace('/', '.').replace('.py', '')
                    if module_name.startswith('src.'):
                        module_name = module_name[4:]  # Убираем 'src.'
                    
                    # Для utils.config
                    if module_name == 'utils.config':
                        from src.utils.config import load_config, save_config
                        print(f"   ✅ Функции найдены: load_config, save_config")
                    
                    # Для data.dataset  
                    elif module_name == 'data.dataset':
                        from src.data.dataset import PlantDataset, create_data_loaders
                        print(f"   ✅ Классы/функции найдены: PlantDataset, create_data_loaders")
                    
                    # Для models.resnet
                    elif module_name == 'models.resnet':
                        from src.models.resnet import ResNet50, ResNetConfig
                        print(f"   ✅ Классы найдены: ResNet50, ResNetConfig")
                    
                    # Для training.trainer
                    elif module_name == 'training.trainer':
                        from src.training.trainer import PlantTrainer
                        print(f"   ✅ Класс найден: PlantTrainer")
                        
                except ImportError as e:
                    print(f"   ❌ Ошибка импорта: {e}")
        else:
            print(f"❌ {file_path} НЕ СУЩЕСТВУЕТ")

if __name__ == "__main__":
    check_all_files()
