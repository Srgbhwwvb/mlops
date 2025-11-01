import pytest
import tempfile
import os
from src.utils.config import load_config, save_config


def test_load_config():
    """Test loading config from YAML file."""
    # Создаем временный конфиг файл
    config_content = """
data:
  train_path: "test/path/*.png"
  val_size: 0.2
model:
  num_classes: 5
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        config = load_config(config_path)
        assert config["data"]["train_path"] == "test/path/*.png"
        assert config["data"]["val_size"] == 0.2
        assert config["model"]["num_classes"] == 5
    finally:
        os.unlink(config_path)


def test_save_config():
    """Test saving config to YAML file."""
    config = {"data": {"train_path": "test/path"}, "model": {"num_classes": 10}}

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "test_config.yaml")
        save_config(config, config_path)

        # Проверяем что файл создан
        assert os.path.exists(config_path)

        # Проверяем содержимое
        with open(config_path, "r") as f:
            content = f.read()
            assert "train_path" in content
            assert "num_classes" in content
