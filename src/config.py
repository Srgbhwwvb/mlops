from pathlib import Path
from typing import Any, Optional

import yaml


class APIConfig:
    port: int

    def __init__(self, config: dict[str, Any]) -> None:
        """Init from a dictionary."""
        self.port = int(config["port"])

    def into_dict(self) -> dict[str, Any]:
        return {
            "port": self.port,
        }


class DataConfig:
    train_path: Path
    test_path: Path
    val_size: float
    random_seed: int

    def __init__(self, config: dict[str, Any]) -> None:
        """Init from a dictionary."""
        self.train_path = Path(config["train_path"])
        self.test_path = Path(config["test_path"])
        self.val_size = float(config["val_size"])
        self.random_seed = int(config["random_seed"])

    def into_dict(self) -> dict[str, Any]:
        return {
            "train_path": self.train_path,
            "test_path": self.test_path,
            "val_size": self.val_size,
            "random_seed": self.random_seed,
        }


class ModelConfig:
    name: str
    num_classes: int
    pretrained: bool

    def __init__(self, config: dict[str, Any]) -> None:
        """Init from a dictionary."""
        self.name = str(config["name"])
        self.num_classes = int(config["num_classes"])
        self.pretrained = bool(config["pretrained"])

    def into_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_classes": self.num_classes,
            "pretrained": self.pretrained,
        }


class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int
    optimizer: str
    scheduler: str
    scheduler_config: Any
    seed: Optional[int]

    def __init__(self, config: dict[str, Any]) -> None:
        """Init from a dictionary."""
        self.batch_size = int(config["batch_size"])
        self.epochs = int(config["epochs"])
        self.learning_rate = float(config["learning_rate"])
        self.weight_decay = float(config["weight_decay"])
        self.patience = int(config["patience"])
        self.optimizer = str(config["optimizer"])
        self.scheduler = str(config["scheduler"])
        self.scheduler_config = config["scheduler_config"]
        self.seed = config.get("seed")

    def into_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "scheduler_config": self.scheduler_config,
            "seed": self.seed,
        }


class TransformConfig:
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    random_horizontal_flip: float
    random_rotation: float
    val: Any

    def __init__(self, config: dict[str, Any]) -> None:
        """Init from a dictionary."""
        self.image_size = int(config["image_size"])

        mean_list = config["mean"]
        self.mean: tuple[float, float, float] = (
            float(mean_list[0]),
            float(mean_list[1]),
            float(mean_list[2]),
        )
        del mean_list

        std_list = config["std"]
        self.std = (float(std_list[0]), float(std_list[1]), float(std_list[2]))
        del std_list

        self.random_horizontal_flip = float(config["train"]["RandomHorizontalFlip"])
        self.random_rotation = int(config["train"]["RandomRotation"])

        self.val = config.get("val")

    def into_dict(self) -> dict[str, Any]:
        return {
            "image_size": self.image_size,
            "mean": self.mean,
            "std": self.std,
            "train": {
                "RandomHorizontalFlip": self.random_horizontal_flip,
                "RandomRotation": self.random_rotation,
            },
            "val": self.val,
        }


class OutputConfig:
    log_dir: Path
    model_dir: Path
    save_frequency: int

    def __init__(self, config: dict[str, Any]) -> None:
        """Init from a dictionary."""
        self.log_dir = Path(config["log_dir"])
        self.model_dir = Path(config["model_dir"])
        self.save_frequency = int(config["save_frequency"])

    def into_dict(self) -> dict[str, Any]:
        return {
            "log_dir": self.log_dir,
            "model_dir": self.model_dir,
            "save_frequency": self.save_frequency,
        }


class Config:
    api_config: APIConfig
    data_config: DataConfig
    model_config: ModelConfig
    training_config: TrainingConfig
    transform_config: TransformConfig
    output_config: OutputConfig

    def __init__(self, config: dict[str, Any]) -> None:
        """Init from a dictionary."""
        self.api_config = APIConfig(config["api"])
        self.data_config = DataConfig(config["data"])
        self.model_config = ModelConfig(config["model"])
        self.training_config = TrainingConfig(config["training"])
        self.transform_config = TransformConfig(config["transforms"])
        self.output_config = OutputConfig(config["output"])

    @classmethod
    def from_file(cls, config_path: Path) -> "Config":
        """Save configuration to YAML file."""
        if not config_path.exists():
            raise ValueError(f"Config file not found: {config_path}")

        with config_path.open(encoding="utf-8") as f:
            try:
                config = yaml.safe_load(f)
                return Config(config)
            except Exception as e:
                raise ValueError(f"Error during loading config `{config_path}`") from e

    def into_dict(self) -> dict[str, Any]:
        return {
            "api": self.api_config.into_dict(),
            "data": self.data_config.into_dict(),
            "model": self.model_config.into_dict(),
            "training": self.training_config.into_dict(),
            "transforms": self.transform_config.into_dict(),
            "output": self.output_config.into_dict(),
        }


def save_config(config: dict[str, Any], save_path: Path) -> None:
    """Save configuration to YAML file."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def create_test_config_dict() -> dict[str, Any]:
    """Create a minimal test configuration for unit testing."""
    return {
        "api": {"port": "25565"},
        "data": {
            "train_path": "./data/initial_data/train/*/*.png",
            "test_path": "./data/initial_data/test/*.png",
            "val_size": 0.2,
            "random_seed": 42,
        },
        "model": {"name": "resnet50", "num_classes": 12, "pretrained": True},
        "training": {
            "batch_size": 2,  # Small for testing
            "epochs": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.001,
            "patience": 5,
            "optimizer": "adam",
            "scheduler": "reduce_on_plateau",
            "scheduler_config": {"mode": "min", "factor": 0.5, "patience": 3},
        },
        "transforms": {
            "image_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "train": {
                "RandomHorizontalFlip": 0.5,
                "RandomVerticalFlip": 0.3,
                "RandomRotation": 30,
            },
            "val": {},
        },
        "output": {
            "log_dir": "./test_logs",
            "model_dir": "./test_models",
            "save_frequency": 1,
        },
    }


def create_test_config() -> Config:
    """Create a minimal test configuration for unit testing."""
    return Config(create_test_config_dict())
