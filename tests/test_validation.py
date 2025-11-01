import pytest
import yaml

from src.utils.validation import validate_config_structure

class TestConfigValidation:
    def test_valid_config(self):
        """Test validation of correct config."""
        valid_config = {
            'data': {
                'train_path': '/path/to/train',
                'val_size': 0.2
            },
            'model': {
                'name': 'resnet50',
                'num_classes': 12
            },
            'training': {
                'batch_size': 16,
                'epochs': 10,
                'learning_rate': 0.001
            },
            'transforms': {
                'image_size': 224
            },
            'output': {
                'model_dir': './models'
            }
        }
        
        errors = validate_config_structure(valid_config)
        assert len(errors) == 0

    def test_invalid_config(self):
        """Test validation of incorrect config."""
        invalid_config = {
            'data': {
                'val_size': 1.5  # Invalid value
            }
            # Missing required sections
        }
        
        errors = validate_config_structure(invalid_config)
        assert len(errors) > 0
        assert any('Missing required section' in error for error in errors)