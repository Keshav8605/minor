import unittest
import torch
from unittest.mock import MagicMock, patch
from src.baseline.classifier import MultimodalClassifier
from src.baseline.trainer import train_baseline
import shutil
from pathlib import Path

class MockConfig:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

class MockEncoder:
    def __init__(self, hidden_size):
        self.config = MockConfig(hidden_size)

class TestClassifier(unittest.TestCase):
    @patch('src.baseline.classifier.ImageEncoder')
    @patch('src.baseline.classifier.TextEncoder')
    def test_dynamic_classifier_shapes(self, MockText, MockImage):
        # Setup mocks with synthetic dimensions
        img_mock_instance = MagicMock()
        img_mock_instance.vit.config = MockConfig(512)
        MockImage.return_value = img_mock_instance
        
        txt_mock_instance = MagicMock()
        txt_mock_instance.roberta.config = MockConfig(256)
        MockText.return_value = txt_mock_instance
        
        clf = MultimodalClassifier(
            image_model_id="mock_img", 
            text_model_id="mock_txt", 
            hidden_size=128, 
            dropout=0.1, 
            mode="multimodal"
        )
        
        # In multimodal mode, input to MLP should be img_dim + txt_dim = 512 + 256 = 768
        self.assertEqual(clf.mlp[0].in_features, 768)
        
        # Test output shape
        dummy_fused = torch.rand(2, 768)
        out = clf.mlp(dummy_fused)
        self.assertEqual(out.shape, (2, 1))

    @patch('src.baseline.trainer.torch.save')
    @patch('src.baseline.trainer.nn.BCEWithLogitsLoss')
    def test_trainer_best_model_saving(self, mock_bce, mock_save):
        mock_criterion = MagicMock()
        mock_bce.return_value = mock_criterion
        
        mock_loss = MagicMock()
        mock_criterion.return_value = mock_loss
        
        # 1 batch for train, 1 for val. Over 3 epochs = 6 calls.
        mock_loss.item.side_effect = [
            0.9, 0.8, # Epoch 1
            0.7, 0.5, # Epoch 2 (best)
            0.6, 0.6  # Epoch 3
        ]
        
        # Create a real dummy model that doesn't fail when to(device) or train() is called
        import torch.nn as nn
        mock_model = nn.Linear(1, 1)
        # Mock its forward call so we don't need real inputs
        mock_model.forward = MagicMock()
        mock_model.forward.return_value = MagicMock()
        
        # Dummy dataloaders with 1 batch each
        dummy_batch = {
            "pixel_values": MagicMock(),
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
            "labels": MagicMock()
        }
        mock_train_loader = [dummy_batch]
        mock_val_loader = [dummy_batch]
        
        test_save_dir = Path("test_checkpoints")
        if test_save_dir.exists():
            shutil.rmtree(test_save_dir)
        test_save_dir.mkdir()
        
        train_baseline(
            model=mock_model,
            train_loader=mock_train_loader,
            val_loader=mock_val_loader,
            epochs=3,
            lr=1e-4,
            device="cpu",
            save_dir=str(test_save_dir)
        )
        
        # Check how many times best_model.pth was saved
        # It should be saved at Epoch 1 (first epoch) and Epoch 2 (new best)
        save_calls = mock_save.call_args_list
        best_model_calls = [call for call in save_calls if str(call[0][1]).endswith('best_model.pth')]
        
        self.assertEqual(len(best_model_calls), 2, "best_model.pth should be saved exactly twice for these losses")
        
        # Clean up
        shutil.rmtree(test_save_dir)
