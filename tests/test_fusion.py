import unittest
import torch
from src.baseline.fusion import FeatureFusion

class TestFusion(unittest.TestCase):
    def test_multimodal_fusion(self):
        img_feat = torch.rand(2, 768)
        txt_feat = torch.rand(2, 768)
        
        fusion = FeatureFusion(mode="multimodal")
        out = fusion(img_feat, txt_feat)
        
        self.assertEqual(out.shape, (2, 1536))
        
    def test_image_only(self):
        img_feat = torch.rand(2, 768)
        txt_feat = torch.rand(2, 768)
        
        fusion = FeatureFusion(mode="image_only")
        out = fusion(img_feat, txt_feat)
        
        self.assertEqual(out.shape, (2, 768))
