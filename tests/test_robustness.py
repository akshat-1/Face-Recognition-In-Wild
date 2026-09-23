import os
import sys
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from losses.curricular_loss import CurricularFaceLoss
from losses.broadface_queue import BroadFaceMemoryQueue
from models.ddrc_solver import LISTASparseSolver, DDRCClassifier
from models.anet_attribute import ANetAttributeParser
from models.pim_frontalizer import PIMFrontalizationGAN, D2SCGANSuperRes
from models.backbone import ResNet100Backbone, vit_face_base
from models.detector import WildFaceDetector, soft_nms_pytorch
from pipeline.wild_face_pipeline import OccuPoseBroadDictPipeline

class TestOccuPoseBroadDictNet(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_vit_face_backbone(self):
        vit = vit_face_base(embedding_dim=512).to(self.device)
        x = torch.randn(2, 3, 112, 112, device=self.device)
        
        embeddings = vit(x)
        self.assertEqual(embeddings.shape, (2, 512))
        norm = torch.norm(embeddings, p=2, dim=1)
        self.assertTrue(torch.allclose(norm, torch.ones_like(norm), atol=1e-4))
        print("✓ Vision Transformer (ViT-Face) Backbone Test Passed: Feature Norms = 1.000")

    def test_curricular_face_loss(self):
        loss_fn = CurricularFaceLoss(in_features=512, num_classes=50, s=64.0, m=0.5).to(self.device)
        embeddings = torch.randn(8, 512, device=self.device)
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=self.device)
        
        loss = loss_fn(embeddings, labels)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss_fn.t.item(), -1.0)
        print(f"✓ CurricularFace Loss Test Passed: Loss = {loss.item():.4f}, t = {loss_fn.t.item():.4f}")

    def test_broadface_memory_queue(self):
        queue = BroadFaceMemoryQueue(queue_size=100, feature_dim=512).to(self.device)
        
        embeds1 = F.normalize(torch.randn(40, 512, device=self.device), p=2, dim=1)
        labels1 = torch.randint(0, 50, (40,), device=self.device)
        queue.update(embeds1, labels1)
        self.assertEqual(int(queue.queue_ptr.item()), 40)
        
        embeds2 = F.normalize(torch.randn(80, 512, device=self.device), p=2, dim=1)
        labels2 = torch.randint(0, 50, (80,), device=self.device)
        queue.update(embeds2, labels2)
        
        self.assertTrue(queue.is_full.item())
        q_embeds, q_labels = queue.get_queue_samples()
        self.assertEqual(q_embeds.shape, (100, 512))
        print("✓ BroadFace Memory Queue Test Passed")

    def test_ddrc_lista_classifier(self):
        ddrc = DDRCClassifier(feature_dim=512, num_classes=10, atoms_per_class=5).to(self.device)
        
        # Test enrolled identity feature
        f_known = torch.randn(2, 512, device=self.device)
        preds, confs, residuals = ddrc(f_known)
        
        self.assertEqual(len(preds), 2)
        self.assertEqual(residuals.shape, (2, 10))
        print(f"✓ DDRC Classifier Test Passed: Predictions = {preds}, Confidences = {confs.tolist()}")

    def test_anet_attribute_parser(self):
        anet = ANetAttributeParser().to(self.device)
        x = torch.randn(4, 3, 112, 112, device=self.device)
        
        logits, occ_mask, is_occluded = anet(x)
        self.assertEqual(logits.shape, (4, 40))
        self.assertEqual(occ_mask.shape, (4, 1, 7, 7))
        self.assertEqual(is_occluded.shape, (4,))
        print("✓ ANet Attribute & Occlusion Parser Test Passed")

    def test_pim_frontalization(self):
        pim = PIMFrontalizationGAN().to(self.device)
        x_profile = torch.randn(2, 3, 112, 112, device=self.device)
        
        x_frontal = pim(x_profile, yaw_angle=45.0)
        self.assertEqual(x_frontal.shape, (2, 3, 112, 112))
        print("✓ PIM Frontalization & D2SC-GAN Super-Res Test Passed")

    def test_resnet_backbone(self):
        backbone = ResNet100Backbone(embedding_dim=512).to(self.device)
        x = torch.randn(2, 3, 112, 112, device=self.device)
        
        embeddings = backbone(x)
        self.assertEqual(embeddings.shape, (2, 512))
        norm = torch.norm(embeddings, p=2, dim=1)
        self.assertTrue(torch.allclose(norm, torch.ones_like(norm), atol=1e-4))
        print("✓ ResNet-100 Backbone Test Passed: Feature Norms = 1.000")

    def test_wild_face_pipeline(self):
        pipeline = OccuPoseBroadDictPipeline(num_enrolled_classes=10, feature_dim=512).to(self.device)
        
        # Zero score head weight and set bias positive so synthetic image guarantees >0.95 detection scores
        nn.init.zeros_(pipeline.detector.score_head.weight)
        nn.init.constant_(pipeline.detector.score_head.bias, 5.0)
        nn.init.zeros_(pipeline.detector.bbox_head.weight)
        nn.init.constant_(pipeline.detector.bbox_head.bias, 0.5)
        
        img = torch.randn(1, 3, 224, 224, device=self.device)
        
        results = pipeline(img, score_threshold=0.1)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # Verify structure of output dict
        first_face = results[0]
        self.assertIn('box', first_face)
        self.assertIn('identity', first_face)
        self.assertIn('confidence', first_face)
        self.assertIn('is_occluded', first_face)
        self.assertIn('attributes', first_face)
        
        print(f"✓ End-to-End OccuPose-BroadDictNet Pipeline Test Passed: {len(results)} faces detected & processed. First face identity: '{first_face['identity']}', confidence: {first_face['confidence']:.4f}, occluded: {first_face['is_occluded']}")

if __name__ == "__main__":
    unittest.main()
