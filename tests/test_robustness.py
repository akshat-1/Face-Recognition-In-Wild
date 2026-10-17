import os
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import DatasetConfigurationError, NameLabeledFaceDataset, UnlabeledWildDataset
from losses.broadface_queue import BroadFaceMemoryQueue
from losses.curricular_loss import CurricularFaceLoss
from models.ddrc_solver import DDRCClassifier
from models.detector import soft_nms_pytorch
from models.gcn_cluster import GCNLinkPredictor
from pipeline.alignment import align_face_crop, estimate_pose_from_landmarks


class TestCorrectedContracts(unittest.TestCase):
    def test_dataset_requires_real_data_and_indexes_valid_samples(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                NameLabeledFaceDataset(root_dir=os.path.join(root, "missing"))
            image_dir = os.path.join(root, "alice_wearing_mask")
            os.makedirs(image_dir)
            Image.new("RGB", (32, 32), (120, 80, 40)).save(os.path.join(image_dir, "face.png"))
            dataset = NameLabeledFaceDataset(root_dir=root, is_train=False)
            self.assertEqual(len(dataset), 1)
            image, label, is_labeled = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 112, 112))
            self.assertEqual(label, 0)
            self.assertTrue(is_labeled)
            with self.assertRaises(IndexError):
                dataset[1]

    def test_unlabeled_dataset_does_not_fabricate_samples(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises((FileNotFoundError, DatasetConfigurationError)):
                UnlabeledWildDataset(root_dir=root)

    def test_alignment_and_landmark_pose(self):
        crop = torch.zeros(1, 3, 112, 112)
        landmarks = torch.tensor([[[38.3, 51.7], [73.5, 51.5], [56.0, 71.7], [41.5, 92.4], [70.7, 92.2]]])
        aligned, used = align_face_crop(crop, landmarks)
        self.assertEqual(tuple(aligned.shape), (1, 3, 112, 112))
        self.assertTrue(used)
        pose = estimate_pose_from_landmarks(landmarks[0])
        self.assertEqual(pose["valid"], 1.0)
        self.assertTrue(-90.0 <= pose["yaw"] <= 90.0)

    def test_soft_nms_preserves_landmark_source_indices(self):
        boxes = torch.tensor([[10.0, 10.0, 30.0, 30.0], [60.0, 10.0, 80.0, 30.0]])
        scores = torch.tensor([0.4, 0.9])
        _, _, indices = soft_nms_pytorch(boxes, scores, score_threshold=0.1, return_indices=True)
        self.assertEqual(indices.tolist(), [1, 0])

    def test_curricular_loss_and_queue(self):
        loss_fn = CurricularFaceLoss(in_features=8, num_classes=4)
        embeddings = F.normalize(torch.randn(3, 8), dim=1)
        labels = torch.tensor([0, 1, 2])
        loss = loss_fn(embeddings, labels)
        self.assertTrue(torch.isfinite(loss))
        queue = BroadFaceMemoryQueue(in_features=8, num_classes=4, queue_size=4)
        queue.enqueue(embeddings, labels)
        self.assertEqual(int(queue.queue_ptr.item()), 3)
        with self.assertRaises(ValueError):
            queue.enqueue(embeddings, torch.tensor([0, 1, 9]))

    def test_dictionary_classifier_known_unknown_and_one_class(self):
        classifier = DDRCClassifier(feature_dim=2, atoms_per_class=1)
        classifier.enroll_prototypes(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), ["alice", "bob"])
        predictions, confidences, residuals = classifier(torch.tensor([[1.0, 0.0], [0.7071, 0.7071]]), residual_threshold=0.55, margin_threshold=0.10)
        self.assertEqual(predictions[0], "alice")
        self.assertEqual(predictions[1], "unknown")
        self.assertEqual(tuple(residuals.shape), (2, 2))
        self.assertTrue(torch.isfinite(confidences).all())
        one_class = DDRCClassifier(feature_dim=2, atoms_per_class=1)
        one_class.enroll_prototypes(torch.tensor([[1.0, 0.0]]), ["alice"])
        self.assertEqual(one_class(torch.tensor([[1.0, 0.0]]))[0][0], "alice")

    def test_gcn_global_labels_are_valid(self):
        predictor = GCNLinkPredictor(feature_dim=4, hidden_dim=8, k_neighbors=2)
        predictor.eval()
        embeddings = F.normalize(torch.randn(8, 4), dim=1)
        labels, active = predictor.generate_global_pseudo_labels(embeddings, confidence_threshold=0.99, start_class_idx=10)
        self.assertEqual(tuple(labels.shape), (8,))
        self.assertEqual(tuple(active.shape), (8,))
        self.assertTrue(((labels == -1) | (labels >= 10)).all())

    def test_checkpoint_round_trip_for_queue(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "queue.pt")
            original = BroadFaceMemoryQueue(in_features=4, num_classes=2, queue_size=3)
            original.enqueue(F.normalize(torch.randn(2, 4), dim=1), torch.tensor([0, 1]))
            torch.save(original.state_dict(), path)
            restored = BroadFaceMemoryQueue(in_features=4, num_classes=2, queue_size=3)
            restored.load_state_dict(torch.load(path, map_location="cpu"))
            self.assertTrue(torch.equal(original.feature_mb, restored.feature_mb))
            self.assertTrue(torch.equal(original.label_mb, restored.label_mb))


if __name__ == "__main__":
    unittest.main()
