import os
import torch
import torch.nn as nn
import torch.optim as optim

from losses.curricular_loss import CurricularFaceLoss
from losses.broadface_queue import BroadFaceMemoryQueue
from models.backbone import ResNet100Backbone
from pipeline.wild_face_pipeline import OccuPoseBroadDictPipeline

def train_one_epoch_demo(num_classes: int = 50, batch_size: int = 8):
    """
    Demonstrates training a mini-batch with CurricularFace adaptive loss and BroadFace memory queueing.
    """
    print("=== Step 1: Initializing OccuPose-BroadDictNet Training Components ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Instantiate ResNet-100 backbone (embedding dim = 512)
    backbone = ResNet100Backbone(embedding_dim=512, layers=(3, 4, 6, 3)).to(device)
    
    # 2. Instantiate CurricularFace adaptive loss
    curricular_loss_fn = CurricularFaceLoss(in_features=512, num_classes=num_classes, s=64.0, m=0.5).to(device)
    
    # 3. Instantiate BroadFace memory queue
    broadface_queue = BroadFaceMemoryQueue(queue_size=1024, feature_dim=512).to(device)
    
    # Optimizer
    optimizer = optim.SGD(
        list(backbone.parameters()) + list(curricular_loss_fn.parameters()),
        lr=0.1, momentum=0.9, weight_decay=5e-4
    )
    
    print("\n=== Step 2: Running Mini-Batch Forward & Backward Pass ===")
    backbone.train()
    curricular_loss_fn.train()
    
    # Generate synthetic training mini-batch
    dummy_faces = torch.randn(batch_size, 3, 112, 112, device=device)
    dummy_labels = torch.randint(0, num_classes, (batch_size,), device=device)
    
    # Forward pass through backbone
    embeddings = backbone(dummy_faces) # (batch_size, 512)
    
    # Calculate CurricularFace adaptive loss
    loss = curricular_loss_fn(embeddings, dummy_labels)
    
    # Backward pass & optimization step
    optimizer.zero_grad()
    loss.backward()
    
    # Save old classifier weights for BroadFace drift compensation
    with torch.no_grad():
        old_weight = curricular_loss_fn.weight.clone()
        
    optimizer.step()
    
    # BroadFace queue update & weight drift compensation
    broadface_queue.update(embeddings.detach(), dummy_labels)
    broadface_queue.compensate_weight_drift(old_weight, curricular_loss_fn.weight)
    
    print(f"CurricularFace Training Loss: {loss.item():.4f}")
    print(f"CurricularFace Dynamic Curriculum Parameter t: {curricular_loss_fn.t.item():.4f}")
    print(f"BroadFace Memory Queue Enqueued Samples: {broadface_queue.queue_ptr.item()}/1024")
    
    # Save checkpoint demo
    checkpoint_path = "checkpoint_occu_pose_broad_dict.pt"
    torch.save({
        'backbone_state': backbone.state_dict(),
        'curricular_loss_state': curricular_loss_fn.state_dict(),
        'broadface_queue_state': broadface_queue.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }, checkpoint_path)
    print(f"\nSaved training checkpoint to: {checkpoint_path}")

def run_inference_demo():
    """
    Demonstrates running end-to-end inference on a wild image using OccuPoseBroadDictPipeline.
    """
    print("\n=== Step 3: Running End-to-End Inference Engine Pipeline ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pipeline = OccuPoseBroadDictPipeline(num_enrolled_classes=50, feature_dim=512).to(device)
    
    # Prepare dummy input image (224x224)
    dummy_wild_image = torch.randn(1, 3, 224, 224, device=device)
    
    # Initialize score head so detection succeeds
    nn.init.zeros_(pipeline.detector.score_head.weight)
    nn.init.constant_(pipeline.detector.score_head.bias, 5.0)
    nn.init.zeros_(pipeline.detector.bbox_head.weight)
    nn.init.constant_(pipeline.detector.bbox_head.bias, 0.5)
    
    results = pipeline(dummy_wild_image, score_threshold=0.1)
    
    print(f"Detected and processed {len(results)} faces in wild scene:")
    for idx, face in enumerate(results):
        print(f"\nFace #{idx + 1}:")
        print(f"  Bounding Box: {face['box']}")
        print(f"  Predicted Identity: {face['identity']}")
        print(f"  Confidence Score: {face['confidence']}")
        print(f"  Occluded Flag: {face['is_occluded']}")
        print(f"  Key Attribute Predictions: {list(face['attributes'].items())[:3]}")

if __name__ == "__main__":
    train_one_epoch_demo()
    run_inference_demo()
