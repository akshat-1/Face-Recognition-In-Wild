import torch
import torch.nn as nn
import torch.nn.functional as F

class BroadFaceMemoryQueue(nn.Module):
    """
    BroadFace: Looking at tens of thousands of people at once for face recognition (Kim et al., ECCV 2020)
    
    Maintains a large FIFO queue of past feature embeddings (e.g. N_q = 32,768) and handles
    weight update drift compensation so past queue embeddings remain valid representations of current weights.
    """
    def __init__(self, queue_size: int = 32768, feature_dim: int = 512, momentum: float = 0.99):
        super(BroadFaceMemoryQueue, self).__init__()
        self.queue_size = queue_size
        self.feature_dim = feature_dim
        self.momentum = momentum
        
        # FIFO Queue buffers
        self.register_buffer("queue_embeddings", torch.randn(queue_size, feature_dim))
        self.register_buffer("queue_labels", torch.zeros(queue_size, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("is_full", torch.zeros(1, dtype=torch.bool))
        
        # Normalize initial queue embeddings
        self.queue_embeddings = F.normalize(self.queue_embeddings, p=2, dim=1)

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Enqueues new mini-batch embeddings and labels into the FIFO queue.
        Args:
            embeddings: (batch_size, feature_dim) normalized embeddings
            labels: (batch_size,) identity labels
        """
        batch_size = embeddings.size(0)
        ptr = int(self.queue_ptr.item())
        
        # If batch size exceeds remaining queue capacity, wrap around
        if ptr + batch_size <= self.queue_size:
            self.queue_embeddings[ptr:ptr + batch_size] = embeddings
            self.queue_labels[ptr:ptr + batch_size] = labels
            ptr = (ptr + batch_size) % self.queue_size
        else:
            overflow = (ptr + batch_size) - self.queue_size
            first_part = batch_size - overflow
            
            self.queue_embeddings[ptr:self.queue_size] = embeddings[:first_part]
            self.queue_labels[ptr:self.queue_size] = labels[:first_part]
            
            self.queue_embeddings[0:overflow] = embeddings[first_part:]
            self.queue_labels[0:overflow] = labels[first_part:]
            
            ptr = overflow
            self.is_full[0] = True
            
        self.queue_ptr[0] = ptr

    def compensate_weight_drift(self, old_weight: torch.Tensor, new_weight: torch.Tensor):
        """
        Applies BroadFace compensation to old queue embeddings using weight update delta ΔW.
        f_compensated = f_old + η * (W_new - W_old) * f_old
        """
        if not self.training:
            return
            
        with torch.no_grad():
            delta_w = new_weight - old_weight
            if delta_w.abs().sum() < 1e-8:
                return
                
            # Apply momentum adjustment to queue embeddings
            drift = torch.matmul(self.queue_embeddings, delta_w.T)
            adjusted_embeddings = self.queue_embeddings + 0.01 * torch.matmul(drift, delta_w)
            self.queue_embeddings.copy_(F.normalize(adjusted_embeddings, p=2, dim=1))

    def get_queue_samples(self):
        """
        Returns active queue embeddings and identity labels.
        """
        if not self.is_full:
            ptr = int(self.queue_ptr.item())
            return self.queue_embeddings[:ptr], self.queue_labels[:ptr]
        return self.queue_embeddings, self.queue_labels
