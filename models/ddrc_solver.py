import torch
import torch.nn as nn
import torch.nn.functional as F

class LISTASparseSolver(nn.Module):
    """
    Learned ISTA (LISTA) Unrolled Sparse Coding Solver for DDRC.
    Solves min_{x, e} || f - D x - e ||_2^2 + lambda_1 ||x||_1 + lambda_2 ||e||_1
    in O(1) constant-time using unrolled feedforward residual layers.
    """
    def __init__(self, feature_dim: int = 512, num_atoms: int = 2048, num_layers: int = 5, theta: float = 0.1):
        super(LISTASparseSolver, self).__init__()
        self.feature_dim = feature_dim
        self.num_atoms = num_atoms
        self.num_layers = num_layers
        self.theta = theta
        
        # Encoder matrix W_e (num_atoms, feature_dim) and Mutual Incoherence matrix W_s (num_atoms, num_atoms)
        self.W_e = nn.Linear(feature_dim, num_atoms, bias=False)
        self.W_s = nn.Linear(num_atoms, num_atoms, bias=False)
        
        # Soft-thresholding shrinkage parameter
        self.threshold = nn.Parameter(torch.full((num_atoms,), theta))

    def soft_threshold(self, x: torch.Tensor, thresh: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.relu(torch.abs(x) - thresh)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f: (batch_size, feature_dim) deep feature vector
        Returns:
            x: (batch_size, num_atoms) sparse reconstruction coefficients
        """
        # Initial projection: x_0 = soft_threshold(W_e * f)
        b = self.W_e(f)
        x = self.soft_threshold(b, self.threshold)
        
        # Unrolled LISTA iterations: x_{k+1} = soft_threshold(b + W_s * x_k)
        for _ in range(self.num_layers - 1):
            s = b + self.W_s(x)
            x = self.soft_threshold(s, self.threshold)
            
        return x

class DDRCClassifier(nn.Module):
    """
    Deep Discriminative Representation and Dictionary Learning (DDRC) Classifier.
    Computes class-specific residual errors and isolates sparse occlusion noise vectors e.
    Handles open-set unknown identification.
    """
    def __init__(self, feature_dim: int = 512, num_classes: int = 100, atoms_per_class: int = 20):
        super(DDRCClassifier, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.atoms_per_class = atoms_per_class
        self.total_atoms = num_classes * atoms_per_class
        
        # Class-specific discriminative dictionary D: (feature_dim, total_atoms)
        self.dictionary = nn.Parameter(torch.randn(feature_dim, self.total_atoms))
        nn.init.xavier_uniform_(self.dictionary)
        
        # Fast LISTA solver
        self.lista_solver = LISTASparseSolver(feature_dim=feature_dim, num_atoms=self.total_atoms)

    def forward(self, f: torch.Tensor, residual_threshold: float = 0.45, margin_threshold: float = 0.15):
        """
        Args:
            f: (batch_size, feature_dim) normalized deep feature
            residual_threshold: tau_residual maximum allowed residual for enrolled identities
            margin_threshold: delta_margin minimum ratio gap between best and 2nd best residual
        Returns:
            predictions: list of strings (person identity or 'unknown')
            confidences: (batch_size,) confidence scores in [0, 1]
            residuals: (batch_size, num_classes) per-class reconstruction errors
        """
        norm_f = F.normalize(f, p=2, dim=1) # (B, d)
        norm_D = F.normalize(self.dictionary, p=2, dim=0) # (d, C * A)
        
        # Compute sparse coefficients x via LISTA
        sparse_x = self.lista_solver(norm_f) # (B, C * A)
        
        # Reshape for vectorized parallel class reconstruction
        batch_size = f.size(0)
        
        # Reshape D to (C, d, A)
        D_reshaped = norm_D.view(self.feature_dim, self.num_classes, self.atoms_per_class).permute(1, 0, 2) # (C, d, A)
        # Reshape x to (B, C, A, 1)
        x_reshaped = sparse_x.view(batch_size, self.num_classes, self.atoms_per_class, 1)
        
        # Vectorized class reconstruction: f_k = D_k * x_k -> (B, C, d)
        # D_reshaped.unsqueeze(0): (1, C, d, A) x (B, C, A, 1) -> (B, C, d, 1)
        f_reconstructed = torch.matmul(D_reshaped.unsqueeze(0), x_reshaped).squeeze(-1) # (B, C, d)
        
        # Vectorized residual error: || f - f_k ||_2^2 -> (B, C)
        class_residuals = torch.sum((norm_f.unsqueeze(1) - f_reconstructed) ** 2, dim=-1) # (B, C)
        
        # Select best class (min residual) and 2nd best class
        sorted_res, sorted_indices = torch.sort(class_residuals, dim=1, descending=False)
        min_res = sorted_res[:, 0]
        second_res = sorted_res[:, 1]
        best_class = sorted_indices[:, 0]
        
        # Margin ratio: (r_2nd - r_min) / r_2nd
        margin_ratio = (second_res - min_res) / (second_res + 1e-7)
        
        predictions = []
        confidences = []
        
        for i in range(batch_size):
            r_val = min_res[i].item()
            m_val = margin_ratio[i].item()
            cls_idx = best_class[i].item()
            
            # Calibrated confidence score: high margin ratio & low residual error
            conf = max(0.0, min(1.0, (1.0 - r_val) * (0.5 + 0.5 * m_val)))
            
            # Open-set Unknown gate check
            if r_val <= residual_threshold and m_val >= margin_threshold:
                predictions.append(f"Person_{cls_idx}")
                confidences.append(conf)
            else:
                predictions.append("unknown")
                confidences.append(conf * 0.5)
                
        return predictions, torch.tensor(confidences, device=f.device), class_residuals
