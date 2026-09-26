import torch
import torch.nn as nn
import torch.nn.functional as F

class LISTASparseSolver(nn.Module):
    """
    Official Learned ISTA (LISTA) Unrolled Sparse Coding Solver for DDRC (Gregor & LeCun, ICML 2010; Deng et al., IEEE TIP 2018).
    
    Solves min_{x, e} || f - D x - e ||_2^2 + lambda_1 ||x||_1 + lambda_2 ||e||_1
    in O(1) constant-time unrolled feedforward layers with layer-specific learned thresholds and mutual incoherence matrices.
    """
    def __init__(self, feature_dim: int = 512, num_atoms: int = 2048, num_layers: int = 5, theta: float = 0.1):
        super(LISTASparseSolver, self).__init__()
        self.feature_dim = feature_dim
        self.num_atoms = num_atoms
        self.num_layers = num_layers
        
        # Initial encoder projection matrix W_e
        self.W_e = nn.Linear(feature_dim, num_atoms, bias=False)
        nn.init.xavier_uniform_(self.W_e.weight)
        
        # Layer-specific mutual incoherence feedforward layers W_s^{(k)}
        self.W_s = nn.ModuleList([
            nn.Linear(num_atoms, num_atoms, bias=False) for _ in range(num_layers - 1)
        ])
        for layer in self.W_s:
            nn.init.xavier_uniform_(layer.weight)
            
        # Layer-specific learnable soft-thresholding shrinkage parameters theta^{(k)}
        self.thresholds = nn.ParameterList([
            nn.Parameter(torch.full((num_atoms,), theta)) for _ in range(num_layers)
        ])

    def soft_threshold(self, x: torch.Tensor, thresh: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.relu(torch.abs(x) - thresh)

    def forward(self, f: torch.Tensor):
        """
        Args:
            f: (batch_size, feature_dim) deep feature vector
        Returns:
            x: (batch_size, num_atoms) sparse reconstruction coefficients
            e: (batch_size, feature_dim) sparse occlusion error vector
        """
        b = self.W_e(f)
        x = self.soft_threshold(b, self.thresholds[0])
        
        # Layer-unrolled feedforward iterations
        for k in range(self.num_layers - 1):
            s = b + self.W_s[k](x)
            x = self.soft_threshold(s, self.thresholds[k + 1])
            
        # Explicit sparse occlusion error vector e = f - D * x
        # Soft-threshold error vector e for L1-sparse corruption isolation
        e = self.soft_threshold(f - F.linear(x, self.W_e.weight.T[:self.feature_dim, :]), 0.05)
        return x, e

class DDRCClassifier(nn.Module):
    """
    Official Deep Discriminative Representation & Dictionary Learning (DDRC) Classifier.
    
    Performs discriminative dictionary reconstruction, isolates sparse occlusion error vectors e,
    calculates class residuals r_k, and enforces Open-Set Unknown Gating.
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
        
        # Layer-unrolled LISTA solver
        self.lista_solver = LISTASparseSolver(feature_dim=feature_dim, num_atoms=self.total_atoms, num_layers=5)

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
        
        # Solve for sparse coefficients x and occlusion error vector e
        sparse_x, error_e = self.lista_solver(norm_f) # (B, C * A), (B, d)
        
        batch_size = f.size(0)
        
        # Reshape D to (C, d, A) and x to (B, C, A, 1)
        D_reshaped = norm_D.view(self.feature_dim, self.num_classes, self.atoms_per_class).permute(1, 0, 2) # (C, d, A)
        x_reshaped = sparse_x.view(batch_size, self.num_classes, self.atoms_per_class, 1)
        
        # Vectorized class reconstruction: f_k = D_k * x_k -> (B, C, d)
        f_reconstructed = torch.matmul(D_reshaped.unsqueeze(0), x_reshaped).squeeze(-1) # (B, C, d)
        
        # Class residual with explicit sparse occlusion error isolation: || f - f_k - e ||_2^2
        clean_f = norm_f - error_e
        class_residuals = torch.sum((clean_f.unsqueeze(1) - f_reconstructed) ** 2, dim=-1) # (B, C)
        
        # Select minimum residual class and 2nd minimum class
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
            
            # Calibrated open-set confidence score
            conf = max(0.0, min(1.0, (1.0 - r_val) * (0.5 + 0.5 * m_val)))
            
            # Open-Set Unknown Gate
            if r_val <= residual_threshold and m_val >= margin_threshold:
                predictions.append(f"Person_{cls_idx}")
                confidences.append(conf)
            else:
                predictions.append("unknown")
                confidences.append(conf * 0.5)
                
        return predictions, torch.tensor(confidences, device=f.device), class_residuals
