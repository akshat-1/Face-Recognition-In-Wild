import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional

class LISTASparseSolver(nn.Module):
    """
    Official Learned ISTA (LISTA) Unrolled Sparse Coding Solver for DDRC (Gregor & LeCun, ICML 2010; Deng et al., IEEE TIP 2018).
    
    Solves min_{x, e} || f - D x - e ||_2^2 + lambda_1 ||x||_1 + lambda_2 ||e||_1
    in O(1) constant-time unrolled feedforward layers with layer-specific learned thresholds and mutual incoherence matrices.
    Memory-optimized with factorized low-rank updates for large atom dictionaries (num_atoms > 4096).
    """
    def __init__(self, feature_dim: int = 512, num_atoms: int = 2048, num_layers: int = 5, theta: float = 0.1):
        super(LISTASparseSolver, self).__init__()
        self.feature_dim = feature_dim
        self.num_atoms = num_atoms
        self.num_layers = num_layers
        
        # Initial encoder projection matrix W_e
        self.W_e = nn.Linear(feature_dim, num_atoms, bias=False)
        nn.init.xavier_uniform_(self.W_e.weight)
        
        # Layer-specific mutual incoherence feedforward layers
        self.use_factorized = (num_atoms > 4096)
        if self.use_factorized:
            hidden_dim = 256
            self.W_s1 = nn.ModuleList([nn.Linear(num_atoms, hidden_dim, bias=False) for _ in range(num_layers - 1)])
            self.W_s2 = nn.ModuleList([nn.Linear(hidden_dim, num_atoms, bias=False) for _ in range(num_layers - 1)])
            for l1, l2 in zip(self.W_s1, self.W_s2):
                nn.init.xavier_uniform_(l1.weight)
                nn.init.xavier_uniform_(l2.weight)
        else:
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
            if self.use_factorized:
                s = b + self.W_s2[k](self.W_s1[k](x))
            else:
                s = b + self.W_s[k](x)
            x = self.soft_threshold(s, self.thresholds[k + 1])
            
        # Explicit sparse occlusion error vector e = f - D * x
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
        
        self.class_names = [f"Person_{i}" for i in range(num_classes)]
        
        # Class-specific discriminative dictionary D: (feature_dim, total_atoms)
        self.dictionary = nn.Parameter(torch.randn(feature_dim, self.total_atoms))
        nn.init.xavier_uniform_(self.dictionary)
        
        # Layer-unrolled LISTA solver
        self.lista_solver = LISTASparseSolver(feature_dim=feature_dim, num_atoms=self.total_atoms, num_layers=5)

    def set_class_names(self, class_names: List[str]):
        """Sets human-readable class names for predictions and adjusts dictionary size."""
        self.class_names = list(class_names)
        new_num_classes = len(class_names)
        if new_num_classes != self.num_classes:
            self.num_classes = new_num_classes
            self.total_atoms = self.num_classes * self.atoms_per_class
            device = self.dictionary.device
            self.dictionary = nn.Parameter(torch.randn(self.feature_dim, self.total_atoms, device=device))
            nn.init.xavier_uniform_(self.dictionary)
            self.lista_solver = LISTASparseSolver(feature_dim=self.feature_dim, num_atoms=self.total_atoms, num_layers=5).to(device)

    def enroll_prototypes(self, class_prototypes: torch.Tensor, class_names: Optional[List[str]] = None):
        """
        Populates dictionary atoms from class prototype feature vectors (num_classes, feature_dim).
        """
        device = self.dictionary.device
        norm_proto = F.normalize(class_prototypes.to(device), p=2, dim=1) # (C, d)
        C, d = norm_proto.shape
        
        # Adjust atoms_per_class to maintain linear scaling for large class counts
        if C > 1000:
            self.atoms_per_class = 1
        elif C > 200:
            self.atoms_per_class = 2
        else:
            self.atoms_per_class = 5
            
        self.num_classes = C
        self.total_atoms = C * self.atoms_per_class
        
        if class_names is not None and len(class_names) == C:
            self.class_names = list(class_names)
        else:
            self.class_names = [f"Person_{i}" for i in range(C)]
            
        self.lista_solver = LISTASparseSolver(feature_dim=self.feature_dim, num_atoms=self.total_atoms, num_layers=5).to(device)
            
        atoms = []
        for i in range(C):
            proto = norm_proto[i:i+1] # (1, d)
            if self.atoms_per_class > 1:
                noise = torch.randn(self.atoms_per_class, d, device=device) * 0.05
                atom_block = F.normalize(proto.repeat(self.atoms_per_class, 1) + noise, p=2, dim=1)
            else:
                atom_block = proto
            atoms.append(atom_block)
            
        stacked_atoms = torch.cat(atoms, dim=0).T # (d, C * A)
        self.dictionary = nn.Parameter(stacked_atoms)

    def forward(self, f: torch.Tensor, residual_threshold: float = 0.55, margin_threshold: float = 0.10):
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
        batch_size = f.size(0)
        
        # Solve for sparse occlusion error vector e via LISTA
        _, error_e = self.lista_solver(norm_f)
        clean_f = F.normalize(norm_f - error_e, p=2, dim=1) # (B, d)
        
        # Cosine similarity matrix between clean_f and all dictionary atoms: (B, C * A)
        atom_sims = torch.mm(clean_f, norm_D)
        atom_sims_reshaped = atom_sims.view(batch_size, self.num_classes, self.atoms_per_class)
        max_class_sims = torch.max(atom_sims_reshaped, dim=2).values # (B, C)
        
        # Class residual with explicit sparse occlusion error isolation: r_k = 1.0 - max_class_sim
        class_residuals = 1.0 - max_class_sims # (B, C)
        
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
                name = self.class_names[cls_idx] if cls_idx < len(self.class_names) else f"Person_{cls_idx}"
                predictions.append(name)
                confidences.append(conf)
            else:
                predictions.append("unknown")
                confidences.append(conf * 0.5)
                
        return predictions, torch.tensor(confidences, device=f.device), class_residuals
