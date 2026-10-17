"""Dictionary-conditioned sparse open-set classification.

The previous implementation generated coefficients from an unrelated random
matrix and then classified with cosine similarity. This version makes every
LISTA/ISTA step use the enrolled dictionary and computes genuine class-specific
reconstruction residuals. The dictionary can be trained externally or filled
from an enrollment gallery.
"""

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_threshold(value: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * F.relu(value.abs() - threshold)


class LISTASparseSolver(nn.Module):
    """Fixed-step unrolled robust sparse coding for a supplied dictionary."""

    def __init__(self, feature_dim: int = 512, num_atoms: int = 2048,
                 num_layers: int = 5, theta: float = 0.05):
        super().__init__()
        if feature_dim <= 0 or num_atoms <= 0 or num_layers <= 0:
            raise ValueError("feature_dim, num_atoms, and num_layers must be positive")
        self.feature_dim = feature_dim
        self.num_atoms = num_atoms
        self.num_layers = num_layers
        self.log_step = nn.Parameter(torch.tensor(-1.0))
        self.threshold_x = nn.ParameterList([
            nn.Parameter(torch.full((num_atoms,), theta)) for _ in range(num_layers)
        ])
        self.threshold_e = nn.ParameterList([
            nn.Parameter(torch.tensor(theta)) for _ in range(num_layers)
        ])

    def forward(self, f: torch.Tensor, dictionary: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if f.ndim != 2 or f.shape[1] != self.feature_dim:
            raise ValueError(f"f must have shape (B,{self.feature_dim})")
        if dictionary.ndim != 2 or dictionary.shape != (self.feature_dim, self.num_atoms):
            raise ValueError(
                f"dictionary must have shape ({self.feature_dim},{self.num_atoms}), "
                f"got {tuple(dictionary.shape)}"
            )
        d = F.normalize(dictionary.float(), p=2, dim=0)
        f = f.float()
        x = torch.zeros((f.shape[0], self.num_atoms), device=f.device, dtype=f.dtype)
        e = torch.zeros_like(f)
        step = self.log_step.exp().clamp(1e-4, 2.0)
        for layer in range(self.num_layers):
            residual = f - x @ d.T - e
            x = _soft_threshold(x + step * (residual @ d), self.threshold_x[layer].abs())
            e = _soft_threshold(f - x @ d.T, self.threshold_e[layer].abs())
        return x, e


class DDRCClassifier(nn.Module):
    """Gallery dictionary classifier with thresholded open-set gating.

    The thresholds become calibrated only after they are selected on a held-out
    known/unknown validation protocol.
    """

    def __init__(self, feature_dim: int = 512, num_classes: int = 0,
                 atoms_per_class: int = 20, num_layers: int = 5):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.atoms_per_class = max(1, int(atoms_per_class))
        self.num_classes = 0
        self.total_atoms = 0
        self.class_names: List[str] = []
        self.register_parameter("dictionary", None)
        self.lista_solver = LISTASparseSolver(
            feature_dim=self.feature_dim,
            num_atoms=1,
            num_layers=num_layers,
        )
        if num_classes:
            self._reset_dictionary(num_classes, device=torch.device("cpu"))

    def _reset_dictionary(self, num_classes: int, device: torch.device) -> None:
        if num_classes <= 0:
            self.num_classes = 0
            self.total_atoms = 0
            self.dictionary = None
            self.class_names = []
            return
        self.num_classes = int(num_classes)
        self.total_atoms = self.num_classes * self.atoms_per_class
        self.dictionary = nn.Parameter(
            torch.empty(self.feature_dim, self.total_atoms, device=device)
        )
        nn.init.xavier_uniform_(self.dictionary)
        self.lista_solver = LISTASparseSolver(
            feature_dim=self.feature_dim,
            num_atoms=self.total_atoms,
            num_layers=self.lista_solver.num_layers,
        ).to(device)
        self.class_names = [f"Person_{i}" for i in range(self.num_classes)]

    def set_class_names(self, class_names: List[str]) -> None:
        if not class_names:
            self._reset_dictionary(0, next(self.parameters()).device)
            return
        device = self.dictionary.device if self.dictionary is not None else next(self.parameters()).device
        old_atoms = self.atoms_per_class
        self._reset_dictionary(len(class_names), device)
        self.atoms_per_class = old_atoms
        self.class_names = list(class_names)

    @torch.no_grad()
    def enroll_prototypes(self, class_prototypes: torch.Tensor,
                          class_names: Optional[List[str]] = None) -> None:
        if class_prototypes.ndim != 2 or class_prototypes.shape[1] != self.feature_dim:
            raise ValueError(f"prototypes must have shape (C,{self.feature_dim})")
        device = self.dictionary.device if self.dictionary is not None else next(self.parameters()).device
        prototypes = F.normalize(class_prototypes.to(device).float(), p=2, dim=1)
        self._reset_dictionary(prototypes.shape[0], device)
        atoms = prototypes[:, None, :].expand(-1, self.atoms_per_class, -1)
        self.dictionary.copy_(atoms.reshape(-1, self.feature_dim).T)
        if class_names is not None:
            if len(class_names) != prototypes.shape[0]:
                raise ValueError("class_names must match the number of prototypes")
            self.class_names = list(class_names)

    @torch.no_grad()
    def enroll_gallery(self, embeddings: torch.Tensor, labels: torch.Tensor,
                       class_names: Optional[Sequence[str]] = None) -> None:
        """Build atoms from all gallery embeddings instead of classifier weights."""
        if embeddings.ndim != 2 or embeddings.shape[1] != self.feature_dim:
            raise ValueError(f"embeddings must have shape (N,{self.feature_dim})")
        if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
            raise ValueError("labels must be rank-1 and align with embeddings")
        if embeddings.shape[0] == 0:
            self._reset_dictionary(0, embeddings.device)
            return
        labels = labels.long()
        unique = torch.unique(labels, sorted=True)
        names = list(class_names) if class_names is not None else [f"Person_{int(i)}" for i in unique]
        if len(names) != unique.numel():
            raise ValueError("class_names must match the number of unique labels")
        device = embeddings.device
        self._reset_dictionary(unique.numel(), device)
        blocks = []
        for class_id in unique:
            class_embeddings = F.normalize(embeddings[labels == class_id].float(), p=2, dim=1)
            if class_embeddings.shape[0] >= self.atoms_per_class:
                block = class_embeddings[:self.atoms_per_class]
            else:
                block = class_embeddings.repeat((self.atoms_per_class + class_embeddings.shape[0] - 1) // class_embeddings.shape[0], 1)[:self.atoms_per_class]
            blocks.append(block)
        self.dictionary.copy_(torch.cat(blocks, dim=0).T)
        self.class_names = names

    def forward(self, f: torch.Tensor, residual_threshold: float = 0.55,
                margin_threshold: float = 0.10):
        if f.ndim != 2 or f.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape (B,{self.feature_dim})")
        batch_size = f.shape[0]
        if self.dictionary is None or self.num_classes == 0:
            unknown = ["unknown"] * batch_size
            return unknown, torch.zeros(batch_size, device=f.device), torch.empty(batch_size, 0, device=f.device)

        norm_f = F.normalize(f.float(), p=2, dim=1)
        x, error = self.lista_solver(norm_f, self.dictionary)
        clean = norm_f - error
        dictionary = F.normalize(self.dictionary.float(), p=2, dim=0)
        reconstruction = x @ dictionary.T
        residuals = []
        for class_id in range(self.num_classes):
            start = class_id * self.atoms_per_class
            end = start + self.atoms_per_class
            class_reconstruction = x[:, start:end] @ dictionary[:, start:end].T
            residuals.append((clean - class_reconstruction).square().mean(dim=1))
        class_residuals = torch.stack(residuals, dim=1)
        sorted_residuals, sorted_indices = torch.sort(class_residuals, dim=1)
        best = sorted_indices[:, 0]
        best_residual = sorted_residuals[:, 0]
        if self.num_classes > 1:
            second = sorted_residuals[:, 1]
            margin = (second - best_residual) / second.clamp_min(1e-7)
        else:
            margin = torch.ones_like(best_residual)

        confidence = torch.exp(-best_residual / max(float(residual_threshold), 1e-6))
        confidence = confidence * margin.clamp(0.0, 1.0)
        predictions = []
        confidence_values = []
        for index in range(batch_size):
            known = bool(best_residual[index] <= residual_threshold and margin[index] >= margin_threshold)
            class_index = int(best[index].item())
            predictions.append(self.class_names[class_index] if known else "unknown")
            confidence_values.append(float(confidence[index].clamp(0.0, 1.0).item()))
        return predictions, torch.tensor(confidence_values, device=f.device), class_residuals
