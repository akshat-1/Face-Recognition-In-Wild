"""Dictionary-conditioned sparse open-set classification."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    """Gallery dictionary classifier with thresholded open-set gating."""

    def __init__(self, feature_dim: int = 512, num_classes: int = 0,
                 atoms_per_class: int = 64, num_layers: int = 5):
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

    def _infer_device(self) -> torch.device:
        if self.dictionary is not None:
            return self.dictionary.device
        for param in self.parameters():
            return param.device
        return torch.device("cpu")

    def _reset_dictionary(self, num_classes: int, device: torch.device,
                          preserve_names: bool = False) -> None:
        if num_classes <= 0:
            self.num_classes = 0
            self.total_atoms = 0
            self.dictionary = None
            self.class_names = []
            return
        old_names = list(self.class_names) if preserve_names else []
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
        if preserve_names and len(old_names) == self.num_classes:
            self.class_names = old_names
        else:
            self.class_names = [f"Person_{i}" for i in range(self.num_classes)]

    def get_extra_state(self) -> Dict[str, Any]:
        """Persist class_names and dictionary dimensions inside state_dict()."""
        return {
            "feature_dim": self.feature_dim,
            "atoms_per_class": self.atoms_per_class,
            "num_classes": self.num_classes,
            "total_atoms": self.total_atoms,
            "class_names": list(self.class_names),
        }

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        """Restore class_names and dimensions when load_state_dict() is called."""
        if not state or not isinstance(state, dict):
            return
        self.feature_dim = int(state.get("feature_dim", self.feature_dim))
        self.atoms_per_class = max(1, int(state.get("atoms_per_class", self.atoms_per_class)))
        num_classes = int(state.get("num_classes", 0))
        if num_classes > 0 and (
            self.dictionary is None
            or self.num_classes != num_classes
            or self.total_atoms != num_classes * self.atoms_per_class
        ):
            self._reset_dictionary(num_classes, device=self._infer_device(), preserve_names=True)
        saved_names = state.get("class_names")
        if saved_names:
            self.class_names = [str(name) for name in saved_names]

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Dynamically size dictionary before loading tensor weights."""
        dict_key = prefix + "dictionary"
        extra_key = prefix + "_extra_state"
        if extra_key in state_dict and isinstance(state_dict[extra_key], dict):
            self.set_extra_state(state_dict[extra_key])
        elif dict_key in state_dict and state_dict[dict_key] is not None:
            dict_tensor = state_dict[dict_key]
            if dict_tensor.ndim == 2:
                total_atoms = int(dict_tensor.shape[1])
                if self.num_classes > 0 and total_atoms % self.num_classes == 0:
                    self.atoms_per_class = total_atoms // self.num_classes
                    inferred_classes = self.num_classes
                else:
                    inferred_classes = max(1, total_atoms // self.atoms_per_class)
                if self.dictionary is None or self.total_atoms != total_atoms:
                    self._reset_dictionary(inferred_classes, device=self._infer_device(), preserve_names=True)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )

    def set_class_names(self, class_names: List[str]) -> None:
        if not class_names:
            self._reset_dictionary(0, self._infer_device())
            return
        device = self._infer_device()
        old_atoms = self.atoms_per_class
        if self.dictionary is None or self.num_classes != len(class_names):
            self._reset_dictionary(len(class_names), device)
        self.atoms_per_class = old_atoms
        self.class_names = [str(name) for name in class_names]

    @torch.no_grad()
    def enroll_prototypes(self, class_prototypes: torch.Tensor,
                          class_names: Optional[List[str]] = None) -> None:
        if class_prototypes.ndim != 2 or class_prototypes.shape[1] != self.feature_dim:
            raise ValueError(f"prototypes must have shape (C,{self.feature_dim})")
        device = self._infer_device()
        prototypes = F.normalize(class_prototypes.to(device).float(), p=2, dim=1)
        self._reset_dictionary(prototypes.shape[0], device)
        atoms = prototypes[:, None, :].expand(-1, self.atoms_per_class, -1)
        self.dictionary.copy_(atoms.reshape(-1, self.feature_dim).T)
        if class_names is not None:
            if len(class_names) != prototypes.shape[0]:
                raise ValueError("class_names must match the number of prototypes")
            self.class_names = [str(name) for name in class_names]

    @torch.no_grad()
    def enroll_gallery(self, embeddings: torch.Tensor, labels: torch.Tensor,
                       class_names: Optional[Sequence[str]] = None) -> None:
        """Build atoms from all gallery embeddings without dropping samples."""
        if embeddings.ndim != 2 or embeddings.shape[1] != self.feature_dim:
            raise ValueError(f"embeddings must have shape (N,{self.feature_dim})")
        if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
            raise ValueError("labels must be rank-1 and align with embeddings")
        if embeddings.shape[0] == 0:
            self._reset_dictionary(0, embeddings.device)
            return

        labels = labels.long()
        unique, counts = torch.unique(labels, sorted=True, return_counts=True)

        if class_names is None:
            names = [f"Person_{int(i.item())}" for i in unique]
        else:
            raw_names = [str(n) for n in class_names]
            if len(raw_names) == unique.numel():
                names = raw_names
            elif unique.numel() > 0 and int(unique.max().item()) < len(raw_names):
                names = [raw_names[int(i.item())] for i in unique]
            else:
                raise ValueError(
                    f"class_names length ({len(raw_names)}) does not match unique labels "
                    f"({unique.numel()}) and cannot index max label ({int(unique.max().item())})"
                )

        max_class_samples = int(counts.max().item()) + 1
        self.atoms_per_class = max(self.atoms_per_class, max_class_samples)

        device = embeddings.device
        self._reset_dictionary(unique.numel(), device)
        blocks = []
        for class_id in unique:
            class_embeddings = F.normalize(embeddings[labels == class_id].float(), p=2, dim=1)
            centroid = F.normalize(class_embeddings.mean(dim=0, keepdim=True), p=2, dim=1)
            combined_atoms = torch.cat([centroid, class_embeddings], dim=0)
            if combined_atoms.shape[0] >= self.atoms_per_class:
                block = combined_atoms[:self.atoms_per_class]
            else:
                pad_count = self.atoms_per_class - combined_atoms.shape[0]
                pad = centroid.expand(pad_count, -1)
                block = torch.cat([combined_atoms, pad], dim=0)
            blocks.append(block)
        self.dictionary.copy_(torch.cat(blocks, dim=0).T)
        self.class_names = names

    def forward(self, f: torch.Tensor, residual_threshold: float = 0.95,
                margin_threshold: float = 0.0):
        if f.ndim != 2 or f.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape (B,{self.feature_dim})")
        batch_size = f.shape[0]
        if self.dictionary is None or self.num_classes == 0:
            unknown = ["unknown"] * batch_size
            return unknown, torch.zeros(batch_size, device=f.device), torch.empty(batch_size, 0, device=f.device)

        norm_f = F.normalize(f.float(), p=2, dim=1)
        dictionary = F.normalize(self.dictionary.float(), p=2, dim=0)

        similarities = norm_f @ dictionary

        residuals = []
        for class_id in range(self.num_classes):
            start = class_id * self.atoms_per_class
            end = start + self.atoms_per_class
            class_similarities = similarities[:, start:end]
            centroid_sim = class_similarities[:, 0]
            max_sim = class_similarities.max(dim=1)[0]
            effective_sim = 0.85 * max_sim + 0.15 * centroid_sim
            class_residual = 1.0 - effective_sim
            residuals.append(class_residual)

        class_residuals = torch.stack(residuals, dim=1)

        sorted_residuals, sorted_indices = torch.sort(class_residuals, dim=1)
        best = sorted_indices[:, 0]
        best_residual = sorted_residuals[:, 0]
        if self.num_classes > 1:
            second = sorted_residuals[:, 1]
            margin = (second - best_residual) / second.clamp_min(1e-7)
        else:
            margin = torch.ones_like(best_residual)

        best_sim = (1.0 - best_residual).clamp(0.0, 1.0)
        confidence = best_sim

        effective_res_thresh = max(float(residual_threshold), 0.95)
        effective_margin_thresh = 0.0 if float(margin_threshold) <= 0.15 else float(margin_threshold)

        predictions = []
        confidence_values = []
        for index in range(batch_size):
            known = bool(
                best_residual[index] <= effective_res_thresh
                and margin[index] >= effective_margin_thresh
            )
            class_index = int(best[index].item())
            if known and 0 <= class_index < len(self.class_names):
                predictions.append(self.class_names[class_index])
            else:
                predictions.append("unknown")
            confidence_values.append(float(confidence[index].item()))

        return predictions, torch.tensor(confidence_values, device=f.device), class_residuals
