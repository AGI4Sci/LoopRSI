from __future__ import annotations

import torch
from torch import nn


class TargetFeatureCRPM(nn.Module):
    """CRPM variant whose target conditioning can run for unseen target genes.

    The scientific objective remains perturbation response prediction. The change is
    only the target-conditioning parameterization: fixed target features are encoded
    into low-rank CRPM weights. Optional learned target-id corrections are valid only
    for targets observed during training and must be disabled for official H1
    validation/test cold-start inference.
    """

    def __init__(
        self,
        baselines: torch.Tensor,
        feature_dim: int,
        rank: int,
        n_train_targets: int,
        n_batches: int,
        learned_id_correction: bool = False,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or rank <= 0:
            raise ValueError("feature_dim and rank must be positive")
        self.register_buffer("baselines", baselines)
        self.target_encoder = nn.Sequential(nn.Linear(feature_dim, rank), nn.Tanh())
        self.learned_target = nn.Embedding(n_train_targets, rank) if learned_id_correction else None
        self.batch_weights = nn.Embedding(n_batches, rank)
        self.gene_programs = nn.Parameter(torch.empty(rank, baselines.shape[1]))
        nn.init.zeros_(self.batch_weights.weight)
        if self.learned_target is not None:
            nn.init.zeros_(self.learned_target.weight)
        nn.init.normal_(self.gene_programs, std=0.01)

    def condition(self, features: torch.Tensor, target_id: torch.Tensor | None = None, allow_id: bool = False) -> torch.Tensor:
        weights = self.target_encoder(features)
        if allow_id and self.learned_target is not None:
            if target_id is None:
                raise ValueError("target_id is required when learned ID correction is enabled")
            weights = weights + self.learned_target(target_id)
        return weights

    def forward(
        self,
        features: torch.Tensor,
        batch_id: torch.Tensor,
        target_id: torch.Tensor | None = None,
        allow_id: bool = False,
    ) -> torch.Tensor:
        weights = self.condition(features, target_id, allow_id=allow_id) + self.batch_weights(batch_id)
        return self.baselines[batch_id] + weights @ self.gene_programs
