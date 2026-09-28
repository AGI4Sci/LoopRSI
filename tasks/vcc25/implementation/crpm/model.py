from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CRPM(nn.Module):
    """Control baseline + target/batch weights over shared gene programs."""

    def __init__(self, baselines: torch.Tensor, n_targets: int, rank: int,
                 use_batch_calibration: bool = True, n_guides: int = 0,
                 use_target_batch_interaction: bool = False,
                 use_guide_prior: bool = False,
                 guide_parent_targets: torch.Tensor | None = None,
                 guide_hierarchy_membership: torch.Tensor | None = None,
                 hierarchy_node_layers: torch.Tensor | None = None,
                 hierarchy_embedding_dim: int = 0,
                 hierarchy_alpha_learnable: bool = False,
                 hierarchy_dropout: float = 0.0):
        super().__init__()
        n_batches, n_genes = baselines.shape
        self.register_buffer("baselines", baselines)
        self.target_weights = nn.Embedding(n_targets, rank)
        self.guide_residuals = nn.Embedding(n_guides, rank) if n_guides else None
        self.guide_prior = nn.Linear(rank, rank) if use_guide_prior else None
        self.hierarchy_embeddings = None
        self.hierarchy_projection = None
        self.hierarchy_mode = None
        self.hierarchy_dropout = float(hierarchy_dropout)
        if not 0.0 <= self.hierarchy_dropout < 1.0:
            raise ValueError("hierarchy_dropout must be in [0, 1)")
        if guide_hierarchy_membership is not None:
            if hierarchy_embedding_dim <= 0:
                raise ValueError("hierarchy_embedding_dim must be positive")
            if guide_hierarchy_membership.ndim != 2 or len(guide_hierarchy_membership) != n_guides:
                raise ValueError("guide_hierarchy_membership must have shape [n_guides, n_nodes]")
            if hierarchy_node_layers is None or len(hierarchy_node_layers) != guide_hierarchy_membership.shape[1]:
                raise ValueError("hierarchy_node_layers must have one entry per hierarchy node")
            self.register_buffer("guide_hierarchy_membership", guide_hierarchy_membership.float())
            self.register_buffer("hierarchy_node_layers", hierarchy_node_layers.long())
            self.hierarchy_embeddings = nn.Embedding(
                guide_hierarchy_membership.shape[1], hierarchy_embedding_dim
            )
            self.hierarchy_projection = nn.Linear(hierarchy_embedding_dim, rank, bias=False)
            n_layers = int(hierarchy_node_layers.max().item()) + 1
            layer_scale = torch.ones(n_layers, dtype=torch.float32)
            if hierarchy_alpha_learnable:
                self.hierarchy_alpha = nn.Parameter(layer_scale)
            else:
                self.register_buffer("hierarchy_alpha", layer_scale)
            self.hierarchy_mode = "go_dag"
        elif guide_parent_targets is not None:
            if hierarchy_embedding_dim <= 0:
                raise ValueError("hierarchy_embedding_dim must be positive")
            if len(guide_parent_targets) != n_guides:
                raise ValueError("guide_parent_targets must have one entry per guide")
            self.register_buffer("guide_parent_targets", guide_parent_targets.long())
            self.hierarchy_embeddings = nn.Embedding(n_targets + 1, hierarchy_embedding_dim)
            self.hierarchy_projection = nn.Linear(hierarchy_embedding_dim, rank, bias=False)
            alpha = torch.ones(2, dtype=torch.float32)
            if hierarchy_alpha_learnable:
                self.hierarchy_alpha = nn.Parameter(alpha)
            else:
                self.register_buffer("hierarchy_alpha", alpha)
            self.hierarchy_mode = "root_target_guide"
        self.gene_programs = nn.Parameter(torch.empty(rank, n_genes))
        self.batch_weights = nn.Embedding(n_batches, rank) if use_batch_calibration else None
        self.n_batches = n_batches
        self.target_batch_weights = (
            nn.Embedding(n_targets * n_batches, rank)
            if use_target_batch_interaction else None
        )
        nn.init.zeros_(self.target_weights.weight)
        if self.guide_residuals is not None:
            nn.init.zeros_(self.guide_residuals.weight)
        if self.guide_prior is not None:
            nn.init.zeros_(self.guide_prior.weight)
            nn.init.zeros_(self.guide_prior.bias)
        if self.hierarchy_embeddings is not None:
            nn.init.normal_(self.hierarchy_embeddings.weight, std=0.01)
            nn.init.normal_(self.hierarchy_projection.weight, std=0.01)
        nn.init.normal_(self.gene_programs, std=0.01)
        if self.batch_weights is not None:
            nn.init.zeros_(self.batch_weights.weight)
        if self.target_batch_weights is not None:
            nn.init.zeros_(self.target_batch_weights.weight)

    def forward(self, target: torch.Tensor, batch: torch.Tensor,
                guide: torch.Tensor | None = None,
                guide_keep_mask: torch.Tensor | None = None,
                target_guide_prior: torch.Tensor | None = None) -> torch.Tensor:
        target_weights = self.target_weights(target)
        weights = target_weights
        if self.hierarchy_embeddings is not None:
            if guide is None:
                raise ValueError("guide ids are required by the hierarchy-conditioned model")
            if self.hierarchy_mode == "go_dag":
                scaled_embeddings = self.hierarchy_embeddings.weight * self.hierarchy_alpha[
                    self.hierarchy_node_layers
                ][:, None]
                hierarchy_features = self.guide_hierarchy_membership[guide] @ scaled_embeddings
            else:
                parent = self.guide_parent_targets[guide]
                root = torch.full_like(parent, self.hierarchy_embeddings.num_embeddings - 1)
                hierarchy_features = (
                    self.hierarchy_alpha[0] * self.hierarchy_embeddings(root)
                    + self.hierarchy_alpha[1] * self.hierarchy_embeddings(parent)
                )
            hierarchy_features = F.dropout(
                hierarchy_features, p=self.hierarchy_dropout, training=self.training
            )
            hierarchy_weights = self.hierarchy_projection(hierarchy_features)
            weights = weights + hierarchy_weights
            self.last_hierarchy_abs_sum = float(
                hierarchy_weights.detach().abs().sum().cpu()
            )
        if target_guide_prior is not None:
            weights = weights + target_guide_prior[target]
        if self.guide_residuals is not None:
            guide_prior = (
                self.guide_prior(target_weights)
                if self.guide_prior is not None else None
            )
            if guide is None and guide_prior is None:
                raise ValueError("guide ids are required by the guide-aware model")
            if guide is None:
                guide_weights = guide_prior
            else:
                guide_weights = self.guide_residuals(guide)
                if guide_keep_mask is not None:
                    guide_weights = guide_weights * guide_keep_mask[:, None]
                if guide_prior is not None:
                    guide_weights = guide_weights + guide_prior
            weights = weights + guide_weights
        if self.batch_weights is not None:
            weights = weights + self.batch_weights(batch)
        if self.target_batch_weights is not None:
            weights = weights + self.target_batch_weights(target * self.n_batches + batch)
        return self.baselines[batch] + weights @ self.gene_programs

    def shrinkage_penalty(self, eps: float = 1e-6) -> torch.Tensor:
        """Smooth group sparsity: jointly prune weak target/program columns."""
        target_scale = torch.sqrt(self.target_weights.weight.square().mean(dim=0) + eps)
        gene_scale = torch.sqrt(self.gene_programs.square().mean(dim=1) + eps)
        penalty = (target_scale * gene_scale).sum()
        if self.batch_weights is not None:
            batch_scale = torch.sqrt(self.batch_weights.weight.square().mean(dim=0) + eps)
            penalty = penalty + (batch_scale * gene_scale).sum()
        if self.guide_residuals is not None:
            guide_scale = torch.sqrt(self.guide_residuals.weight.square().mean(dim=0) + eps)
            penalty = penalty + (guide_scale * gene_scale).sum()
        if self.target_batch_weights is not None:
            interaction_scale = torch.sqrt(
                self.target_batch_weights.weight.square().mean(dim=0) + eps)
            penalty = penalty + (interaction_scale * gene_scale).sum()
        if self.hierarchy_embeddings is not None:
            hierarchy_scale = torch.sqrt(
                self.hierarchy_projection.weight.square().mean(dim=1) + eps
            )
            penalty = penalty + hierarchy_scale.sum()
        return penalty
