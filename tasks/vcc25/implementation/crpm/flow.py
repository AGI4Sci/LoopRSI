from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


class ConditionalFlow(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_targets: int,
        n_guides: int,
        n_batches: int,
        condition_dim: int,
        hidden_dim: int,
        depth: int,
        hierarchy_membership: torch.Tensor | None = None,
        hierarchy_node_layers: torch.Tensor | None = None,
        hierarchy_embedding_dim: int = 32,
        hierarchy_dropout: float = 0.0,
        hierarchy_alpha_learnable: bool = True,
    ):
        super().__init__()
        self.target_embedding = nn.Embedding(n_targets, condition_dim)
        self.guide_embedding = nn.Embedding(n_guides, condition_dim)
        self.batch_embedding = nn.Embedding(n_batches, condition_dim)
        self.hierarchy_embeddings = None
        self.hierarchy_projection = None
        self.hierarchy_dropout = float(hierarchy_dropout)
        if hierarchy_membership is not None:
            if hierarchy_node_layers is None:
                raise ValueError("hierarchy_node_layers are required")
            self.register_buffer("hierarchy_membership", hierarchy_membership.float())
            self.register_buffer("hierarchy_node_layers", hierarchy_node_layers.long())
            self.hierarchy_embeddings = nn.Embedding(
                hierarchy_membership.shape[1], hierarchy_embedding_dim
            )
            self.hierarchy_projection = nn.Linear(
                hierarchy_embedding_dim, condition_dim, bias=False
            )
            scales = torch.ones(int(hierarchy_node_layers.max().item()) + 1)
            if hierarchy_alpha_learnable:
                self.hierarchy_alpha = nn.Parameter(scales)
            else:
                self.register_buffer("hierarchy_alpha", scales)
        blocks: list[nn.Module] = [nn.Linear(n_genes + condition_dim + 1, hidden_dim), nn.SiLU()]
        for _ in range(max(depth - 1, 0)):
            blocks.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        blocks.append(nn.Linear(hidden_dim, n_genes))
        self.velocity = nn.Sequential(*blocks)
        for embedding in (self.target_embedding, self.guide_embedding, self.batch_embedding):
            nn.init.normal_(embedding.weight, std=0.01)
        if self.hierarchy_embeddings is not None:
            nn.init.normal_(self.hierarchy_embeddings.weight, std=0.01)
            nn.init.normal_(self.hierarchy_projection.weight, std=0.01)

    def condition(self, target: torch.Tensor, guide: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        condition = (
            self.target_embedding(target)
            + self.guide_embedding(guide)
            + self.batch_embedding(batch)
        )
        if self.hierarchy_embeddings is not None:
            scaled = self.hierarchy_embeddings.weight * self.hierarchy_alpha[
                self.hierarchy_node_layers
            ][:, None]
            features = self.hierarchy_membership[guide] @ scaled
            features = nn.functional.dropout(
                features, p=self.hierarchy_dropout, training=self.training
            )
            hierarchy_condition = self.hierarchy_projection(features)
            condition = condition + hierarchy_condition
            self.last_hierarchy_abs_sum = float(
                hierarchy_condition.detach().abs().sum().cpu()
            )
        return condition

    def forward(
        self,
        x_t: torch.Tensor,
        time: torch.Tensor,
        target: torch.Tensor,
        guide: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.condition(target, guide, batch)
        return self.velocity(torch.cat((x_t, time[:, None], condition), dim=1))


@dataclass
class FlowResult:
    predictions: np.ndarray
    history: list[float]
    optimizer_steps: int
    fit_samples: int
    diagnostics: dict


def train_conditional_flow(
    *,
    x_train: np.ndarray,
    target_train: np.ndarray,
    batch_train: np.ndarray,
    guide_train: np.ndarray,
    target_val: np.ndarray,
    batch_val: np.ndarray,
    guide_val: np.ndarray,
    baselines: np.ndarray,
    n_targets: int,
    n_guides: int,
    n_batches: int,
    device: torch.device,
    seed: int,
    epochs: int,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    aggregate_count_power: float,
    condition_dim: int,
    hidden_dim: int,
    depth: int,
    inference_steps: int,
    hierarchy_membership: np.ndarray | None = None,
    hierarchy_node_layers: np.ndarray | None = None,
    hierarchy_embedding_dim: int = 32,
    hierarchy_dropout: float = 0.0,
    hierarchy_alpha_learnable: bool = True,
    train_initial: np.ndarray | None = None,
    val_initial: np.ndarray | None = None,
) -> FlowResult:
    keys = np.stack((target_train, guide_train, batch_train), axis=1)
    order = np.lexsort((batch_train, guide_train, target_train))
    sorted_keys = keys[order]
    starts = np.flatnonzero(np.r_[True, np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)])
    counts = np.diff(np.r_[starts, len(order)]).astype(np.float32)
    x_fit = np.add.reduceat(x_train[order], starts, axis=0) / counts[:, None]
    group_keys = sorted_keys[starts]
    target_fit = group_keys[:, 0].astype(np.int64)
    guide_fit = group_keys[:, 1].astype(np.int64)
    batch_fit = group_keys[:, 2].astype(np.int64)
    if train_initial is None:
        x0_fit = baselines[batch_fit]
    else:
        if train_initial.shape != x_train.shape:
            raise ValueError("train_initial must have the same shape as x_train")
        x0_fit = np.add.reduceat(train_initial[order], starts, axis=0) / counts[:, None]
    if val_initial is not None and val_initial.shape != (len(target_val), x_train.shape[1]):
        raise ValueError("val_initial has an invalid shape")

    membership_tensor = (
        torch.from_numpy(hierarchy_membership).to(device)
        if hierarchy_membership is not None else None
    )
    layers_tensor = (
        torch.from_numpy(hierarchy_node_layers).to(device)
        if hierarchy_node_layers is not None else None
    )
    model = ConditionalFlow(
        x_fit.shape[1], n_targets, n_guides, n_batches,
        condition_dim, hidden_dim, depth,
        membership_tensor, layers_tensor,
        hierarchy_embedding_dim, hierarchy_dropout,
        hierarchy_alpha_learnable,
    ).to(device)
    initial = torch.cat([parameter.detach().flatten() for parameter in model.parameters()]).clone()
    hierarchy_initial = None
    if model.hierarchy_embeddings is not None:
        hierarchy_initial = torch.cat([
            model.hierarchy_embeddings.weight.detach().flatten(),
            model.hierarchy_projection.weight.detach().flatten(),
            model.hierarchy_alpha.detach().flatten(),
        ]).clone()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    dataset = TensorDataset(
        torch.from_numpy(x0_fit.astype(np.float32)),
        torch.from_numpy(x_fit.astype(np.float32)),
        torch.from_numpy(target_fit),
        torch.from_numpy(guide_fit),
        torch.from_numpy(batch_fit),
    )
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    shuffle = True
    if aggregate_count_power > 0:
        weights = np.power(counts.astype(np.float64), aggregate_count_power)
        weights /= max(float(weights.mean()), 1e-12)
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights), len(dataset), replacement=True, generator=generator
        )
        shuffle = False
    history: list[float] = []
    optimizer_steps = 0
    activation_events = 0
    affected_rows = 0
    velocity_abs_sum = 0.0
    hierarchy_contribution_abs_sum = 0.0
    gradient_norm_max = 0.0
    stop = False
    for _ in range(epochs):
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
            generator=generator,
        )
        running = 0.0
        seen = 0
        model.train()
        for x0, x1, target, guide, batch in loader:
            x0, x1 = x0.to(device), x1.to(device)
            target, guide, batch = target.to(device), guide.to(device), batch.to(device)
            time = torch.rand(len(x0), device=device)
            x_t = x0 + time[:, None] * (x1 - x0)
            desired_velocity = x1 - x0
            optimizer.zero_grad(set_to_none=True)
            velocity = model(x_t, time, target, guide, batch)
            loss = (velocity - desired_velocity).square().mean()
            loss.backward()
            grad_sq = sum(
                float(parameter.grad.detach().square().sum().cpu())
                for parameter in model.parameters() if parameter.grad is not None
            )
            gradient_norm_max = max(gradient_norm_max, grad_sq ** 0.5)
            optimizer.step()
            activation_events += 1
            affected_rows += len(x0)
            velocity_abs_sum += float(velocity.detach().abs().sum().cpu())
            hierarchy_contribution_abs_sum += float(
                getattr(model, "last_hierarchy_abs_sum", 0.0)
            )
            running += float(loss.detach()) * len(x0)
            seen += len(x0)
            optimizer_steps += 1
            if max_steps > 0 and optimizer_steps >= max_steps:
                stop = True
                break
        history.append(running / max(seen, 1))
        if stop:
            break

    model.eval()
    predictions = []
    trajectory_displacement_abs_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(target_val), batch_size):
            target = torch.from_numpy(target_val[start:start + batch_size]).to(device)
            guide = torch.from_numpy(guide_val[start:start + batch_size]).to(device)
            batch = torch.from_numpy(batch_val[start:start + batch_size]).to(device)
            initial_chunk = (
                baselines[batch_val[start:start + batch_size]]
                if val_initial is None else val_initial[start:start + batch_size]
            )
            state = torch.from_numpy(initial_chunk.astype(np.float32, copy=False)).to(device)
            initial_state = state.clone()
            for step in range(inference_steps):
                time = torch.full(
                    (len(state),), (step + 0.5) / inference_steps, device=device
                )
                state = state + model(state, time, target, guide, batch) / inference_steps
            trajectory_displacement_abs_sum += float(
                (state - initial_state).abs().sum().cpu()
            )
            predictions.append(state.cpu().numpy())
    final = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
    diagnostics = {
        "activation_events": activation_events,
        "affected_rows": affected_rows,
        "transformed_feature_count": x_fit.shape[1],
        "gradient_norm_max": gradient_norm_max,
        "parameter_update_l2": float(torch.linalg.vector_norm(final - initial).cpu()),
        "velocity_abs_sum": velocity_abs_sum,
        "trajectory_displacement_abs_sum": trajectory_displacement_abs_sum,
    }
    if model.hierarchy_embeddings is not None:
        hierarchy_final = torch.cat([
            model.hierarchy_embeddings.weight.detach().flatten(),
            model.hierarchy_projection.weight.detach().flatten(),
            model.hierarchy_alpha.detach().flatten(),
        ])
        diagnostics.update({
            "hierarchy_parameter_update_l2": float(
                torch.linalg.vector_norm(hierarchy_final - hierarchy_initial).cpu()
            ),
            "hierarchy_contribution_abs_sum": hierarchy_contribution_abs_sum,
            "hierarchy_embedding_norm": float(
                torch.linalg.vector_norm(model.hierarchy_embeddings.weight.detach()).cpu()
            ),
            "hierarchy_layer_scales": model.hierarchy_alpha.detach().cpu().tolist(),
        })
    return FlowResult(
        predictions=np.concatenate(predictions),
        history=history,
        optimizer_steps=optimizer_steps,
        fit_samples=len(dataset),
        diagnostics=diagnostics,
    )
