from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=8)
def _load_embedding_vectors(embedding_npz: str, gene_ids: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Load a bounded target subset once per process from the immutable NPZ."""
    vectors: dict[str, np.ndarray] = {}
    with np.load(embedding_npz, allow_pickle=False) as embeddings:
        available = set(embeddings.files)
        for gene_id in gene_ids:
            if gene_id in available:
                vector = np.asarray(embeddings[gene_id], dtype=np.float32)
                if vector.ndim != 1:
                    raise ValueError(f"gene embedding for {gene_id} is not a vector")
                vectors[gene_id] = vector
    return vectors


def hashed_gene_symbol_features(targets: list[str], official_genes: list[str], dim: int) -> np.ndarray:
    """Return deterministic test-time target features from gene identity only.

    This intentionally does not use expression, validation labels, or test reference
    values. It is a legal cold-start baseline representation, not a strong biology
    feature substitute.
    """
    if dim <= 0:
        raise ValueError("feature dimension must be positive")
    gene_index = {gene: index for index, gene in enumerate(official_genes)}
    features = np.zeros((len(targets), dim), dtype=np.float32)
    for row, gene in enumerate(targets):
        symbol = str(gene).upper()
        tokens = ["bias", f"symbol:{symbol}"]
        tokens.extend(f"char:{char}" for char in symbol)
        tokens.extend(f"bigram:{symbol[i:i + 2]}" for i in range(max(0, len(symbol) - 1)))
        if gene in gene_index:
            index = gene_index[gene]
            tokens.extend([
                "in_official_gene_panel",
                f"gene_index_bucket:{index // 128}",
                f"gene_index_mod:{index % 257}",
            ])
            features[row, 0] = (index + 1) / max(len(official_genes), 1)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            col = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            features[row, col] += sign
        norm = float(np.linalg.norm(features[row]))
        if norm > 0:
            features[row] /= norm
    return features


def projected_gene_embedding_features(
    targets: list[str],
    symbol_to_gene_id: dict[str, str],
    embedding_npz: str,
    dim: int,
    seed: int = 20260908,
) -> tuple[np.ndarray, dict]:
    """Project fixed gene embeddings into the cold-start target feature space.

    The embedding file and symbol-to-gene-id mapping must be fixed public/runtime
    assets. This function never reads expression values or validation/test labels;
    unmapped targets receive an all-zero vector and are reported to the caller.
    """
    if dim <= 0:
        raise ValueError("feature dimension must be positive")
    rng = np.random.default_rng(seed)
    projection: np.ndarray | None = None
    features = np.zeros((len(targets), dim), dtype=np.float32)
    mapped = []
    unmapped = []
    gene_ids = tuple(sorted({gene_id for target in targets if (gene_id := symbol_to_gene_id.get(target))}))
    vectors = _load_embedding_vectors(embedding_npz, gene_ids)
    for row, target in enumerate(targets):
        gene_id = symbol_to_gene_id.get(target)
        if gene_id is None or gene_id not in vectors:
            unmapped.append(target)
            continue
        vec = vectors[gene_id]
        if projection is None:
            projection = rng.normal(
                loc=0.0,
                scale=1.0 / np.sqrt(float(vec.shape[0])),
                size=(vec.shape[0], dim),
            ).astype(np.float32)
        if vec.shape[0] != projection.shape[0]:
            raise ValueError(f"gene embedding dimension mismatch for {target}")
        projected = vec @ projection
        norm = float(np.linalg.norm(projected))
        if norm > 0:
            projected = projected / norm
        features[row] = projected.astype(np.float32)
        mapped.append(target)
    return features, {
        "type": "projected_fixed_gene_embedding_features",
        "feature_dim": dim,
        "source_embedding": embedding_npz,
        "projection": "deterministic_gaussian_random_projection",
        "projection_seed": seed,
        "targets": len(targets),
        "mapped_targets": len(mapped),
        "unmapped_targets": len(unmapped),
        "coverage": len(mapped) / max(len(targets), 1),
        "unmapped_sample": unmapped[:20],
        "uses_expression": False,
        "test_time_available": True,
    }



def pca_gene_embedding_features(
    targets: list[str],
    fit_targets: list[str],
    symbol_to_gene_id: dict[str, str],
    embedding_npz: str,
    dim: int,
) -> tuple[np.ndarray, dict]:
    """Project fixed embeddings through PCA fitted only on training targets."""
    if dim <= 0:
        raise ValueError("feature dimension must be positive")
    target_ids = {
        target: symbol_to_gene_id[target]
        for target in sorted(set(targets) | set(fit_targets))
        if target in symbol_to_gene_id
    }
    loaded = _load_embedding_vectors(embedding_npz, tuple(sorted(set(target_ids.values()))))
    vectors = {target: loaded[gene_id] for target, gene_id in target_ids.items() if gene_id in loaded}
    fit = [vectors[target] for target in fit_targets if target in vectors]
    if len(fit) < 2:
        raise ValueError("PCA requires at least two mapped training targets")
    fit_matrix = np.asarray(fit, dtype=np.float32)
    mean = fit_matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(fit_matrix - mean, full_matrices=False)
    components = vt[: min(dim, vt.shape[0])].T
    features = np.zeros((len(targets), dim), dtype=np.float32)
    mapped = []
    for row, target in enumerate(targets):
        if target not in vectors:
            continue
        projected = (vectors[target][None, :] - mean) @ components
        features[row, : projected.shape[1]] = projected[0]
        norm = float(np.linalg.norm(features[row]))
        if norm > 0:
            features[row] /= norm
        mapped.append(target)
    return features, {
        "type": "train_target_pca_fixed_gene_embedding_features",
        "feature_dim": dim,
        "fitted_components": int(components.shape[1]),
        "fit_targets": len(fit_targets),
        "mapped_fit_targets": len(fit),
        "mapped_targets": len(mapped),
        "uses_expression": False,
        "uses_validation_labels": False,
        "uses_test_expression": False,
        "test_time_available": True,
    }



def go_bp_hash_features(targets: list[str], node_table: str, edge_table: str, dim: int) -> tuple[np.ndarray, dict]:
    """Build deterministic GO-BP target features from a materialized GO DAG.

    The graph must already be fixed before model training/inference. Targets with
    no direct GO annotation get an all-zero biological vector and are reported as
    unmapped; callers must decide whether that coverage is scientifically usable.
    """
    import csv
    from collections import defaultdict, deque

    if dim <= 0:
        raise ValueError("feature dimension must be positive")
    labels: dict[str, tuple[str, str]] = {}
    with open(node_table, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            labels[row["node_id"]] = (row["node_type"], row["label"])
    parents: dict[str, list[tuple[str, str]]] = defaultdict(list)
    direct_target_terms: dict[str, list[str]] = defaultdict(list)
    with open(edge_table, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            parent, child, edge_type = row["parent_id"], row["child_id"], row["edge_type"]
            parents[child].append((parent, edge_type))
            if child.startswith("target:") and edge_type == "annotates_target":
                direct_target_terms[child.split(":", 1)[1]].append(parent)
    features = np.zeros((len(targets), dim), dtype=np.float32)
    mapped = []
    unmapped = []
    for row, target in enumerate(targets):
        direct = direct_target_terms.get(target, [])
        if not direct:
            unmapped.append(target)
            continue
        mapped.append(target)
        seen = set()
        queue = deque(direct)
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            for parent, edge_type in parents.get(node, []):
                if edge_type in {"go_parent", "ontology_root"}:
                    queue.append(parent)
        for node in seen:
            digest = hashlib.sha256(f"go:{node}".encode("utf-8")).digest()
            col = int.from_bytes(digest[:4], "little") % dim
            features[row, col] += 1.0 / max(len(seen), 1)
        norm = float(np.linalg.norm(features[row]))
        if norm > 0:
            features[row] /= norm
    return features, {
        "type": "go_bp_materialized_graph_hash_features",
        "feature_dim": dim,
        "targets": len(targets),
        "mapped_targets": len(mapped),
        "unmapped_targets": len(unmapped),
        "coverage": len(mapped) / max(len(targets), 1),
        "unmapped_sample": unmapped[:20],
        "uses_expression": False,
        "test_time_available": True,
    }


def go_bp_ic_features(
    targets: list[str],
    fit_targets: list[str],
    node_table: str,
    edge_table: str,
    dim: int,
) -> tuple[np.ndarray, dict]:
    """Hash GO-BP ancestors with information content fitted on train targets only."""
    import csv
    import math
    from collections import defaultdict, deque

    if dim <= 0:
        raise ValueError("feature dimension must be positive")
    parents: dict[str, list[str]] = defaultdict(list)
    direct: dict[str, list[str]] = defaultdict(list)
    with open(edge_table, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            parent, child, edge_type = row["parent_id"], row["child_id"], row["edge_type"]
            if edge_type in {"go_parent", "ontology_root"}:
                parents[child].append(parent)
            elif child.startswith("target:") and edge_type == "annotates_target":
                direct[child.split(":", 1)[1]].append(parent)

    def ancestors(target: str) -> set[str]:
        seen: set[str] = set()
        queue = deque(direct.get(target, []))
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            queue.extend(parents.get(node, []))
        return seen

    target_terms = {target: ancestors(target) for target in set(targets) | set(fit_targets)}
    document_frequency: dict[str, int] = defaultdict(int)
    for target in fit_targets:
        for term in target_terms[target]:
            document_frequency[term] += 1
    n_fit = max(len(fit_targets), 1)
    information_content = {
        term: math.log((n_fit + 1.0) / (count + 1.0))
        for term, count in document_frequency.items()
    }
    features = np.zeros((len(targets), dim), dtype=np.float32)
    mapped = []
    for row, target in enumerate(targets):
        terms = target_terms[target]
        if not terms:
            continue
        mapped.append(target)
        for term in terms:
            weight = information_content.get(term, 0.0)
            digest = hashlib.sha256(f"go-ic:{term}".encode("utf-8")).digest()
            col = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            features[row, col] += sign * weight
        norm = float(np.linalg.norm(features[row]))
        if norm > 0:
            features[row] /= norm
    return features, {
        "type": "train_only_information_content_weighted_go_bp",
        "feature_dim": dim,
        "fit_targets": len(fit_targets),
        "targets": len(targets),
        "mapped_targets": len(mapped),
        "coverage": len(mapped) / max(len(targets), 1),
        "uses_expression": False,
        "uses_validation_labels": False,
        "uses_test_expression": False,
        "test_time_available": True,
    }
