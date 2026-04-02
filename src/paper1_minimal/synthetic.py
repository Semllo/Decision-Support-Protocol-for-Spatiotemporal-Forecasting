from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticConfig:
    num_nodes: int = 12
    num_steps: int = 24 * 28
    graph_seed: int = 20260402
    train_ratio: float = 0.6
    val_ratio: float = 0.2


def row_normalize(adj: np.ndarray) -> np.ndarray:
    row_sum = adj.sum(axis=1, keepdims=True)
    return np.divide(adj, row_sum, out=np.zeros_like(adj), where=row_sum > 0)


def build_graph(num_nodes: int, graph_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(graph_seed)
    positions = rng.uniform(0.0, 1.0, size=(num_nodes, 2)).astype(np.float32)
    distances = np.sqrt(((positions[:, None, :] - positions[None, :, :]) ** 2).sum(axis=2)).astype(
        np.float32
    )
    np.fill_diagonal(distances, np.inf)

    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for node_idx in range(num_nodes):
        nearest = np.argsort(distances[node_idx])[:3]
        local_scale = float(np.mean(distances[node_idx, nearest])) + 1e-6
        for nbr_idx in nearest:
            weight = float(np.exp(-distances[node_idx, nbr_idx] / local_scale))
            adj[node_idx, nbr_idx] = max(adj[node_idx, nbr_idx], weight)
            adj[nbr_idx, node_idx] = max(adj[nbr_idx, node_idx], weight)

    shift_map = np.argmin(distances, axis=1).astype(np.int32)
    return positions, adj, shift_map


def build_split_labels(num_steps: int, train_ratio: float, val_ratio: float) -> np.ndarray:
    train_end = int(num_steps * train_ratio)
    val_end = int(num_steps * (train_ratio + val_ratio))
    labels = np.full(num_steps, "test", dtype=object)
    labels[:train_end] = "train"
    labels[train_end:val_end] = "val"
    return labels


def simulate_clean_signal(num_steps: int, adj_norm: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    num_nodes = adj_norm.shape[0]
    steps = np.arange(num_steps, dtype=np.float32)

    daily_amp = rng.uniform(2.5, 6.5, size=num_nodes).astype(np.float32)
    weekly_amp = rng.uniform(0.8, 2.2, size=num_nodes).astype(np.float32)
    daily_phase = rng.uniform(0.0, 2.0 * np.pi, size=num_nodes).astype(np.float32)
    weekly_phase = rng.uniform(0.0, 2.0 * np.pi, size=num_nodes).astype(np.float32)
    node_bias = rng.normal(0.0, 1.8, size=num_nodes).astype(np.float32)

    seasonal = (
        daily_amp[None, :] * np.sin((2.0 * np.pi * steps[:, None] / 24.0) + daily_phase[None, :])
        + weekly_amp[None, :]
        * np.sin((2.0 * np.pi * steps[:, None] / (24.0 * 7.0)) + weekly_phase[None, :])
    ).astype(np.float32)

    shocks = np.zeros((num_steps, num_nodes), dtype=np.float32)
    num_events = max(6, num_steps // 96)
    for _ in range(num_events):
        start = int(rng.integers(12, max(13, num_steps - 18)))
        duration = int(rng.integers(4, 18))
        center = int(rng.integers(0, num_nodes))
        amplitude = float(rng.uniform(4.0, 10.0))

        influence = np.zeros(num_nodes, dtype=np.float32)
        influence[center] = 1.0
        walk = influence.copy()
        spatial_profile = influence.copy()
        for hop in range(1, 3):
            walk = adj_norm @ walk
            spatial_profile += (0.7**hop) * walk
        spatial_profile /= spatial_profile.max() + 1e-6

        for t in range(start, min(num_steps, start + duration)):
            decay = np.exp(-(t - start) / max(duration / 3.0, 1.0))
            shocks[t] += np.float32(amplitude * decay) * spatial_profile

    signal = np.zeros((num_steps, num_nodes), dtype=np.float32)
    signal[0] = 15.0 + node_bias + seasonal[0] + rng.normal(0.0, 0.8, size=num_nodes).astype(np.float32)
    for t in range(1, num_steps):
        diffusion = adj_norm @ signal[t - 1]
        noise = rng.normal(0.0, 0.7, size=num_nodes).astype(np.float32)
        signal[t] = (
            0.72 * signal[t - 1]
            + 0.14 * diffusion
            + 0.18 * seasonal[t]
            + shocks[t]
            + noise
        ).astype(np.float32)
    return np.clip(signal, 0.0, None).astype(np.float32)


def materialize_references(clean: np.ndarray, shift_map: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    num_steps, num_nodes = clean.shape

    references: dict[str, np.ndarray] = {"clean": clean.copy()}

    lag_p1 = np.full_like(clean, np.nan)
    lag_p1[:-1] = clean[1:]
    references["lag_p1"] = lag_p1

    lag_p3 = np.full_like(clean, np.nan)
    lag_p3[:-3] = clean[3:]
    references["lag_p3"] = lag_p3

    references["node_shift_1hop"] = clean[:, shift_map].copy()

    support_drop = clean.copy()
    dropped_nodes = rng.choice(num_nodes, size=max(1, int(round(num_nodes * 0.3))), replace=False)
    support_drop[:, dropped_nodes] = np.nan
    references["support_drop_30"] = support_drop

    missing = clean.copy()
    missing_mask = rng.random(missing.shape) < 0.3
    missing[missing_mask] = np.nan
    references["missing_30"] = missing
    return references


def make_timestamps(num_steps: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01 00:00:00", periods=num_steps, freq="h")
