from __future__ import annotations

import numpy as np


def _repeat(vector: np.ndarray, output_length: int) -> np.ndarray:
    return np.repeat(vector[None, :], output_length, axis=0).astype(np.float32)


def predict_persistence(context: np.ndarray, output_length: int, shift_map: np.ndarray) -> np.ndarray:
    del shift_map
    return _repeat(context[-1], output_length)


def predict_moving_average(context: np.ndarray, output_length: int, shift_map: np.ndarray) -> np.ndarray:
    del shift_map
    window = context[-6:] if context.shape[0] >= 6 else context
    return _repeat(window.mean(axis=0), output_length)


def predict_neighbor_last(context: np.ndarray, output_length: int, shift_map: np.ndarray) -> np.ndarray:
    neighbor_last = context[-1, shift_map]
    return _repeat(neighbor_last, output_length)


def predict_linear_trend(context: np.ndarray, output_length: int, shift_map: np.ndarray) -> np.ndarray:
    del shift_map
    tail = context[-6:] if context.shape[0] >= 6 else context
    if tail.shape[0] <= 1:
        return _repeat(tail[-1], output_length)
    slope = np.diff(tail, axis=0).mean(axis=0)
    last = tail[-1]
    steps = np.arange(1, output_length + 1, dtype=np.float32)[:, None]
    return (last[None, :] + steps * slope[None, :]).astype(np.float32)


BASELINE_FUNCTIONS = {
    "persistence": predict_persistence,
    "moving_average": predict_moving_average,
    "neighbor_last": predict_neighbor_last,
    "linear_trend": predict_linear_trend,
}
