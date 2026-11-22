"""Plan a file order from document means, token counts and calibration shortlists."""
import os
from pathlib import Path
import tempfile

import numpy as np


def _ids(values, count):
    ids = np.asarray(values)
    if ids.dtype.kind not in "iu" and ids.size:
        raise ValueError("Document IDs must be integers")
    if ids.size and (ids.min() < 0 or ids.max() >= count):
        raise ValueError("Document ID outside this snapshot")
    return ids.astype(np.int64, copy=False)


class DiskLayout:
    """Map logical document rows to dense storage IDs and back."""

    def __init__(self, order):
        order = np.asarray(order)
        if order.ndim != 1 or not order.size:
            raise ValueError("Expected a nonempty permutation")
        order = _ids(order, len(order)).copy()
        if not np.array_equal(np.sort(order), np.arange(len(order))):
            raise ValueError("Storage order must contain each logical row exactly once")
        self._order = order
        self._inverse = np.argsort(order)
        self._order.flags.writeable = False
        self._inverse.flags.writeable = False

    def storage_ids(self, logical_rows):
        return self._inverse[_ids(logical_rows, self._order.size)]

    def logical_ids(self, storage_rows):
        return self._order[_ids(storage_rows, self._order.size)]

    def __len__(self):
        return self._order.size

    def save(self, path):
        """Publish a new snapshot atomically without replacing an existing ID map."""
        path = Path(path)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
            temp_path = Path(tmp.name)
        try:
            with temp_path.open("wb") as out:
                np.savez_compressed(out, version=np.int64(1), order=self._order)
            os.link(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            if saved["version"].shape != () or saved["version"].item() != 1:
                raise ValueError("Unsupported layout version")
            return cls(saved["order"])


def _normalized(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)


def _local_path(ids, features):
    """Bisect large groups, then order each leaf by greedy cosine similarity."""
    if len(ids) < 2:
        return ids
    points = features[ids]
    center = points.mean(axis=0)
    if len(ids) > 384:
        first = np.argmax(np.sum((points - center) ** 2, axis=1))
        last = np.argmax(np.sum((points - points[first]) ** 2, axis=1))
        projection = points @ (points[last] - points[first])
        split = ids[np.lexsort((ids, projection))]
        middle = len(split) // 2
        return np.concatenate((_local_path(split[:middle], features),
                               _local_path(split[middle:], features)))
    similarity = points @ points.T
    visited = np.zeros(len(ids), dtype=bool)
    order = np.empty_like(ids)
    current = np.argmin(points @ center)
    for step in range(len(ids)):
        order[step] = ids[current]
        visited[current] = True
        current = np.argmax(np.where(visited, -np.inf, similarity[current]))
    return order


def fit_layout(tokens, lengths, shortlists):
    """Fit on historical candidate lists. Input rows are unpadded document tokens."""
    tokens = np.asarray(tokens)
    lengths = np.asarray(lengths)
    if (tokens.ndim != 2 or tokens.shape[1] == 0 or tokens.dtype != np.float16
            or lengths.ndim != 1 or lengths.size == 0 or lengths.dtype.kind not in "iu"
            or np.any(lengths <= 0) or sum(map(int, lengths)) != len(tokens)):
        raise ValueError("Expected FP16 [tokens, dim] and positive document lengths covering all rows")
    offsets = np.r_[0, np.cumsum(lengths, dtype=np.int64)]
    means = np.asarray([tokens[a:b].mean(axis=0, dtype=np.float32)
                        for a, b in zip(offsets, offsets[1:])], dtype=np.float64)
    if not np.isfinite(means).all():
        raise ValueError("Expected finite token vectors")
    means /= np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-12)
    means = _normalized(means)
    features = _normalized(means - means.mean(axis=0))
    frequency = np.zeros(len(lengths), dtype=np.int64)
    for shortlist in shortlists:
        ids = _ids(shortlist, len(lengths))
        if ids.ndim != 1 or ids.size == 0:
            raise ValueError("Each calibration shortlist must be a nonempty vector of document IDs")
        frequency[np.unique(ids)] += 1
    if not frequency.any():
        raise ValueError("At least one calibration shortlist is needed")
    popular = np.lexsort((np.arange(len(lengths)), -frequency))
    hot = np.cumsum(lengths[popular]) <= 0.10 * lengths.sum()
    order = np.concatenate([_local_path(np.sort(popular[mask]), features) for mask in (hot, ~hot)])
    return DiskLayout(order)
