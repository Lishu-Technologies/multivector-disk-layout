"""Offline document placement from calibration shortlists and document means."""
import numpy as np


def normalized(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)


def traffic_features(n_docs, traffic):
    if len(traffic) == 0:
        raise ValueError("At least one calibration shortlist is required")
    signatures = np.zeros((n_docs, len(traffic)), dtype=np.float32)
    for col, row in enumerate(traffic):
        row = np.asarray(row)
        if row.ndim != 1 or row.dtype.kind not in "iu" or not len(row) or row.min()<0 or row.max()>=n_docs:
            raise ValueError("Invalid calibration document IDs")
        signatures[row,col] = 1
    frequency = signatures.sum(axis=1)
    # Diagonalize the small query-query Gram matrix on the research-scale traces.
    if len(traffic)>32:
        _, axes = np.linalg.eigh(signatures.T @ signatures)
        signatures = signatures @ axes[:,-32:]
    return normalized(signatures), frequency


def local_path(ids, features):
    """Bisect large groups, then greedily put similar documents next to one another."""
    if len(ids)<2:
        return ids
    points = features[ids]
    if len(ids)>384:
        center = points.mean(axis=0)
        a = np.argmax(np.sum((points-center)**2,axis=1))
        b = np.argmax(np.sum((points-points[a])**2,axis=1))
        projection = points @ (points[b]-points[a])
        split = ids[np.lexsort((ids,projection))]
        m = len(ids)//2
        return np.r_[local_path(split[:m],features),local_path(split[m:],features)]
    similarities = points @ points.T
    used = np.zeros(len(ids),bool)
    answer = []
    j = int(np.argmin(points @ points.mean(axis=0)))
    for _ in range(len(ids)):
        answer.append(ids[j])
        used[j] = True
        j = int(np.argmax(np.where(used,-np.inf,similarities[j])))
    return np.asarray(answer,dtype=np.int64)


def layout_order(means, lengths, traffic, method="blend"):
    lengths = np.asarray(lengths,dtype=np.int64)
    n = len(lengths)
    if len(means)!=n or not n or np.any(lengths<=0):
        raise ValueError("Document lengths and means must cover the same non-empty corpus")
    if method=="identity":
        return np.arange(n)
    signatures,frequency = traffic_features(n,traffic)
    ids = np.arange(n)
    popularity = np.lexsort((ids,-frequency))
    if method=="frequency":
        return popularity
    semantic = normalized(normalized(means)-normalized(means).mean(axis=0))
    if method=="semantic":
        features = semantic
    elif method=="traffic":
        features = signatures
    elif method=="blend":
        features = np.column_stack((semantic*np.float32(.7), signatures))
    else:
        raise ValueError(f"Unknown layout {method}")
    # Isolate frequently cached documents so they do not separate two cold records.
    hot = np.cumsum(lengths[popularity]) <= .10*lengths.sum()
    pieces = [local_path(np.sort(popularity[mask]),features) for mask in (hot,~hot)]
    return np.concatenate(pieces)


def physical_starts(order, offsets):
    if not np.array_equal(np.sort(order),np.arange(len(offsets)-1)):
        raise ValueError("Layout must be a permutation of all document IDs")
    starts = np.empty(len(order),dtype=np.int64)
    starts[order] = np.r_[0,np.cumsum(np.diff(offsets)[order][:-1])]
    return starts
