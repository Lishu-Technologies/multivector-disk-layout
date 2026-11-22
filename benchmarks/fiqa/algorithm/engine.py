"""Deterministic top-k inside a supplied shortlist using compact support-function bounds."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import nnls


def cap(cosine, height, radius):
    """Max along a unit query over a ball clipped by one half-space."""
    c = np.clip(cosine, -1.0, 1.0)
    h = np.clip(height, -radius, radius)
    tangent = np.sqrt(np.maximum(0.0, (1-c)*(1+c)))
    rim = np.sqrt(np.maximum(0.0, (radius-h)*(radius+h)))
    return np.where(radius*c <= h, radius, c*h+tangent*rim)


@dataclass
class Summary:
    anchors: np.ndarray
    heights: np.ndarray
    radii: np.ndarray
    scale: float

    def bounds(self, query, candidates, combine=True):
        """Only visit indexed summaries. No raw document vectors are accepted here."""
        q = np.asarray(query, dtype=np.float64)
        norms = np.linalg.norm(q, axis=1)
        directions = q / np.maximum(norms[:,None], 1e-300)
        # Indexed anchors retain their actual norms after storage as float32.
        anchors = self.anchors.astype(np.float64)
        anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)
        similarities = directions @ anchors.T
        neighbors = min(12,len(anchors))
        selected = np.argpartition(-similarities, neighbors-1, axis=1)[:,:neighbors]
        radii = self.radii[candidates].astype(np.float64)
        upper = np.zeros(len(candidates))
        approx = np.zeros(len(candidates))
        for t, (direction, ids) in enumerate(zip(directions,selected)):
            if norms[t] == 0:
                continue
            angles = similarities[t,ids]
            values = self.heights[np.ix_(candidates,ids)].astype(np.float64)
            values = values*(2*self.scale/255)-self.scale
            bound = cap(angles[None,:],values,radii[:,None]).min(axis=1)
            approx += norms[t]*values[:,np.argmax(angles)]
            if combine:
                # A nonnegative combination preserves the one-sided inequality.
                # Normalize the combined anchor, then intersect with the ball again.
                local = anchors[ids].T
                weights, _ = nnls(local,direction)
                axis = local @ weights
                length = np.linalg.norm(axis)
                if length > 1e-8:
                    cosine = float(direction @ axis / length)
                    combined = cap(cosine, values @ weights/length, radii)
                    bound = np.minimum(bound,combined)
            upper += norms[t]*bound
        # Outward guard far larger than rounding of FP64 norm/cap operations
        # for the bounded unit-scale data checked by the builder.
        return upper + 0.00001*(1+norms.sum()), approx


def stable_top(ids, scores, k):
    return np.lexsort((ids,-scores))[:k]


def safe_shortlist(candidates, upper, approx, k, scorer):
    """Read k seed documents, then batch all remaining documents that might beat kth."""
    ids = np.asarray(candidates,dtype=np.int64)
    if len(ids)!=len(np.unique(ids)) or not 0<k<=len(ids):
        raise ValueError("Expected a unique candidate set with 0 < k <= its size")
    scores = np.full(len(ids),-np.inf)
    seed = stable_top(ids,approx,k)
    scores[seed] = scorer(ids[seed])
    threshold = scores[seed].min()
    keep = upper>=threshold
    keep[seed] = False
    rows = np.flatnonzero(keep)
    if len(rows):
        scores[rows] = scorer(ids[rows])
    result = ids[stable_top(ids,scores,k)]
    return result, np.r_[seed,rows]
