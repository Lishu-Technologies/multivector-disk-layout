"""Compare the installed release planner against all archived FiQA storage IDs."""
import argparse
from pathlib import Path

import numpy as np

from multivector_disk_layout import fit_layout


def run(corpus, archive):
    tokens = np.load(corpus / "documents.npy", mmap_mode="r")
    lengths = np.load(corpus / "doclens.npy")
    queries = np.load(corpus / "queries.npy")
    query_lengths = np.load(corpus / "query_lens.npy")
    offsets = np.r_[0, lengths.cumsum(dtype=np.int64)]
    means = np.asarray([tokens[a:b].mean(axis=0, dtype="f4")
                        for a, b in zip(offsets[:-1], offsets[1:])], dtype="f8")
    means /= np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-12)
    rows = np.arange(len(lengths))
    with np.load(archive) as trace:
        train = []
        for i in trace["train"]:
            pooled = queries[i, :query_lengths[i]].mean(axis=0, dtype="f8")
            train.append(np.lexsort((rows, -(means @ pooled)))[:200])
        layout = fit_layout(tokens, lengths, train)
        np.testing.assert_array_equal(layout.logical_ids(rows), trace["order"])
        np.testing.assert_array_equal(layout.storage_ids(trace["order"]), rows)
    print(f"Verified the layout and inverse map for all {len(rows):,} documents.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("traces", type=Path)
    args = parser.parse_args()
    run(args.corpus, args.traces)
