"""Build a new, disposable Qdrant collection and verify a mapped MaxSim query."""
import argparse
from contextlib import closing

import numpy as np
from qdrant_client import QdrantClient, models

from multivector_disk_layout import fit_layout


def run(url, collection):
    rng = np.random.default_rng(3741)
    docs = [rng.normal(size=(rng.integers(6, 14), 16)).astype("f2") for _ in range(48)]
    history = [rng.choice(len(docs), 12, replace=False) for _ in range(24)]
    layout = fit_layout(np.concatenate(docs), [len(d) for d in docs], history)
    query = rng.normal(size=(3, 16)).astype("f2").astype("f4")
    candidates = np.array([2, 5, 11, 16, 18, 21, 23, 29, 33, 39, 45, 47])
    with closing(QdrantClient(url=url)) as client:
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=16, distance=models.Distance.DOT, on_disk=True,
                datatype=models.Datatype.FLOAT16,
                multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM)),
            hnsw_config=models.HnswConfigDiff(m=0),
        )
        points = [models.PointStruct(id=slot, vector=docs[layout.logical_ids(slot)].tolist())
                  for slot in range(len(layout))]
        client.upsert(collection, points=points, wait=True)
        result = client.query_points(collection, query=query.tolist(),
            query_filter=models.Filter(must=[models.HasIdCondition(has_id=layout.storage_ids(candidates).tolist())]),
            search_params=models.SearchParams(exact=True), limit=5, with_payload=False).points
    logical = layout.logical_ids([p.id for p in result])
    scores = [(query.astype("f8") @ docs[i].astype("f8").T).max(axis=1).sum() for i in candidates]
    expected = candidates[np.lexsort((candidates, -np.asarray(scores)))[:5]]
    np.testing.assert_array_equal(logical, expected)
    print(f"Verified logical top 5: {logical.tolist()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:6333")
    parser.add_argument("--collection", default="multivector_layout_demo", help="A new collection name")
    args = parser.parse_args()
    run(args.url, args.collection)
