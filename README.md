# Multivector Disk Layout

An offline layout planner for teams that rerank a shortlist against token vectors stored on SSD. It places semantically similar documents near each other and gives each document a reversible numeric storage ID. Your retriever keeps its original document IDs and MaxSim scoring rule. The planner runs on CPU and requires only NumPy.

[中文说明](README.zh-CN.md) · [FiQA benchmark and raw records](benchmarks/fiqa/README.md)

## Install and try it

```sh
git clone https://github.com/Lishu-Technologies/multivector-disk-layout.git
cd multivector-disk-layout
python -m pip install .
```

```python
import numpy as np
from multivector_disk_layout import DiskLayout, fit_layout

rng = np.random.default_rng(42)
lengths = np.array([3, 7, 4, 6, 2])
tokens = rng.normal(size=(lengths.sum(), 96)).astype("float16")
history = [[0, 1, 3], [1, 4], [1, 2, 4]]

layout = fit_layout(tokens, lengths, history)
layout.save("layout.npz")
saved = DiskLayout.load("layout.npz")
candidate_ids = [4, 0, 2]
point_ids = saved.storage_ids(candidate_ids)
assert saved.logical_ids(point_ids).tolist() == candidate_ids
```

For your own corpus, concatenate the token matrices in logical document order. Pass one positive length per document, plus historical candidate lists from the first-stage retriever. Logical IDs are zero-based document row numbers. Keep your application's external ID table beside the layout snapshot. The CLI accepts an FP16 token matrix, integer lengths, and either a two-dimensional integer NPY file or a JSON array of variable-length candidate lists.

```sh
multivector-disk-layout documents.npy lengths.npy shortlists.json layout.npz
```

Use a new output path for each snapshot. Deploy the matching collection and ID map together when the corpus or layout changes.

## Use with Qdrant

1. Create a **new** collection with on-disk multivectors and the MaxSim comparator. For storage slot `s`, upload the token matrix of logical document `layout.logical_ids(s)` using Qdrant point ID `s`.
2. Translate each shortlist with `layout.storage_ids(candidate_ids)`. Pass those IDs in the Qdrant `has_id` filter and run the multivector query.
3. Translate returned point IDs with `layout.logical_ids(result_ids)` before returning results to the application.

In our Qdrant 1.20.2 test, compaction restored the original order when we changed only insertion order. Assigning numeric IDs by layout position preserved all 57,638 target positions in the resulting vector files. This is an observation for the tested version and collection settings. The benchmark includes the file audit.

The example creates a fresh collection on a running local Qdrant server, uploads synthetic documents and checks the returned logical top five against NumPy exhaustive scoring.

```sh
python -m pip install ".[qdrant]"
python examples/qdrant_demo.py --url http://127.0.0.1:6333 --collection layout_demo_v1
```

## What improved on FiQA?

Qdrant 1.20.2, 57,638 documents, 7.31 million token vectors, fixed shortlists of 200 documents. Each comparison changes the document layout inside the same engine.

| Qdrant workload | Original P95 | Semantic layout P95 |
| --- | ---: | ---: |
| First access, mean-vector candidates | 50.46 ms | 42.41 ms |
| First access, BM25 candidates | 65.48 ms | 63.94 ms |
| After a full workload pass, mean-vector candidates | 8.63 ms | 8.81 ms |
| After a full workload pass, BM25 candidates | 11.68 ms | 11.66 ms |

The first-access P95 reduction for mean-vector candidates was **16.0%** (paired query-bootstrap 95% interval 8.6%–23.7%). The BM25 interval crossed zero. Across 128 mixed first-access requests per fresh copy, volume reads fell from about 599–600 MiB to 544 MiB. Warm-cache results were similar. All 160 evaluation queries for each candidate generator returned the same ordered top ten as exhaustive scoring in the main run. The results describe single-client reranking on one Windows NVMe machine. [Methodology and reproduction](benchmarks/fiqa/README.md)

## How the planner works

The planner counts candidate appearances on calibration queries and sets aside the most frequently selected documents up to 10% of corpus tokens. Within each tier it centers normalized document means, recursively bisects groups larger than 384 documents, then greedily orders each small group by cosine similarity. Only the permutation is stored. Fitting never reads evaluation queries.

The process uses one frequency counter per document and computes document means one document at a time. Build on a static snapshot and remeasure with your candidate generator and cache budget. The benchmark also includes the standalone direct-I/O reference ranker and the original experimental implementation.

## License

Copyright 2026 Lishu Technologies. Software is available under [Apache-2.0](LICENSE). See [benchmark data attribution](DATA-LICENSES.md) for the FiQA materials.
