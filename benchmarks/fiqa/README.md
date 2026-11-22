# FiQA and Qdrant 1.20.2

FiQA contains 57,638 documents and 7,314,136 FP16 token vectors from the answerai-colbert-small encoder. A fixed hash split supplied 324 calibration queries. The main replay used 160 different queries, two candidate generators, 200 documents per shortlist and three repeats. Scores were checked against exhaustive FP64 MaxSim within each shortlist.

## Check the published measurements

From this directory, run the following with Python and NumPy. Unpacking creates a local `results/` directory. The archive includes all 11,136 query-method observations, reference scores, query splits, collection settings and the physical file audit.

```sh
python -m zipfile -e records.zip .
python verify_results.py results
```

The `*_steady.json` summaries use the two repeats after both layouts completed the full workload once. The `*_stats.json` files retain all three repeats. The first-touch test used two fresh on-disk copies of each layout, with 64 queries and two candidate generators per copy. Qdrant started with `no_populate`. Windows volume counters measured bytes read across the drive. The uncertainty intervals resample queries and preserve their repeated observations.

## Reproduce from the pinned inputs

The recorded hardware was an Intel i5-12400, 15.7 GiB RAM, RTX 3060 12 GB and a Great Wall GW3000 1 TB NVMe drive on Windows. For the full hardware replay, use a fresh workspace with roughly 40 GB free. The commands below assume a Windows Python environment with CUDA-enabled PyTorch, NumPy, SciPy, requests, PyArrow and qdrant-client installed. The measured environment used Python 3.12.4, NumPy 2.5.3, SciPy 1.18.1, PyTorch 2.7.1+cu118, PyArrow 26.0.1, qdrant-client 1.20.0 and grpcio 1.85.3.

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
python fetch.py data
python fetch_text.py data
python prepare.py data
python server.py data
```

Keep the server terminal open. In a second terminal in the same directory, run each import to completion before starting the next.

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
python load_q.py data original
python load_q.py data semantic
python load_q.py data semantic --datatype float32
python load_q.py data semantic --physical-ids
python load_q.py data semantic --physical-ids --datatype float32
python audit_files.py data data/verified_order.json
python measure.py data
```

The `algorithm/` directory freezes the original benchmark implementation. The installable planner was checked separately against its FiQA permutation. The physical audit reads the pinned Qdrant version's block format, restores logical row IDs and checks each document's hash at its expected file position. Of the 57,638 rows, 57,594 have distinct vector contents.

The source data, revisions and licenses are in [data attribution](../../DATA-LICENSES.md). See the [detailed Chinese report](report.zh-CN.txt) for P50, P95, paired intervals, FP32 controls and the independent direct-I/O results.

With the package installed, the records extracted and the public vectors downloaded, run `python check_planner.py data/fiqa results/traces.npz` to compare the release planner and its inverse map against all 57,638 archived positions. Use `OPENBLAS_NUM_THREADS=1` to match the reference run.
