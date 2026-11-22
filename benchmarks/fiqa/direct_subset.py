"""Run the existing direct-I/O CPU-cache/GPU ranker on the very same first-touch query subset."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from first_touch import DriveReads
sys.path.insert(0,str(Path(__file__).resolve().parent/"algorithm"))
from rerank import Ranker

def run(root,path):
    trace=np.load(root/"traces.npz")
    q=np.load(root/"fiqa"/"queries.npy")
    lens=np.load(root/"fiqa"/"query_lens.npy")
    queries=trace["queries"][:64]
    pairs=[]
    for pool in ("dense","bm25"):
        pairs.extend((pool,int(qi),ids,ids[np.lexsort((ids,-values))[:10]]) for qi,ids,values in zip(queries,trace[pool],trace[pool+"_scores"]))
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(2)
    counter=DriveReads(root.resolve().drive)
    with path.open("x") as out:
        for rd,layouts in enumerate((("original","semantic"),("semantic","original"))):
            seq=np.random.default_rng(91031+rd).permutation(len(pairs))
            for layout in layouts:
                with Ranker(root/layout,"cuda",True) as ranker:
                    # Isolate CUDA initialization from query latency on both layouts.
                    qi=trace["queries"][-1]
                    ranker.topk(q[qi,:lens[qi]],trace["dense"][-1])
                    baseline=counter.sample()
                    for j in seq:
                        pool,qi,ids,truth=pairs[j]
                        calls,bytes_read=ranker.reader.calls,ranker.reader.bytes
                        t=time.perf_counter()
                        top=ranker.topk(q[qi,:lens[qi]],ids)
                        ms=(time.perf_counter()-t)*1000
                        np.testing.assert_array_equal(top,truth)
                        out.write(json.dumps(dict(type="query",pool=pool,query=qi,layout=layout,repeat=rd,ms=ms,
                                                 calls=ranker.reader.calls-calls,read_bytes=ranker.reader.bytes-bytes_read,
                                                 top=top.tolist()))+"\n")
                    out.write(json.dumps(dict(type="phase",layout=layout,repeat=rd,drive_read_bytes=counter.sample()-baseline))+"\n")
    counter.close()

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("new_jsonl",type=Path)
    a=p.parse_args()
    run(a.root,a.new_jsonl)
