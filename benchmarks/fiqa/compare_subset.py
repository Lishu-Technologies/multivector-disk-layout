"""Pair direct-file times with the two fresh Qdrant sweeps by query, pool and repeat."""
import argparse
import json
from pathlib import Path
import numpy as np

def compare(cold,direct):
    qrows=[json.loads(x) for x in cold.read_text().splitlines()]
    files=[json.loads(x) for x in direct.read_text().splitlines()]
    result={}
    for pool in ("dense","bm25","both"):
        def arm(rows,layout,phase):
            rr=sorted((r for r in rows if r["type"]=="query" and r["layout"]==layout and (pool=="both" or r["pool"]==pool)
                       and (phase is None or r["phase"]==phase)),key=lambda r:(r["query"],r["pool"],r["repeat"]))
            assert len(rr)==(256 if pool=="both" else 128)
            return [(r["query"],r["pool"],r["repeat"]) for r in rr],np.asarray([r["ms"] for r in rr]).reshape(64,-1,2)
        idx,x=arm(files,"semantic",None)
        draws=np.random.default_rng(27819).integers(64,size=(3000,64))
        a=np.quantile(x,.95)
        ax=np.quantile(x[draws],.95,axis=(1,2,3))
        for layout in ("original","semantic"):
            pos,y=arm(qrows,layout,"first")
            assert idx==pos
            b=np.quantile(y,.95)
            by=np.quantile(y[draws],.95,axis=(1,2,3))
            result[f"{pool}/qdrant_{layout}"]=dict(local_semantic_p95_ms=float(a),qdrant_p95_ms=float(b),
                local_reduction_pct=float(100*(1-a/b)),ci95=np.quantile(100*(1-ax/by),[.025,.975]).tolist())
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for field in ("cold","direct","new_json"):
        p.add_argument(field,type=Path)
    args=p.parse_args()
    with args.new_json.open("x") as f:
        json.dump(compare(args.cold,args.direct),f,indent=2)
