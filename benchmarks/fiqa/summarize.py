"""Validate the replay matrix and compute paired query-bootstrap P95 changes."""
import argparse
import json
from pathlib import Path
import numpy as np

def read(path):
    with path.open(encoding="utf8") as f:
        meta=json.loads(next(f))
        rows=[json.loads(line) for line in f]
    return meta,rows

def summary(path,after_full_pass=False):
    meta,rows=read(path)
    repeats=meta["repeats"]
    n=meta["test_queries"]
    control=bool(meta.get("precision_control"))
    engines=("qdrant","cpu_file","cuda_direct","qdrant_f32") if control else ("qdrant","cpu_file","cuda_direct")
    layouts=("semantic",) if control else ("original","semantic")
    assert len(rows)==2*len(engines)*len(layouts)*n*repeats
    if after_full_pass:
        rows=[dict(r,repeat=r["repeat"]-1) for r in rows if r["repeat"]>0]
        repeats-=1
    out={"queries_per_pool":n,"repeats":repeats,"rows":len(rows),"pools":{}}
    for pool in ("dense","bm25"):
        arms={}
        stats={}
        ids=sorted({r["query"] for r in rows if r["pool"]==pool})
        assert len(ids)==n
        draw=np.random.default_rng(69103).integers(n,size=(3000,n))
        def intervals(matrix):
            return np.quantile(matrix,.95),np.quantile(matrix[draw],.95,axis=(-2,-1))
        distributions={}
        for e in engines:
            for l in layouts:
                key=e+"/"+l
                rr=sorted((r for r in rows if (r["pool"],r["engine"],r["layout"])==(pool,e,l)),
                          key=lambda r:(r["query"],r["repeat"]))
                assert [(r["query"],r["repeat"]) for r in rr]==[(i,j) for i in ids for j in range(repeats)]
                def matrix(field):
                    return np.array([r[field] for r in rr]).reshape(n,repeats)
                assert all(len(r["top"])==len(set(r["top"]))==10 and r["overlap"]==len(set(r["top"])&set(r["reference"])) for r in rr)
                ms=matrix("ms")
                distributions[key]=intervals(ms)
                arms[key]=rr
                st=dict(p95_ms=float(np.quantile(ms,.95)),p50_ms=float(np.median(ms)),p95_each=np.quantile(ms,.95,axis=0).tolist(),
                        ordered_each=matrix("ordered").sum(axis=0).tolist(),equivalent_each=matrix("tie_equivalent").sum(axis=0).tolist(),
                        set_equal_each=(matrix("overlap")==10).sum(axis=0).tolist(),mean_overlap=float(matrix("overlap").mean()))
                if e.startswith("qdrant"):
                    st["server_p95_ms"]=float(np.quantile(matrix("server_ms"),.95))
                    distributions[key+"_server"]=intervals(matrix("server_ms"))
                    st["max_score_error"]=float(matrix("max_abs_score_error").max())
                else:
                    st.update(calls=float(matrix("calls").mean()),read_MiB=float(matrix("read_bytes").mean()/2**20))
                    assert matrix("ordered").all()
                stats[key]=st
        assert all([r["reference"] for r in rr]==[r["reference"] for r in arms["qdrant/semantic"]] for rr in arms.values())
        comparisons={}
        def compare(a,b):
            (x,xb),(y,yb)=distributions[a],distributions[b]
            boot=100*(1-xb/yb)
            comparisons[a+" vs "+b]={"p95_reduction_pct":float(100*(1-x/y)), "ci95":np.quantile(boot,[.025,.975]).tolist(),
                                       "ratio":float(x/y)}
        if not control:
            for e in engines:
                compare(e+"/semantic",e+"/original")
            compare("qdrant/semantic_server","qdrant/original_server")
        else:
            compare("qdrant_f32/semantic","qdrant/semantic")
        for e in ("cpu_file","cuda_direct"):
            for l in layouts:
                compare(e+"/semantic","qdrant/"+l)
            if control:
                compare(e+"/semantic","qdrant_f32/semantic")
        out["pools"][pool]={"arms":stats,"comparisons":comparisons}
    return out

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("results",type=Path)
    p.add_argument("new_json",type=Path)
    p.add_argument("--after-full-pass",action="store_true")
    a=p.parse_args()
    with a.new_json.open("x",encoding="utf8") as f:
        json.dump(summary(a.results,a.after_full_pass),f,indent=2)
