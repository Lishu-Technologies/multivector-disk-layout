"""Within-query randomized full-corpus FiQA reranking on real Qdrant and the unchanged local ranker."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from load_q import URL,session
sys.path.insert(0,str(Path(__file__).resolve().parent/"algorithm"))
from rerank import Ranker

def run(root,destination,precision_control=False):
    s=session()
    trace=np.load(root/"traces.npz")
    queries=np.load(root/"fiqa"/"queries.npy")
    lengths=np.load(root/"fiqa"/"query_lens.npy")
    logical_ids=trace["order"]
    storage_ids=np.argsort(logical_ids)
    test=int(trace["eval_count"])
    rng=np.random.default_rng(70219631)
    placements=("original","semantic")
    engines=("qdrant","cpu_file","cuda_direct")
    arms=[(e,l) for e in engines for l in placements]
    if precision_control:
        arms=[(e,"semantic") for e in (*engines,"qdrant_f32")]
    receipts={}
    for layout in (("placed","placed_f32") if precision_control else ("original","placed")):
        url=f"{URL}/collections/fiqa_{layout}"
        state=s.get(url,timeout=10).json()["result"]
        assert state["status"]=="green" and state["points_count"]==57638 and state["optimizer_status"]=="ok"
        receipts[layout]=state
    cases=[]
    for pool in ("dense","bm25"):
        for pos,qi in enumerate(trace["queries"]):
            ids=trace[pool][pos]
            q=queries[qi,:lengths[qi]]
            values=trace[pool+"_scores"][pos]
            ranked=np.lexsort((ids,-values))
            want=ids[ranked[:10]]
            bodies={}
            for layout,candidate_ids in (("original",ids),("semantic",storage_ids[ids])):
                bodies[layout]=json.dumps({"query":q.tolist(),"filter":{"must":[{"has_id":candidate_ids.tolist()}]},"params":{"exact":True},
                                           "limit":10,"with_payload":False,"with_vector":False},separators=(",",":")).encode()
            cases.append((pool,pos,int(qi),ids,q,want,dict(zip(ids,values)),float(values[ranked[9]]-values[ranked[10]]),bodies))
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    with ExitStack() as stack,destination.open("x",encoding="utf8") as out:
        rankers={}
        for e,l in arms:
            if e.startswith("qdrant"):
                continue
            model=stack.enter_context(Ranker(root/l,"cuda" if e=="cuda_direct" else "cpu",e=="cuda_direct"))
            if e=="cpu_file":
                model.cache.clear()
            rankers[e,l]=model
        meta={"type":"metadata","server":s.get(URL).json(),"state":receipts,"torch":torch.__version__,
              "cuda":torch.cuda.get_device_name(),"cpu_threads":torch.get_num_threads(),"repeats":3,"test_queries":test,
              "precision_control":precision_control,"hot_docs":len(rankers["cuda_direct","semantic"].cache),
              "hot_payload":sum(x.nbytes for x in rankers["cuda_direct","semantic"].cache.values())}
        out.write(json.dumps(meta)+"\n")
        def execute(ix,rd):
            pool,pos,qi,ids,q,want,truth,gap,bodies=cases[ix]
            for a in rng.permutation(len(arms)):
                e,l=arms[a]
                row=dict(pool=pool,query=qi,repeat=rd,engine=e,layout=l)
                if e.startswith("qdrant"):
                    suffix="_f32" if e=="qdrant_f32" else ""
                    collection="placed" if l=="semantic" else "original"
                    url=f"{URL}/collections/fiqa_{collection}{suffix}/points/query"
                    t=time.perf_counter()
                    result=s.post(url,data=bodies[l],headers={"Content-Type":"application/json"},timeout=45)
                    if not result.ok:
                        raise RuntimeError(f"Qdrant {result.status_code} {result.text[:700]}")
                    data=result.json()
                    results=data["result"]["points"]
                    got=np.array([p["id"] for p in results])
                    if l=="semantic":
                        got=logical_ids[got]
                    row["ms"]=(time.perf_counter()-t)*1000
                    row["server_ms"]=data["time"]*1000
                    row["usage"]=data.get("usage",{})
                    row["returned_scores"]=[p["score"] for p in results]
                    row["max_abs_score_error"]=max(abs(truth[i]-p["score"]) for i,p in zip(got,results))
                else:
                    ranker=rankers[e,l]
                    reader=ranker.reader
                    calls,requested=reader.calls,reader.bytes
                    t=time.perf_counter()
                    got=ranker.topk(q,ids,10)
                    row["ms"]=(time.perf_counter()-t)*1000
                    row.update(calls=reader.calls-calls,read_bytes=reader.bytes-requested)
                if len(got)!=10 or len(set(got))!=10 or not set(got)<=truth.keys():
                    raise ValueError("Invalid result IDs")
                same=np.array_equal(got,want)
                if not e.startswith("qdrant") and not same:
                    raise ValueError(f"Local precision regression: {e}/{l}/{qi}")
                score_diff=max(abs(truth[x]-truth[y]) for x,y in zip(want,got))
                row.update(top=got.tolist(),reference=want.tolist(),ordered=bool(same),tie_equivalent=bool(score_diff<=1e-12),
                           overlap=len(set(got)&set(want)),boundary_gap=gap)
                if rd>=0:
                    out.write(json.dumps(row,separators=(",",":"))+"\n")
        warm=[j for j,c in enumerate(cases) if c[1]>=test]
        active=[j for j,c in enumerate(cases) if c[1]<test]
        for j in rng.permutation(warm):
            execute(j,-1)
        for rd in range(3):
            for i,j in enumerate(rng.permutation(active)):
                execute(j,rd)
                if i%80==79:
                    out.flush()
                    print("Repeat",rd+1,"cases",i+1,"/",len(active),flush=True)
    s.close()

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("new_jsonl",type=Path)
    p.add_argument("--precision-control",action="store_true")
    a=p.parse_args()
    run(a.root,a.new_jsonl,a.precision_control)
