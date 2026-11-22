"""Two disk reads with per-token residual bounds and a calibration-trained RAM cache."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from blockio import BlockReader
from engine import stable_top
from planes import unpack_nibble
from replay import reduce_exact, token_rows, trained_cache
from study import load_data, splits, candidates_for


def build(root,index):
    docs,offset,queries,lens=load_data(root)
    train,_=splits(root)
    q=np.concatenate([queries[i,:lens[i]] for i in train]).astype("f8")
    _,_,right=np.linalg.svd(q,full_matrices=False)
    axis=right[0].astype("f4")
    high=np.memmap(index/"high.bin",mode="r",dtype="u1",shape=docs.shape)
    dims=docs.shape[1]
    mid,low=[np.memmap(index/f"{name}.bin",mode="r",dtype="u1",shape=(len(docs),dims//2)) for name in ["middle","low"]]
    centers=np.load(index/"center0.npy")
    meta=np.empty((len(docs),2),dtype="f2")
    leak=0.
    with (index/"rest.bin").open("wb") as f:
        for a in range(0,len(docs),16384):
            b=min(len(docs),a+16384)
            ((unpack_nibble(mid[a:b])<<4)|unpack_nibble(low[a:b])).tofile(f)
            delta=docs[a:b].astype("f8")-centers[high[a:b]].astype("f8")
            coeff=(delta@axis).astype("f2")
            remainder=delta-coeff.astype("f8")[:,None]*axis[None,:]
            leak=max(leak,np.abs(remainder@axis).max())
            meta[a:b,0]=coeff
            meta[a:b,1]=np.nextafter((np.linalg.norm(remainder,axis=1)+1e-9).astype("f2"),np.float16(np.inf))
    np.save(index/"oneaxis.npy",axis)
    np.save(index/"projection.npy",meta)
    np.save(index/"oneleak.npy",np.array(float(leak)+1e-8))


class Hybrid:
    def __init__(self,index,device):
        self.device=device
        self.center=torch.from_numpy(np.load(index/"center0.npy")).to(device)
        self.axis=torch.from_numpy(np.load(index/"oneaxis.npy")).to(device)
        self.projections=np.load(index/"projection.npy")
        self.leak=float(np.load(index/"oneleak.npy"))
        self.decode=np.load(index/"decode.npy").view("<f2")

    def bounds(self,codes,rows,lens,query):
        q=torch.from_numpy(query).to(self.device)
        projected=q@self.axis
        tail=(q-projected[:,None]*self.axis).norm(dim=1)
        meta=torch.from_numpy(self.projections[rows].astype("f4")).to(self.device)
        vectors=self.center[torch.from_numpy(codes.astype("i4")).to(self.device)]
        value=vectors@q.T+meta[:,0,None]*projected
        delta=meta[:,1,None]*tail+self.leak*projected.abs()
        ind=torch.repeat_interleave(torch.arange(len(lens),device=self.device),torch.as_tensor(lens,device=self.device))
        def maximum(scores):
            best=torch.full((len(lens),len(q)),-torch.inf,device=self.device)
            best.scatter_reduce_(0,ind[:,None].expand_as(scores),scores,reduce="amax",include_self=True)
            return best.sum(dim=1,dtype=torch.float64).cpu().numpy()
        eps=0.00005*(1+len(q))
        return maximum(value-delta)-eps,maximum(value+delta)+eps


def full_score(root,reader,offset,ids,q,cache,header,device):
    hit=np.array([i in cache for i in ids])
    miss=ids[~hit]
    raw=reader.fetch(root/"documents.npy",offset,miss,192,header) if len(miss) else np.empty((0,192),"u1")
    array=raw.view("<f2").reshape(-1,96)
    if hit.any():
        edges=np.r_[0,np.cumsum(np.diff(offset)[miss])]
        parts={k:array[a:b] for k,a,b in zip(miss,edges,edges[1:])}
        array=np.concatenate([cache[k] if k in cache else parts[k] for k in ids])
    return reduce_exact(array,q,np.diff(offset)[ids],device),raw.nbytes,len(miss)


def hybrid_score(index,reader,offset,ids,q,cache,scorer,device,k=10,high_cache=None):
    hit=np.array([k in cache for k in ids])
    hit_ids,miss=ids[hit],ids[~hit]
    lens=np.diff(offset)
    s=np.empty(0)
    if len(hit_ids):
        s=reduce_exact(np.concatenate([cache[k] for k in hit_ids]),q,lens[hit_ids],device)
    if not len(miss):
        return hit_ids[stable_top(hit_ids,s,k)],None,0,0
    if high_cache:
        cold=np.array([i for i in miss if i not in high_cache],dtype=np.int64)
        raw=reader.fetch(index/"high.bin",offset,cold,96) if len(cold) else np.empty((0,96),"u1")
        fetched=raw.nbytes
        edges=np.r_[0,np.cumsum(lens[cold])]
        parts={i:raw[a:b] for i,a,b in zip(cold,edges,edges[1:])}
        high=np.concatenate([high_cache[i] if i in high_cache else parts[i] for i in miss])
    else:
        high=reader.fetch(index/"high.bin",offset,miss,96)
        fetched=high.nbytes
    lo,hi=scorer.bounds(high,token_rows(offset,miss),lens[miss],q)
    kth=np.partition(np.r_[s,lo],-k)[-k]
    keep=np.flatnonzero(hi>=kth)
    read=miss[keep]
    extra=0
    if len(read):
        rows=token_rows(np.r_[0,np.cumsum(lens[miss])],keep)
        rest=reader.fetch(index/"rest.bin",offset,read,96)
        extra=rest.nbytes
        packed=(high[rows].astype("u2")<<8)|rest
        s=np.r_[s,reduce_exact(scorer.decode[packed],q,lens[read],device)]
    final=np.r_[hit_ids,read]
    return final[stable_top(final,s,k)],(miss,lo,hi),fetched+extra,len(read)


def bench(root,base,index,report,device,direct,repeats):
    docs,offset,queries,lens=load_data(root)
    train,test=splits(root)
    means=np.load(base/"means.npy")
    # Keep first-stage representation in FP32, including scoring; it belongs to all methods.
    # Use the identical published pilot shortlist computation (FP64 arithmetic on FP32 means).
    common=means.nbytes+offset.nbytes
    metadata=sum((index/f).stat().st_size for f in
                 ["center0.npy","decode.npy","oneaxis.npy","projection.npy","oneleak.npy"])
    budget=int(docs.nbytes*.10)-common
    norm_means=means.astype("f8")
    norm_means/=np.maximum(1e-12,np.linalg.norm(norm_means,axis=1,keepdims=True))
    hot=trained_cache(norm_means,queries,lens,root,docs,offset,budget)
    shared=trained_cache(norm_means,queries,lens,root,docs,offset,budget-metadata)
    # Shortlists are supplied to a reranker. Precompute them once; the means are
    # conservatively charged above even though the timed reranker doesn't need them.
    shortlists={int(i):candidates_for(norm_means,queries[i,:lens[i]]) for i in test}
    scorer=Hybrid(index,device)
    reader=BlockReader(direct)
    rng=np.random.default_rng(53998)
    rows=[]
    try:
        for repeat in range(repeats):
            order=rng.permutation(test)
            for j,qi in enumerate([order[0],*order]):
                ids=shortlists[int(qi)]
                q=queries[qi,:lens[qi]]
                outcome={}
                for name in rng.permutation(["full","cached","hybrid"]):
                    last=(reader.bytes,reader.calls)
                    start=time.perf_counter()
                    if name=="hybrid":
                        rank,interval,payload,reads=hybrid_score(index,reader,offset,ids,q,shared,scorer,device)
                    else:
                        scores,payload,reads=full_score(root,reader,offset,ids,q,hot if name=="cached" else {},docs.offset,device)
                        rank=ids[stable_top(ids,scores,10)]
                        interval=None
                        if name=="full":
                            truth=scores
                    elapsed=(time.perf_counter()-start)*1000
                    outcome[name]=(rank,interval,payload,reads,elapsed,reader.bytes-last[0],reader.calls-last[1])
                if j==0:
                    continue
                lookup=dict(zip(ids,truth))
                for name,(rank,interval,payload,reads,ms,count,calls) in outcome.items():
                    violations=0
                    min_slack=None
                    if interval:
                        mids,left,right=interval
                        true=np.array([lookup[i] for i in mids])
                        slack=np.minimum(true-left,right-true)
                        violations=int(np.count_nonzero(slack<0))
                        min_slack=float(slack.min())
                    rows.append(dict(query=int(qi),repeat=repeat,method=str(name),same=bool(np.array_equal(rank,outcome["full"][0])),
                                     violations=violations,min_slack=min_slack,io_bytes=count,payload=payload,ms=ms,
                                     calls=calls,full_vectors_read=reads))
    finally:
        reader.close()
    totals={}
    for n,meta in (("full",common),("cached",common+sum(x.nbytes for x in hot.values())),
                   ("hybrid",common+metadata+sum(x.nbytes for x in shared.values()))):
        rr=[r for r in rows if r["method"]==n]
        totals[n]=dict(metadata_fraction=meta/docs.nbytes,queries=len(test),runs=len(rr),same=sum(r["same"] for r in rr),
                       violations=sum(r["violations"] for r in rr),io_bytes=sum(r["io_bytes"] for r in rr),
                       payload=sum(r["payload"] for r in rr),p50_ms=np.median([r["ms"] for r in rr]),
                       p95_ms=np.quantile([r["ms"] for r in rr],.95),calls=sum(r["calls"] for r in rr))
    out=dict(dataset=root.name,baseline_cached_docs=len(hot),hybrid_cached_docs=len(shared),training_queries=len(train),
             windows_direct=direct,device=device,repeats=repeats,totals=totals,rows=rows)
    with report.open("x",encoding="utf8") as f:
        json.dump(out,f,indent=2)
    print(root.name,json.dumps(totals),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["build","bench"])
    for name in ["corpus","base_index","plane_index"]:
        p.add_argument(name,type=Path)
    p.add_argument("--report",type=Path)
    p.add_argument("--device",default="cpu")
    p.add_argument("--windows-direct",action="store_true")
    p.add_argument("--repeats",type=int,default=2)
    a=p.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(2)
    if a.action=="build":
        build(a.corpus,a.plane_index)
    elif a.report and a.repeats>=1:
        bench(a.corpus,a.base_index,a.plane_index,a.report,a.device,a.windows_direct,a.repeats)
    else:
        p.error("Provide --report and a positive repeat count")
