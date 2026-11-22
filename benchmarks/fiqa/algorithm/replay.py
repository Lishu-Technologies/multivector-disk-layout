"""Replay candidate reranking through real files; evaluate ordered top-k and I/O."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

from blockio import BlockReader
from engine import stable_top
from planes import BoundScorer, surviving, unpack_nibble
from residual import DirectionScorer
from study import load_data, splits, candidates_for


def token_rows(offset,ids):
    return np.concatenate([np.arange(offset[i],offset[i+1]) for i in ids])


def reduce_exact(vectors,query,lengths,device):
    v = torch.from_numpy(vectors.astype("f4")).to(device=device,dtype=torch.float64)
    q = torch.from_numpy(query).to(device=device,dtype=torch.float64)
    values = v@q.T
    ind = torch.repeat_interleave(torch.arange(len(lengths),device=device),torch.as_tensor(lengths,device=device))
    out = torch.full((len(lengths),len(q)),-torch.inf,device=device,dtype=torch.float64)
    out.scatter_reduce_(0,ind[:,None].expand_as(values),values,reduce="amax",include_self=True)
    return out.sum(axis=1).cpu().numpy()


def plane_score(reader,index,scorer,offset,ids,q,device):
    lengths = np.diff(offset)[ids]
    high = reader.fetch(index/"high.bin",offset,ids,96)
    payload = high.nbytes
    bounds = []
    for stage,name in enumerate(["middle.bin","low.bin"]):
        lower,upper = scorer.bounds(ids,offset,q,high,token_rows(offset,ids),stage)
        bounds.append((ids,lower,upper))
        keep = surviving(ids,lower,upper,10)
        local = np.r_[0,np.cumsum(lengths)]
        high = high[token_rows(local,keep)].astype("u2")
        ids,lengths = ids[keep],lengths[keep]
        part = reader.fetch(index/name,offset,ids,48)
        payload += part.nbytes
        high = (high<<4)|unpack_nibble(part)
    exact = reduce_exact(scorer.decode[high],q,lengths,device)
    return ids[stable_top(ids,exact,10)], bounds,payload,len(ids)


def extra_metadata(index,directional):
    files = ["angles.npy","norms.npy","decode.npy","center0.npy","center1.npy"]
    if directional:
        files += ["residual.npy","directions.npy","leak.npy"]
    return sum((index/n).stat().st_size for n in files)


def trained_cache(means,queries,qlens,root,docs,offset,budget):
    train,_ = splits(root)
    counts = np.zeros(len(means),dtype="i4")
    for qi in train:
        counts[candidates_for(means,queries[qi,:qlens[qi]])] += 1
    cache = {}
    for i in np.lexsort((np.arange(len(means)),-counts)):
        size = (offset[i+1]-offset[i])*192
        if size<=budget:
            cache[int(i)] = np.array(docs[offset[i]:offset[i+1]])
            budget -= size
    return cache


def run(root,summary_root,plane_root,direct,report,limit,device,directional=False):
    docs,offset,queries,qlens = load_data(root)
    _,test = splits(root)
    test = test[:limit] if limit else test
    means = np.load(summary_root/"means.npy").astype(np.float64)
    means /= np.maximum(1e-12,np.linalg.norm(means,axis=1,keepdims=True))
    names = ["full","cached","adaptive","direction"] if directional else ["full","direct","adaptive"]
    indexes = {n:plane_root/("adaptive" if n=="direction" else n) for n in names if n not in {"full","cached"}}
    scorers = {n:(DirectionScorer if n=="direction" else BoundScorer)(path,device) for n,path in indexes.items()}
    cache = trained_cache(means,queries,qlens,root,docs,offset,extra_metadata(indexes["direction"],True)) if directional else {}
    for s in scorers.values():
        s.angles = np.array(s.angles)
    reader = BlockReader(direct)
    rng = np.random.default_rng(187)
    rows = []
    try:
        # Warm up all kernels; no warmup request contributes a measurement.
        for trial,qi in enumerate([test[0],*test]):
            q = queries[qi,:qlens[qi]]
            ids = candidates_for(means,q)
            lengths = np.diff(offset)[ids]
            truth = None
            outcomes = {}
            for name in rng.permutation(names):
                pre_bytes,pre_calls = reader.bytes,reader.calls
                start = time.perf_counter()
                if name in {"full","cached"}:
                    missing = ids if name=="full" else np.array([i for i in ids if i not in cache],dtype=np.int64)
                    raw = reader.fetch(root/"documents.npy",offset,missing,192,docs.offset) if len(missing) else np.empty((0,192),"u1")
                    vectors = raw.view("<f2").reshape(-1,96)
                    if name=="cached":
                        boundaries = np.r_[0,np.cumsum(np.diff(offset)[missing])]
                        parts = {i:vectors[a:b] for i,a,b in zip(missing,boundaries,boundaries[1:])}
                        vectors = np.concatenate([cache[int(i)] if i in cache else parts[i] for i in ids])
                    scores = reduce_exact(vectors,q,lengths,device)
                    if name=="full":
                        truth = scores
                    ranking,bounds,payload,count = ids[stable_top(ids,scores,10)],[],raw.nbytes,len(missing)
                else:
                    ranking,bounds,payload,count = plane_score(reader,indexes[name],scorers[name],offset,ids,q,device)
                ms = (time.perf_counter()-start)*1000
                outcomes[name] = (ranking,bounds,payload,count,ms,reader.bytes-pre_bytes,reader.calls-pre_calls)
            if trial==0:
                continue
            lookup = dict(zip(ids,truth))
            for name,(ranking,bounds,payload,count,ms,file_bytes,requests) in outcomes.items():
                violations = 0
                slack = None
                for candidates,lo,hi in bounds:
                    scores = np.array([lookup[x] for x in candidates])
                    gap = np.minimum(scores-lo,hi-scores)
                    violations += int(np.sum(gap<0))
                    slack = min(float(gap.min()),slack) if slack is not None else float(gap.min())
                rows.append({"query":int(qi),"method":str(name),"same":bool(np.array_equal(ranking,outcomes["full"][0])),
                             "violations":violations,"min_slack":slack,"payload":payload,"io_bytes":file_bytes,
                             "read_calls":requests,"ms":ms,"exact_docs":count,
                             "stage12_docs":len(bounds[1][0]) if bounds else 0})
    finally:
        reader.close()
    totals = {}
    for name in names:
        s = [r for r in rows if r["method"]==name]
        totals[name] = {"queries":len(s),"matches":sum(r["same"] for r in s),"violations":sum(r["violations"] for r in s),
                        "avg_documents_fully_read":np.mean([r["exact_docs"] for r in s]),
                        "avg_documents_read_at_12bits":np.mean([r["stage12_docs"] for r in s]),
                        "payload_bytes":sum(r["payload"] for r in s),"io_bytes":sum(r["io_bytes"] for r in s),
                        "calls":sum(r["read_calls"] for r in s),
                        "p50_ms":np.median([r["ms"] for r in s]),"p95_ms":np.quantile([r["ms"] for r in s],.95)}
        for metric in ("payload_bytes","io_bytes","p95_ms"):
            totals[name][metric+"_saving"] = 1-totals[name][metric]/totals["full"][metric]
        if name!="full":
            side_bytes = sum(v.nbytes for v in cache.values()) if name=="cached" else extra_metadata(indexes[name],name=="direction")
            # Conservatively charge the first-stage means + document offsets too.
            totals[name]["metadata_fraction"] = (side_bytes+means.size*4+offset.nbytes)/docs.nbytes
    result={"corpus":root.name,"device":device,"unbuffered_windows_reads":direct,"totals":totals,"rows":rows}
    with report.open("x",encoding="utf8") as f:
        json.dump(result,f,indent=2)
    print(json.dumps(totals),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("corpus","summary_index","plane_index","report"):
        p.add_argument(arg,type=Path)
    p.add_argument("--windows-direct",action="store_true")
    p.add_argument("--directional",action="store_true")
    p.add_argument("--limit",type=int)
    p.add_argument("--device",default="cpu")
    a=p.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(2)
    run(a.corpus,a.summary_index,a.plane_index,a.windows_direct,a.report,a.limit,a.device,a.directional)
