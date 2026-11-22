"""Frozen two-corpus pilot. Set OPENBLAS_NUM_THREADS=1 for comparable CPU timings."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from engine import Summary, safe_shortlist, stable_top


BUDGETS = (512, 1536, 3584)


def splits(root):
    names = np.load(root/"queries_ids.npy")
    order = sorted(range(len(names)),key=lambda i:hashlib.sha256(("anchor-v1:"+str(names[i])).encode()).digest())
    return np.array(order[:len(order)//2]),np.array(order[len(order)//2:])


def load_data(root):
    docs = np.load(root/"documents.npy",mmap_mode="r")
    lengths = np.load(root/"doclens.npy").astype(np.int64)
    offsets = np.r_[0,np.cumsum(lengths)]
    queries, qlens = np.load(root/"queries.npy"), np.load(root/"query_lens.npy")
    if offsets[-1]!=len(docs) or min(lengths)<=0 or min(qlens)<=0:
        raise ValueError("Invalid corpus lengths")
    return docs,offsets,queries,qlens


def build(root, destination, device):
    docs, offsets, queries, qlens = load_data(root)
    train,_ = splits(root)
    tokens = np.concatenate([queries[i,:qlens[i]] for i in train])
    tokens = tokens[np.linalg.norm(tokens,axis=1)>1e-8]
    perm = np.random.default_rng(21419).permutation(len(tokens))[:max(BUDGETS)]
    anchors = tokens[perm].astype(np.float64)
    anchors = (anchors/np.linalg.norm(anchors,axis=1,keepdims=True)).astype(np.float32)
    if len(anchors)!=max(BUDGETS):
        raise ValueError("Insufficient calibration tokens")
    norm = np.linalg.norm(anchors.astype(np.float64),axis=1)
    basis = torch.from_numpy(anchors).to(device)
    n = len(offsets)-1
    heights = np.empty((n,len(anchors)),dtype=np.uint8)
    means = np.empty((n,docs.shape[1]),dtype=np.float32)
    radii = np.empty(n,dtype=np.float32)
    scale = 1.002
    # Run the summary writer with TF32 disabled. With d=96 and ||v||,||a||<1.002,
    # a FP32 dot product has error below gamma_192*1.002^2 < 1.2e-5.
    if docs.shape[1]!=96:
        raise ValueError("The builder's rounding budget is for this 96D pilot")
    started = time.perf_counter()
    for begin in range(0,n,40):
        ids = np.arange(begin,min(begin+40,n))
        lengths = offsets[ids+1]-offsets[ids]
        rel = np.arange(lengths.max())
        valid = rel[None,:]<lengths[:,None]
        indices = offsets[ids,None]+np.minimum(rel[None,:],lengths[:,None]-1)
        x = np.asarray(docs[indices],dtype=np.float32)
        mag = np.sqrt(np.einsum("btd,btd->bt",x,x,dtype=np.float64)).max(axis=1)
        if not np.all(np.isfinite(mag)) or np.any(mag>=scale-0.0001):
            raise ValueError("Document vectors are not within the declared norm range")
        radii[ids] = np.nextafter(mag.astype(np.float32),np.float32(np.inf))
        means[ids] = np.einsum("btd,bt->bd",x,valid)/lengths[:,None]
        x_gpu = torch.from_numpy(x).to(device)
        # Padding repeats a true vector. It cannot change the maximum.
        for c in range(0,len(anchors),384):
            part = basis[c:c+384]
            scores = (x_gpu @ part.T).amax(dim=1).cpu().numpy().astype(np.float64)
            scores = scores/norm[c:c+len(part)] + 0.00003
            quantized = np.ceil((scores+scale)*(255/(2*scale)))
            heights[ids,c:c+len(part)] = np.clip(quantized,0,255).astype(np.uint8)
    destination.mkdir(parents=True,exist_ok=True)
    for name, data in (("anchors",anchors),("heights",heights),("radii",radii),("means",means)):
        np.save(destination/f"{name}.npy",data)
    print("Index built",root.name,round(time.perf_counter()-started,2),"s",flush=True)


def exact(q,docs,offsets,ids):
    q = np.asarray(q,dtype=np.float64)
    chunks = [docs[offsets[i]:offsets[i+1]] for i in ids]
    vectors = np.concatenate(chunks).astype(np.float64)
    start = np.r_[0,np.cumsum([len(c) for c in chunks])[:-1]]
    return np.maximum.reduceat(vectors@q.T,start,axis=0).sum(axis=1)


def candidates_for(means, q):
    pooled = q.mean(axis=0,dtype=np.float64)
    return stable_top(np.arange(len(means)),means@pooled,200)


def index_arrays(dest):
    return [np.load(dest/f"{name}.npy",mmap_mode="r") for name in ("anchors","heights","radii","means")]


def bench(root,dest,report):
    docs,offsets,queries,qlens = load_data(root)
    anchors,heights,radii,means = index_arrays(dest)
    norm_means = means.astype(np.float64)
    norm_means /= np.maximum(1e-12,np.linalg.norm(norm_means,axis=1,keepdims=True))
    train,tests = splits(root)
    # Ground-truth file uses query and document row indices. It is used for audit only.
    published = np.loadtxt(root/"gt_top100.tsv")
    audit_errors = []
    rows = []
    for qi in tests:
        q = queries[qi,:qlens[qi]]
        candidates = candidates_for(norm_means,q)
        lengths = np.diff(offsets)[candidates]
        scores = exact(q,docs,offsets,candidates)
        gold = stable_top(candidates,scores,10)
        threshold = scores[gold[-1]]
        common = {int(r[1]):r[3] for r in published[published[:,0]==qi]}
        audit_errors.extend(abs(score-common[i]) for i,score in zip(candidates,scores) if i in common)
        for size,combine in [(v,True) for v in BUDGETS]+[(BUDGETS[-1],False)]:
            summary = Summary(anchors[:size],heights[:,:size],radii,1.002)
            started = time.perf_counter()
            upper, approx = summary.bounds(q,candidates,combine)
            ms = (time.perf_counter()-started)*1000
            # Full scores are already present solely for offline validation. This callback
            # returns only explicitly requested scores. Physical I/O is a separate replay.
            score_lookup = dict(zip(candidates,scores))
            predicted, read = safe_shortlist(candidates,upper,approx,10,
                                              lambda ids:np.array([score_lookup[i] for i in ids]))
            rows.append({"query":int(qi),"size":size,"combined":combine,"same":bool(np.array_equal(predicted,candidates[gold])),
                         "pairs":len(candidates),"min_slack":float(np.min(upper-scores)),
                         "violations":int(np.sum(upper<scores)),"docs_read":len(read),
                         "full_bytes":int(lengths.sum()*docs.shape[1]*docs.itemsize),
                         "bytes_read":int(lengths[read].sum()*docs.shape[1]*docs.itemsize),
                         "oracle_read":int(lengths[upper>=threshold].sum()*docs.shape[1]*docs.itemsize),
                         "bound_ms":ms})
    if not audit_errors or max(audit_errors)>0.002:
        raise AssertionError(f"Published score check failed: {max(audit_errors,default=None)}")
    metrics = []
    for size,combine in [(v,True) for v in BUDGETS]+[(BUDGETS[-1],False)]:
        subset = [r for r in rows if r["size"]==size and r["combined"]==combine]
        raw = sum(r["full_bytes"] for r in subset)
        # Store means, offsets, radii, plus one scale and the shared dictionary; counts
        # refer to payload sizes in a packed index, not padded .npy files on disk.
        meta = means.nbytes+offsets.nbytes+radii.nbytes+size*(len(radii)+anchors.shape[1]*anchors.itemsize)+8
        metrics.append({"anchors":size,"combined":combine,"summary_fraction":meta/docs.nbytes,
                        "ordered_top10_matches":sum(r["same"] for r in subset),"queries":len(subset),
                        "violations":sum(r["violations"] for r in subset),"pairs":len(subset)*200,
                        "min_slack":min(r["min_slack"] for r in subset),
                        "logical_byte_saving":1-sum(r["bytes_read"] for r in subset)/raw,
                        "oracle_saving_ceiling":1-sum(r["oracle_read"] for r in subset)/raw,
                        "bound_ms_p95":float(np.quantile([r["bound_ms"] for r in subset],.95))})
    output={"dataset":root.name,"docs":len(offsets)-1,"tokens":len(docs),"dimensions":docs.shape[1],
            "calibration_queries":len(train),"evaluation_queries":len(tests),"published_score_pairs":len(audit_errors),
            "max_published_score_delta":max(audit_errors),"metrics":metrics,"rows":rows}
    with report.open("x",encoding="utf8") as f:
        json.dump(output,f,indent=2)
    print(root.name,json.dumps(metrics),flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["build","bench"])
    p.add_argument("corpus",type=Path)
    p.add_argument("index",type=Path)
    p.add_argument("--report",type=Path)
    p.add_argument("--device",default="cpu")
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)
    if args.command=="build":
        build(args.corpus,args.index,args.device)
    else:
        if not args.report:
            p.error("--report is required for bench")
        bench(args.corpus,args.index,args.report)
