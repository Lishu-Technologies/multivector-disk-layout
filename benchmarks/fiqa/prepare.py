"""Frozen FiQA protocol. Fit the previous semantic layout and build independent candidate lists."""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parent/"algorithm"))
from rerank import _layout_means, build_full_index
from study import candidates_for, exact, load_data, splits

def lexical(root,queries):
    data=root/"fiqa"
    query_table=pq.read_table(root/"queries.parquet").to_pylist()
    query_text={str(x["_id"]):x["text"] for x in query_table}
    all_query_ids=np.load(data/"queries_ids.npy")
    words=re.compile(r"[a-z0-9]+")
    query_terms=[set(words.findall(query_text[str(all_query_ids[i])].lower())) for i in queries]
    vocabulary=set.union(*query_terms)
    doc_ids=np.load(data/"doc_ids.npy")
    ids={str(v):i for i,v in enumerate(doc_ids)}
    n=len(ids)
    lens=np.full(n,-1,dtype="i4")
    postings=defaultdict(list)
    for batch in pq.ParquetFile(root/"corpus.parquet").iter_batches(batch_size=512):
        for row in batch.to_pylist():
            i=ids[str(row["_id"])]
            terms=words.findall(((row.get("title") or "")+" "+row["text"]).lower())
            counts=Counter(terms)
            lens[i]=len(terms)
            for t in counts.keys() & vocabulary:
                postings[t].append((i,counts[t]))
    assert len(lens)==57638 and np.all(lens>=0)
    weights={}
    for term,hits in postings.items():
        doc,tf=np.asarray(hits).T
        idf=np.log1p((n-len(doc)+.5)/(len(doc)+.5))
        score=idf*2.2*tf/(tf+1.2*(.25+.75*lens[doc]/lens.mean()))
        weights[term]=doc,score
    out=[]
    loc=np.arange(n)
    for terms in query_terms:
        scores=np.zeros(n)
        for term in sorted(terms & weights.keys()):
            idx,values=weights[term]
            scores[idx]+=values
        out.append(np.lexsort((loc,-scores))[:200])
    return np.asarray(out)

def run(root):
    data=root/"fiqa"
    docs,offset,q,qlens=load_data(data)
    tr,held=splits(data)
    evaluation,warmup=held[:160],held[160:168]
    queries=np.r_[evaluation,warmup]
    means=_layout_means(docs,offset)
    train=[candidates_for(means,q[i,:qlens[i]]) for i in tr]
    dense=np.array([candidates_for(means,q[i,:qlens[i]]) for i in queries])
    target=root/"semantic"
    budget=int(docs.nbytes*.1)-(len(offset)-1)*(96*4+8)-offset.nbytes
    start=time.perf_counter()
    if not target.exists():
        build_full_index(docs,np.diff(offset),train,target,budget,"semantic")
    orig=root/"original"
    orig.mkdir(exist_ok=True)
    if not (orig/"documents.npy").exists():
        os.link(data/"documents.npy",orig/"documents.npy")
    for name in ("hot_ids.npy","offset.npy"):
        shutil.copyfile(target/name,orig/name)
    np.save(orig/"starts.npy",offset[:-1])
    spec=json.loads((target/"index.json").read_bytes())
    spec["layout"]="identity"
    (orig/"index.json").write_text(json.dumps(spec))
    placed=np.load(target/"documents.npy",mmap_mode="r")
    starts=np.load(target/"starts.npy")
    for i,a in enumerate(starts):
        np.testing.assert_array_equal(docs[offset[i]:offset[i+1]].view("u2"),placed[a:a+(offset[i+1]-offset[i])].view("u2"))
    print("Verified",len(starts),"relocated documents",round(time.perf_counter()-start,1),"s",flush=True)
    bm=lexical(root,queries)
    traces=dict(queries=queries,train=tr,eval_count=len(evaluation),dense=dense,bm25=bm,
                order=np.argsort(starts),source_revision="6fe7f906b70351d889119b771961b85145bb60a1")
    for k,lists in (("dense",dense),("bm25",bm)):
        traces[k+"_scores"]=np.stack([exact(q[i,:qlens[i]],docs,offset,ids) for i,ids in zip(queries,lists)])
    np.savez_compressed(root/"traces.npz",**traces)
    print("Frozen",len(tr),"fit,",len(evaluation),"test and",len(warmup),"warmups; candidate-list hash",
          hashlib.sha256(dense.tobytes()+bm.tobytes()).hexdigest(),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    run(p.parse_args().root)
