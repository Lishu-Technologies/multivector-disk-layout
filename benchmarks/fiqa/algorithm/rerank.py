"""Build and query a standalone local float16 MaxSim index. Document IDs are row numbers."""
import argparse
import json
from pathlib import Path
import uuid
import numpy as np
import torch
from blockio import BlockReader
from planes import scalar_code
from hybrid import Hybrid, hybrid_score, full_score
from engine import stable_top
from layout import layout_order, physical_starts
from plane_cache import admit


def _layout_means(docs,offset):
    pooled=np.asarray([docs[a:b].mean(axis=0,dtype="f4") for a,b in zip(offset,offset[1:])],dtype="f8")
    if not np.isfinite(pooled).all():
        raise ValueError("Expected finite document tokens")
    return pooled/np.maximum(np.linalg.norm(pooled,axis=1,keepdims=True),1e-12)


def _write(documents, lengths, calibration, destination, hot_order, cache_bytes, traffic, layout_method, refinements):
    """documents [sum(lengths),96] fp16. calibration [number of query tokens,96]."""
    x=np.asarray(documents)
    lengths=np.asarray(lengths)
    q=np.asarray(calibration,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=96 or x.dtype!=np.float16:
        raise ValueError("Expected a [tokens, 96] FP16 array")
    if lengths.ndim!=1 or lengths.dtype.kind not in "iu" or not len(lengths) or np.any(lengths<=0) or sum(lengths)!=len(x):
        raise ValueError("Document lengths must be positive integers and cover the token array")
    if q.ndim!=2 or q.shape[1]!=96 or not len(q) or not np.isfinite(q).all():
        raise ValueError("Expected finite calibration query tokens")
    if cache_bytes<0 or (refinements is not None and (traffic is None or hot_order is not None)):
        raise ValueError("Use a nonnegative budget and provide traffic without hot_order for plane admission")
    root=Path(destination)
    root.mkdir(parents=True,exist_ok=False)
    offset=np.r_[0,np.cumsum(lengths,dtype=np.int64)]
    if traffic is not None:
        traffic=list(traffic)
        means=_layout_means(x,offset)
        order=layout_order(means,lengths,traffic,layout_method)
        starts=physical_starts(order,offset)
        if hot_order is None and cache_bytes>0 and refinements is None:
            counts=np.zeros(len(lengths),dtype=np.int64)
            for row in traffic:
                counts[np.unique(row)]+=1
            hot_order=np.lexsort((np.arange(len(lengths)),-counts))
    counts=np.zeros(65536,dtype=np.int64)
    for a in range(0,len(x),16384):
        batch=x[a:a+16384]
        mags=np.linalg.norm(batch.astype("f8"),axis=1)
        if not np.isfinite(mags).all() or np.any(mags>1.01):
            raise ValueError("Only finite document token vectors with norm <= 1.01 are supported")
        counts+=np.bincount(batch.view("u2").ravel(),minlength=65536)
    encode,decode,tables=scalar_code(counts,True)
    axis=np.linalg.svd(q,full_matrices=False)[2][0].astype("f4")
    projection=np.empty((len(x),2),dtype="f2")
    leak=0.
    with (root/"high.bin").open("xb") as hi,(root/"rest.bin").open("xb") as lo:
        for a in range(0,len(x),16384):
            b=min(a+16384,len(x))
            code=encode[x[a:b].view("u2")]
            high=(code>>8).astype("u1")
            high.tofile(hi)
            (code&255).astype("u1").tofile(lo)
            delta=x[a:b].astype("f8")-tables[0][high].astype("f8")
            coeff=(delta@axis).astype("f2")
            rem=delta-coeff.astype("f8")[:,None]*axis[None,:]
            projection[a:b,0]=coeff
            projection[a:b,1]=np.nextafter((np.linalg.norm(rem,axis=1)+1e-9).astype("f2"),np.float16(np.inf))
            leak=max(leak,np.abs(rem@axis).max())
    if traffic is not None:
        for name in ("high.bin","rest.bin"):
            source=np.memmap(root/name,mode="r",dtype="u1",shape=x.shape)
            new_file=root/(name+".placed")
            with new_file.open("xb") as out:
                for i in order:
                    source[offset[i]:offset[i+1]].tofile(out)
            del source
            new_file.replace(root/name)
        np.save(root/"starts.npy",starts)
    selected=[]
    if refinements is not None:
        selected,high_only=admit(lengths,traffic,list(refinements),cache_bytes)
        np.save(root/"hot_high_ids.npy",high_only)
    elif hot_order is not None:
        for doc in dict.fromkeys(int(i) for i in hot_order):
            if not 0<=doc<len(lengths):
                raise ValueError("Invalid hot document ID")
            needed=int(lengths[doc])*192
            if needed<=cache_bytes:
                selected.append(doc)
                cache_bytes-=needed
    for name,values in [("offset",offset),("hot_ids",np.array(selected,dtype="i8")),("decode",decode),
                        ("center0",tables[0]),("oneaxis",axis),("projection",projection),("oneleak",np.array(leak+1e-8))]:
        np.save(root/f"{name}.npy",values)
    # Readers open an index only after all binary data and metadata have been written.
    spec={"version":1 if traffic is None else 2,"dims":96,"documents":len(lengths)}
    if traffic is not None:
        spec.update(layout=layout_method,calibration_lists=len(traffic),plane_cache=refinements is not None)
    (root/"index.json").write_text(json.dumps(spec),encoding="utf8")


def _publish(destination,write):
    root=Path(destination)
    if root.exists():
        raise FileExistsError(root)
    staging=root.with_name(root.name+"."+uuid.uuid4().hex+".partial")
    try:
        write(staging)
        staging.rename(root)
    finally:
        if staging.exists():
            for entry in staging.iterdir():
                entry.unlink()
            staging.rmdir()


def build_index(documents, lengths, calibration, destination, hot_order=None, cache_bytes=0,
                traffic=None, layout_method="blend", refinements=None):
    """Publish a two-plane index under a new path. Failed attempts can be retried."""
    _publish(destination,lambda root:_write(documents,lengths,calibration,root,hot_order,cache_bytes,
                                            traffic,layout_method,refinements))


def build_full_index(documents,lengths,traffic,destination,cache_bytes=0,layout_method="semantic"):
    """Physically group full FP16 records and cache popular documents. No token quantization."""
    docs=np.asarray(documents)
    lengths=np.asarray(lengths)
    if (docs.ndim!=2 or docs.shape[1]!=96 or docs.dtype!=np.float16 or lengths.ndim!=1
            or not len(lengths) or lengths.dtype.kind not in "iu" or np.any(lengths<=0)
            or int(lengths.sum())!=len(docs) or cache_bytes<0):
        raise ValueError("Expected FP16 [tokens,96], positive document lengths and nonnegative cache budget")
    offsets=np.r_[0,np.cumsum(lengths,dtype=np.int64)]
    means=_layout_means(docs,offsets)
    traffic=list(traffic)
    order=layout_order(means,lengths,traffic,layout_method)
    freq=np.zeros(len(lengths),dtype=np.int64)
    for row in traffic:
        freq[np.unique(row)]+=1
    hot=[]
    for i in np.lexsort((np.arange(len(lengths)),-freq)):
        cost=int(lengths[i])*192
        if cost<=cache_bytes:
            hot.append(i)
            cache_bytes-=cost
    def write(root):
        root.mkdir(parents=True,exist_ok=False)
        out=np.lib.format.open_memmap(root/"documents.npy",mode="w+",dtype=docs.dtype,shape=docs.shape)
        starts=physical_starts(order,offsets)
        for i in order:
            out[starts[i]:starts[i]+lengths[i]]=docs[offsets[i]:offsets[i+1]]
        out.flush()
        del out
        for name,arr in (("starts",starts),("offset",offsets),("hot_ids",np.asarray(hot,dtype="i8"))):
            np.save(root/(name+".npy"),arr)
        spec=dict(version=2,storage="full",dims=96,documents=len(lengths),layout=layout_method)
        (root/"index.json").write_text(json.dumps(spec),encoding="utf8")
    _publish(destination,write)


class Ranker:
    def __init__(self,index,device="cpu",windows_direct=False):
        self.index=Path(index)
        spec=json.loads((self.index/"index.json").read_bytes())
        if spec["version"] not in (1,2) or spec["dims"]!=96:
            raise ValueError("Unsupported index")
        if torch.is_autocast_enabled() or (device.startswith("cuda") and torch.backends.cuda.matmul.allow_tf32):
            raise ValueError("Disable mixed precision and TF32 for certified bounds")
        self.device=device
        self.offset=np.load(self.index/"offset.npy")
        starts=np.load(self.index/"starts.npy") if spec["version"]==2 else None
        if starts is not None:
            if starts.shape!=(len(self.offset)-1,) or starts.dtype.kind not in "iu":
                raise ValueError("Invalid physical address map")
            if not np.array_equal(physical_starts(np.argsort(starts),self.offset),starts):
                raise ValueError("Physical records do not tile the index")
        self.cache={}
        self.high_cache={}
        self.full=spec.get("storage")=="full"
        if self.full:
            raw=np.load(self.index/"documents.npy",mmap_mode="r")
            self.header=raw.offset
            for i in np.load(self.index/"hot_ids.npy"):
                self.cache[i]=np.array(raw[starts[i]:starts[i]+self.offset[i+1]-self.offset[i]])
            self.reader=BlockReader(windows_direct,physical_starts=starts)
            return
        self.scorer=Hybrid(self.index,device)
        shape=(self.offset[-1],96)
        hi,lo=[np.memmap(self.index/f,mode="r",dtype="u1",shape=shape) for f in ["high.bin","rest.bin"]]
        for i in np.load(self.index/"hot_ids.npy"):
            a=self.offset[i] if starts is None else starts[i]
            b=a+(self.offset[i+1]-self.offset[i])
            self.cache[i]=self.scorer.decode[(hi[a:b].astype("u2")<<8)|lo[a:b]]
        if spec.get("plane_cache"):
            for i in np.load(self.index/"hot_high_ids.npy"):
                a=starts[i]
                self.high_cache[i]=np.array(hi[a:a+(self.offset[i+1]-self.offset[i])])
        self.reader=BlockReader(windows_direct,physical_starts=starts)

    def topk(self,query,candidate_ids,k=10):
        ids=np.asarray(candidate_ids)
        if (ids.ndim!=1 or ids.dtype.kind not in "iu" or len(np.unique(ids))!=len(ids) or
                not 0<k<=len(ids) or ids.min()<0 or ids.max()>=len(self.offset)-1):
            raise ValueError("Expected distinct valid candidate row IDs and 0 < k <= candidate count")
        q=np.ascontiguousarray(query,dtype="f4")
        if q.ndim!=2 or q.shape[1]!=96 or not len(q) or not np.isfinite(q).all() or np.linalg.norm(q.astype("f8"),axis=1).max()>1.01:
            raise ValueError("Expected finite [tokens,96] query with per-token norm <= 1.01")
        if self.device.startswith("cuda") and torch.backends.cuda.matmul.allow_tf32:
            raise ValueError("Disable TF32 for certified bounds")
        with torch.autocast(device_type=torch.device(self.device).type,enabled=False):
            if self.full:
                scores,_,_=full_score(self.index,self.reader,self.offset,ids,q,self.cache,self.header,self.device)
                return ids[stable_top(ids,scores,k)]
            return hybrid_score(self.index,self.reader,self.offset,ids,q,self.cache,self.scorer,self.device,k,self.high_cache)[0]

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.reader.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    ap=p.add_subparsers(dest="action",required=True)
    b=ap.add_parser("build")
    for name in ("documents","lengths","calibration","destination"):
        b.add_argument(name,type=Path)
    b.add_argument("--traffic",type=Path,help="Integer matrix of calibration shortlists in original document row IDs")
    b.add_argument("--refinements",type=Path,help="JSON list of calibration document ID lists requiring the low plane")
    b.add_argument("--layout",choices=("blend","semantic","traffic","frequency"),default="blend")
    b.add_argument("--cache-bytes",type=int,default=0,help="Payload budget for hot documents selected from --traffic")
    f=ap.add_parser("build-full",help="Build a physically grouped full-precision index")
    for name in ("documents","lengths","traffic","destination"):
        f.add_argument(name,type=Path)
    f.add_argument("--layout",choices=("blend","semantic","traffic","frequency"),default="semantic")
    f.add_argument("--cache-bytes",type=int,default=0)
    run=ap.add_parser("query")
    for name in ("index","query","candidates"):
        run.add_argument(name,type=Path)
    run.add_argument("-k",type=int,default=10)
    run.add_argument("--device",default="cpu")
    a=p.parse_args()
    if a.action=="build":
        if a.cache_bytes<0 or ((a.cache_bytes or a.refinements) and not a.traffic):
            p.error("--cache-bytes must be nonnegative and a positive budget requires --traffic")
        build_index(np.load(a.documents,mmap_mode="r"),np.load(a.lengths),np.load(a.calibration),a.destination,
                    traffic=np.load(a.traffic) if a.traffic else None,layout_method=a.layout,cache_bytes=a.cache_bytes,
                    refinements=json.loads(a.refinements.read_bytes()) if a.refinements else None)
    elif a.action=="build-full":
        build_full_index(np.load(a.documents,mmap_mode="r"),np.load(a.lengths),np.load(a.traffic),
                         a.destination,a.cache_bytes,a.layout)
    else:
        torch.backends.cuda.matmul.allow_tf32=False
        with Ranker(a.index,a.device) as index:
            print(json.dumps(index.topk(np.load(a.query),np.load(a.candidates),a.k).tolist()))
