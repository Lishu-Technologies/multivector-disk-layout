"""Read-only physical-slot audit for the pinned Qdrant 1.20.2 Windows segment format."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def audit(root):
    data=root/"fiqa"
    docs=np.load(data/"documents.npy",mmap_mode="r")
    lens=np.load(data/"doclens.npy").astype("i8")
    off=np.r_[0,np.cumsum(lens)]
    hashes=[hashlib.sha256(docs[a:b]).digest() for a,b in zip(off,off[1:])]
    order=np.load(root/"traces.npz")["order"]
    result={}
    for name,kind,expected in (("original","<f2",np.arange(len(lens))),("semantic","<f2",order),("semantic_f32","<f4",order),
                               ("placed","<f2",order),("placed_f32","<f4",order)):
        segments=root/"instance"/"storage"/"collections"/("fiqa_"+name)/"0"/"segments"
        if not segments.is_dir():
            continue
        seen=[]
        for seg in segments.iterdir():
            base=seg/"vector_storage"
            offsets=base/"offsets"
            if not offsets.is_dir():
                continue
            count=int.from_bytes((offsets/"status.dat").read_bytes(),"little")
            if count==0:
                continue
            rows=np.fromfile(offsets/"chunk_0.mmap",dtype="<u4",count=count*3).reshape(-1,3)
            vectors=base/"vectors"
            block=json.loads((vectors/"config.json").read_text())["chunk_size_vectors"]
            chunks={int(p.stem.split("_")[1]):np.memmap(p,mode="r",dtype=kind).reshape(-1,96) for p in vectors.glob("chunk_*.mmap")}
            part=[]
            for start,size,capacity in rows[np.argsort(rows[:,0])]:
                k,a=divmod(int(start),block)
                size=int(size)
                assert 0<size<=int(capacity)
                raw=chunks[k][a:a+size]
                if len(raw)<size:
                    raw=np.concatenate((raw,chunks[k+1][:size-len(raw)]))
                packed=raw.astype("<f2",copy=False)
                if kind=="<f4" and not np.array_equal(packed.astype("f4"),raw):
                    raise ValueError("Stored FP32 values differ from FP16 source alphabet")
                part.append(hashlib.sha256(packed).digest())
            seen.append(part)
        flat=[v for segment in seen for v in segment]
        assert Counter(flat)==Counter(hashes)
        # This corpus has a single populated segment per collection. If a future
        # optimizer splits it, comparing the arbitrary directory order is invalid.
        positional=sum(a==hashes[b] for a,b in zip(flat,expected)) if len(seen)==1 else None
        result[name]=dict(records=len(flat),populated_segments=len(seen),matching_contents=len(flat),
                          matching_physical_positions=positional,positions_matching_original=sum(a==b for a,b in zip(flat,hashes))
                          if len(seen)==1 else None,unique_document_contents=len(set(flat)))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("new_json",type=Path)
    a=p.parse_args()
    result=audit(a.root)
    with a.new_json.open("x") as f:
        json.dump(result,f,indent=2)
    print(result)
