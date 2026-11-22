"""Directional residual certificates for the high-byte prefix of a lossless index."""
import argparse
from pathlib import Path
import numpy as np
import torch
from planes import BoundScorer
from study import splits, load_data


def build(root,index):
    docs,offset,queries,lens=load_data(root)
    train,_=splits(root)
    q=np.concatenate([queries[i,:lens[i]] for i in train]).astype("f8")
    _,_,right=np.linalg.svd(q,full_matrices=False)
    basis=right[:4].T.astype("f4")
    u=basis.astype("f8")
    high=np.memmap(index/"high.bin",mode="r",dtype="u1",shape=docs.shape)
    center=np.load(index/"center0.npy")
    meta=np.empty((len(docs),5),dtype="f2")
    leak=0.0
    for a in range(0,len(docs),16384):
        b=min(a+16384,len(docs))
        residual=docs[a:b].astype("f8")-center[high[a:b]].astype("f8")
        coeff=(residual@u).astype("f2")
        rest=residual-coeff.astype("f8")@u.T
        radius=np.linalg.norm(rest,axis=1)
        leak=max(leak,np.linalg.norm(rest@u,axis=1).max())
        meta[a:b,:4]=coeff
        meta[a:b,4]=np.nextafter((radius+1e-9).astype("f2"),np.float16(np.inf))
    np.save(index/"directions.npy",basis)
    np.save(index/"residual.npy",meta)
    np.save(index/"leak.npy",np.array(float(leak)+1e-8))
    print("Residual certificate built",root.name,flush=True)


class DirectionScorer(BoundScorer):
    def __init__(self,index,device="cpu"):
        super().__init__(index,device)
        self.basis=torch.from_numpy(np.load(index/"directions.npy")).to(device)
        self.residual=np.load(index/"residual.npy")
        self.leak=float(np.load(index/"leak.npy"))

    def bounds(self,ids,offset,query,prefix,rows,stage):
        if stage!=0:
            return super().bounds(ids,offset,query,prefix,rows,stage)
        q=torch.from_numpy(query).to(self.device)
        coefficients=q@self.basis
        qrest=(q-coefficients@self.basis.T).norm(dim=1)
        indices=torch.from_numpy(prefix.astype("i4")).to(self.device)
        center=self.centers[0][indices]
        meta=torch.from_numpy(self.residual[rows].astype("f4")).to(self.device)
        score=center@q.T + meta[:,:4]@coefficients.T
        width=meta[:,4,None]*qrest + self.leak*coefficients.norm(dim=1)
        lens=offset[ids+1]-offset[ids]
        doc_index=torch.repeat_interleave(torch.arange(len(ids),device=self.device),torch.as_tensor(lens,device=self.device))
        bounds=[]
        for value,sign in ((score-width,-1),(score+width,1)):
            maxima=torch.full((len(ids),len(q)),-torch.inf,device=self.device)
            maxima.scatter_reduce_(0,doc_index[:,None].expand_as(value),value,reduce="amax",include_self=True)
            # Unit-scale vectors, 96D fp32 dot products, TF32 disabled. Reserve
            # 5e-5 per token (over four times the conservative dot error budget).
            bounds.append(maxima.sum(dim=1,dtype=torch.float64).cpu().numpy()+sign*0.00005*(1+len(q)))
        return bounds


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("corpus",type=Path)
    p.add_argument("index",type=Path)
    a=p.parse_args()
    build(a.corpus,a.index)
