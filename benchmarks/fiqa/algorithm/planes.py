"""Lossless FP16 planes, optionally recoded using a shared monotonic scalar dictionary."""
import argparse
from pathlib import Path
import numpy as np


def scalar_code(counts, adaptive):
    bits = np.arange(65536,dtype=np.uint16)
    present = bits[counts>0]
    ordered = present[np.argsort(present.view(np.float16).astype(float),kind="stable")]
    if not np.isfinite(ordered.view(np.float16)).all():
        raise ValueError("Nonfinite FP16 input")
    encode = bits.copy()
    if adaptive:
        values = np.arcsinh(ordered.view(np.float16).astype(float)/0.08)
        def groups(width):
            left = 0
            while left<len(values):
                right = min(left+256,np.searchsorted(values,values[left]+width,side="right"))
                yield left,right
                left = right
        lo,hi = 0.0,float(np.ptp(values))+1e-7
        for _ in range(42):
            mid = (lo+hi)/2
            if sum(1 for _ in groups(mid))<=256:
                hi = mid
            else:
                lo = mid
        for i,(left,right) in enumerate(groups(hi)):
            sub = np.arange(right-left)*256//(right-left)
            encode[ordered[left:right]] = i*256+sub
    codes = encode[ordered]
    if len(np.unique(codes))!=len(ordered):
        raise ValueError("Non-injective code on the indexed alphabet")
    decode = np.zeros(65536,dtype=np.uint16)
    decode[codes] = ordered
    reps = []
    for shift in (8,4):
        group = codes >> shift
        n = 65536>>shift
        weight = counts[ordered]
        sums = np.bincount(group,weights=weight*ordered.view(np.float16).astype(float),minlength=n)
        mass = np.bincount(group,weights=weight,minlength=n)
        reps.append((sums/np.maximum(1,mass)).astype(np.float32))
    return encode,decode,reps


def pack_nibble(x):
    return ((x[:,::2]<<4)|x[:,1::2]).astype(np.uint8)


def unpack_nibble(raw):
    packed = np.asarray(raw,dtype=np.uint8)
    full = np.empty((len(packed),packed.shape[1]*2),dtype=np.uint8)
    full[:,::2] = packed>>4
    full[:,1::2] = packed&15
    return full


def build_planes(root, target, adaptive):
    target.mkdir(parents=True,exist_ok=True)
    docs = np.load(root/"documents.npy",mmap_mode="r")
    counts = np.zeros(65536,dtype=np.int64)
    for i in range(0,len(docs),12000):
        counts += np.bincount(docs[i:i+12000].view(np.uint16).ravel(),minlength=65536)
    encode,decode,reps = scalar_code(counts,adaptive)
    np.save(target/"decode.npy",decode)
    for i,table in enumerate(reps):
        np.save(target/f"center{i}.npy",table)
    lengths = np.load(root/"doclens.npy").astype(np.int64)
    offset = np.r_[0,np.cumsum(lengths)]
    angles = np.empty((len(docs),2),dtype=np.float16)
    norm_ranges = np.empty((len(lengths),2),dtype=np.float32)
    with (target/"high.bin").open("wb") as hi, (target/"middle.bin").open("wb") as mid, (target/"low.bin").open("wb") as low:
        for idx in range(0,len(lengths),64):
            stop = min(idx+64,len(lengths))
            start,end = offset[idx],offset[stop]
            x = docs[start:end]
            codes = encode[x.view(np.uint16)]
            if not np.array_equal(decode[codes],x.view(np.uint16)):
                raise AssertionError("Bitwise round-trip failed")
            (codes>>8).astype("u1").tofile(hi)
            pack_nibble((codes>>4)&15).tofile(mid)
            pack_nibble(codes&15).tofile(low)
            true = x.astype(np.float64)
            mag = np.linalg.norm(true,axis=1)
            if mag.min()<=0 or mag.max()>1.01:
                raise ValueError("Expected finite near-unit nonzero vectors")
            rel = offset[idx:stop]-start
            norm_ranges[idx:stop,0] = np.nextafter(np.minimum.reduceat(mag,rel).astype("f4"),np.float32(-np.inf))
            norm_ranges[idx:stop,1] = np.nextafter(np.maximum.reduceat(mag,rel).astype("f4"),np.float32(np.inf))
            true /= mag[:,None]
            for j,shift in enumerate((8,4)):
                proxy = reps[j][codes>>shift].astype(np.float64)
                size = np.linalg.norm(proxy,axis=1)
                proxy /= np.maximum(size[:,None],1e-100)
                chord = np.linalg.norm(true-proxy,axis=1)
                angle = 2*np.arcsin(np.minimum(1,chord/2))
                angle[size==0] = np.pi
                angles[start:end,j] = np.nextafter((angle+1e-8).astype("f2"),np.float16(np.inf))
    np.save(target/"angles.npy",angles)
    np.save(target/"norms.npy",norm_ranges)
    print("Lossless planes written",target,flush=True)


class BoundScorer:
    """Reference CPU path. Both caller-selected candidates and output ordering are explicit."""
    def __init__(self,index,device="cpu"):
        import torch
        self.torch = torch
        self.device = device
        self.angles = np.load(index/"angles.npy",mmap_mode="r")
        self.norms = np.load(index/"norms.npy")
        self.centers = [torch.from_numpy(np.load(index/f"center{i}.npy")).to(device) for i in (0,1)]
        self.decode = np.load(index/"decode.npy").view(np.float16)

    def bounds(self,ids,offset,query,prefix,rows,stage):
        t = self.torch
        lens = offset[ids+1]-offset[ids]
        starts = np.r_[0,np.cumsum(lens)[:-1]]
        cos_angles = self.angles[rows,stage].astype("f4")
        indices = t.from_numpy(prefix.astype("i4")).to(self.device)
        proxy = self.centers[stage][indices]
        proxy = t.nn.functional.normalize(proxy,dim=1)
        q = t.from_numpy(np.asarray(query,dtype="f4")).to(self.device)
        magnitudes = t.linalg.vector_norm(q,dim=1)
        q = t.nn.functional.normalize(q,dim=1)
        c = (proxy@q.T).clamp(-1,1)
        theta = t.from_numpy(cos_angles).to(self.device)[:,None]
        # 96D FP32 cosine error incl. norm + matrix multiplication is < 5e-5
        # at these magnitudes (TF32 and mixed precision disabled in both entry points).
        # Widen the cosine before applying the angle transformation.
        hi = (c+0.00005).clamp(-1,1)
        lo = (c-0.00005).clamp(-1,1)
        cos_a,sin_a = t.cos(theta),t.sin(theta)
        amax = t.where((hi>=cos_a)|(theta>=np.pi),1.,
                       hi*cos_a+t.sqrt(((1-hi)*(1+hi)).clamp_min(0))*sin_a)
        amin = t.where((lo<=-cos_a)|(theta>=np.pi),-1.,
                       lo*cos_a-t.sqrt(((1-lo)*(1+lo)).clamp_min(0))*sin_a)
        # First reduce in token space; common document-wide norm intervals apply
        # monotonically to both positive and negative maxima.
        bounds = []
        for values,is_upper in ((amin,False),(amax,True)):
            # One maximum per document and query token, without copying the score matrix to CPU.
            ind = t.repeat_interleave(t.arange(len(ids),device=self.device),t.as_tensor(lens,device=self.device))
            maxima = t.full((len(ids),c.shape[1]),-t.inf,device=self.device)
            maxima.scatter_reduce_(0,ind[:,None].expand_as(values),values,reduce="amax",include_self=True)
            maxima = maxima.cpu().numpy().astype("f8")
            left,right = self.norms[ids,0,None],self.norms[ids,1,None]
            mag = np.where(maxima>=0,right if is_upper else left,left if is_upper else right)
            score = ((maxima*mag)*np.linalg.norm(query.astype("f8"),axis=1)).sum(axis=1)
            bounds.append(score+(1 if is_upper else -1)*0.00001*(1+len(query)))
        return bounds


def surviving(ids, lo, hi,k):
    if k>=len(ids):
        return np.arange(len(ids))
    cutoff = np.partition(lo,-k)[-k]
    return np.flatnonzero(hi>=cutoff)


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("corpus",type=Path)
    p.add_argument("index",type=Path)
    p.add_argument("--direct",action="store_true")
    a=p.parse_args()
    build_planes(a.corpus,a.index,not a.direct)
