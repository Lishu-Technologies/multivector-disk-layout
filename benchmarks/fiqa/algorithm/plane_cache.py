"""Spend the same payload budget on frequently used precision levels of each record."""
import numpy as np


def admit(lengths, traffic, refinements, budget):
    if len(traffic)!=len(refinements):
        raise ValueError("Each calibration shortlist requires its own refined subset")
    n=len(lengths)
    counts=np.zeros((n,2),dtype=np.int64)
    for row,refined in zip(traffic,refinements):
        row=np.unique(row)
        refined=np.unique(refined)
        if row.dtype.kind not in "iu" or np.any(row<0) or np.any(row>=n):
            raise ValueError("Invalid shortlist")
        if len(refined) and (refined.dtype.kind not in "iu" or not np.isin(refined,row).all()):
            raise ValueError("Refinement IDs must belong to the same shortlist")
        counts[row,0]+=1
        if len(refined):
            counts[refined,1]+=1
    doc,level=np.indices(counts.shape).reshape(2,-1)
    priority=np.lexsort((doc,level,-counts.ravel()))
    chosen=np.zeros_like(counts,dtype=bool)
    for j in priority:
        i,l=doc[j],level[j]
        if counts[i,l]==0:
            break
        size=int(lengths[i])*96
        if size<=budget and (l==0 or chosen[i,0]):
            chosen[i,l]=True
            budget-=size
    return np.flatnonzero(chosen[:,1]),np.flatnonzero(chosen[:,0]&~chosen[:,1])
