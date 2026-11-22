"""Recompute expected IDs and summaries directly from the archived trace and raw measurements."""
import argparse
import json
from pathlib import Path
import numpy as np
from summarize import summary
from cold_stats import summarize as cold_summary
from compare_subset import compare

def verify(root):
    trace=np.load(root/"traces.npz")
    n=int(trace["eval_count"])
    assert len(set(trace["train"])&set(trace["queries"]))==0
    gold={}
    for pool in ("dense","bm25"):
        for i,ids,scores in zip(trace["queries"][:n],trace[pool],trace[pool+"_scores"]):
            assert len(set(ids))==len(ids)==200
            gold[pool,int(i)]=(set(ids),ids[np.lexsort((ids,-scores))[:10]].tolist())
    for filename,count in (("main_final",5760),("precision_final",3840),("first_touch",1024),("direct_subset",512)):
        k=0
        with (root/(filename+".jsonl")).open() as f:
            for row in map(json.loads,f):
                if "top" not in row:
                    continue
                ids,truth=gold[row["pool"],row["query"]]
                top=row["top"]
                assert len(top)==len(set(top))==10 and set(top)<=ids and row["ms"]>0
                if "ordered" in row:
                    assert row["ordered"]==(top==truth)
                if "reference" in row:
                    assert row["reference"]==truth
                if filename=="direct_subset" or row.get("engine") in ("cpu_file","cuda_direct"):
                    assert top==truth
                k+=1
        assert k==count
    for name in ("main","precision"):
        assert summary(root/(name+"_final.jsonl"))==json.loads((root/(name+"_stats.json")).read_bytes())
        assert summary(root/(name+"_final.jsonl"),True)==json.loads((root/(name+"_steady.json")).read_bytes())
    assert cold_summary(root/"first_touch.jsonl")==json.loads((root/"cold_stats.json").read_bytes())
    assert compare(root/"first_touch.jsonl",root/"direct_subset.jsonl")==json.loads((root/"subset_stats.json").read_bytes())
    order=json.loads((root/"physical_final.json").read_bytes())
    assert all(order[k]["matching_physical_positions"]==57638 for k in ("original","placed","placed_f32"))
    print("Verified 11,136 query-method observations, six summaries, candidate IDs and the physical-order audit.")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("results",type=Path)
    verify(p.parse_args().results)
