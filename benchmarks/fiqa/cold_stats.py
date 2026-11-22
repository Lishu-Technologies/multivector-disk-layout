"""Paired layout comparisons within startup and replay phases."""
import argparse
import json
from pathlib import Path
import numpy as np

def summarize(path):
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    phases=[r for r in rows if r["type"]=="phase"]
    obs=[r for r in rows if r["type"]=="query"]
    assert len(phases)==8 and len(obs)==1024
    result={"observations":len(obs),"phases":phases,"comparisons":{}}
    for phase in ("first","replay"):
        for kind in ("dense","bm25","both"):
            arms={}
            dist={}
            for layout in ("original","semantic"):
                rs=sorted((r for r in obs if r["phase"]==phase and r["layout"]==layout and (kind=="both" or r["pool"]==kind)),
                          key=lambda r:(r["query"],r["pool"],r["repeat"]))
                assert len(rs)==(256 if kind=="both" else 128)
                if layout=="original":
                    positions=[(r["query"],r["pool"],r["repeat"]) for r in rs]
                else:
                    assert positions==[(r["query"],r["pool"],r["repeat"]) for r in rs]
                ms=np.array([r["ms"] for r in rs]).reshape(64,-1,2)
                draw=np.random.default_rng(9230).integers(64,size=(3000,64))
                dist[layout]=np.quantile(ms[draw],.95,axis=(1,2,3))
                arms[layout]=dict(p95=float(np.quantile(ms,.95)),p50=float(np.median(ms)),runs=len(rs),
                                  p95_each=np.quantile(ms,.95,axis=(0,1)).tolist(),ordered=sum(r["ordered"] for r in rs))
            pct=100*(1-arms["semantic"]["p95"]/arms["original"]["p95"])
            boot=100*(1-dist["semantic"]/dist["original"])
            result["comparisons"][phase+"/"+kind]=dict(arms=arms,p95_reduction_pct=pct,
                                                       ci95=np.quantile(boot,[.025,.975]).tolist())
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("raw",type=Path)
    p.add_argument("new_json",type=Path)
    a=p.parse_args()
    with a.new_json.open("x") as f:
        json.dump(summarize(a.raw),f,indent=2)
