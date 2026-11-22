"""Run both benchmark blocks strictly serially, audit files, stop the private server, test first access."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import requests

def run(root):
    src=Path(__file__).resolve().parent
    env=dict(os.environ,OPENBLAS_NUM_THREADS="1")
    def invoke(script,*args):
        subprocess.run([sys.executable,"-B","-X","utf8",str(src/script),*map(str,args)],env=env,check=True)
    for name in ("fiqa_original","fiqa_placed","fiqa_placed_f32"):
        if not (root/(name+".json")).is_file():
            raise RuntimeError("Finish all three Qdrant imports before measuring")
    verified=json.loads((root/"verified_order.json").read_bytes())
    for label in ("original","placed","placed_f32"):
        if verified[label]["matching_physical_positions"]!=57638:
            raise RuntimeError(f"Physical order has not been verified for {label}")
    for name,flags in (("main",()),("precision",("--precision-control",))):
        invoke("bench.py",root,root/(name+"_final.jsonl"),*flags)
    invoke("audit_files.py",root,root/"physical_final.json")
    (root/"instance"/"request-stop").touch()
    s=requests.Session()
    s.trust_env=False
    for _ in range(30):
        try:
            s.get("http://127.0.0.1:17633",timeout=1)
        except requests.ConnectionError:
            break
        time.sleep(.5)
    else:
        raise RuntimeError("Private source server did not exit")
    invoke("first_touch.py",root,root/"first_touch.jsonl",root/"fresh_copies")
    invoke("direct_subset.py",root,root/"direct_subset.jsonl")
    invoke("compare_subset.py",root/"first_touch.jsonl",root/"direct_subset.jsonl",root/"subset_stats.json")
    invoke("cold_stats.py",root/"first_touch.jsonl",root/"cold_stats.json")
    for name in ("main","precision"):
        invoke("summarize.py",root/(name+"_final.jsonl"),root/(name+"_stats.json"))
        invoke("summarize.py",root/(name+"_final.jsonl"),root/(name+"_steady.json"),"--after-full-pass")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data",type=Path)
    run(parser.parse_args().data.resolve())
