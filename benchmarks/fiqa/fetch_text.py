from pathlib import Path
import argparse
import requests
from fetch import get_file

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("data",type=Path)
    root=p.parse_args().data
    rev="979c07a7cb5ccc6ca009792241fa1250b98055dd"
    url="https://huggingface.co/api/datasets/BeIR/fiqa/tree/"+rev
    r=requests.get(url,params={"recursive":"true"},timeout=20)
    r.raise_for_status()
    for item in r.json():
        path=item["path"]
        if path.endswith(".parquet"):
            target=root/(path.split("/")[0]+".parquet")
            get_file(f"https://huggingface.co/datasets/BeIR/fiqa/resolve/{rev}/{path}",target,item["size"],item["lfs"]["oid"])
