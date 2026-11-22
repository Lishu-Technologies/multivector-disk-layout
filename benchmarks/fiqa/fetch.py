"""Fetch the pinned public FiQA vectors and the official Windows Qdrant binary."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import time
import zipfile
import requests

REV="6fe7f906b70351d889119b771961b85145bb60a1"
REPO="robro612/fiqa_answerai_colbert_small"
TAG="v1.20.2"
DIGEST="7a5290b9ef1b204b250a1626ddafec580647ba3cc62ba1630b3a8efdc2718b32"

def checksum(path,oid,size):
    h=hashlib.sha256() if len(oid)==64 else hashlib.sha1()
    if len(oid)==40:
        h.update(f"blob {size}\0".encode())
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(2**20),b""):
            h.update(chunk)
    return h.hexdigest()

def get_file(url,path,size,oid):
    if path.exists() and checksum(path,oid,size)==oid:
        return
    tmp=path.with_suffix(".partial")
    def fetch_part(span):
        a,b=span
        part=path.with_name(f"{path.name}.{a}.part")
        if part.exists() and part.stat().st_size==b-a:
            return part
        multiple=size>64*2**20
        for attempt in range(5):
            try:
                headers={"Range":f"bytes={a}-{b-1}"} if multiple else {}
                u=url+f"?download=true&part={a}" if multiple else url
                with requests.get(u,headers=headers,stream=True,timeout=(20,100)) as r:
                    r.raise_for_status()
                    if multiple and (r.status_code!=206 or r.headers.get("Content-Range","").split("/")[0]!=f"bytes {a}-{b-1}"):
                        raise ValueError("Server did not return the requested byte interval")
                    with part.open("wb") as f:
                        for c in r.iter_content(2**20):
                            f.write(c)
                if part.stat().st_size!=b-a:
                    raise IOError("Incomplete part")
                return part
            except (requests.RequestException,IOError,ValueError):
                if attempt==4:
                    raise
                time.sleep(1+attempt)
    step=32*2**20 if size>64*2**20 else size
    spans=[(a,min(a+step,size)) for a in range(0,size,step)]
    with ThreadPoolExecutor(6) as pool,tmp.open("wb") as f:
        for part in pool.map(fetch_part,spans):
            with part.open("rb") as src:
                for c in iter(lambda:src.read(2**20),b""):
                    f.write(c)
    if checksum(tmp,oid,size)!=oid:
        raise ValueError(f"Checksum mismatch {path.name}")
    tmp.replace(path)
    for a,_ in spans:
        path.with_name(f"{path.name}.{a}.part").unlink()
    print("Verified",path.name,size,flush=True)

def run(root):
    root.mkdir(parents=True,exist_ok=True)
    root=(root.resolve())
    z=root/"qdrant.zip"
    get_file(f"https://github.com/qdrant/qdrant/releases/download/{TAG}/qdrant-x86_64-pc-windows-msvc.zip",z,32545356,DIGEST)
    with zipfile.ZipFile(z) as package:
        exe=next(n for n in package.namelist() if Path(n).name.lower()=="qdrant.exe")
        (root/"qdrant.exe").write_bytes(package.read(exe))
    r=requests.get(f"https://huggingface.co/api/datasets/{REPO}/tree/{REV}",timeout=25)
    r.raise_for_status()
    items={i["path"]:i for i in r.json()}
    data=root/"fiqa"
    data.mkdir(exist_ok=True)
    for name in ("doclens.npy","queries.npy","query_lens.npy","queries_ids.npy","doc_ids.npy","documents.npy"):
        i=items[name]
        get_file(f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/{name}",data/name,i["size"],i.get("lfs",{}).get("oid") or i["oid"])

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("destination",type=Path)
    run(p.parse_args().destination)
