"""Compare Qdrant layouts on fresh non-buffered NTFS copies, then replay the same queries."""
import argparse
import ctypes
from ctypes import wintypes as w
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import requests

class RawCounter(ctypes.Structure):
    _fields_=[("status",w.DWORD),("time",w.FILETIME),("first",ctypes.c_longlong),
              ("second",ctypes.c_longlong),("multi",w.DWORD)]

class DriveReads:
    def __init__(self,drive):
        self.dll=ctypes.WinDLL("pdh")
        self.q=ctypes.c_void_p()
        self.h=ctypes.c_void_p()
        assert self.dll.PdhOpenQueryW(None,0,ctypes.byref(self.q))==0
        path=rf"\LogicalDisk({drive})\Disk Read Bytes/sec"
        assert self.dll.PdhAddEnglishCounterW(self.q,path,0,ctypes.byref(self.h))==0
    def sample(self):
        assert self.dll.PdhCollectQueryData(self.q)==0
        value=RawCounter()
        kind=w.DWORD()
        assert self.dll.PdhGetRawCounterValue(self.h,ctypes.byref(kind),ctypes.byref(value))==0
        assert value.status in (0,1)
        return value.first
    def close(self):
        self.dll.PdhCloseQuery(self.q)

def run(root,output,clones):
    root=root.resolve()
    clones=clones.resolve()
    clones.mkdir(parents=True,exist_ok=False)
    src=root/"instance"/"storage"
    trace=np.load(root/"traces.npz")
    query_vectors=np.load(root/"fiqa"/"queries.npy")
    qlens=np.load(root/"fiqa"/"query_lens.npy")
    logical_ids=trace["order"]
    storage_ids=np.argsort(logical_ids)
    query_ids=trace["queries"][:64]
    cases=[]
    for pool in ("dense","bm25"):
        for qi,ids,scores in zip(query_ids,trace[pool],trace[pool+"_scores"]):
            q=query_vectors[qi,:qlens[qi]]
            bodies={}
            for layout,candidates in (("original",ids),("semantic",storage_ids[ids])):
                bodies[layout]=json.dumps({"query":q.tolist(),"filter":{"must":[{"has_id":candidates.tolist()}]},"params":{"exact":True},
                                           "limit":10,"with_payload":False,"with_vector":False},separators=(",",":")).encode()
            want=ids[np.lexsort((ids,-scores))[:10]].tolist()
            cases.append((pool,int(qi),bodies,want))
    s=requests.Session()
    s.trust_env=False
    url="http://127.0.0.1:17653"
    try:
        s.get(url,timeout=1)
    except requests.ConnectionError:
        pass
    else:
        raise RuntimeError("The private test port is in use")
    counter=DriveReads(root.drive)
    with output.open("x",encoding="utf8") as out:
        for rd,layouts in enumerate((("original","semantic"),("semantic","original"))):
            seq=np.random.default_rng(91031+rd).permutation(len(cases))
            for layout in layouts:
                folder=clones/f"{rd}_{layout}"
                dst=folder/"storage"
                dst.mkdir(parents=True)
                shutil.copyfile(src/"raft_state.json",dst/"raft_state.json")
                collection="fiqa_"+("placed" if layout=="semantic" else "original")
                local=dst/"collections"/collection
                local.parent.mkdir()
                args=["robocopy",str(src/"collections"/collection),str(local),"/E","/J","/COPY:DAT",
                      "/R:1","/W:0","/NFL","/NDL","/NJH","/NJS","/NP"]
                cp=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                if cp.returncode>=8:
                    raise RuntimeError(cp.stdout.decode(errors="replace"))
                env=dict(os.environ,QDRANT__SERVICE__HOST="127.0.0.1",QDRANT__SERVICE__HTTP_PORT="17653",
                         QDRANT__SERVICE__GRPC_PORT="17654",QDRANT__CLUSTER__ENABLED="false",
                         QDRANT__TELEMETRY_DISABLED="true",QDRANT__LOG_LEVEL="WARN",QDRANT__STORAGE__LOW_MEMORY_MODE="no_populate")
                with (folder/"qdrant.log").open("wb") as log:
                    proc=subprocess.Popen([str(root/"qdrant.exe")],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,
                                          creationflags=subprocess.CREATE_NO_WINDOW)
                    try:
                        endpoint=url+"/collections/"+collection
                        for _ in range(120):
                            if proc.poll() is not None:
                                raise RuntimeError(f"Qdrant startup failed. See {folder}/qdrant.log")
                            try:
                                info=s.get(endpoint,timeout=2).json().get("result",{})
                                if info.get("status")=="green" and info["points_count"]==57638:
                                    break
                            except (requests.RequestException,ValueError):
                                pass
                            time.sleep(.25)
                        else:
                            raise RuntimeError("Timed out loading fresh collection")
                        idle=counter.sample()
                        time.sleep(.5)
                        quiet=counter.sample()-idle
                        for phase in ("first","replay"):
                            start=counter.sample()
                            begin=time.perf_counter()
                            for j in seq:
                                pool,qi,bodies,want=cases[j]
                                t=time.perf_counter()
                                reply=s.post(endpoint+"/points/query",data=bodies[layout],headers={"Content-Type":"application/json"},timeout=60)
                                reply.raise_for_status()
                                data=reply.json()
                                top=[p["id"] for p in data["result"]["points"]]
                                if layout=="semantic":
                                    top=logical_ids[top].tolist()
                                ms=(time.perf_counter()-t)*1000
                                out.write(json.dumps(dict(type="query",layout=layout,repeat=rd,phase=phase,pool=pool,
                                                         query=qi,ms=ms,server_ms=data["time"]*1000,ordered=top==want,top=top))+"\n")
                            elapsed=time.perf_counter()-begin
                            read=counter.sample()-start
                            out.write(json.dumps(dict(type="phase",layout=layout,repeat=rd,phase=phase,
                                                     seconds=elapsed,drive_read_bytes=read,idle_half_second_bytes=quiet,
                                                     copied_with_robocopy_J=cp.returncode,queries=len(seq)))+"\n")
                            out.flush()
                            print(rd+1,layout,phase,round(elapsed,2),"s, D: reads",round(read/2**20,1),"MiB",flush=True)
                    finally:
                        if proc.poll() is None:
                            proc.terminate()
                        proc.wait(timeout=20)
    counter.close()
    s.close()

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("data",type=Path)
    p.add_argument("new_jsonl",type=Path)
    p.add_argument("new_clone_directory",type=Path)
    a=p.parse_args()
    run(a.data,a.new_jsonl,a.new_clone_directory)
