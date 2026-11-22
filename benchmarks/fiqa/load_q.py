"""Load all logical IDs into two real Qdrant collections in the requested insertion orders."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import grpc
import numpy as np
import requests
from qdrant_client import grpc as api

URL="http://127.0.0.1:17633"

def session():
    s=requests.Session()
    s.trust_env=False
    return s

def run(root,layout,datatype,physical_ids=False):
    s=session()
    server=s.get(URL,timeout=5).json()
    assert server["version"]=="1.20.2"
    data=root/"fiqa"
    docs=np.load(data/"documents.npy",mmap_mode="r")
    lengths=np.load(data/"doclens.npy").astype("i8")
    offset=np.r_[0,np.cumsum(lengths)]
    order=np.arange(len(lengths)) if layout=="original" else np.argsort(np.load(root/layout/"starts.npy"))
    if physical_ids and layout!="semantic":
        raise ValueError("Position-based IDs are intended for the semantic layout")
    assigned=np.argsort(order) if physical_ids else np.arange(len(order))
    name="fiqa_"+("placed" if physical_ids else layout)+("_f32" if datatype=="float32" else "")
    url=URL+"/collections/"+name
    checkpoint=root/(name+".cursor")
    info=s.get(url,timeout=10)
    if info.status_code==404:
        config={"vectors":{"size":96,"distance":"Dot","datatype":datatype,"on_disk":True,
                           "multivector_config":{"comparator":"max_sim"}},
                "hnsw_config":{"m":0},"shard_number":1,"replication_factor":1,
                "optimizers_config":{"default_segment_number":1,"max_segment_size":4000000,"indexing_threshold":0}}
        r=s.put(url,json=config,timeout=30)
        if not r.ok:
            raise RuntimeError(r.text)
        checkpoint.write_text("0")
    elif not info.ok or not checkpoint.exists():
        raise RuntimeError("Unrecognized collection or missing import cursor")
    cursor=int(checkpoint.read_text())
    channel=grpc.insecure_channel("127.0.0.1:17634",options=[("grpc.enable_http_proxy",0),("grpc.max_send_message_length",64*1024**2)])
    stub=api.PointsStub(channel)
    start=time.perf_counter()
    for a in range(cursor,len(order),96):
        batch=order[a:a+96]
        points=[]
        for i in batch:
            arr=docs[offset[i]:offset[i+1]]
            vec=api.Vector(data=arr.astype("f4").ravel().tolist(),vectors_count=len(arr))
            points.append(api.PointStruct(id=api.PointId(num=int(assigned[i])),vectors=api.Vectors(vector=vec)))
        reply=stub.Upsert(api.UpsertPoints(collection_name=name,wait=True,points=points),timeout=120)
        if reply.result.status!=api.UpdateStatus.Completed:
            raise RuntimeError(str(reply.result))
        checkpoint.write_text(str(a+len(batch)))
        if (a//96)%100==0:
            print(name,a+len(batch),"/",len(order),round(time.perf_counter()-start,1),"s",flush=True)
    channel.close()
    r=s.patch(url,json={"optimizers_config":{"indexing_threshold":20000}},timeout=30)
    r.raise_for_status()
    for _ in range(180):
        info=s.get(url,timeout=10).json()["result"]
        if info["status"]=="green" and info["optimizer_status"]=="ok" and info["points_count"]==len(order):
            break
        if info["status"]=="red":
            raise RuntimeError(str(info["optimizer_status"]))
        time.sleep(2)
    else:
        raise RuntimeError("Collection did not stabilize")
    assert info["config"]["params"]["vectors"]["on_disk"] and info["config"]["params"]["vectors"]["datatype"]==datatype
    receipt={"server":server,"collection":name,"order_sha256":hashlib.sha256(order.astype("<i8").tobytes()).hexdigest(),
             "state":info,"physical_ids":physical_ids,"import_elapsed_seconds_this_invocation":time.perf_counter()-start}
    (root/(name+".json")).write_text(json.dumps(receipt,indent=2))
    print(name,info["points_count"],"points",info["segments_count"],"segments",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("layout",choices=["original","semantic"])
    p.add_argument("--datatype",choices=["float16","float32"],default="float16")
    p.add_argument("--physical-ids",action="store_true",help="Persist the permutation across ID-sorting compaction using a reversible row-ID map")
    args=p.parse_args()
    run(args.root,args.layout,args.datatype,args.physical_ids)
