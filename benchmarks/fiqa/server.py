"""Run the pinned Qdrant binary as a loopback-only, isolated benchmark service."""
import argparse
import os
from pathlib import Path
import subprocess
import time

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    root=p.parse_args().root.resolve()
    instance=root/"instance"
    instance.mkdir(exist_ok=True)
    stop=instance/"request-stop"
    if stop.exists():
        stop.unlink()
    env=dict(os.environ,QDRANT__SERVICE__HOST="127.0.0.1",QDRANT__SERVICE__HTTP_PORT="17633",
             QDRANT__SERVICE__GRPC_PORT="17634",QDRANT__TELEMETRY_DISABLED="true",
             QDRANT__CLUSTER__ENABLED="false",QDRANT__LOG_LEVEL="WARN")
    with (instance/"server.log").open("ab") as log:
        child=subprocess.Popen([str(root/"qdrant.exe")],env=env,cwd=instance,stdout=log,stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        print("Local Qdrant PID",child.pid,flush=True)
        try:
            while child.poll() is None and not stop.exists():
                time.sleep(1)
        finally:
            if child.poll() is None:
                child.terminate()
            status=child.wait()
        if not stop.exists():
            raise SystemExit(status or 1)
