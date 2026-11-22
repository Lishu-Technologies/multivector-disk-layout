"""Parallel coalesced reads. The Windows direct path explicitly bypasses the OS file cache."""
import ctypes
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import threading

import numpy as np


class BlockReader:
    page_size = 4096

    def __init__(self,direct=False,workers=4,physical_starts=None):
        self.direct = direct
        self.physical_starts = physical_starts
        self.workers = ThreadPoolExecutor(workers)
        self.local = threading.local()
        self.handles = []
        if direct:
            if os.name != "nt":
                raise ValueError("Unbuffered replay currently requires Windows")
            from ctypes import wintypes as w
            self.win = ctypes.WinDLL("kernel32",use_last_error=True)
            self.win.CreateFileW.argtypes = [w.LPCWSTR,w.DWORD,w.DWORD,ctypes.c_void_p,w.DWORD,w.DWORD,w.HANDLE]
            self.win.CreateFileW.restype = w.HANDLE
            self.win.SetFilePointerEx.argtypes = [w.HANDLE,ctypes.c_longlong,ctypes.c_void_p,w.DWORD]
            self.win.ReadFile.argtypes = [w.HANDLE,ctypes.c_void_p,w.DWORD,ctypes.POINTER(w.DWORD),ctypes.c_void_p]
            self.win.CloseHandle.argtypes = [w.HANDLE]
        self.bytes = self.calls = 0

    def _read(self, task):
        path,start,end = task
        if not hasattr(self.local,"files"):
            self.local.files = {}
        if path not in self.local.files:
            if self.direct:
                handle = self.win.CreateFileW(str(path),0x80000000,7,None,3,0x20000000,None)
                if handle == ctypes.c_void_p(-1).value:
                    raise ctypes.WinError(ctypes.get_last_error())
            else:
                handle = path.open("rb",buffering=0)
            self.local.files[path] = handle
            self.handles.append(handle)
        f = self.local.files[path]
        size = end-start
        if not self.direct:
            f.seek(start)
            return f.read(size)
        if not self.win.SetFilePointerEx(f,start,None,0):
            raise ctypes.WinError(ctypes.get_last_error())
        buf = ctypes.create_string_buffer(size+self.page_size)
        address = (ctypes.addressof(buf)+self.page_size-1)//self.page_size*self.page_size
        count = ctypes.c_ulong()
        if not self.win.ReadFile(f,address,size,ctypes.byref(count),None):
            raise ctypes.WinError(ctypes.get_last_error())
        return ctypes.string_at(address,count.value)

    def fetch(self,path,offsets,ids,width,header=0):
        path = Path(path)
        starts = offsets if self.physical_starts is None else self.physical_starts
        spans = [(int(starts[i])*width+header, int(starts[i]+offsets[i+1]-offsets[i])*width+header,j)
                 for j,i in enumerate(ids)]
        runs = []
        for start,end,j in sorted(spans):
            left = start//self.page_size*self.page_size
            right = (end+self.page_size-1)//self.page_size*self.page_size
            if runs and left<=runs[-1][1] and right-runs[-1][0]<=2**20:
                runs[-1][1] = max(runs[-1][1],right)
                runs[-1][2].append((start,end,j))
            else:
                runs.append([left,right,[(start,end,j)]])
        buffers = self.workers.map(self._read,[(path,a,b) for a,b,_ in runs])
        parts = [None]*len(ids)
        for (a,b,items),buffer in zip(runs,buffers):
            for start,end,j in items:
                if end-a>len(buffer):
                    raise IOError("Short document read")
                parts[j] = memoryview(buffer)[start-a:end-a]
        self.bytes += sum(b-a for a,b,_ in runs)
        self.calls += len(runs)
        return np.frombuffer(b"".join(parts),dtype=np.uint8).reshape(-1,width)

    def close(self):
        self.workers.shutdown()
        for handle in self.handles:
            if self.direct:
                self.win.CloseHandle(handle)
            else:
                handle.close()
