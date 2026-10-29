"""Durable, atomic local experiment records."""
import json
import os
import time
from pathlib import Path


def replace_with_retry(temporary,path):
    # Windows readers, antivirus, and sync clients may briefly hold the old
    # destination without delete sharing. Keep the fsynced temporary intact.
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if os.name!='nt' or attempt==19:raise
            time.sleep(min(.25,.01*2**attempt))


def atomic_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8',newline='\n') as handle:
        json.dump(value,handle,indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    replace_with_retry(temporary,path)


def append_json(path,value):
    with Path(path).open('a',encoding='utf-8',newline='\n') as handle:
        handle.write(json.dumps(value)+'\n')
        handle.flush()
        os.fsync(handle.fileno())


def durable_torch_save(value,path):
    import torch
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as handle:
        torch.save(value,handle)
        handle.flush()
        os.fsync(handle.fileno())
    replace_with_retry(temporary,path)


def durable_npz(path,**arrays):
    import numpy as np
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle,**arrays)
        handle.flush()
        os.fsync(handle.fileno())
    replace_with_retry(temporary,path)
