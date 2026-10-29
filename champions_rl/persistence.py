"""Durable, atomic local experiment records."""
import json
import os
from pathlib import Path


def atomic_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8',newline='\n') as handle:
        json.dump(value,handle,indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


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
    temporary.replace(path)


def durable_npz(path,**arrays):
    import numpy as np
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle,**arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
