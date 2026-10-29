"""Frozen paired validation in separate processes, outside fast rollout workers."""
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .persistence import atomic_json


def process_running(pid):
    if not pid:return False
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:return ctypes.get_last_error()==5  # access denied is not proof of exit
        try:
            code=wintypes.DWORD()
            return not kernel.GetExitCodeProcess(handle,ctypes.byref(code)) or code.value==259
        finally:kernel.CloseHandle(handle)
    try:os.kill(pid,0)
    except ProcessLookupError:return False
    except PermissionError:return True
    return True


class RecoveredProcess:
    """Observe an evaluation that survived an interrupted trainer process."""
    def __init__(self,job):self.pid=job['pid'];self.job=job;self.returncode=None
    def poll(self):
        if process_running(self.pid):return None
        summary=Path(self.job['output'])/'summary.json'
        self.returncode=0 if summary.exists() and json.loads(summary.read_text())['complete'] else 1
        return self.returncode


class ValidationQueue:
    def __init__(self,output,initial,workers=4,games=256):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.initial=str(Path(initial).resolve());self.workers=workers;self.games=games
        self.pending=deque();self.active=None;self.finished=[]
        saved=self.output/'queue.json'
        if saved.exists():
            state=json.loads(saved.read_text())
            self.pending.extend(state['pending']);self.finished=state['finished']
            job=state.get('active')
            if job:
                summary=Path(job['output'])/'summary.json'
                if process_running(job.get('pid')):
                    self.active=(RecoveredProcess(job),None,job)
                elif summary.exists() and json.loads(summary.read_text())['complete']:
                    self.finished.append({**job,'exit_code':0})
                else:
                    # Preserve the interrupted reports; retry in a new attempt folder.
                    self.pending.appendleft({**job,'attempt':job.get('attempt',0)+1})

    def schedule(self,checkpoint,step):
        known=[*self.pending,*self.finished]+([self.active[2]] if self.active else [])
        if any(job['step']==step for job in known):return
        self.pending.append({'checkpoint':str(Path(checkpoint).resolve()),'step':step})
        self.poll()

    def poll(self):
        if self.active and self.active[0].poll() is not None:
            process,log,job=self.active
            if log:log.close()
            self.finished.append({**job,'exit_code':process.returncode})
            print(json.dumps({'event':'paired_validation_finished',**self.finished[-1]}),flush=True)
            self.active=None
        if self.active is None and self.pending:
            job=self.pending.popleft();folder=self.output/f"step_{job['step']:09d}"
            if job.get('attempt'):folder=folder.with_name(folder.name+f"_retry_{job['attempt']}")
            command=[sys.executable,'-u','-m','champions_rl.validation_queue','--checkpoint',job['checkpoint'],
                     '--initial',self.initial,'--output',str(folder),'--workers',str(self.workers),'--games',str(self.games)]
            log=(self.output/(folder.name+'.log')).open('w',encoding='utf-8')
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            self.active=(process,log,{**job,'output':str(folder),'pid':process.pid})
        atomic_json(self.output/'queue.json',{'pending':list(self.pending),'active':self.active[2] if self.active else None,'finished':self.finished})

    def close(self):
        while self.active or self.pending:
            self.poll()
            if self.active:time.sleep(2)


def suite(checkpoint,initial,output,workers=4,games=256):
    from .evaluation import evaluate_parallel
    output=Path(output)
    common=dict(checkpoint=checkpoint,games=games,workers=workers,deterministic=True,seed=92341)
    manifest=evaluate_parallel(output=output/'manifest',opponents=['balanced','hyperoffense','setup_sweeper','attrition','pivot','frozen_initial'],
                               frozen_opponents={'frozen_initial':initial},**common)
    saved=evaluate_parallel(output=output/'saved',opponents=['balanced'],roster='saved',**common)
    atomic_json(output/'summary.json',{'manifest':manifest['results'],'saved':saved['results'],
                                      'complete':manifest['complete'] and saved['complete']})
    if not manifest['complete'] or not saved['complete']:
        raise RuntimeError('Validation ended before the requested game counts were completed')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','initial','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--games',type=int,default=256)
    suite(**vars(parser.parse_args()))
