"""Frozen paired validation in separate processes, outside fast rollout workers."""
from collections import deque
import json
from pathlib import Path
import subprocess
import sys
import time
from .persistence import atomic_json


class ValidationQueue:
    def __init__(self,output,initial,workers=4,games=256):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.initial=str(Path(initial).resolve());self.workers=workers;self.games=games
        self.pending=deque();self.active=None;self.finished=[]

    def schedule(self,checkpoint,step):
        self.pending.append({'checkpoint':str(Path(checkpoint).resolve()),'step':step})
        self.poll()

    def poll(self):
        if self.active and self.active[0].poll() is not None:
            process,log,job=self.active;log.close()
            self.finished.append({**job,'exit_code':process.returncode})
            print(json.dumps({'event':'paired_validation_finished',**self.finished[-1]}),flush=True)
            self.active=None
        if self.active is None and self.pending:
            job=self.pending.popleft();folder=self.output/f"step_{job['step']:09d}"
            command=[sys.executable,'-u','-m','champions_rl.validation_queue','--checkpoint',job['checkpoint'],
                     '--initial',self.initial,'--output',str(folder),'--workers',str(self.workers),'--games',str(self.games)]
            log=(self.output/(folder.name+'.log')).open('w',encoding='utf-8')
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            self.active=(process,log,{**job,'output':str(folder)})
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


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','initial','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--games',type=int,default=256)
    suite(**vars(parser.parse_args()))
