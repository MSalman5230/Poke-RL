"""Rebuild cached demonstration features from saved public observations."""
import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
from multiprocessing.util import Finalize
from pathlib import Path
import numpy as np
from .engine import Engine
from .features import Encoder,FEATURE_VERSION
from .knowledge import knowledge
from .policies import ScriptedBot,TRAINING_STYLES
from .persistence import atomic_json,durable_npz
from .provenance import feature_signature
from .rewards import potential


def initialize():
    global _engine,_encoder
    _engine=Engine();_encoder=Encoder(engine=_engine)
    Finalize(None,_engine.close,exitpriority=10)


def rebuild(job):
    source,output=map(Path,job)
    meta=json.loads(source.with_suffix('.json').read_text(encoding='utf-8'))
    if meta['knowledge_sha256']!=knowledge().sha256 or meta.get('error') or meta.get('reference_fallbacks'):
        raise ValueError('Incompatible/degraded source demonstration')
    with gzip.open(source.with_suffix('.observations.json.gz'),'rt',encoding='utf-8') as stream:raw=json.load(stream)
    rows=[];labels=[];changed=0
    scripted=(*TRAINING_STYLES,'damage','setup','defensive','stochastic')
    teacher=ScriptedBot(meta['teacher'],encoder=_encoder) if meta['teacher'] in scripted else None
    for row in raw['rows']:
        encoded=_encoder.encode(row['observation']);rows.append(encoded)
        label=int(np.argmax(teacher.scores(row['observation'],encoded))) if teacher is not None else row['expert_action']
        changed+=label!=row['expert_action'];labels.append(label);row['expert_action']=label
    with np.load(source,allow_pickle=False) as old:
        payload={k:old[k].copy() for k in old.files if k not in rows[0]}
    payload.update({k:np.stack([r[k] for r in rows]) for k in rows[0]})
    payload['labels']=np.asarray(labels,dtype=np.int64)
    payload['outcomes']=np.asarray([raw['outcome']-.15*potential(r['observation']) for r in raw['rows']],dtype=np.float32)
    durable_npz(output,**payload)
    raw['feature_signature']=feature_signature()
    with gzip.open(output.with_suffix('.observations.json.gz'),'wt',encoding='utf-8') as stream:json.dump(raw,stream)
    meta.update(feature_version=FEATURE_VERSION,feature_signature=feature_signature(),
                reencoded_from=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                relabeled_decisions=changed,
                relabeling='Scripted teacher recomputed on each recorded position; frozen-reference labels preserved. Outcome belongs to the original behavior trajectory.')
    atomic_json(output.with_suffix('.json'),meta)
    return {'samples':len(rows),'changed_script_labels':changed}


def run(source,output,workers=4):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    jobs=[];skipped=0
    for path in source.glob('*.npz'):
        if not path.with_suffix('.json').exists():continue
        target=output/path.name
        if target.with_suffix('.json').exists():
            meta=json.loads(target.with_suffix('.json').read_text(encoding='utf-8'))
            if meta.get('feature_signature')!=feature_signature() or meta.get('source_sha256')!=hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError('Existing regenerated shard differs from source or current features')
            skipped+=1;continue
        jobs.append((str(path),str(target)))
    samples=changed=0
    with mp.get_context('spawn').Pool(workers,initializer=initialize) as pool:
        for index,row in enumerate(pool.imap_unordered(rebuild,jobs),1):
            samples+=row['samples'];changed+=row['changed_script_labels']
            if index%64==0:print(json.dumps({'rebuilt':index,'jobs':len(jobs),'samples':samples}),flush=True)
    report={'source':str(source.resolve()),'output':str(output.resolve()),'rebuilt_games':len(jobs),'already_rebuilt':skipped,
            'samples':samples,'changed_script_labels':changed,'feature_signature':feature_signature()}
    atomic_json(output/'reencoding.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    run(**vars(parser.parse_args()))
