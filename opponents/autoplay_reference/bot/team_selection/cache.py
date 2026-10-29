"""Versioned, atomic, roster-specific preview preparation. No live DB access."""
import copy
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import zlib
from opponents.autoplay_reference.bot.domain import catalog
from opponents.autoplay_reference.bot.runtime import config
from opponents.autoplay_reference.bot.simulator import bridge
from opponents.autoplay_reference.bot.team_selection import duels

_stores={}


def environment(client,data):
    if client.identity is None:
        raise ValueError('Showdown must be verified before preview-cache preparation')
    from opponents.autoplay_reference.bot.opponent import prediction, mega
    return {'profile_version':3,'catalog':catalog.get().identity,
        'inputs':duels.sha([inspect.getsource(f) for f in (duels.member_input,duels.options,duels.task)]
                          +[sorted(duels.CONTEXTUAL),sorted(duels.PRIOR_REFERENCE)]),
        'prediction_sources':{Path(m.__file__).name:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                              for m in (prediction,mega)},
        'simulator':{k:client.identity[k] for k in ('revision','format','worker_sha256','lock_sha256',
            'battle_sha256','champions_sha256','preview_sha256')},
        'datasets':{k:{a:b for a,b in v.items() if a!='path'} for k,v in data.identities.items()}}


def team_identity(roster):
    # Read timestamps/notes do not change a battle build. Roster order is retained.
    return duels.sha([duels.member_input(n,m) for n,m in roster.members.items()])


class PreviewCache:
    def __init__(self,roster,client,data):
        self.team_identity=team_identity(roster)
        self.environment=environment(client,data)
        self.identity=duels.sha([self.team_identity,self.environment])
        self.path=Path(config.runtime_path('state','preview_cache',self.identity+'.json.gz'))
        self.pairs={}
        self.prepared=False
        self.preparation={}
        self.error=None
        self.score_dirty=False
        self.score_identity=duels.scoring_identity()
        self.load()

    def load(self):
        if not self.path.is_file():
            return
        try:
            payload=json.loads(gzip.decompress(self.path.read_bytes()))
            if not isinstance(payload,dict):
                raise ValueError('Preview cache envelope must be an object')
            if payload.get('schema_version')!=2 or payload['identity']!=self.identity or payload['environment']!=self.environment:
                raise ValueError('Incompatible preview cache identity')
            if payload['checksum']!=duels.sha(payload['pairs']):
                raise ValueError('Preview cache checksum mismatch')
            for key,row in payload['pairs'].items():
                if key!=duels.sha(row['input']):
                    raise ValueError('Preview cache build key mismatch')
                duels.validate(row['profile'],row['input'])
            self.pairs=payload['pairs']
            self.prepared=bool(payload.get('prepared'))
            self.preparation=payload.get('preparation',{})
            if payload.get('score_identity')!=self.score_identity:
                failures=self.evaluate()
                raw_failures=self.preparation.get('profile_failures',self.preparation.get('failures',[]))
                self.preparation.update(assumptions=duels.ASSUMPTIONS,profile_failures=raw_failures,
                    scoring_failures=failures,failures=raw_failures+failures)
                self.score_dirty=True
        except (ValueError,KeyError,TypeError,OSError,EOFError,zlib.error) as exc:
            self.pairs={}
            self.error=repr(exc)

    def lookup(self,input_task):
        key=duels.sha(input_task)
        row=self.pairs.get(key)
        if row is None or row['input']!=input_task:
            raise KeyError('Matchup not prepared for these exact builds/forms')
        return key,row

    def add(self,input_task,profile):
        duels.validate(profile,input_task)
        key=duels.sha(input_task)
        self.pairs[key]={'input':copy.deepcopy(input_task),'profile':profile,
                         'evaluation':None}
        return key

    def evaluate(self):
        """Score only after all raw first-form dependencies are available."""
        failures=[]
        for key,row in self.pairs.items():
            try:
                row['evaluation']=duels.score(row['profile'],row['input'],
                    activation_lookup=lambda task:self.lookup(task)[1]['profile'])
                row.pop('evaluation_error',None)
            except (KeyError,ValueError) as exc:
                row['evaluation']=None
                row['evaluation_error']=repr(exc)
                failures.append({'key':key,'error':repr(exc)})
        return failures

    def save(self):
        payload={'schema_version':2,'identity':self.identity,'team_identity':self.team_identity,
                 'environment':self.environment,'score_identity':self.score_identity,'prepared':self.prepared,
                 'pairs':self.pairs,'checksum':duels.sha(self.pairs),'preparation':self.preparation}
        content=gzip.compress(duels.canonical(payload).encode(),compresslevel=1,mtime=0)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temp_root=Path(config.runtime_path('tmp'))
        temp_root.mkdir(parents=True,exist_ok=True)
        fd,name=tempfile.mkstemp(prefix='preview-cache-',suffix='.tmp',dir=temp_root)
        try:
            with os.fdopen(fd,'wb') as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name,self.path)
            self.score_dirty=False
        finally:
            if os.path.exists(name):
                os.remove(name)


def get(roster,data,client=None):
    client=client or bridge.get()
    key=(str(Path(config.RUNTIME_DATA).resolve()),team_identity(roster),duels.sha(environment(client,data)),
         duels.sha(duels.scoring_identity()))
    if key not in _stores:
        _stores[key]=PreviewCache(roster,client,data)
    return _stores[key]
