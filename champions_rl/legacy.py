"""Read an explicitly frozen runtime without reinterpreting old model inputs."""
from functools import lru_cache
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys


@lru_cache(maxsize=4)
def frozen_package(runtime):
    root=Path(runtime).resolve()
    manifest=json.loads((root/'BASELINE_MANIFEST.json').read_text(encoding='utf-8'))
    for relative,sha in manifest['files'].items():
        if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=sha:
            raise ValueError('Frozen baseline source changed: '+relative)
    name='_champions_frozen_'+hashlib.sha256(str(root).encode()).hexdigest()[:16]
    spec=importlib.util.spec_from_file_location(name,root/'champions_rl/__init__.py',
                                               submodule_search_locations=[str(root/'champions_rl')])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return name


def load_frozen_policy(checkpoint,runtime,device='cpu',engine=None):
    name=frozen_package(str(Path(runtime).resolve()))
    model,metadata=importlib.import_module(name+'.model').BattleModel.load(checkpoint,device)
    encoder=importlib.import_module(name+'.features').Encoder(engine=engine)
    signature=importlib.import_module(name+'.provenance').feature_signature()
    return model,metadata,encoder,signature
