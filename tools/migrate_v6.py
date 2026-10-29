"""Explicit architecture warm start, not a compatible reload or a promotion."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from champions_rl.features import Encoder,FEATURE_VERSION
from champions_rl.model import BattleModel


def migrate(source,runtime,output):
    if Path(output).exists():raise ValueError('Refusing to overwrite a migration candidate')
    old=torch.load(source,map_location='cpu',weights_only=True)
    if old['feature_version']!=6 or FEATURE_VERSION!=7:raise ValueError('This migration is only v6 -> v7')
    runtime=Path(runtime).resolve()
    manifest=json.loads((runtime/'BASELINE_MANIFEST.json').read_text())
    for relative,expected in manifest['files'].items():
        if hashlib.sha256((runtime/relative).read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen runtime changed: '+relative)
    name='_frozen_v6_migration'
    spec=importlib.util.spec_from_file_location(name,runtime/'champions_rl/__init__.py',
                                               submodule_search_locations=[str(runtime/'champions_rl')])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    prior_encoder=importlib.import_module(name+'.features').Encoder(native=False)
    signature=importlib.import_module(name+'.provenance').feature_signature()
    if signature!=old['feature_signature']:raise ValueError('Frozen runtime does not match the v6 weights')
    encoder=Encoder(native=False);model=BattleModel(**old['config'])
    state=model.state_dict();previous=old['model'];copied=[]
    for key,value in state.items():
        if key in previous and value.shape==previous[key].shape:
            value.copy_(previous[key]);copied.append(key)
    with torch.no_grad():
        # Initialize the set encoder to the average learned move embedding.
        # GELU(x) - GELU(-x) == x, leaving room for new move attributes to learn.
        first,last=model.move_set[0],model.move_set[2]
        first.weight.zero_();first.bias.zero_();last.weight.zero_();last.bias.zero_()
        for i in range(24):
            first.weight[i,i]=1;first.weight[24+i,i]=-1
            last.weight[i,i]=1;last.weight[i,24+i]=-1
        model.mon.weight.zero_()
        for i,name in enumerate(encoder.mon_names):
            if name in prior_encoder.mon_names:
                model.mon.weight[:,i]=previous['mon.weight'][:,prior_encoder.mon_names.index(name)]
        a=len(encoder.mon_names);b=len(prior_encoder.mon_names)
        model.mon.weight[:,a:a+32]=previous['mon.weight'][:,b:b+32]
        model.mon.weight[:,a+32:a+56]=previous['mon.weight'][:,b+32:b+32+96].reshape(-1,4,24).sum(1)
        model.mon.weight[:,-32:]=previous['mon.weight'][:,-32:]
        model.field.weight.zero_()
        for i,name in enumerate(encoder.field_names):
            if name in prior_encoder.field_names:
                model.field.weight[:,i]=previous['field.weight'][:,prior_encoder.field_names.index(name)]
        # Semantic move labels retain their meaning; the broken slot-specific
        # switch head is deliberately replaced by a fresh shared scorer.
        count=model.opponent_action.out_features
        model.opponent_action.weight.copy_(previous['opponent_action.weight'][:count])
        model.opponent_action.bias.copy_(previous['opponent_action.bias'][:count])
    report={'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            'source_feature_signature':signature,'source_runtime':str(runtime),
            'copied_tensors':copied,'parameters':sum(p.numel() for p in model.parameters()),
            'interpretation':'Warm start only: new move set encoder and destination head need training and paired validation. No optimizer state transferred; frozen v6 unchanged.'}
    model.save(output,phase='v7_architecture_warm_start',**report)
    Path(output).with_suffix('.migration.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='copied_tensors'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','runtime','output'):parser.add_argument('--'+name,type=Path,required=True)
    migrate(**vars(parser.parse_args()))
