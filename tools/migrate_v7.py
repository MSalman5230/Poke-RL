"""Explicit v7 -> v8 warm start; never resume old optimizers or cached features."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from champions_rl.features import Encoder,FEATURE_VERSION
from champions_rl.legacy import load_frozen_policy
from champions_rl.model import BattleModel


def migrate(source,runtime,output):
    source=Path(source);output=Path(output)
    if output.exists():raise ValueError('Refusing to overwrite a migrated checkpoint')
    data=torch.load(source,map_location='cpu',weights_only=True)
    if data['feature_version']!=7 or FEATURE_VERSION!=8:raise ValueError('This migration is only v7 -> v8')
    old,_,encoder,signature=load_frozen_policy(source,runtime)
    model=BattleModel(**old.config);previous=old.state_dict()
    with torch.no_grad():
        for key,value in model.state_dict().items():
            if key in previous and value.shape==previous[key].shape:value.copy_(previous[key])
        model.mon.weight.zero_()
        names=Encoder(native=False).mon_names
        for index,name in enumerate(names):
            if name in encoder.mon_names:model.mon.weight[:,index].copy_(previous['mon.weight'][:,encoder.mon_names.index(name)])
        model.mon.weight[:,len(names):].copy_(previous['mon.weight'][:,len(encoder.mon_names):])
        # Keep the starting policy stable. The new target-conditioned branch
        # can learn meaningful reserve order during subsequent corrected training.
        model.preview_disguise.weight.zero_()
    report={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'source_feature_signature':signature,'source_runtime':str(runtime),
            'new_parameters_initialized_to_zero':['identity_uncertain input column','preview_disguise projection'],
            'optimizer_transferred':False,'parameters':sum(p.numel() for p in model.parameters()),
            'status':'Warm start, not a promoted policy; v8 data must be rebuilt.'}
    model.save(output,phase='v8_defect_warm_start',**report)
    output.with_suffix('.migration.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','runtime','output'):parser.add_argument('--'+name,type=Path,required=True)
    print(json.dumps(migrate(**vars(parser.parse_args())),indent=2))
