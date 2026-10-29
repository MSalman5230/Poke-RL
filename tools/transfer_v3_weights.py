"""Explicit v3/v4 -> v5 weight transfer. Never relabel encoded data or optimizer state."""
import argparse
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from champions_rl.features import FEATURE_VERSION
from champions_rl.knowledge import knowledge
from champions_rl.model import BattleModel


def transfer(source,output):
    data=torch.load(source,map_location='cpu',weights_only=True)
    digest=hashlib.sha256(str(data.get('feature_version')).encode())
    for name in ('data/knowledge.json','data/sets.json'):
        digest.update(name.encode());digest.update((ROOT/name).read_bytes())
    if FEATURE_VERSION!=5 or data.get('feature_version') not in (3,4) or data.get('knowledge_sha256')!=knowledge().sha256 or data.get('feature_signature')!=digest.hexdigest():
        raise ValueError('Only the explicitly audited v3/v4-to-v5 weight transfer is supported')
    model=BattleModel(**data['config'])
    model.load_state_dict(data['model'],strict=True)
    if Path(output).exists():raise FileExistsError(output)
    model.save(output,phase='weight_transfer_requires_revalidation',source=str(Path(source).resolve()),
               source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),source_metadata=data.get('metadata'),
               transfer='Weights only. Corrected v5 features are recomputed; old vectors and optimizer state are incompatible.')
    print(str(Path(output).resolve()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();transfer(args.source,args.output)
