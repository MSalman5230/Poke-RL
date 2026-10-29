"""Compare frozen evaluations using paired seed blocks, retaining both seats."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from champions_rl.persistence import atomic_json


def interval(blocks,seed=240011):
    values=np.asarray([np.mean(v) for _,v in sorted(blocks.items())],dtype=float)
    if len(values)<2:raise ValueError('At least two independent seed blocks are required')
    rng=np.random.default_rng(seed)
    samples=values[rng.integers(len(values),size=(10000,len(values)))].mean(1)
    return {'candidate_minus_baseline':float(values.mean()),'paired_seed_blocks':len(values),
            'bootstrap_95':np.quantile(samples,[.025,.975]).tolist()}


def compare(baseline,candidate):
    baseline=Path(baseline);candidate=Path(candidate)
    a=json.loads((baseline/'report.json').read_text());b=json.loads((candidate/'report.json').read_text())
    for report in (a,b):
        if not report.get('complete'):raise ValueError('An evaluation is incomplete')
        if any(r['games']!=report['requested_games_per_opponent'] for r in report['results'].values()):
            raise ValueError('The requested game count was not completed')
        if any(r['errors'] or r['fallbacks'] for r in report['results'].values()):
            raise ValueError('Invalid games or fallback decisions cannot certify a paired improvement')
    for key in ('feature_signature','knowledge_sha256','team_manifest_sha256','reference_manifest_sha256',
                'seed','split','roster','reference_teams','frozen_opponents','requested_games_per_opponent'):
        if a.get(key)!=b.get(key):raise ValueError(f'Incompatible evaluation schedules: {key}')
    if set(a['results'])!=set(b['results']):raise ValueError('Opponent schedules differ')
    all_blocks={};styles={}
    for style in sorted(a['results']):
        left={r['game']:r for r in json.loads((baseline/f'{style}_games.json').read_text())}
        right={r['game']:r for r in json.loads((candidate/f'{style}_games.json').read_text())}
        expected=set(range(a['requested_games_per_opponent']))
        if set(left)!=expected or set(right)!=expected:raise ValueError('Incomplete paired game records')
        if len(expected)%2:raise ValueError('Paired comparisons require complete two-seat blocks')
        blocks={}
        for game,row in left.items():
            other=right[game]
            if row.get('error') or other.get('error'):raise ValueError('Invalid paired game')
            if row['team_ids']!=other['team_ids'] or row['seat']!=other['seat']:
                raise ValueError('Mismatched teams or seats')
            difference=float(other['won'])-float(row['won'])
            blocks.setdefault(game//2,[]).append(difference)
            all_blocks.setdefault(game//2,[]).append(difference)
        styles[style]={'baseline':a['results'][style],'candidate':b['results'][style],**interval(blocks)}
    changed=[]
    for path in (baseline/'worker_0'/'code_snapshot').rglob('*'):
        if not path.is_file():continue
        relative=path.relative_to(baseline/'worker_0'/'code_snapshot')
        other=candidate/'worker_0'/'code_snapshot'/relative
        if not other.exists() or path.read_bytes()!=other.read_bytes():changed.append(relative.as_posix())
    return {'baseline':str(baseline),'candidate':str(candidate),
            'baseline_checkpoint_sha256':a['checkpoint_sha256'],'candidate_checkpoint_sha256':b['checkpoint_sha256'],
            'feature_signature':a['feature_signature'],'baseline_implementation_sha256':a['implementation_sha256'],
            'candidate_implementation_sha256':b['implementation_sha256'],'changed_implementation_files':changed,
            'baseline_deterministic':a['deterministic'],'candidate_deterministic':b['deterministic'],
            'baseline_search':a.get('search',False),'candidate_search':b.get('search',False),
            'by_opponent':styles,'aggregate':interval(all_blocks),
            'interpretation':'Exploratory validation. Bootstrap retains both seats and all opponents within each seed block. '
                             'Repeated checkpoint selection and per-opponent comparisons are not multiplicity-adjusted. '
                             'Promotion still requires a separately frozen test benchmark; truncations are reported, not treated as wins.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','candidate','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();report=compare(args.baseline,args.candidate)
    atomic_json(args.output,report);print(json.dumps(report['aggregate'],indent=2))
