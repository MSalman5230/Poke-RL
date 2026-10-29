"""Summarize completed review experiments; never treat interrupted runs as wins."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from champions_rl.persistence import atomic_json

RUNS={
    'small_imitation_deterministic':'eval_bc_v3_deterministic',
    'dagger_deterministic':'eval_dagger_v3_deterministic',
    'dagger_search':'eval_dagger_v3_search_retry',
    'search_distilled':'eval_search_distilled_v3',
    'ppo_deterministic':'eval_ppo_v3_deterministic_retry',
    'ppo_stochastic':'eval_ppo_v3_stochastic',
    'ppo_competitive':'eval_ppo_v3_competitive',
    'full_imitation_deterministic':'eval_bc_v3_full_deterministic',
    'full_imitation_stochastic':'eval_bc_v3_full_stochastic',
    'full_imitation_competitive':'eval_bc_v3_full_competitive',
    'full_imitation_reference':'eval_bc_v3_full_reference',
    'corrected_full_deterministic':'eval_corrected_full_v3_deterministic',
    'corrected_full_reference':'eval_corrected_full_v3_reference',
    'ppo_corrected_deterministic':'eval_ppo_corrected_v3_deterministic',
}


def paired_difference(first,second):
    a=json.loads((first/'report.json').read_text())
    b=json.loads((second/'report.json').read_text())
    for report,folder in ((a,first),(b,second)):
        if 'feature_signature' not in report:
            import torch
            report['feature_signature']=torch.load(folder/'evaluated_model.pt',map_location='cpu',weights_only=True)['feature_signature']
    for key in ('split','seed','team_manifest_sha256','reference_teams','feature_signature'):
        if a[key]!=b[key]:raise ValueError(f'Incompatible schedules: {key}')
    differences=defaultdict(list)
    for style in sorted(set(a['results'])&set(b['results'])):
        if style=='reference':continue
        left={r['game']:r for r in json.loads((first/f'{style}_games.json').read_text())}
        right={r['game']:r for r in json.loads((second/f'{style}_games.json').read_text())}
        if left.keys()!=right.keys():raise ValueError('Incomplete paired schedule')
        for game,row in left.items():
            other=right[game]
            if row.get('error') or other.get('error'):raise ValueError('Invalid game in paired comparison')
            if row['team_ids']!=other['team_ids'] or row['seat']!=other['seat']:raise ValueError('Mismatched teams/seats')
            differences[game//2].append(float(other['won'])-float(row['won']))
    values=np.asarray([np.mean(v) for v in differences.values()])
    rng=np.random.default_rng(991)
    bootstrap=values[rng.integers(len(values),size=(10000,len(values)))].mean(1)
    return {'second_minus_first':float(values.mean()),'paired_seed_blocks':len(values),
            'bootstrap_95':np.quantile(bootstrap,[.025,.975]).tolist(),
            'same_implementation_hash':a['implementation_sha256']==b['implementation_sha256'],
            'interpretation':'Exploratory validation; bootstrap keeps both seats and all styles within each seed block. Not a test-set certification.'}


def summarize():
    runs={};pending=[]
    for label,folder in RUNS.items():
        path=ROOT/'runtime_data'/folder/'report.json'
        if not path.exists():pending.append(label);continue
        report=json.loads(path.read_text())
        if not report.get('complete') or any(r['games']!=report['requested_games_per_opponent'] for r in report['results'].values()):
            pending.append(label);continue
        total=sum(r['games'] for r in report['results'].values())
        wins=sum(r['wins'] for r in report['results'].values())
        runs[label]={'report':path.relative_to(ROOT).as_posix(),'games':total,'wins':wins,'win_rate':wins/total,
                     'split':report['split'],'deterministic':report['deterministic'],'search':report.get('search',False),
                     'checkpoint_sha256':report['checkpoint_sha256'],'implementation_sha256':report['implementation_sha256'],
                     'results':report['results'],'acceptance':report['acceptance']}
    comparisons={}
    for first,second in [('small_imitation_deterministic','full_imitation_deterministic'),
                         ('dagger_deterministic','dagger_search'),('full_imitation_deterministic','full_imitation_stochastic'),
                         ('full_imitation_deterministic','ppo_deterministic'),
                         ('corrected_full_deterministic','ppo_corrected_deterministic'),
                         ('full_imitation_deterministic','corrected_full_deterministic')]:
        if first in runs and second in runs:
            comparisons[first+' -> '+second]=paired_difference(ROOT/'runtime_data'/RUNS[first],ROOT/'runtime_data'/RUNS[second])
    collections=[]
    for folder in ('demos_v3','search_demos_v3'):
        for path in sorted((ROOT/'runtime_data'/folder).glob('summary_*.json')):
            collections.append({'path':path.relative_to(ROOT).as_posix(),**json.loads(path.read_text())})
    report={'date':'2026-10-29','runs':runs,'pending_or_incomplete':pending,'paired_comparisons':comparisons,
            'collections':collections,'actor_parameters':574786,'training_only_critic_parameters':428841,
            'excluded_interrupted_runs':['runtime_data/eval_dagger_v3_search','runtime_data/eval_ppo_v3_deterministic'],
            'limitations':['Validation games select experiments; the held-out test split has not been used.',
                           'Different implementation hashes are retained explicitly; comparisons do not isolate a single code change.',
                           'No measured superiority to Jaxcalibur or completed 60% reference acceptance gate.']}
    atomic_json(ROOT/'benchmarks/2026-10-29-review-experiments.json',report)
    print(json.dumps({key:(value['wins'],value['games']) for key,value in runs.items()},indent=2))


if __name__=='__main__':summarize()
