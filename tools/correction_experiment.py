"""Prepare one fixed-budget correction ablation; --run requires training unpaused."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from champions_rl.persistence import atomic_json
from champions_rl.provenance import feature_signature
from champions_rl.opponent_labels import switch_offset
from champions_rl.training import demonstration_heldout


def preflight(folder):
    counts={'samples':0,'training':0,'heldout':0};files={}
    for path in sorted(Path(folder).glob('*.npz')):
        meta=json.loads(path.with_suffix('.json').read_text())
        if meta.get('error') or meta.get('reference_fallbacks'):raise ValueError('Degraded demonstration: '+str(path))
        if meta.get('feature_signature')!=feature_signature():raise ValueError('Rebuild incompatible demonstrations: '+str(path))
        with np.load(path,allow_pickle=False) as data:
            labels=data['labels'];mask=data['mask']
            if not mask[np.arange(len(labels)),labels].all():raise ValueError('Illegal teacher action in '+str(path))
            enemy=data['enemy_labels'];switch=enemy>=switch_offset()
            destinations=enemy[switch]-switch_offset()
            if (destinations>=6).any() or not data['opponent_switch_mask'][switch,destinations].all():
                raise ValueError('Masked opponent supervision in '+str(path))
            size=len(labels)
        heldout=demonstration_heldout(path,meta)
        counts['samples']+=size;counts['heldout' if heldout else 'training']+=size
        files[path.name]={'data_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                         'metadata_sha256':hashlib.sha256(path.with_suffix('.json').read_bytes()).hexdigest()}
    if not counts['training'] or not counts['heldout']:raise ValueError('Need distinct training and held-out games in '+str(folder))
    return {**counts,'shards':files}


def commands(output):
    output=Path(output);python=sys.executable
    base=['runtime_data/demos_v8','runtime_data/distill_v6_to_v8']
    corrections='runtime_data/mistakes_v8_reference/corrections';jobs=[]
    for arm in ('control','corrections'):
        jobs.append((arm+'_train',[python,'-u','-m','champions_rl','pretrain','--initial','runtime_data/v8_pilot_warm_start.pt',
          '--data',*base,*([corrections] if arm=='corrections' else []),'--output',str(output/(arm+'.pt')),
          '--steps','400','--batch-size','128','--seed','942811','--learning-rate','0.00001','--identity-dropout','0.25',
          '--correction-mass','0.05' if arm=='corrections' else '0']))
    # The primary comparison is direct cross-play, with a full paired schedule.
    jobs.append(('head_to_head',[python,'-u','-m','champions_rl','evaluate','--checkpoint',str(output/'corrections.last.pt'),
      '--opponents','frozen_control','--frozen-opponent','frozen_control='+str(output/'control.last.pt'),
      '--games','2048','--workers','2','--seed','492617','--deterministic','--output',str(output/'head_to_head')]))
    for arm in ('control','corrections'):
        common=[python,'-u','-m','champions_rl','evaluate','--checkpoint',str(output/(arm+'.last.pt')),
                '--workers','2','--seed','92341','--deterministic']
        for schedule,opponents,extra,games in [
          ('broad','balanced,hyperoffense,setup_sweeper,attrition,pivot',[],256),
          ('cores','balanced',['--split','validation_cores'],256),
          ('saved','balanced',['--roster','saved'],256),
          ('saved_vs_varied','balanced',['--roster','saved_vs_varied'],256),
          ('reference','reference',[],256),
          ('past_policy','frozen_past',['--frozen-opponent','frozen_past=runtime_data/ppo_v7_generalization/step_000690798.pt',
            '--frozen-opponent-runtime','frozen_past=runtime_data/baselines/v7_e3e2d8a'],2048)]:
            jobs.append((arm+'_'+schedule,common+['--opponents',opponents,'--games',str(games),*extra,'--output',str(output/arm/schedule)]))
    # Retention is judged against frozen v6 as well as the matched v8 control.
    # Rerun the original validation schedule, never its consumed test seeds.
    for schedule,opponents,extra in [('saved','balanced',['--roster','saved']),
                                    ('saved_vs_varied','balanced',['--roster','saved_vs_varied']),
                                    ('reference','reference',[])]:
        jobs.append(('v6_anchor_'+schedule,[python,'-u','-m','champions_rl','evaluate',
          '--checkpoint','runtime_data/candidates/v6_2m_review/policy.pt',
          '--legacy-runtime','runtime_data/baselines/v6_3cc5468','--opponents',opponents,
          '--games','256','--workers','2','--seed','92341','--deterministic',*extra,'--output',str(output/'v6_anchor'/schedule)]))
    return jobs


def prepare(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    data={folder:preflight(folder) for folder in ('runtime_data/demos_v8','runtime_data/distill_v6_to_v8','runtime_data/mistakes_v8_reference/corrections')}
    plan={'status':'prepared_not_started','feature_signature':feature_signature(),
      'training_implementation':{name:hashlib.sha256((ROOT/'champions_rl'/name).read_bytes()).hexdigest()
                                 for name in ('training.py','model.py','augmentation.py','opponent_labels.py')},
      'initial_sha256':hashlib.sha256((ROOT/'runtime_data/v8_pilot_warm_start.pt').read_bytes()).hexdigest(),
      'variable':'Add 5% sampling mass of public-teacher corrections from fresh learner losses; retain earlier data.',
      'fixed':'Same v8 start, 400 updates, seed 942811, LR 1e-5, batch 128, ID dropout 25%; evaluate fixed last weights, not per-arm best-BC selections.',
      'primary':'2048 head-to-head games (1024 complete seed pairs), treatment versus control. About 80%+ normal-approximation power for a five-point improvement under the conservative maximum pair-score variance; actual power depends on outcome correlation.',
      'secondary':'Broad/core/saved/saved-versus-varied and reference checks, plus 2048 games per arm against the best prior broad-validation checkpoint. Frozen v6 anchors all saved-team checks. Small secondary schedules do not promise five-point precision.',
      'past_opponent_selection':'Step 690798 led completed v7 broad validation (58.67% overall, 73.03% of decisive pairs); not claimed universally strongest without cross-play.',
      'agreed_promotion_targets':{'broad_decisive_pair_share':.6,'core_decisive_pair_share':.6,'head_to_head_win_rate':.6,
        'saved_team':'No regression; report paired uncertainty and retain the frozen baseline if unclear.'},
      'interpretation':'Teacher preferences are validated in the saved-reference domain only. Individual labels are not counterfactual win proofs. Value targets retain actual learner outcomes. No automatic promotion; prospective test untouched.',
      'resources':'Sequential jobs; at most two battle/evaluation workers; no overnight PPO or model-size change.',
      'data':data,'jobs':[{'name':name,'argv':argv} for name,argv in commands(output)]}
    path=output/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Experiment plan changed; use a new folder')
    atomic_json(path,plan)
    return plan


def run(output):
    stop=ROOT/'runtime_data/STOP'
    if stop.exists():raise RuntimeError('Training is explicitly paused; obtain the requested resume instruction first')
    plan=prepare(output);output=Path(output)
    for job in plan['jobs']:
        if stop.exists():raise RuntimeError('Stopped before '+job['name'])
        marker=output/(job['name']+'.complete.json')
        if marker.exists():continue
        log=output/(job['name']+'.log')
        # Never silently overwrite a partial benchmark or optimizer run.
        if log.exists():raise RuntimeError('Unfinished experiment job exists; preserve it and use a new output folder: '+str(log))
        print(json.dumps({'event':'experiment_job','name':job['name'],'log':str(log)}),flush=True)
        with log.open('w',encoding='utf-8') as stream:
            process=subprocess.Popen(job['argv'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            try:
                while process.poll() is None:
                    if stop.exists():raise RuntimeError('Stopped at user request')
                    time.sleep(1)
            finally:
                if process.poll() is None:
                    if os.name=='nt':subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
                    else:process.kill()
                    process.wait(timeout=10)
            if process.returncode:raise RuntimeError('Experiment failed; see '+str(log))
        atomic_json(marker,{'argv':job['argv'],'log':str(log),'exit_code':0})
    from tools.compare_evaluations import compare
    comparisons={}
    for schedule in ('broad','cores','saved','saved_vs_varied','reference','past_policy'):
        comparisons[schedule]=compare(output/'control'/schedule,output/'corrections'/schedule)
    retention={arm:{schedule:compare(output/'v6_anchor'/schedule,output/arm/schedule,allow_feature_change=True)
                    for schedule in ('saved','saved_vs_varied','reference')} for arm in ('control','corrections')}
    atomic_json(output/'comparison.json',{'status':'completed_no_automatic_promotion','comparisons':comparisons,
                                        'frozen_v6_retention':retention,
                                        'head_to_head':json.loads((output/'head_to_head/report.json').read_text())})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runtime_data/correction_experiment_v8'))
    parser.add_argument('--run',action='store_true',help='Execute only after the user explicitly resumes training')
    args=parser.parse_args()
    if args.run:run(args.output)
    else:
        result=prepare(args.output)
        print(json.dumps({'status':result['status'],'jobs':len(result['jobs']),'data':{k:{n:v for n,v in r.items() if n!='shards'} for k,r in result['data'].items()}},indent=2))
