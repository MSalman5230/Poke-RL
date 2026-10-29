"""Record observed review-cycle results without promoting incomplete experiments."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from champions_rl.persistence import atomic_json


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize():
    runtime=ROOT/'runtime_data';reports={}
    for folder in sorted(runtime.glob('eval_v6_*')):
        path=folder/'report.json'
        if not path.exists():continue
        report=read(path)
        complete=report.get('complete',False) and all(r['games']==report['requested_games_per_opponent'] for r in report['results'].values())
        reports[folder.name]={key:report.get(key) for key in ('checkpoint_sha256','implementation_sha256','feature_signature',
                            'seed','split','roster','reference_teams','deterministic','search','results','acceptance')}
        reports[folder.name].update(complete=complete,report=path.relative_to(ROOT).as_posix(),report_sha256=digest(path))
    metrics=runtime/'ppo_v6_full/metrics.jsonl'
    updates=[json.loads(line) for line in metrics.read_text().splitlines()] if metrics.exists() else []
    run=runtime/'ppo_v6_full'
    final=read(run/'summary.json') if (run/'summary.json').exists() else None
    training={'complete':bool(final and not final['stopped'] and final['steps']>=2000000),
              'target_decisions':2000000,'updates':len(updates),'output':run.relative_to(ROOT).as_posix(),
              'initial_training_code_commit':'a4bd4e9'}
    if updates:
        last=updates[-1]
        training.update({key:last[key] for key in ('steps','seconds','decisions_per_second','private_value_mse',
                        'actor_completed_epochs','critic_completed_epochs','mean_current_batch','win_rates')})
        training.update(actor_coverage_min=min(r['actor_sample_coverage'] for r in updates),
                        critic_coverage_min=min(r['critic_sample_coverage'] for r in updates),
                        actor_kl_stops=sum(r['kl_early_stop'] for r in updates),metrics_sha256=digest(metrics))
        phases={}
        for row in updates:phases[row.get('phase_start_steps',0)]=max(phases.get(row.get('phase_start_steps',0),0),row['seconds'])
        training['elapsed_training_wall_seconds']=sum(phases.values())
        training['training_phases']=phases
        training['aggregate_decisions_per_wall_second']=last['steps']/sum(phases.values())
        awake_phases={}
        for row in updates:
            if 'awake_seconds' in row:awake_phases[row.get('phase_start_steps',0)]=row['awake_seconds']
        training['measured_awake_phase_seconds']=awake_phases
        training['awake_time_available_for_all_phases']=set(awake_phases)==set(phases)
        training['timing_note']='Wall time includes Windows sleep. Awake time is recorded only in newer phases; missing values are not inferred from performance.'
    training['paired_validations']={p.parent.name:read(p) for p in (run/'validation').glob('step_*/summary.json')}
    speed=runtime/'throughput_v5/report.json'
    benchmark=read(speed)
    speed_result={'source':speed.relative_to(ROOT).as_posix(),'source_sha256':digest(speed),'medians':benchmark['medians'],
                  'checkpoint_sha256':benchmark['checkpoint_sha256'],'settings':benchmark['settings'],
                  'limits':'Short cold-cache v5 probes, before the final faint-counter fix; two epochs and no concurrent evaluation. '
                           'Not current full-run throughput or an isolated caching ablation. The 1000 decisions/s target is unmet.'}
    imitation=[json.loads(line) for line in (runtime/'bc_v6_final.metrics.jsonl').read_text().splitlines()]
    best=min(imitation,key=lambda r:r['balanced_validation_loss'])
    audit=runtime/'value_audit_v4/report.json'
    value=read(audit)
    value_result={'source':audit.relative_to(ROOT).as_posix(),'source_sha256':digest(audit),
                  'feature_signature':value['feature_signature'],'overall':value['overall'],'by_turn':value['by_turn'],
                  'limits':'Early v4 probe, not the final v6 model. Weak initial prediction justified deferring search promotion. '
                           'Correlated position diagnostics do not prove correct ranking of alternative actions.'}
    current_audits={}
    for path in sorted(runtime.glob('value_audit_v6_*/report.json')):
        result=read(path)
        current_audits[path.parent.name]={key:result[key] for key in ('settings','checkpoint_sha256','critic_sha256','feature_signature','truncated_games','limits')}
        current_audits[path.parent.name]['by_turn']={name:{kind:{k:v[kind][k] for k in ('return_mse','explained_variance','brier')} for kind in ('public','private')}
                                                    for name,v in {'overall':result['overall'],**result['by_turn']}.items()}
    correction=runtime/'dagger_v6_saved/summary_balanced_261001.json'
    report={'updated_utc':datetime.now(timezone.utc).isoformat(),'actor_parameters':574786,'training_only_critic_parameters':428841,
            'feedback_disposition':{
                'tactical_correctness':'Implemented and versioned; actual simulator outcome tests, public counters, restoration, reliability flags, semantic fingerprint, and raw-observation re-encoding.',
                'rollout_utilization':'Separate critic optimizer; representative actor KL sample; 512 minibatch and reduced actor LR; sample coverage and epoch logging.',
                'feature_cost':'Prepared matchup reuse, safe raw-roll caching, fused requests, cached per-observation stats, and skipped overwritten approximations. Speed target remains unmet.',
                'imitation_selection':'Step-zero candidate and per-teacher/balanced validation added. New BC candidate is not promoted on agreement alone.',
                'search':'Default off. Private-critic sampled leaves and paired uncertainty gate implemented. New distillation requires measured battle improvement; no qualifying certificate exists.',
                'expert_coverage':'Added varied expert own teams and prospective core exclusions. Collected 128 saved-roster learner-led games with scripted teacher corrections; tested original and 25% correction sampling mass separately.',
                'opponent_diversity':'Merged near-duplicate training styles, added extremes and pinned historical checkpoints; old styles retained for comparison.',
                'long_private_ppo':'Two-million-decision run with frozen paired validation every ten updates; current completion recorded separately.',
                'saved_roster':'Separate saved-roster and excluded-core evaluation schedules are tracked below.',
                'readme_encoding':'UTF-8 repaired; workflow and search guidance updated.'},
            'demonstrations':{'valid_games':1175,'decisions':13337,'reference_decisions':1545,'varied_reference_valid_games':55,
                              'varied_reference_errors_excluded':9,'reencoded_script_labels_changed':36},
            'imitation':{'step_zero':imitation[0],'selected':best,'promotion':'Not established by teacher agreement; use paired battle reports.'},
            'throughput_probe':speed_result,'initial_value_audit':value_result,'current_value_audits':current_audits,
            'value_calibration_diagnostics':{p.name:read(p) for p in (ROOT/'benchmarks').glob('*-v6-value-calibration-*.json')},
            'learner_corrections':read(correction) if correction.exists() else None,'training':training,'evaluations':reports,
            'limitations':['No final test-set certification or demonstrated superiority to Jaxcalibur.',
                           'The reference timeout incident is retained as invalid data, not counted as a valid benchmark.',
                           'Transferred old weights may have encountered now-excluded team cores.',
                           'No equal-wall-clock causal ablation of each fix has been completed.',
                           'Repeated validation selects candidates; it is not an independent final strength claim.']}
    path=ROOT/'benchmarks/2026-10-29-v6-review.json';atomic_json(path,report)
    print(json.dumps({'file':str(path),'training':{k:v for k,v in training.items() if k!='paired_validations'},
                      'evaluations':{k:v['complete'] for k,v in reports.items()}},indent=2))


if __name__=='__main__':summarize()
