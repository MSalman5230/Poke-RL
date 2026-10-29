"""Retry only invalid scheduled reference games, retaining every failed attempt."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from champions_rl.evaluation import acceptance,wilson
from champions_rl.persistence import atomic_json


def recover(source,output):
    source=Path(source).resolve();output=Path(output).resolve()
    if source==output:raise ValueError('Recovery must preserve the original output folder')
    report=json.loads((source/'report.json').read_text())
    if set(report['results'])!={'reference'}:raise ValueError('Use a reference-only evaluation')
    records=json.loads((source/'reference_games.json').read_text())
    games=report['requested_games_per_opponent']
    if {r['game'] for r in records}!=set(range(games)):raise ValueError('Original schedule has missing games')
    output.mkdir(parents=True,exist_ok=True)
    repairs=[];source_hashes={report['implementation_sha256']}
    for index,record in enumerate(records):
        if not record.get('error'):continue  # never rerun a valid loss
        folder=output/f"retry_{record['game']:06d}"
        completed=folder/'reference_games.json'
        if not completed.exists():
            command=[sys.executable,'-u','-m','champions_rl','evaluate','--checkpoint',str(source/'evaluated_model.pt'),
                     '--opponents','reference','--games',str(games),'--seed',str(report['seed']),
                     '--split',report['split'],'--roster',report['roster'],'--reference-teams',report['reference_teams'],
                     '--shard-count',str(games),'--shard-index',str(record['game']),'--output',str(folder)]
            if report['deterministic']:command.append('--deterministic')
            with (output/(folder.name+'.log')).open('w',encoding='utf-8') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,
                               creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        retry_report=json.loads((folder/'report.json').read_text())
        for key in ('checkpoint_sha256','feature_signature','knowledge_sha256','team_manifest_sha256',
                    'reference_manifest_sha256','seed','split','roster','reference_teams','deterministic'):
            if report.get(key)!=retry_report.get(key):raise ValueError(f'Recovery changed {key}')
        retried=json.loads(completed.read_text())
        if len(retried)!=1 or retried[0]['game']!=record['game'] or retried[0].get('error'):
            raise RuntimeError(f'Retry still invalid; original and retry retained at {folder}')
        repairs.append({'game':record['game'],'original_invalid_attempt':record,'retry':str(folder),
                        'retry_implementation_sha256':retry_report['implementation_sha256']})
        source_hashes.add(retry_report['implementation_sha256']);records[index]=retried[0]
        atomic_json(output/'recovery_progress.json',repairs)
        print(json.dumps({'event':'reference_recovered','game':record['game'],'won':retried[0]['won']}),flush=True)
    wins=sum(r['won'] for r in records);draws=sum(r['winner'] is None for r in records)
    row={'games':games,'wins':wins,'losses':games-wins-draws,'draws':draws,'errors':0,
         'fallbacks':0,'truncations':sum(r['truncated'] for r in records),'win_rate':wins/games,'confidence':wilson(wins,games)}
    if any(a.get('errors') or a.get('deadline_passed') for r in records for a in r.get('reference_decisions',[])):
        raise ValueError('An allegedly valid reference game contains a failed decision audit')
    report.update(results={'reference':row},complete=True,recovered=True,original_report=str(source/'report.json'),
                  invalid_attempts_preserved=repairs,implementation_sources=sorted(source_hashes),
                  recovery_policy='Only runtime-invalid scheduled games retried; valid losses never rerun. Original failure evidence retained. This validation recovery is not a clean test certification.')
    report['implementation_sha256']=hashlib.sha256(json.dumps(sorted(source_hashes)).encode()).hexdigest()
    report['acceptance']=acceptance(report['results'],report['split'],report['roster'])
    report['acceptance']['passed']=False
    shutil.copyfile(source/'evaluated_model.pt',output/'evaluated_model.pt')
    atomic_json(output/'reference_games.json',records);atomic_json(output/'report.json',report)
    print(json.dumps(row),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    recover(**vars(parser.parse_args()))
