"""Retry only invalid scheduled reference games, retaining every failed attempt."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
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
from champions_rl.power import keep_awake


@keep_awake()
def recover(source,output,workers=1):
    if not 1<=workers<=8:raise ValueError('Use 1 to 8 isolated recovery workers')
    source=Path(source).resolve();output=Path(output).resolve()
    if source==output:raise ValueError('Recovery must preserve the original output folder')
    report=json.loads((source/'report.json').read_text())
    if set(report['results'])!={'reference'}:raise ValueError('Use a reference-only evaluation')
    merged=source/'reference_games.json'
    records=json.loads(merged.read_text()) if merged.exists() else [row for path in sorted(source.glob('worker_*/reference_games.json')) for row in json.loads(path.read_text())]
    games=report['requested_games_per_opponent']
    by_game={r['game']:r for r in records}
    if len(by_game)!=len(records) or not set(by_game)<=set(range(games)):raise ValueError('Invalid original game IDs')
    records=[by_game.get(game,{'game':game,'error':'Game not completed before source interruption',
                               'interrupted_before_completion':True}) for game in range(games)]
    output.mkdir(parents=True,exist_ok=True)
    repairs=[];source_hashes={report['implementation_sha256']}
    def repair(index,record):
        retried=None
        for attempt in range(3):
            folder=output/(f"retry_{record['game']:06d}"+(f'_attempt{attempt+1}' if attempt else ''))
            completed=folder/'reference_games.json'
            if folder.exists() and not completed.exists():continue  # preserve an interrupted retry
            if not completed.exists():
                command=[sys.executable,'-u','-m','champions_rl','evaluate','--checkpoint',str(source/'evaluated_model.pt'),
                     '--opponents','reference','--games',str(games),'--seed',str(report['seed']),
                     '--split',report['split'],'--roster',report['roster'],'--reference-teams',report['reference_teams'],
                     '--shard-count',str(games),'--shard-index',str(record['game']),'--output',str(folder)]
                if report['deterministic']:command.append('--deterministic')
                with (output/(folder.name+'.log')).open('w',encoding='utf-8') as log:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,
                               creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            candidate=json.loads(completed.read_text())
            if len(candidate)!=1 or candidate[0]['game']!=record['game']:raise ValueError('Unexpected retry game IDs')
            if not candidate[0].get('error'):
                retried=candidate;break
        if retried is None:raise RuntimeError(f'Three incomplete/invalid retries; retained at {output}')
        retry_report=json.loads((folder/'report.json').read_text())
        for key in ('checkpoint_sha256','feature_signature','knowledge_sha256','team_manifest_sha256',
                    'reference_manifest_sha256','seed','split','roster','reference_teams','deterministic'):
            if report.get(key)!=retry_report.get(key):raise ValueError(f'Recovery changed {key}')
        provenance={'game':record['game'],'original_invalid_attempt':record,'retry':str(folder),
                    'retry_implementation_sha256':retry_report['implementation_sha256']}
        return index,retried[0],provenance

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending=[pool.submit(repair,index,record) for index,record in enumerate(records) if record.get('error')]
        for completed in as_completed(pending):
            index,record,provenance=completed.result()
            repairs.append(provenance);repairs.sort(key=lambda row:row['game'])
            source_hashes.add(provenance['retry_implementation_sha256']);records[index]=record
            atomic_json(output/'recovery_progress.json',repairs)
            print(json.dumps({'event':'reference_recovered','game':record['game'],'won':record['won']}),flush=True)
    wins=sum(r['won'] for r in records);draws=sum(r['winner'] is None for r in records)
    row={'games':games,'wins':wins,'losses':games-wins-draws,'draws':draws,'errors':0,
         'fallbacks':0,'truncations':sum(r['truncated'] for r in records),'win_rate':wins/games,'confidence':wilson(wins,games)}
    if any(a.get('errors') or a.get('deadline_passed') for r in records for a in r.get('reference_decisions',[])):
        raise ValueError('An allegedly valid reference game contains a failed decision audit')
    report.update(results={'reference':row},complete=True,recovered=True,recovery_workers=workers,original_report=str(source/'report.json'),
                  invalid_attempts_preserved=repairs,implementation_sources=sorted(source_hashes),
                  recovery_policy='Only interrupted or runtime-invalid scheduled games retried in separate processes; valid losses never rerun. Original failure evidence retained. This validation recovery is not a clean test certification.')
    report['implementation_sha256']=hashlib.sha256(json.dumps(sorted(source_hashes)).encode()).hexdigest()
    report['acceptance']=acceptance(report['results'],report['split'],report['roster'])
    report['acceptance']['passed']=False
    shutil.copyfile(source/'evaluated_model.pt',output/'evaluated_model.pt')
    atomic_json(output/'reference_games.json',records);atomic_json(output/'report.json',report)
    print(json.dumps(row),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=1,help='Concurrent isolated game processes; completed valid games are never replayed')
    recover(**vars(parser.parse_args()))
