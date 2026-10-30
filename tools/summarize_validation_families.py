"""Descriptive validation weaknesses by opponent and each player's team family."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from champions_rl.persistence import atomic_json


def group_results(games):
    groups=defaultdict(list)
    for style,row in games:
        if row.get('error'):raise ValueError('Cannot summarize failed validation games as battle outcomes')
        if len(row.get('team_families',[]))!=2:raise ValueError('Validation record lacks both team families')
        own=row['team_families'][row['seat']];enemy=row['team_families'][1-row['seat']]
        for axis,label in [('opponent',style),('own_family',own),('enemy_family',enemy),
                           ('opponent_and_own_family',style+'/'+own)]:
            groups[axis,label].append(row)
    result={}
    for (axis,label),rows in sorted(groups.items()):
        wins=sum(bool(r['won']) for r in rows)
        result.setdefault(axis,{})[label]={'games':len(rows),'wins':wins,'win_rate':wins/len(rows),
            'seed_blocks':len({r['game']//2 for r in rows}),
            'truncations':sum(r.get('truncated',False) for r in rows)}
    return result


def summarize(folder):
    folder=Path(folder);report_path=folder/'report.json';report=json.loads(report_path.read_text())
    if not report.get('complete'):raise ValueError('Complete the validation schedule first')
    records=[];sources={report_path.name:hashlib.sha256(report_path.read_bytes()).hexdigest()}
    for style,counts in report['results'].items():
        path=folder/(style+'_games.json');rows=json.loads(path.read_text())
        if counts['errors'] or counts['fallbacks'] or len(rows)!=counts['games']:
            raise ValueError('Incomplete or invalid validation records')
        records.extend((style,row) for row in rows)
        sources[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    return {'source':str(folder),'source_sha256':sources,'checkpoint_sha256':report['checkpoint_sha256'],
            'split':report['split'],'roster':report['roster'],'deterministic':report['deterministic'],
            'groups':group_results(records),
            'limits':'Descriptive validation only. Repeated seeds, teams and opponent styles are correlated. '
                     'Small family cells do not establish a ranking or causal explanation of failures.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluations',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    atomic_json(args.output,{'evaluations':[summarize(folder) for folder in args.evaluations]})
