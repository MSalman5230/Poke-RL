"""Repeatable feature timing on a fixed public-position corpus (no training)."""
import argparse
import cProfile
import json
import pickle
import pstats
import random
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from champions_rl.arena import load_teams,play_match
from champions_rl.engine import Engine
from champions_rl.features import Encoder
from champions_rl.policies import ScriptedBot


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--corpus',type=Path,default=ROOT/'runtime_data/feature_profile_positions.pkl')
    parser.add_argument('--games',type=int,default=24)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.corpus.exists():
        rng=random.Random(839281)
        teams=load_teams('validation')
        positions=[]
        with Engine() as engine:
            encoder=Encoder(engine=engine)
            bots=[ScriptedBot('balanced',encoder=encoder),ScriptedBot('stochastic',seed=7,encoder=encoder)]
            for game in range(args.games):
                play_match(engine,[rng.choice(teams),rng.choice(teams)],bots,seed=839281+game,
                           capture=lambda side,view,action:positions.append(view))
        args.corpus.parent.mkdir(parents=True,exist_ok=True)
        args.corpus.write_bytes(pickle.dumps(positions))
    positions=pickle.loads(args.corpus.read_bytes())  # locally generated only
    with Engine() as engine:
        encoder=Encoder(engine=engine)
        profile=cProfile.Profile()
        started=time.monotonic()
        profile.enable()
        for position in positions:encoder.encode(position)
        profile.disable()
        elapsed=time.monotonic()-started
        profile.dump_stats(str(args.output/'features.prof'))
        with (args.output/'profile.txt').open('w',encoding='utf-8') as f:
            pstats.Stats(profile,stream=f).strip_dirs().sort_stats('cumulative').print_stats(45)
        report={'positions':len(positions),'seconds_with_profiler':elapsed,
                'positions_per_second_with_profiler':len(positions)/elapsed,
                'native_cache':getattr(encoder.native,'metrics',{})}
        (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
