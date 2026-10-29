"""Compare choices on identical public positions; outcome ties alone are not duplication."""
import json
import pickle
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from champions_rl.engine import Engine
from champions_rl.features import Encoder
from champions_rl.policies import ScriptedBot,TRAINING_STYLES
from champions_rl.persistence import atomic_json

positions=pickle.loads((ROOT/'runtime_data/feature_profile_positions.pkl').read_bytes())
styles=['damage','balanced','setup','defensive',*TRAINING_STYLES[2:]]
choices={name:[] for name in styles}
with Engine() as engine:
    encoder=Encoder(engine=engine)
    bots={n:ScriptedBot(n,encoder=encoder) for n in styles}
    for obs in positions:
        encoded=encoder.encode(obs)
        for name,bot in bots.items():choices[name].append(int(np.argmax(bot.scores(obs,encoded))))
report={'positions':len(positions),'non_preview_positions':sum(p['phase']!='preview' for p in positions),
        'agreement':{a:{b:float(np.mean(np.array(choices[a])==choices[b])) for b in styles} for a in styles},
        'training_styles':TRAINING_STYLES,
        'notes':'One fixed 24-game public-position corpus; measures choice overlap, not playing strength. Legacy styles remain available only for historical comparisons.'}
atomic_json(ROOT/'benchmarks/2026-10-29-style-agreement-v4.json',report)
print(json.dumps(report),flush=True)
