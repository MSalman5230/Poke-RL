"""Simulator-verified immediate winning endgames, independent of value/search scores."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from champions_rl.engine import Engine
from champions_rl.features import Encoder
from champions_rl.model import BattleModel,NeuralBot
from champions_rl.teams import TeamPool


def fixtures():
    pool=TeamPool();reference=pool.data['reference_team']
    def set_(species,moves,item='Lum Berry',ability=None):
        row=copy.deepcopy(pool.by_species[species][0]['set']);row.update(moves=moves.split(','),item=item)
        if ability:row['ability']=ability
        return row
    def team(lead):return [lead]+[copy.deepcopy(s) for s in reference if s['species']!=lead['species']][:5]
    lucario=set_('Lucario','Extreme Speed,Close Combat,Bullet Punch,Swords Dance')
    garchomp=set_('Garchomp','Earthquake,Outrage,Rock Slide,Poison Jab','Choice Scarf','Sand Veil')
    gholdengo=set_('Gholdengo','Make It Rain,Shadow Ball,Thunderbolt,Psychic','Choice Scarf')
    grimmsnarl=set_('Grimmsnarl','Scary Face,Taunt,Spirit Break,Reflect','Light Clay','Prankster')
    grimmsnarl['nature']='Jolly';grimmsnarl['evs']={'hp':2,'atk':32,'spe':32}
    slow_gold=set_('Gholdengo','Make It Rain,Shadow Ball,Flash Cannon,Dazzling Gleam','Air Balloon')
    slow_gold['nature']='Quiet';slow_gold['evs']={'hp':32,'spa':32,'spd':2}
    cases=[('priority_over_raw_power',lucario,garchomp,[30,25]),
           ('priority_and_ghost_immunity',lucario,gholdengo,[30,10]),
           ('status_immunity_winning_attack',grimmsnarl,slow_gold,[1,10])]
    result=[]
    for name,own,foe,hp in cases:
        a=team(own);b=team(foe)
        # Substitute harmless unused item duplicates only in the background
        # roster. The active complete sets are kept exactly as authored.
        for party in (a,b):
            items=set()
            for mon in party:
                if mon['item'] in items:mon['item']=''
                if mon['item']:items.add(mon['item'])
        result.append({'name':name,'teams':[a,b],'hp':hp})
    return result


def audit(checkpoint,output):
    torch.set_num_threads(1)
    source=fixtures()
    process=subprocess.run(['node',str(ROOT/'simulator/exercises.cjs')],input=json.dumps(source),
                            text=True,encoding='utf-8',capture_output=True,timeout=120,
                            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if process.returncode:raise RuntimeError(process.stderr)
    verified=json.loads(process.stdout);model,_=BattleModel.load(checkpoint)
    with Engine() as engine:
        bot=NeuralBot(model,Encoder(engine=engine),deterministic=True)
        for row in verified:
            row['policy_action']=bot.choose(row['observation']);row['solved']=row['policy_action'] in row['winning_actions']
    report={'checkpoint':str(checkpoint),'solved':sum(r['solved'] for r in verified),'exercises':len(verified),
            'results':[{k:v for k,v in r.items() if k!='observation'} for r in verified],
            'training_performed':False,'limits':'A small diagnostic suite. Never substitute these scores for paired battle validation.'}
    Path(output).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    audit(**vars(parser.parse_args()))
