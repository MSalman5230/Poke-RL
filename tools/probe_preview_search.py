"""Time one public preview search and record its complete terminal evidence."""
import argparse
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from champions_rl.arena import load_teams
from champions_rl.engine import Engine
from champions_rl.features import Encoder
from champions_rl.model import BattleModel
from champions_rl.preview_search import TerminalPreviewBot


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu');args=parser.parse_args()
    torch.set_num_threads(1);model,_=BattleModel.load(args.checkpoint,args.device)
    teams=load_teams('validation_cores')[:2]
    with Engine() as engine:
        obs=engine.request('new',game='probe',teams=teams,seed=[17,31,48,92])['views'][0]
        bot=TerminalPreviewBot(model,Encoder(engine=engine),engine=engine,seed=120031)
        action=bot.choose(obs)
        result={'action':action,**bot.last_search,'statistics':dict(bot.statistics)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
