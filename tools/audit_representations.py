"""Probe move permutations, destination binding and counter visibility on fixed public states."""
import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[1]
    parser.add_argument('--runtime',type=Path,default=root)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--positions',type=Path,default=root/'runtime_data/feature_profile_positions.pkl')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=208)
    args=parser.parse_args()
    sys.path.insert(0,str(args.runtime.resolve()))
    import numpy as np
    import torch
    from champions_rl.engine import Engine
    from champions_rl.features import Encoder
    from champions_rl.model import BattleModel,tensor_batch
    from champions_rl.opponent_labels import switch_offset
    from champions_rl.provenance import feature_signature
    torch.set_num_threads(2)
    model,_=BattleModel.load(args.checkpoint)
    corpus=pickle.loads(args.positions.read_bytes())
    # Include preview and later positions throughout the fixed profiling corpus.
    positions=[corpus[i] for i in np.linspace(0,len(corpus)-1,min(args.samples,len(corpus)),dtype=int)]
    rng=np.random.default_rng(32071)
    rows=[];switch_errors=[]
    with Engine() as engine,torch.no_grad():
        encoder=Encoder(engine=engine)
        for obs in positions:
            changed=copy.deepcopy(obs);mapping=np.arange(134)
            for mon in changed['own']:
                moves=mon.get('moves',[])
                # Conversion's first native move is a real mechanic, not an arbitrary ordering.
                if any((m.get('id') if isinstance(m,dict) else m)=='conversion' for m in moves):continue
                order=rng.permutation(len(moves))
                mon['moves']=[moves[i] for i in order]
                if mon.get('set') and len(mon['set']['moves'])==len(order):
                    mon['set']['moves']=[mon['set']['moves'][i] for i in order]
                if mon.get('active'):
                    inverse=np.argsort(order)
                    for old,new in enumerate(inverse):
                        mapping[old]=new;mapping[old+4]=new+4
                    for a in changed['legal']:
                        if a['kind']=='move':
                            old=a['id'];a['slot']=int(inverse[a['slot']]);a['id']=a['slot']+4*bool(a.get('mega'))
            a=encoder.encode(obs);b=encoder.encode(changed)
            logits,_,foe=model(tensor_batch([a]));other=model(tensor_batch([b]))[0]
            p=logits[0].softmax(-1).numpy();q=other[0].softmax(-1).numpy()[mapping]
            rows.append({'phase':obs['phase'],'probability_max_error':float(abs(p-q).max()),
                         'decision_changed':bool(p.argmax()!=q.argmax())})
            benches=[i for i,m in enumerate(obs['opponent']) if not m.get('active') and not m.get('fainted')]
            if len(benches)>=2:
                left=copy.deepcopy(obs);left['history']=[]
                right=copy.deepcopy(left);i,j=benches[:2]
                right['opponent'][i],right['opponent'][j]=right['opponent'][j],right['opponent'][i]
                x=model(tensor_batch([encoder.encode(left)]))[2][0,switch_offset():].numpy()
                y=model(tensor_batch([encoder.encode(right)]))[2][0,switch_offset():].numpy()
                y[[i,j]]=y[[j,i]]
                switch_errors.append(float(abs(x-y).max()))
        obs=copy.deepcopy(positions[0])
        actor=obs['own'][0];actor['times_attacked']=0
        before=encoder.encode(obs,entities_only=True)
        actor['times_attacked']=5
        after=encoder.encode(obs,entities_only=True)
        hits_visible=any(not np.array_equal(before[k],after[k]) for k in before)
        obs['faints']=[[],[]];before=encoder.encode(obs,entities_only=True)
        obs['faints'][0]=[{'species':actor['species'],'turn':0}]*4
        after=encoder.encode(obs,entities_only=True)
        faints_visible=any(not np.array_equal(before[k],after[k]) for k in before)
    report={'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'feature_signature':feature_signature(),'corpus_sha256':hashlib.sha256(args.positions.read_bytes()).hexdigest(),
            'positions':len(rows),'by_phase':{phase:{'positions':sum(r['phase']==phase for r in rows),
               'changed':sum(r['decision_changed'] for r in rows if r['phase']==phase),
               'max_probability_error':max(r['probability_max_error'] for r in rows if r['phase']==phase)}
               for phase in sorted({r['phase'] for r in rows})},
            'switch_destination_max_logit_error':max(switch_errors,default=0.),
            'rage_fist_counter_visible':hits_visible,'cumulative_faints_visible':faints_visible,
            'interpretation':'Representation audit, not a win-rate comparison. Move tests exclude Conversion because its first slot has mechanical meaning.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
