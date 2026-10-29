"""Separate value discrimination from overconfidence using whole-game folds."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from champions_rl.persistence import atomic_json


def metrics(prediction,target):
    variance=np.var(target)
    return {'return_mse':float(np.mean((prediction-target)**2)),
            'explained_variance':float(1-np.var(target-prediction)/variance) if variance else None,
            'correlation':float(np.corrcoef(prediction,target)[0,1]) if np.var(prediction) and variance else None}


def crossfit(rows,name):
    prediction=np.asarray([r[name] for r in rows],dtype=float);target=np.asarray([r['outcome']-.15*r['potential'] for r in rows],dtype=float)
    fold=np.asarray([(r['game']//2)%2 for r in rows])
    calibrated=np.empty_like(prediction);fits=[]
    for heldout in (0,1):
        train=fold!=heldout;test=~train
        if not train.any() or not test.any():raise ValueError('Both seed-block folds must contain complete games')
        x=prediction[train];y=target[train]
        # Constrain this diagnostic to shrinkage: do not invert or amplify a
        # critic by fitting the evaluation outcomes it is being scored on.
        slope=float(np.clip(np.mean((x-x.mean())*(y-y.mean()))/max(1e-12,np.var(x)),0,1))
        intercept=float(y.mean()-slope*x.mean())
        calibrated[test]=intercept+slope*prediction[test]
        fits.append({'heldout_fold':heldout,'slope':slope,'intercept':intercept,
                     'training_games':len({r['game'] for r,keep in zip(rows,train) if keep}),
                     'heldout_games':len({r['game'] for r,keep in zip(rows,test) if keep})})
    return {'raw':metrics(prediction,target),'out_of_fold_shrinkage':metrics(calibrated,target),'fits':fits}


def audit(folder,output):
    folder=Path(folder);source=folder/'positions.json';rows=json.loads(source.read_text())
    provenance=json.loads((folder/'report.json').read_text())
    report={'source':str(folder),'positions_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'checkpoint_sha256':provenance['checkpoint_sha256'],'critic_sha256':provenance['critic_sha256'],
            'positions':len(rows),'games':len({r['game'] for r in rows}),
            'public':crossfit(rows,'public'),'private':crossfit(rows,'private'),
            'limits':'Two folds keep both seats and all positions of a seed block together. Diagnostic only: '
                     'no model weights or deployment scores changed. Better calibration across observed games '
                     'does not establish correct ranking of alternative actions or justify search promotion.'}
    atomic_json(output,report);print(json.dumps({name:report[name] for name in ('public','private')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.audit,args.output)
