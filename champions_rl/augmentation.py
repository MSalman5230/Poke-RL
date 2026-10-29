"""Explicit, replayable input augmentation; never randomize inside model.forward."""
import numpy as np
import torch
from .knowledge import knowledge


ID_FIELDS={'species':('species','history_species'),
           'moves':('moves','action_moves','history_moves','last_moves'),
           'items':('items','items_prior'),'abilities':('abilities','abilities_prior')}


def identity_dropout(batch,probability=.25,masks=None):
    """Drop the same token everywhere within a position, keeping numeric facts.

    Returns masks reusable for the private view. PPO must store the returned
    augmented observation with its old log probability, then reuse it unchanged
    during optimization. Applying fresh dropout at update time is off-policy.
    UNKNOWN (0), known NONE (1), move validity and action legality are preserved.
    """
    if not 0<=probability<1:raise ValueError('Identity dropout must be in [0,1)')
    if not probability and masks is None:return batch,{}
    result=dict(batch);masks={} if masks is None else dict(masks)
    sample=next(iter(batch.values()));size=len(sample);device=sample.device
    for kind,fields in ID_FIELDS.items():
        if kind not in masks:
            mask=torch.rand(size,len(knowledge().vocab[kind])+2,device=device)<probability
            mask[:,:2]=False;masks[kind]=mask
        for field in fields:
            if field not in batch:continue
            ids=batch[field]
            drop=masks[kind].gather(1,ids.reshape(size,-1)).reshape_as(ids)
            result[field]=ids.masked_fill(drop,0)
    return result,masks


def permute_encoded_moves(encoded,orders,label=None):
    """Reorder presentation slots, carrying attributes, legal actions and labels.

    orders[entity, new_slot] gives the old slot. Mechanically meaningful first
    move flags stay attached to their move: this changes tensor presentation,
    not Conversion's true first move. Semantic opponent-move labels don't move.
    """
    orders=np.asarray(orders)
    if orders.shape!=(12,4) or not np.all(np.sort(orders,axis=1)==np.arange(4)):
        raise ValueError('Expected twelve permutations of four slots')
    result={k:v.copy() for k,v in encoded.items()}
    for key in ('moves','move_features','move_mask'):
        result[key]=encoded[key][np.arange(12)[:,None],orders]
    if 'actions' not in result:return result,label
    active=int(encoded['pokemon'][:6,3].argmax())
    mapping=np.r_[orders[active],orders[active]+4,np.arange(8,134)]
    for key in ('actions','action_moves','mask','guide'):
        if key in result:result[key]=encoded[key][mapping]
    label=None if label is None else int(np.argsort(mapping)[label])
    return result,label
