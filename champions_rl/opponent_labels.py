"""Semantic reply labels: move identity or public roster slot, never move slot."""
from .knowledge import knowledge


def switch_offset():
    return len(knowledge().vocab['moves'])+2


def label_count():
    return switch_offset()+6


def opponent_loss(logits,targets):
    """Fail on impossible supervision instead of hiding a billion-point loss."""
    import torch
    valid=targets>=0
    if not valid.any():return logits.sum()*0
    labels=targets[valid]
    if (labels>=logits.shape[-1]).any():raise ValueError('Opponent target outside the model vocabulary')
    selected=logits[valid].gather(1,labels.unsqueeze(1)).squeeze(1)
    impossible=(selected<=-1e8)|~torch.isfinite(selected)
    if impossible.any():
        raise ValueError('Masked/nonfinite opponent training target: '+str(labels[impossible].detach().cpu().tolist()))
    return torch.nn.functional.cross_entropy(logits[valid],labels)


def action_label(view, action_id):
    action=next((a for a in view['legal'] if a['id']==action_id),None)
    if action is None or action['kind']=='preview':
        return -100
    if action['kind']=='move':
        return knowledge().token('moves',action['move'])
    return switch_offset()+action['slot']
