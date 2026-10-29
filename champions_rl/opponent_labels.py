"""Semantic reply labels: move identity or public roster slot, never move slot."""
from .knowledge import knowledge


def switch_offset():
    return len(knowledge().vocab['moves'])+2


def label_count():
    return switch_offset()+6


def action_label(view, action_id):
    action=next((a for a in view['legal'] if a['id']==action_id),None)
    if action is None or action['kind']=='preview':
        return -100
    if action['kind']=='move':
        return knowledge().token('moves',action['move'])
    return switch_offset()+action['slot']
