"""Experimental team-preview search using terminal policy rollouts, no critic."""
from collections import Counter
import copy
import random
import time
import uuid
import numpy as np
import torch
from .engine import Engine
from .features import Encoder
from .knowledge import canonical
from .model import NeuralBot,encode_for_model,tensor_batch
from .search import paired_search_margin


class TerminalPreviewBot(NeuralBot):
    """Compare three distinct trio/lead choices on identical sampled worlds.

    Both seats continue with the frozen base policy, using only their own
    observations. Every score is an actual terminal result. A failed or capped
    continuation invalidates the whole comparison; it is never valued or
    selectively dropped. This prototype deliberately starts at team preview,
    where reconstruction of hidden ongoing effects is unnecessary.
    """
    def __init__(self,model,encoder=None,worlds=16,candidates=3,seed=0,engine=None,max_turns=200):
        super().__init__(model,encoder or Encoder(),deterministic=True,name='terminal_preview')
        if worlds<16 or candidates<2:raise ValueError('Preview search requires at least 16 worlds and two candidates')
        self.worlds=worlds;self.candidates=candidates;self.rng=random.Random(seed)
        self.engine=engine;self.max_turns=max_turns;self.statistics=Counter();self.last_search={}

    def options(self,obs,probabilities):
        choices=[];seen=set()
        for action in sorted(obs['legal'],key=lambda a:-probabilities[a['id']]):
            picks=action['picks'];key=(picks[0],*sorted(picks[1:]))
            if key in seen:continue # exchanging reserve order is not another trio/lead
            choices.append(action['id']);seen.add(key)
            if len(choices)==self.candidates:break
        return choices

    def sample_teams(self,obs):
        if obs['phase']!='preview':raise ValueError('Terminal search currently supports team preview only')
        if any('set' in m or 'stats' in m or 'selected' in m for m in obs['opponent']):
            raise ValueError('Search requires public opponent information')
        own=copy.deepcopy([m['set'] for m in obs['own']])
        for attempt in range(256):
            foe=[];items=set()
            for mon in obs['opponent']:
                compatible=[(s,w) for s,w in self.encoder.beliefs.candidates(mon)
                            if not s['item'] or canonical(s['item']) not in items]
                if not compatible:break
                s=copy.deepcopy(self.rng.choices([s for s,w in compatible],[w for s,w in compatible])[0])
                foe.append(s)
                if s['item']:items.add(canonical(s['item']))
            if len(foe)==6 and not any(self.engine.request('validate',teams=[own,foe])):return [own,foe]
        raise ValueError('Could not sample a complete legal hidden world')

    def policy_actions(self,observations):
        result=[]
        with torch.no_grad():
            for start in range(0,len(observations),128):
                batch=tensor_batch([encode_for_model(self.model,self.encoder,o) for o in observations[start:start+128]],self.device)
                result.extend(self.model(batch)[0].argmax(-1).cpu().tolist())
        return result

    def choose(self,obs):
        self.statistics['decisions']+=1
        if obs['phase']!='preview':
            self.statistics['policy_only']+=1
            return super().choose(obs)
        if any('set' in m or 'stats' in m or 'selected' in m for m in obs['opponent']):
            raise ValueError('Search requires public opponent information')
        with torch.no_grad():
            probabilities=self.model(tensor_batch([encode_for_model(self.model,self.encoder,obs)],self.device))[0][0].softmax(-1).cpu().numpy()
        policy=int(probabilities.argmax());options=self.options(obs,probabilities)
        if len(options)<2:return policy
        if self.engine is None:self.engine=Engine()
        started=time.monotonic();games=[];scores={a:[] for a in options};decisions=0
        try:
            prefix='preview_'+uuid.uuid4().hex
            requests=[]
            for world in range(self.worlds):
                teams=self.sample_teams(obs);seed=[self.rng.randrange(65536) for _ in range(4)]
                for action in options:
                    name=f'{prefix}_{world}_{action}';games.append({'id':name,'world':world,'action':action})
                    requests.append({'op':'new','game':name,'teams':teams,'seed':seed,'max_turns':self.max_turns})
            states=self.engine.batch(requests)
            # Same opponent preview decision in all candidate copies of a world.
            enemy=self.policy_actions([state['views'][1] for state in states]);decisions+=len(enemy)
            states=self.engine.batch([{'op':'step','game':g['id'],'actions':[g['action'],reply]} for g,reply in zip(games,enemy)])
            while any(not (s['ended'] or s['truncated']) for s in states):
                active=[i for i,s in enumerate(states) if not (s['ended'] or s['truncated'])]
                slots=[(i,side) for i in active for side in (0,1) if states[i]['views'][side]['legal']]
                predictions=self.policy_actions([states[i]['views'][side] for i,side in slots]);decisions+=len(predictions)
                actions={i:[None,None] for i in active}
                for (i,side),action in zip(slots,predictions):actions[i][side]=action
                updated=self.engine.batch([{'op':'step','game':games[i]['id'],'actions':actions[i]} for i in active])
                for i,state in zip(active,updated):states[i]=state
            capped=sum(s['truncated'] or not s['ended'] for s in states)
            if capped:raise ValueError(f'{capped} unresolved continuations; retain base policy')
            for game,state in zip(games,states):
                scores[game['action']].append(0. if state['winner'] is None else 1. if state['winner']==0 else -1.)
            scores={a:np.asarray(v) for a,v in scores.items()}
            choice,intervals=paired_search_margin(scores,policy,np.random.default_rng(self.rng.randrange(2**31)),min_worlds=16)
            self.statistics['searched']+=1;self.statistics['overrides']+=choice!=policy
            self.statistics['terminal_games']+=len(states);self.statistics['rollout_decisions']+=decisions
            self.last_search={'used':True,'policy':policy,'chosen':choice,'worlds':self.worlds,'options':options,
                              'terminal_games':len(states),'rollout_decisions':decisions,'intervals':intervals,
                              'mean_terminal_scores':{str(a):float(v.mean()) for a,v in scores.items()},
                              'seconds':time.monotonic()-started,'value_head_used':False}
            return choice
        except (RuntimeError,ValueError,TimeoutError) as error:
            self.statistics['failed_comparisons']+=1;self.statistics['policy_only']+=1
            self.last_search={'used':False,'policy':policy,'chosen':policy,'error':str(error),
                              'rollout_decisions':decisions,'seconds':time.monotonic()-started,'value_head_used':False}
            return policy
        finally:
            if games:self.engine.batch([{'op':'close','game':g['id']} for g in games])
