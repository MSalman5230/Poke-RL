"""Public-evidence-conditioned hypotheses; never consult an opponent's real set.

The complete-set library supplies joint hypotheses. Usage marginals weight them;
they are a prior, not a claim that marginal products recover human joint usage.
"""
from functools import lru_cache
import itertools
import math
from .knowledge import knowledge,canonical
from .teams import TeamPool


def revealed_moves(mon):
    return tuple(canonical(m['id'] if isinstance(m,dict) else m) for m in mon.get('moves',[]))


class SetBeliefs:
    def __init__(self):
        self.k=knowledge()
        self.rows={canonical(n):rows for n,rows in TeamPool().by_species.items()}

    @lru_cache(maxsize=8192)
    def _candidates(self,species,revealed,item,ability):
        k=self.k
        info=k.species_info(species)
        base=canonical(info.get('baseSpecies',species))
        rows=self.rows.get(canonical(species),self.rows.get(base,[]))
        usage=k.usage(species)
        def weight(s):
            value=1.
            for key,choice in [('Items',canonical(s['item'])),('Abilities',canonical(s['ability']))]:
                d=usage.get(key,{})
                value*=max(.005,d.get(choice,0)/max(1,sum(d.values())))
            moves=usage.get('Moves',{})
            value*=math.exp(sum(math.log(max(.01,moves.get(canonical(m),0)/max(1,sum(moves.values())))) for m in s['moves'])/4)
            spread=s['nature']+':'+ '/'.join(str(s['evs'].get(stat,0)) for stat in ('hp','atk','def','spa','spd','spe'))
            spreads=usage.get('Spreads',{})
            value*=max(.01,spreads.get(spread,0)/max(1,sum(spreads.values())))
            return value
        options=[]
        for row in rows:
            s=row['set']
            if not set(revealed).issubset(map(canonical,s['moves'])): continue
            if item is not None and item!='' and canonical(s['item'])!=item: continue
            if ability is not None and ability!='' and canonical(s['ability'])!=ability: continue
            options.append((s,weight(s)*(4 if row.get('curated') else 1)))
        if not options:
            legal=set(info.get('learnset') or k.species_info(base).get('learnset',[]))
            possible=[m for m,_ in k.prior(species,'Moves',12) if m in legal and m not in revealed]
            possible += [m for m in sorted(legal,key=lambda m:-k.moves[m]['basePower']) if m not in possible and m not in revealed]
            moves=list(revealed[:4])
            moves += possible[:max(0,4-len(moves))]
            items=[(item,1.)] if item is not None else k.prior(species,'Items',3) or [('',1.)]
            abilities=[(ability,1.)] if ability is not None else k.prior(species,'Abilities',2)
            for (nature,evs,sw),(iid,iw),(aid,aw) in itertools.product(k.spread_prior(species)[:3],items,abilities):
                s={'species':info.get('baseSpecies',species),'level':50,'moves':moves,'item':iid,'ability':aid,'nature':nature,'evs':evs}
                options.append((s,sw*iw*aw))
        options=sorted(options,key=lambda r:-r[1])[:8]
        total=sum(w for _,w in options)
        return [(s,w/total) for s,w in options] if total else []

    def candidates(self,mon):
        if 'set' in mon: return [(mon['set'],1.)]
        return self._candidates(mon.get('species',''),revealed_moves(mon),mon.get('item'),mon.get('ability'))

    def scenarios(self,mon):
        return [({**mon,'set':s,'item':mon['item'] if mon.get('item') is not None else canonical(s['item']),
                  'ability':mon['ability'] if mon.get('ability') is not None else canonical(s['ability'])},w) for s,w in self.candidates(mon)]

    def moves(self,mon):
        result={m:1. for m in revealed_moves(mon)}
        for s,weight in self.candidates(mon):
            for move in s['moves']:
                mid=canonical(move)
                if mid not in revealed_moves(mon): result[mid]=result.get(mid,0)+weight
        return result

    def sample(self,mon,rng):
        choices=self.candidates(mon)
        return dict(rng.choices([s for s,_ in choices],weights=[w for _,w in choices])[0])
