"""Usage-teammate-guided compositions of intact, simulator-validated sets.

These are synthetic teams, not recovered human teams. Marginal co-occurrence is
used as a prior alongside item/species clauses, weather and defensive coverage.
"""
import copy
from collections import defaultdict
from .knowledge import knowledge,usage_data,canonical

WEATHER={'drizzle':'rain','drought':'sun','sandstream':'sand','snowwarning':'snow'}
DEPENDENT={'swiftswim':'rain','chlorophyll':'sun','solarpower':'sun','sandrush':'sand','slushrush':'snow'}


def team_family(team):
    k=knowledge()
    if any('trickroom' in map(canonical,s['moves']) for s in team):return 'trick_room'
    climates={WEATHER.get(canonical(s['ability'])) for s in team}-{None}
    if climates:return '+'.join(sorted(climates))
    bulky=sum(s.get('evs',{}).get('hp',0)>=24 and max(s.get('evs',{}).get('def',0),s.get('evs',{}).get('spd',0))>=24 for s in team)
    setup=sum(any(k.moves[canonical(m)].get('boosts') for m in s['moves']) for s in team)
    return 'bulky' if bulky>=3 else 'setup_offense' if setup>=3 else 'mixed'


class TeammateSampler:
    def __init__(self,pool):
        self.pool=pool;self.k=knowledge();self.names=list(pool.by_species)
        self.aliases={canonical(n):n for n in self.names}
        for sid,species in self.k.species.items():
            base=species.get('baseSpecies',species['name'])
            if base in pool.by_species:self.aliases[sid]=base
        self.prior=defaultdict(float);counts=defaultdict(lambda:defaultdict(float))
        for sid,row in usage_data().items():
            name=self.aliases.get(sid)
            if name is None:continue
            self.prior[name]+=row.get('usage',0.)
            for friend,count in row.get('Teammates',{}).items():
                target=self.aliases.get(canonical(friend))
                if target and target!=name:counts[name][target]+=max(0.,count)
        self.prior={n:(self.prior[n]+.00005)**.65 for n in self.names}
        total=sum(self.prior.values());self.prior={n:w/total for n,w in self.prior.items()}
        self.conditional={n:{t:w/sum(row.values()) for t,w in row.items()} for n,row in counts.items() if sum(row.values())}
        self.options={n:[self.describe(r) for r in pool.by_species[n]] for n in self.names}

    def describe(self,row):
        set_=row['set'];mon=self.k.species_info(set_['species']);item=self.k.items.get(canonical(set_['item']),{})
        stone=item.get('megaStone');mega=bool(stone)
        if isinstance(stone,dict):stone=stone.get(mon['name'])
        effective=self.k.species_info(stone) if stone else mon
        ability=canonical(next(iter(effective.get('abilities',{}).values()),set_['ability'])) if mega else canonical(set_['ability'])
        moves=[self.k.moves[canonical(m)] for m in set_['moves']]
        return {'row':row,'number':mon['num'],'item':canonical(set_['item']),'mega':mega,
                'weather':WEATHER.get(ability),'dependent':DEPENDENT.get(ability),
                'physical':any(m['category']=='Physical' for m in moves),'special':any(m['category']=='Special' for m in moves),
                'types':effective.get('types',mon['types']),
                'rain_move':'electroshot' in map(canonical,set_['moves']) and canonical(set_['item'])!='powerherb',
                'veil':'auroraveil' in map(canonical,set_['moves'])}

    def sample(self,rng):
        for attempt in range(40):
            selected=[];numbers=set();items=set();weather=None;mega=False
            for slot in range(6):
                allowed={};weights=[];names=[]
                for name in self.names:
                    options=[]
                    for option in self.options[name]:
                        if option['number'] in numbers or option['item'] and option['item'] in items:continue
                        if mega and option['mega']:continue
                        climate=option['weather'] or weather
                        if weather and option['weather'] and weather!=option['weather']:continue
                        if option['dependent'] and option['dependent']!=climate:continue
                        if option['rain_move'] and climate!='rain' or option['veil'] and climate!='snow':continue
                        if slot==5 and (not any(x['physical'] for x in selected) and not option['physical'] or
                                        not any(x['special'] for x in selected) and not option['special']):continue
                        options.append(option)
                    if not options:continue
                    affinity=sum(self.conditional.get(x['row']['set']['species'],{}).get(name,self.prior[name])
                                 for x in selected)/max(1,len(selected))
                    weight=self.prior[name] if not selected else .2*self.prior[name]+.8*affinity
                    # Reduce repeated severe weaknesses without forcing a fixed
                    # six-species template or rebuilding individual sets.
                    types=options[0]['types']
                    for attack in self.k.type_names:
                        if self.k.effectiveness(attack,types)>1:
                            weak=sum(self.k.effectiveness(attack,x['types'])>1 for x in selected)
                            if weak>=2:weight/=1.6**(weak-1)
                    names.append(name);weights.append(weight);allowed[name]=options
                if not names:break
                name=rng.choices(names,weights)[0];options=allowed[name]
                option=rng.choices(options,[5. if x['row'].get('curated') else 1. for x in options])[0]
                selected.append(option);numbers.add(option['number'])
                if option['item']:items.add(option['item'])
                weather=option['weather'] or weather;mega|=option['mega']
            if len(selected)==6:
                team=[copy.deepcopy(x['row']['set']) for x in selected];rng.shuffle(team);return team
        raise RuntimeError('Could not assemble a coherent teammate-guided team')
