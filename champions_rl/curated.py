"""Authored complete sets and coherent archetypes, validated by the pinned format.

These are strategic templates, not attributed tournament teams. Each moveset,
item, ability, nature and spread is chosen together before legality validation.
"""
import copy
from .knowledge import knowledge,canonical,STATS

TEMPLATES=[
 ('Gholdengo','Air Balloon','Good as Gold','Timid','spa spe','Make It Rain,Shadow Ball,Nasty Plot,Recover'),
 ('Garchomp','Choice Scarf','Rough Skin','Adamant','atk spe','Earthquake,Outrage,Rock Slide,Poison Jab'),
 ('Volcarona','Rocky Helmet','Flame Body','Timid','spa spe','Fiery Dance,Bug Buzz,Giga Drain,Quiver Dance'),
 ('Archaludon','Leftovers','Stamina','Modest','hp spa','Electro Shot,Draco Meteor,Flash Cannon,Thunder Wave'),
 ('Lucario','Lucarionite Z','Inner Focus','Jolly','atk spe','Close Combat,Meteor Mash,Bullet Punch,Swords Dance'),
 ('Sylveon','Fairy Feather','Pixilate','Modest','hp spa','Hyper Voice,Shadow Ball,Wish,Protect'),
 ('Pelipper','Damp Rock','Drizzle','Bold','hp def','Hurricane,Weather Ball,U-turn,Roost'),
 ('Basculegion','Mystic Water','Swift Swim','Adamant','atk spe','Wave Crash,Last Respects,Aqua Jet,Flip Turn'),
 ('Torkoal','Heat Rock','Drought','Bold','hp def','Lava Plume,Earth Power,Stealth Rock,Yawn'),
 ('Venusaur','Life Orb','Chlorophyll','Modest','spa spe','Giga Drain,Sludge Bomb,Weather Ball,Growth'),
 ('Tyranitar','Sitrus Berry','Sand Stream','Adamant','hp atk','Stone Edge,Crunch,Earthquake,Dragon Dance'),
 ('Excadrill','Life Orb','Sand Rush','Jolly','atk spe','Earthquake,Iron Head,Rock Slide,Swords Dance'),
 ('Ninetales-Alola','Light Clay','Snow Warning','Timid','hp spe','Aurora Veil,Freeze-Dry,Moonblast,Encore'),
 ('Baxcalibur','Lum Berry','Thermal Exchange','Jolly','atk spe','Icicle Crash,Glaive Rush,Earthquake,Dragon Dance'),
 ('Dragonite','Lum Berry','Multiscale','Adamant','atk spe','Extreme Speed,Earthquake,Fire Punch,Dragon Dance'),
 ('Scizor','Scizorite','Technician','Adamant','hp atk','Bullet Punch,U-turn,Knock Off,Swords Dance'),
 ('Salamence','Salamencite','Intimidate','Jolly','atk spe','Double-Edge,Earthquake,Dragon Dance,Roost'),
 ('Gengar','Gengarite','Cursed Body','Timid','spa spe','Shadow Ball,Sludge Bomb,Focus Blast,Destiny Bond'),
 ('Slowbro','Slowbronite','Regenerator','Bold','hp def','Surf,Psychic,Slack Off,Calm Mind'),
 ('Blaziken','Focus Sash','Speed Boost','Jolly','atk spe','Flare Blitz,Close Combat,Protect,Swords Dance'),
 ('Primarina','Mystic Water','Torrent','Modest','hp spa','Moonblast,Surf,Energy Ball,Psychic'),
 ('Hippowdon','Rocky Helmet','Sand Stream','Impish','hp def','Earthquake,Slack Off,Yawn,Stealth Rock'),
 ('Rotom-Wash','Sitrus Berry','Levitate','Bold','hp def','Hydro Pump,Volt Switch,Will-O-Wisp,Pain Split'),
 ('Corviknight','Leftovers','Mirror Armor','Impish','hp def','Body Press,Iron Defense,Roost,Brave Bird'),
 ('Meowscarada','Focus Sash','Protean','Jolly','atk spe','Flower Trick,Knock Off,U-turn,Play Rough'),
]

ARCHETYPES={
 'rain':['Pelipper','Basculegion','Archaludon','Scizor','Garchomp','Gholdengo'],
 'sun':['Torkoal','Venusaur','Blaziken','Salamence','Sylveon','Gholdengo'],
 'sand':['Tyranitar','Excadrill','Hippowdon','Gengar','Scizor','Garchomp'],
 'snow':['Ninetales-Alola','Baxcalibur','Lucario','Gholdengo','Rotom-Wash','Volcarona'],
 'balance':['Salamence','Primarina','Gholdengo','Hippowdon','Meowscarada','Corviknight'],
 'offense':['Dragonite','Gholdengo','Lucario','Blaziken','Garchomp','Primarina'],
}


def complete_sets():
    k=knowledge();result=[]
    for species,item,ability,nature,stats,moves in TEMPLATES:
        points=dict.fromkeys(STATS,0)
        for stat in stats.split():points[stat]=32
        points[next(s for s in ('hp','spd','spe','def') if not points[s])]=2
        base={'name':species,'species':species,'item':item,'ability':ability,'nature':nature,
              'level':50,'evs':points,'moves':moves.split(',')}
        for variant in range(2):
            s=copy.deepcopy(base)
            if variant:
                s['nature']={'Jolly':'Adamant','Adamant':'Jolly','Timid':'Modest','Modest':'Timid',
                             'Bold':'Calm','Impish':'Careful'}[nature]
                if nature in ('Bold','Impish'):
                    s['evs']['def'],s['evs']['spd']=s['evs']['spd'],s['evs']['def']
            result.append({'set':s,'variant':variant,'curated':True,
                           'usage':k.usage(species).get('usage',0),
                           'source':'Authored complete strategy template; simulator validated, not a tournament provenance claim'})
    return result


def repair_synthetic(s):
    """Remove obvious item/move/spread contradictions in broad-coverage sets."""
    k=knowledge();s=copy.deepcopy(s)
    info=k.species_info(s['species'])
    moves=[canonical(m) for m in s['moves']]
    item=canonical(s['item'])
    def score(mid):
        m=k.moves[mid]
        charge=any(h in m.get('conditionalHandlers',[]) for h in ('onTryMove','onTry')) and mid in ('hyperbeam','gigaimpact','skyattack','solarbeam','solarblade')
        return m['basePower']*(1.5 if m['type'] in info['types'] else 1)*m['accuracy']/100-(100 if charge else 0)
    attacks=sorted([m for m in info['learnset'] if k.moves[m]['category']!='Status'],key=score,reverse=True)
    if item in ('choiceband','choicespecs','choicescarf','assaultvest'):
        moves=[m for m in moves if k.moves[m]['category']!='Status' or (m=='trick' and item!='assaultvest')]
        categories={k.moves[m]['category'] for m in moves if k.moves[m]['category']!='Status'}
        preferred=('Physical' if item=='choiceband' else 'Special' if item=='choicespecs'
                   else next(iter(categories)) if len(categories)==1 else None)
        if preferred:moves=[m for m in moves if k.moves[m]['category'] in (preferred,'Status')]
        candidates=[m for m in attacks if not preferred or k.moves[m]['category']==preferred]
        moves+=(m for m in candidates if m not in moves)
        moves=moves[:4]
    physical=[m for m in moves if k.moves[m]['category']=='Physical' and m not in ('bodypress','foulplay')]
    special=[m for m in moves if k.moves[m]['category']=='Special']
    if physical and not special:
        old=s['evs'].get('spa',0);s['evs']['spa']=0
        add=min(old,32-s['evs'].get('atk',0));s['evs']['atk']=s['evs'].get('atk',0)+add
        s['evs']['hp']=min(32,s['evs'].get('hp',0)+old-add)
        if k.natures[s['nature']].get('minus')=='atk':s['nature']='Jolly'
    elif special and not physical:
        old=s['evs'].get('atk',0);s['evs']['atk']=0
        add=min(old,32-s['evs'].get('spa',0));s['evs']['spa']=s['evs'].get('spa',0)+add
        s['evs']['hp']=min(32,s['evs'].get('hp',0)+old-add)
        if k.natures[s['nature']].get('minus')=='spa':s['nature']='Timid'
    s['moves']=[k.moves[m]['name'] for m in moves]
    return s
