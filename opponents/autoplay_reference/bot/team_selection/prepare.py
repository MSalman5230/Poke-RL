"""Build all reusable matchup profiles before queueing, and after Box changes."""
import itertools
import os
import time
from opponents.autoplay_reference.bot.domain import catalog
from opponents.autoplay_reference.bot.opponent.data import datasets
from opponents.autoplay_reference.bot.opponent.prediction import BuildEstimator, Opponent
from opponents.autoplay_reference.bot.runtime import budget as budgets
from opponents.autoplay_reference.bot.simulator import bridge
from opponents.autoplay_reference.bot.team_selection import cache, duels


def own_variants(roster):
    c=catalog.get()
    result={}
    for name,member in roster.members.items():
        forms=[member['species']]
        mega=c.stones.get(name,{}).get(member['item'])
        if mega:
            forms.append(mega)
        if member['ability']=='Zero to Hero' and member['species']=='Palafin':
            forms.append('Palafin-Hero')
        result[name]={form:duels.member_input(name,member,form) for form in forms}
    return result


def enemy_variants(name,prediction):
    member=prediction.member
    form=prediction.battle_form or member['species']
    forms=[form]
    if member['ability']=='Zero to Hero' and member['species']=='Palafin':
        forms=[member['species'],'Palafin-Hero']
    return {f:duels.member_input(name,member,f) for f in forms}


def corpus(data):
    c=catalog.get()
    names={c.display('species',key) for rows in data.tables.values() for key in rows}
    names.discard(None)
    estimator=BuildEstimator(data)
    predictions=[]
    incomplete=[]
    for name in sorted(names):
        info=c.species(name)
        if any(p in name for p in ('-Mega','-Hero','-Busted')) or info.get('battleOnly'):
            continue
        for selected in ([False,True] if name in c.stones else [False]):
            prediction=estimator.predict(Opponent(name),mega_selected=selected)
            if prediction.complete:
                predictions.append((name,prediction))
            else:
                incomplete.append({'species':name,'mega':selected,'errors':prediction.errors})
    return predictions,incomplete


def measure(client,tasks,deadline):
    """One mechanics batch; caller decides whether complete work can be used."""
    response=client.request('preview_profiles',timeout=deadline.remaining(),tasks=[{'id':duels.sha(t),**t} for t in tasks])
    expected={duels.sha(t):t for t in tasks}
    rows={row['id']:row for row in response['rows']}
    errors={row['id']:row['error'] for row in response['errors']}
    if len(rows)!=len(response['rows']) or set(rows)&set(errors) or set(rows)|set(errors)!=set(expected):
        raise ValueError('Incomplete or duplicate preview-profile response')
    return rows,errors


def prepare(roster, *, client=None, data=None, progress=None, stop_flag=None, enemies=None):
    started=time.monotonic()
    client=client or bridge.get()
    data=data or datasets()
    if client.identity is None:
        client.verify()
    store=cache.get(roster,data,client)
    if store.prepared and enemies is None:
        if store.score_dirty:
            try:
                store.save()
            except OSError:
                store.prepared=False
                raise
        return {**store.preparation,'reused':True,'seconds':time.monotonic()-started,'cache':str(store.path)}
    own=own_variants(roster)
    predicted,incomplete=corpus(data) if enemies is None else (list(enemies),[])
    tasks={}
    for name,prediction in predicted:
        for a,b in itertools.product((v for forms in own.values() for v in forms.values()),enemy_variants(name,prediction).values()):
            task=duels.task(a,b)
            tasks[duels.sha(task)]=task
            if task.get('copy_sides'):
                opening=duels.task(a,b,opening=True)
                tasks[duels.sha(opening)]=opening
    pending=[t for key,t in tasks.items() if key not in store.pairs]
    failures=[]
    for offset in range(0,len(pending),24):
        if stop_flag and os.path.exists(stop_flag):
            raise InterruptedError('Preview preparation stopped before matchmaking')
        batch=pending[offset:offset+24]
        rows,errors=measure(client,batch,budgets.Deadline(budgets.SIMULATOR_VERIFY_TIMEOUT))
        for task in batch:
            key=duels.sha(task)
            if key in rows:
                store.add(task,rows[key])
            else:
                failures.append({'key':key,'members':task['members'],'error':errors[key]})
        if progress:
            progress({'done':min(offset+24,len(pending)),'total':len(pending),'profiles':len(store.pairs),'failures':len(failures)})
    scoring_failures=store.evaluate()
    store.prepared=enemies is None
    store.preparation={'schema_version':2,'ready':store.prepared,'team_identity':store.team_identity,
        'required_profiles':len(tasks),'profiles':len(store.pairs),'calculated':len(pending)-len(failures),
        'profile_failures':failures,'scoring_failures':scoring_failures,
        'failures':failures+scoring_failures,'incomplete_builds':incomplete,'seconds':time.monotonic()-started,
        'cache':str(store.path),'assumptions':duels.ASSUMPTIONS}
    try:
        store.save()
    except OSError:
        store.prepared=False
        raise
    return {**store.preparation,'reused':not pending,'seconds':time.monotonic()-started}
