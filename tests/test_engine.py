import copy
import json
import random
import subprocess
import numpy as np
import pytest
from champions_rl.arena import play_match,load_teams
from champions_rl.engine import ROOT,Engine
from champions_rl.features import Encoder
from champions_rl.policies import ScriptedBot
from champions_rl.teams import TeamPool


def test_simulator_exit_preserves_failed_query_without_silent_restart(tmp_path):
    with Engine(failure_directory=tmp_path) as engine:
        engine.request('ping')
        process=engine.process
        process.kill();process.wait()
        with pytest.raises(RuntimeError,match='diagnostics='):
            engine.request('ping')
        assert engine.process is process and process.poll() is not None
    files=list(tmp_path.glob('*.json'))
    assert len(files)==1
    details=json.loads(files[0].read_text())
    assert details['operation']=='ping' and details['process_exit_code'] is not None


def test_preview_bring_six_pick_three_and_legal_actions():
    team=TeamPool().data["reference_team"]
    encoder=Encoder()
    with Engine() as e:
        r=e.request("new",game="test",teams=[team,team],seed=[1,2,3,4])
        assert [len(v["legal"]) for v in r["views"]]==[120,120]
        assert encoder.encode(r["views"][0])["mask"].sum()==120
        v=r["views"][0]
        matrix=np.array([[np.clip(encoder.best_damage(a,b)-encoder.best_damage(b,a),-2,2) for b in v["opponent"]] for a in v["own"]])
        encoded=encoder.encode(v)
        for action in v["legal"]:
            assert np.array_equal(encoded["actions"][action["id"]],np.asarray(encoder.action_features(v,action,matrix),dtype=np.float32))
        r=e.request("step",game="test",actions=[14,14])
        for view in r["views"]:
            assert sum(p["selected"] for p in view["own"])==3
            assert len(view["opponent"])==6
            assert sum(p["seen"] for p in view["opponent"])==1
            assert not any("tera" in a["choice"] for a in view["legal"])
            x=encoder.encode(view)
            assert all(np.isfinite(value).all() for value in x.values())


def test_hidden_sets_and_future_actions_do_not_enter_observation():
    team=TeamPool().data["reference_team"]
    alternative=copy.deepcopy(team)
    alternative[0]["item"]="Lum Berry"
    team=copy.deepcopy(team)
    team[0]["item"]="Sitrus Berry"
    # Equal visible species/ability; different unseen item and legal moves.
    alternative[0]["moves"][3]="Thunderbolt"
    with Engine() as e:
        a=e.request("new",game="a",teams=[team,team],seed=[1,2,3,4])
        b=e.request("new",game="b",teams=[team,alternative],seed=[9,8,7,6])
        assert a["views"][0]==b["views"][0]
        a=e.request("step",game="a",actions=[14,14])
        b=e.request("step",game="b",actions=[14,14])
        assert a["views"][0]==b["views"][0]
        enemy=a["views"][0]["opponent"][0]
        assert enemy["item"] is None
        assert enemy["moves"]==[]
        assert "set" not in enemy and "stats" not in enemy


def test_recharge_and_regional_form_slots_complete_deterministically():
    teams=[TeamPool().data["reference_team"]]*2
    with Engine() as e:
        a=play_match(e,teams,[ScriptedBot(),ScriptedBot("setup")],123)
        b=play_match(e,teams,[ScriptedBot(),ScriptedBot("setup")],123)
        assert not a["truncated"]
        assert [x for x in a["trace"] if not x.startswith('|t:|')]==[x for x in b["trace"] if not x.startswith('|t:|')]


def test_reviving_mask_and_bench_health_updates():
    script=r"""
const {Game,PublicState}=require('./simulator/worker.cjs');
const fs=require('fs');
const team=JSON.parse(fs.readFileSync('./data/sets.json')).reference_team;
const g=new Game([team,team],[1,2,3,4]);
g.step([14,14]);
const side=g.battle.sides[0], req=side.activeRequest;
side.activeRequest={forceSwitch:[true],side:req.side};
side.choice.forcedSwitchesLeft=1;
side.activeRequest.side.pokemon[0].reviving=true;
side.activeRequest.side.pokemon[1].condition='0 fnt';
const choices=g.legal(0);
if(choices.length!==1||!choices[0].revive||choices[0].party_slot!==1)throw Error('Bad revival mask');
const s=new PublicState(0);
s.feed(['|poke|p1|Pawmot, L50','|poke|p1|Gholdengo, L50','|switch|p1a: unit0|Pawmot, L50|100/100',
'|switch|p1a: unit1|Gholdengo, L50|100/100','|faint|p1a: unit1',
'|switch|p1a: unit0|Pawmot, L50|100/100','|-heal|p1: unit1|50/100']);
if(s.teams[0][1].fainted||s.teams[0][1].hp!==.5||s.teams[0][0].hp!==1)throw Error('Bad bench revival state');
g.battle.destroy();
"""
    subprocess.run(["node","-e",script],cwd=ROOT,check=True,capture_output=True,text=True)


def test_varied_legal_games_finish_without_silent_action_fallbacks():
    teams=load_teams("validation")
    rng=random.Random(971)
    encoder=Encoder()
    with Engine() as e:
        for seed in range(24):
            bots=[ScriptedBot(style,seed,encoder) for style in ("random","stochastic")]
            result=play_match(e,[rng.choice(teams),rng.choice(teams)],bots,seed,max_turns=100)
            assert result["winner"] in (0,1,None)


def test_hidden_trap_retry_does_not_reveal_committed_enemy_action():
    script=r"""
const {Game}=require('./simulator/worker.cjs');
const team=JSON.parse(require('fs').readFileSync('./data/sets.json')).reference_team;
const g=new Game([team,team],[1,2,3,4]);g.step([14,14]);
const side=g.battle.sides[0];side.active[0].trapped='hidden';
delete side.activeRequest.active[0].trapped;side.activeRequest.active[0].maybeTrapped=true;
const sw=g.legal(0).find(a=>a.kind==='switch');const move=g.legal(1).find(a=>a.kind==='move');
const r=g.step([sw.id,move.id]);
if(r.views[0].legal.some(a=>a.kind==='switch'))throw Error('Trap was not learned');
if(r.views[1].legal.length||r.views[1].phase!=='wait')throw Error('Committed opponent asked to choose again');
if(JSON.stringify(r.views[0]).includes('choice.actions'))throw Error('Choice leaked');
g.step([r.views[0].legal[0].id,null]);g.battle.destroy();
"""
    subprocess.run(["node","-e",script],cwd=ROOT,check=True,capture_output=True,text=True)


def test_illusion_reveal_restores_apparent_species_and_expires_single_turn_effects():
    script=r"""
const {PublicState}=require('./simulator/worker.cjs');
const s=new PublicState(0);
s.feed(['|poke|p2|Zoroark, L50','|poke|p2|Gholdengo, L50',
 '|switch|p2a: unit1|Gholdengo, L50|100/100','|turn|1',
 '|move|p2a: unit1|Nasty Plot|p2a: unit1','|-boost|p2a: unit1|spa|2',
 '|replace|p2a: unit0|Zoroark, L50|75/100',
 '|-singleturn|p2a: unit0|Protect','|turn|2']);
if(s.teams[1][1].seen||s.teams[1][1].moves.length)throw Error('Illusion falsely revealed a teammate');
const z=s.teams[1][0];
if(!z.active||z.hp!==.75||z.boosts.spa!==2||z.moves[0]!=='nastyplot')throw Error('Reveal lost public facts');
if(z.volatiles.protect)throw Error('One-turn volatile did not expire');
"""
    subprocess.run(["node","-e",script],cwd=ROOT,check=True,capture_output=True,text=True)
