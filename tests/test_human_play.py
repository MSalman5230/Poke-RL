import subprocess

from champions_rl.engine import ROOT


def test_local_observer_matches_training_view_and_hides_enemy_sets():
    script = r"""
const assert=require('node:assert/strict');
const {Game}=require('./simulator/worker.cjs');
const {playerObservation}=require('./simulator/showdown_observer.cjs');
const team=JSON.parse(require('fs').readFileSync('./data/sets.json')).reference_team;
// Make the simulator's default IVs explicit on both paths.
for(const mon of team)mon.ivs={hp:31,atk:31,def:31,spa:31,spd:31,spe:31};
const hidden=structuredClone(team);
hidden[2].item='Lum Berry';hidden[0].moves[3]='Thunderbolt';
const a=new Game([team,team],[1,2,3,4]);
const b=new Game([team,hidden],[1,2,3,4]);
assert.deepEqual(playerObservation(a.battle,0),playerObservation(b.battle,0));
a.step([17,14]);b.step([17,14]);
assert.deepEqual(playerObservation(a.battle,0),playerObservation(b.battle,0));
b.battle.destroy();
let decisions=0;
while(!a.battle.ended&&!a.truncated){
  for(let side=0;side<2;side++){
    const obs=playerObservation(a.battle,side);
    assert.deepEqual(obs,a.view(side));
    for(const mon of obs.opponent){
      for(const key of ['set','stats','selected','current_hp','max_hp'])assert(!(key in mon));
    }
  }
  a.step([0,1].map(side=>a.legal(side)[0]?.id??null));decisions++;
}
assert(a.battle.ended&&decisions>3);
a.battle.destroy();
"""
    result = subprocess.run(['node', '-e', script], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr[:5000]
