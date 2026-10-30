import subprocess
import shutil

import pytest

from champions_rl.engine import ROOT


def test_waiting_challenges_are_accepted_after_a_busy_game():
    result = subprocess.run(['node', 'tests/showdown_challenges.cjs'], cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr[:5000]


def test_quick_build_has_local_battle_renderer_and_text_before_panel_load(tmp_path):
    client = ROOT / 'runtime_data/showdown_client/play.pokemonshowdown.com'
    if not (client / 'js/battle-animations.js').exists():
        pytest.skip('Optional Showdown browser client is not installed')
    (tmp_path / 'js').mkdir()
    for name in ('battle-animations.js', 'battle-animations-moves.js', 'battle-text-parser.js'):
        shutil.copyfile(client / 'js' / name, tmp_path / 'js' / name)
    shutil.copyfile(client / 'testclient-new.html', tmp_path / 'testclient-new.html')
    script = r"""
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('node:assert/strict');
const {prepareAssets,localHTML}=require('./simulator/showdown_http.cjs');
const root=process.argv[1];
prepareAssets(root);
const html=localHTML(root,8765);
const context={Dex:{resourcePrefix:'',loadTextData:()=>{},text:{getLanguage:()=> 'en'}},
  $:{easing:{},extend:Object.assign},toID:x=>String(x).toLowerCase().replace(/[^a-z0-9]/g,'')};
context.window=context;context.exports=context;
vm.createContext(context);
// Execute the actual pinned quick-build artifacts in page order, without a
// remote fallback. A direct battle URL must have its renderer and text ready.
for(const match of html.matchAll(/<script[^>]+src="([^"]+)"/g)){
  const source=match[1];
  if(['data/graphics.js','data/text/en.js','js/battle-text-parser.js'].includes(source))
    vm.runInContext(fs.readFileSync(path.join(root,source),'utf8'),context);
  if(source==='js/panel-battle.js'){
    assert.equal(typeof context.BattleScene,'function');
    assert.equal(typeof context.BattleMoveAnims,'object');
    assert.match(new context.BattleTextParser().parseArgs(['start'],{}),/Battle started between Player 1 and Player 2/);
  }
}
"""
    result = subprocess.run(['node', '-e', script, str(tmp_path)], cwd=ROOT,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr[:5000]


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
