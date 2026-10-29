'use strict';
// Fully specified curriculum fixtures, never used to alter a user's live game.
const fs=require('fs');
const {Game}=require('./worker.cjs');
const copy=x=>JSON.parse(JSON.stringify(x));
function position(fixture,seed){
  const g=new Game(fixture.teams,seed);
  g.step([14,14]);
  // The other selected members have already appeared and fainted in this toy
  // endgame. Publish that information before emitting any training observation.
  for(const pub of g.public)pub.teams.forEach((team,side)=>team.forEach((m,i)=>{
    const p=g.battle.sides[side].pokemon.find(p=>p.name==='unit'+i);
    if(p){m.nickname=p.name;m.seen=true;m.moves=p.moveSlots.map(x=>x.id);m.item=p.item;m.ability=p.ability;}
  }));
  g.battle.sides.forEach((side,i)=>{
    for(const p of side.pokemon)if(!side.active.includes(p))p.faint();
    const active=side.active[0],hp=fixture.hp[i];
    if(active.hp>hp)g.battle.directDamage(active.hp-hp,active);
  });
  g.battle.resolveFaints();g.battle.makeRequest('move');g.sync();
  return g;
}
function verify(fixture){
  const probe=position(fixture,[5,17,31,47]);
  const observation=probe.view(0),own=probe.legal(0).map(a=>a.id),foe=probe.legal(1).map(a=>a.id);
  probe.battle.destroy();
  const results={};
  for(const action of own){const trials=[];
    for(const reply of foe)for(let seed=0;seed<16;seed++){
      const g=position(fixture,[seed+1,13,317,911]);
      try{const state=g.step([action,reply]);trials.push({reply,seed,ended:state.ended,winner:state.winner});}
      finally{g.battle.destroy();}
    }
    results[action]={trials:trials.length,terminal_wins:trials.filter(t=>t.ended&&t.winner===0).length,
      terminal_losses:trials.filter(t=>t.ended&&t.winner===1).length,unresolved:trials.filter(t=>!t.ended).length};
  }
  return {name:fixture.name,observation,winning_actions:own.filter(a=>results[a].terminal_wins===results[a].trials),results,
    limits:'Immediate terminal wins against every legal enemy reply across 16 simulator seeds. Controlled complete-information fixture; not a proof for arbitrary hidden battles.'};
}
if(require.main===module){try{process.stdout.write(JSON.stringify(JSON.parse(fs.readFileSync(0,'utf8')).map(verify)));}
  catch(e){process.stderr.write(String(e.stack));process.exitCode=1;}}
module.exports={verify};
