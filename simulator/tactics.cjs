'use strict';
// This API accepts hypothetical sets and public conditions only. No live game
// identifier is accepted: it cannot look up the actual hidden opponent state.
const {Battle,Dex}=require('pokemon-showdown');
const id=Dex.toID;
const copy=x=>JSON.parse(JSON.stringify(x));
function build(q){
  const b=new Battle({formatid:'gen9championsbssregmc',seed:[11,22,33,44],send:()=>{}});
  for(let i=0;i<2;i++){
    const m=i?q.target:q.actor;
    const team=Array.from({length:6},(_,j)=>({...copy(m.set),name:'probe'+i+j,level:50}));
    b.setPlayer('p'+(i+1),{name:'probe'+i,team});
  }
  b.makeChoices('team 1,2,3','team 1,2,3');
  for(let i=0;i<2;i++){
    const input=i?q.target:q.actor,p=b.sides[i].active[0];
    if(input.species&&input.species!==p.species.name)p.formeChange(input.species);
    p.hp=Math.max(1,Math.round(input.hp*p.maxhp));
    p.status=input.status||'';p.statusState={id:p.status,target:p};
    p.boosts={atk:0,def:0,spa:0,spd:0,spe:0,accuracy:0,evasion:0,...input.boosts};
    p.item=id(input.item??input.set.item);p.ability=id(input.ability??input.set.ability);
    if(input.stats)Object.assign(p.storedStats,input.stats);
    if(input.types)p.setType(input.types);
    b.sides[i].totalFainted=q.faints?.[i]||0;
    for(const [effect,data] of Object.entries(q.conditions?.[i]||{})){
      b.sides[i].addSideCondition(effect,p);
      const state=b.sides[i].sideConditions[effect];if(state&&data.layers)state.layers=data.layers;
    }
  }
  b.field.clearWeather();b.field.clearTerrain();
  if(q.weather)b.field.setWeather(q.weather,b.p1.active[0]);
  if(q.terrain)b.field.setTerrain(q.terrain,b.p1.active[0]);
  for(const effect of Object.keys(q.pseudo||{}))b.field.addPseudoWeather(effect,b.p1.active[0]);
  if(q.mega)b.actions.runMegaEvo(b.p1.active[0]);
  if(q.target_mega)b.actions.runMegaEvo(b.p2.active[0]);
  return b;
}
function calculate(q){
  const b=build(q),a=b.p1.active[0],d=b.p2.active[0];
  try{
    let move=b.dex.getActiveMove(q.move);
    const speed=[a.getStat('spe'),d.getStat('spe')];
    const priority=b.runEvent('ModifyPriority',a,d,move,move.priority);
    if(move.category==='Status'||move.id==='recharge'||!move.exists)return {min:0,max:0,mean:0,ko:0,speed,priority,method:'native-conditional-damage'};
    b.setActiveMove(move,a,d);
    b.singleEvent('ModifyMove',move,null,a,d,move,move);
    move=b.runEvent('ModifyMove',a,d,move,move)||move;
    // Damage calls do not themselves run absorption/other TryHit immunities.
    const hit=b.runEvent('TryHit',d,a,move);
    if(hit===false||hit===null)return {min:0,max:0,mean:0,ko:0,speed,priority,method:'native-conditional-damage'};
    let hitCounts=[[1,1]];
    if(typeof move.multihit==='number')hitCounts=[[move.multihit,1]];
    else if(Array.isArray(move.multihit))hitCounts=a.hasAbility('skilllink')?[[move.multihit[1],1]]:
      a.hasItem('loadeddice')?[[4,.5],[5,.5]]:[[2,.35],[3,.35],[4,.15],[5,.15]];
    const initial=d.hp,rows=[];
    for(const [hits,weight] of hitCounts)for(let roll=85;roll<=100;roll++){
      d.hp=initial;let damage=0;
      // Standard single-hit rolls are enumerated. Multi-hit damage uses a
      // common roll per hit: its extrema are useful, probability is approximate.
      b.randomizer=value=>Math.floor(value*roll/100);
      for(let n=1;n<=hits;n++){
        const m=b.dex.getActiveMove(move);m.hit=n;
        if(m.willCrit===undefined)m.willCrit=false;
        const value=Number(b.actions.getDamage(a,d,m,true))||0;
        const survive=hits===1&&initial===d.maxhp&&(d.hasItem('focussash')||d.hasAbility('sturdy'));
        damage+=survive?Math.min(value,initial-1):value;
        d.hp=Math.max(1,initial-damage);
      }
      rows.push({damage,weight:weight/16});
    }
    let accuracy=move.accuracy;
    if(accuracy!==true){
      const stage=Math.max(-6,Math.min(6,(move.ignoreAccuracy?0:a.boosts.accuracy)-(move.ignoreEvasion?0:d.boosts.evasion)));
      accuracy*=stage>=0?(3+stage)/3:3/(3-stage);
      accuracy=b.runEvent('ModifyAccuracy',d,a,move,accuracy);
      accuracy=b.runEvent('Accuracy',d,a,move,accuracy);
    }
    accuracy=accuracy===true?1:Math.max(0,Math.min(1,accuracy/100));
    return {min:Math.min(...rows.map(r=>r.damage))/d.maxhp,max:Math.max(...rows.map(r=>r.damage))/d.maxhp,
      mean:rows.reduce((s,r)=>s+r.damage*r.weight,0)/d.maxhp,
      ko:accuracy*rows.reduce((s,r)=>s+(r.damage>=initial?r.weight:0),0),speed,priority,
      method:move.multihit?'native-multihit-correlated-rolls':'native-conditional-damage'};
  }finally{b.destroy();}
}
module.exports={calculate,build};
