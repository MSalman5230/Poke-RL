'use strict';
// This API accepts hypothetical sets and public conditions only. No live game
// identifier is accepted: it cannot look up the actual hidden opponent state.
const {Battle,Dex}=require('pokemon-showdown');
const id=Dex.toID;
const copy=x=>JSON.parse(JSON.stringify(x));
const dex=Dex.mod('champions');
const restoredVolatiles=new Set(['substitute','flashfire','charge','unburden','magnetrise','telekinesis','roost','smackdown','gastroacid','minimize','tarshot','focusenergy']);
const irrelevantVolatiles=new Set(['confusion','taunt','encore','disable','leechseed','yawn','partiallytrapped','trapped','perishsong','aquaring','ingrain','protect','mustrecharge','choicelock']);
const conditionalMoves=new Set(['counter','mirrorcoat','metalburst','comeuppance','beatup','futuresight','doomdesire','assurance','revenge','avalanche','payback','lashout','retaliate','lastresort','endeavor','finalgambit','bide','round','fusionbolt','fusionflare']);
function metadata(){
  // Conservative HP dependencies from the pinned simulator's actual callbacks.
  const hooks=/^on(?:Source|Any|Ally|Foe)?(?:BasePower|Modify(?:Atk|Def|SpA|SpD|Spe|Move|Priority|Damage|Accuracy|STAB)|Accuracy|TryHit|Effectiveness|CriticalHit|Damage)$/;
  const sensitive=(e,all=false)=>Object.entries(e).some(([k,v])=>(all||hooks.test(k))&&typeof v==='function'&&/\.(?:hp|maxhp|baseMaxhp)\b|getUndynamaxedHP\(/.test(v.toString()));
  return {version:6,moves:dex.moves.all().filter(m=>sensitive(m,true)).map(m=>m.id),
    abilities:dex.abilities.all().filter(a=>a.id!=='sturdy'&&sensitive(a)).map(a=>a.id),
    items:dex.items.all().filter(a=>a.id!=='focussash'&&sensitive(a)).map(a=>a.id),
    set_dependent_abilities:dex.abilities.all().filter(a=>Object.values(a).some(v=>typeof v==='function'&&/formeChange|formChange|transformInto/.test(v.toString()))).map(a=>a.id)};
}
function build(q){
  const b=new Battle({formatid:'gen9championsbssregmc',seed:[11,22,33,44],send:()=>{}});
  for(let i=0;i<2;i++){
    const m=i?q.target:q.actor;
    // Damage needs the two active combatants. A one-member hypothetical side
    // avoids constructing 10 unused Pokemon; party-dependent moves are flagged.
    const team=[{...copy(m.set),name:'probe'+i,level:50}];
    b.setPlayer('p'+(i+1),{name:'probe'+i,team});
  }
  b.makeChoices('team 1','team 1');
  for(let i=0;i<2;i++){
    const input=i?q.target:q.actor,p=b.sides[i].active[0];
    if(input.species&&input.species!==p.species.name)p.formeChange(input.species);
    if(input.max_hp)p.baseMaxhp=p.maxhp=input.max_hp;
    p.hp=Math.max(1,Math.round((input.hp??1)*p.maxhp));
    p.status=input.status||'';p.statusState={id:p.status,target:p};
    p.boosts={atk:0,def:0,spa:0,spd:0,spe:0,accuracy:0,evasion:0,...input.boosts};
    p.item=id(input.item??input.set.item);p.ability=id(input.ability??input.set.ability);
    if(input.stats)Object.assign(p.storedStats,input.stats);
    if(input.types)p.setType(input.types);
    p.timesAttacked=input.times_attacked||0;
    p.volatiles={};
    for(const effect of Object.keys(input.volatiles||{})){
      if(!restoredVolatiles.has(effect))continue;
      p.addVolatile(effect,p);
      if(effect==='substitute'&&input.volatiles[effect].hp!==undefined)p.volatiles[effect].hp=input.volatiles[effect].hp;
    }
    b.sides[i].totalFainted=q.faints?.[i]||0;
    p.abilityState={id:p.ability,target:p,...(p.ability==='supremeoverlord'?{fallen:Math.min(5,b.sides[i].totalFainted)}:{})};
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
function snapshot(p){
  // These are the fields touched by damage modifiers, eating a resist berry,
  // and the explicitly supported form/volatile changes below. No full move or
  // residual phase is run here. Differential tests cover prepared-state reuse.
  const fields=['hp','maxhp','baseMaxhp','boosts','storedStats','baseStoredStats','types','addedType','apparentType',
    'species','baseSpecies','speciesState','details','gender','weightkg','weighthg','item','lastItem','itemState',
    'ability','abilityState','status','statusState','ateBerry','usedItemThisTurn','volatiles','illusion','transformed'];
  const state={};
  for(const key of fields){
    const value=p[key];
    state[key]=Array.isArray(value)?value.slice():value&&Object.getPrototypeOf(value)===Object.prototype?{...value}:value;
  }
  state.volatiles=Object.fromEntries(Object.entries(p.volatiles).map(([k,v])=>[k,{...v}]));
  return ()=>{
    for(const [key,value] of Object.entries(state))p[key]=Array.isArray(value)?value.slice():value&&Object.getPrototypeOf(value)===Object.prototype?{...value}:value;
    p.volatiles=Object.fromEntries(Object.entries(state.volatiles).map(([k,v])=>[k,{...v}]));
  };
}
function calculate(q,prepared){
  const b=prepared||build(q),a=b.p1.active[0],d=b.p2.active[0];
  const restoreA=snapshot(a),restoreD=snapshot(d),logLength=b.log.length;
  const restore=()=>{restoreA();restoreD();b.log.length=logLength;};
  try{
    let move=b.dex.getActiveMove(q.move);
    const speed=[a.getStat('spe'),d.getStat('spe')];
    b.setActiveMove(move,a,d);
    const priority=b.runEvent('ModifyPriority',a,d,move,move.priority);
    const unsupported=[];
    if(move.ohko||conditionalMoves.has(move.id))unsupported.push('conditional_move_execution_or_missing_party_history');
    if(move.onPrepareHit||move.onTryMove||move.onTry)unsupported.push('move_preparation_not_simulated');
    if(a.getAbility().onPrepareHit||a.getAbility().onBeforeMove)unsupported.push('attacker_preparation_ability_not_simulated');
    if(d.hasAbility('rivalry')||a.hasAbility('rivalry'))unsupported.push('unobserved_gender');
    for(const m of [q.actor,q.target])for(const effect of Object.keys(m.volatiles||{})){
      if(!restoredVolatiles.has(effect)&&!irrelevantVolatiles.has(effect)&&!effect.startsWith('fallen'))unsupported.push('unrestored_volatile:'+effect);
    }
    if(move.multihit)unsupported.push('correlated_multihit_rolls_and_between_hit_effects');
    if(q.target.volatiles?.substitute?.hp===undefined&&q.target.volatiles?.substitute&&move.multihit)unsupported.push('unknown_substitute_hp');
    const result={rows:[[0,1]],maxhp:d.maxhp,accuracy:1,speed,priority,survive_full:false,survive_chance:0,
      reliable:unsupported.length===0,unsupported,method:'native-conditional-damage-v6'};
    if(move.category==='Status'||move.id==='recharge'||!move.exists)return result;
    b.singleEvent('ModifyMove',move,null,a,d,move,move);
    move=b.runEvent('ModifyMove',a,d,move,move)||move;
    // Damage calls do not themselves run absorption/other TryHit immunities.
    const hit=b.runEvent('TryHit',d,a,move);
    if(hit===false||hit===null||!d.runImmunity(move.type,false))return result;
    let accuracy=move.accuracy;
    if(accuracy!==true){
      const stage=Math.max(-6,Math.min(6,(move.ignoreAccuracy?0:a.boosts.accuracy)-(move.ignoreEvasion?0:d.boosts.evasion)));
      accuracy*=stage>=0?(3+stage)/3:3/(3-stage);
      accuracy=b.runEvent('ModifyAccuracy',d,a,move,accuracy);
      accuracy=b.runEvent('Accuracy',d,a,move,accuracy);
    }
    result.accuracy=accuracy===true?1:Math.max(0,Math.min(1,accuracy/100));
    let hitCounts=[[1,1]];
    if(typeof move.multihit==='number')hitCounts=[[move.multihit,1]];
    else if(Array.isArray(move.multihit))hitCounts=a.hasAbility('skilllink')?[[move.multihit[1],1]]:
      a.hasItem('loadeddice')?[[4,.5],[5,.5]]:[[2,.35],[3,.35],[4,.15],[5,.15]];
    const defenderAbility=name=>d.hasAbility(name)&&!(b.suppressingAbility(d)&&d.getAbility().flags.breakable);
    result.survive_full=!move.multihit&&(d.hasItem('focussash')||defenderAbility('sturdy'));
    result.survive_chance=!move.multihit&&d.hasItem('focusband')&&!(defenderAbility('disguise')&&['mimikyu','mimikyutotem'].includes(d.species.id)) ? 0.1 : 0;
    // The common single-hit calculation only reads the combatants. Expensive
    // restore/cloning is necessary for consuming berries and supported form or
    // substitute changes. Multi-hit estimates always take the conservative path.
    const mutable=move.multihit||d.volatiles.substitute||defenderAbility('disguise')||defenderAbility('iceface')||
      a.getItem().isBerry||d.getItem().isBerry;
    // Preserve preparation such as Aegislash's Stance Change for every roll.
    // Restoring the pre-ModifyMove state would silently return to Shield form.
    const restorePreparedA=mutable?snapshot(a):null,restorePreparedD=mutable?snapshot(d):null;
    const actorHP=a.hp;
    const initial=d.hp,rows=[];
    for(const [hits,weight] of hitCounts)for(let roll=85;roll<=100;roll++){
      if(mutable){restorePreparedA();restorePreparedD();b.log.length=logLength;}else {a.hp=actorHP;d.hp=initial;}
      b.setActiveMove(move,a,d);let damage=0;
      // Standard single-hit rolls are enumerated. Multi-hit damage uses a
      // common roll per hit: its extrema are useful, probability is approximate.
      b.randomizer=value=>Math.floor(value*roll/100);
      for(let n=1;n<=hits;n++){
        const m=b.dex.getActiveMove(move);m.hit=n;
        if(m.willCrit===undefined)m.willCrit=false;
        let value=Number(b.actions.getDamage(a,d,m,true))||0;
        if(m.id==='falseswipe'||m.id==='holdback')value=Math.min(value,d.hp-1);
        const sub=d.volatiles.substitute;
        if(sub&&!m.flags.bypasssub&&!m.infiltrates){
          sub.hp-=value;if(sub.hp<=0)delete d.volatiles.substitute;
          value=0;
        }else if(value&&defenderAbility('disguise')&&['mimikyu','mimikyutotem'].includes(d.species.id)){
          value=Math.floor(d.baseMaxhp/8);d.formeChange(d.species.id==='mimikyu'?'Mimikyu-Busted':'Mimikyu-Busted-Totem');
        }else if(value&&defenderAbility('iceface')&&d.species.id==='eiscue'&&m.category==='Physical'){
          value=0;d.formeChange('Eiscue-Noice');
        }else if(hits>1&&d.hp===d.maxhp&&(d.hasItem('focussash')||defenderAbility('sturdy'))){
          value=Math.min(value,d.hp-1);
        }
        damage+=value;d.hp=Math.max(0,initial-damage);
        if(!d.hp)break;
      }
      rows.push([damage,weight/16]);
    }
    result.rows=rows;
    return result;
  }finally{restore();if(!prepared)b.destroy();}
}
function calculateMany(queries){
  const groups=new Map(),results=new Array(queries.length);
  queries.forEach((q,i)=>{const {move,...state}=q,key=JSON.stringify(state);if(!groups.has(key))groups.set(key,[]);groups.get(key).push([q,i]);});
  for(const group of groups.values()){
    const b=build(group[0][0]);
    try{for(const [q,i] of group)results[i]=calculate(q,b);}finally{b.destroy();}
  }
  return results;
}
module.exports={calculate,calculateMany,build,metadata};
