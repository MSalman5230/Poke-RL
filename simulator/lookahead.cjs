'use strict';
// Public-belief one-turn search. No live game IDs or engine state are accepted.
const {PRNG}=require('pokemon-showdown');
const {hydrate,serialize,clone}=require('./hypothetical.cjs');
const copy=x=>JSON.parse(JSON.stringify(x));

function wrap(b,obs,teams,Game,PublicState){
  const g=Object.create(Game.prototype);
  g.teams=copy(teams);g.maxTurns=200;g.truncated=false;g.training=false;
  g.battle=b;g.trace=[];g.logPos=b.log.length;
  g.public=[0,1].map(viewer=>{
    const pub=new PublicState(viewer);
    pub.turn=obs.turn;pub.teams=copy([obs.own,obs.opponent]);
    pub.teams.forEach((team,side)=>team.forEach((m,j)=>{
      m.nickname='unit'+j;m.moves=(m.moves||[]).map(x=>typeof x==='string'?x:x.id);
    }));
    pub.active=pub.teams.map(team=>team.findIndex(m=>m.active));
    pub.conditions=copy(obs.conditions);pub.weather=obs.weather;pub.terrain=obs.terrain;
    pub.weather_since=obs.weather_since;pub.terrain_since=obs.terrain_since;
    pub.pseudo=copy(obs.pseudo);pub.mega=copy(obs.mega);pub.history=copy(obs.history);
    pub.faints=copy(obs.faints);
    return pub;
  });
  return g;
}

function weighted(choices,probs,rng){
  const total=probs.reduce((a,b)=>a+b,0);
  let value=rng.random()*total;
  for(let i=0;i<choices.length;i++){value-=probs[i];if(value<=0)return choices[i];}
  return choices[choices.length-1];
}

function lookahead(q,Game,PublicState){
  if(q.observation.opponent.some(m=>m.set||m.stats||m.selected!==undefined))throw Error('Search requires a public opponent view');
  const rows=[],errors=[];
  for(let worldIndex=0;worldIndex<q.worlds.length;worldIndex++){
    const world=q.worlds[worldIndex];let base;
    try{
      const prepared=hydrate(world.input);base=prepared.battle;
      base.sides.forEach((side,i)=>{side.name='player'+i;});
      const root=serialize(base),probe=wrap(base,q.observation,world.teams,Game,PublicState);
      const replies=probe.legal(1);
      if(!replies.length)throw Error('Opponent has no simultaneous reply');
      const labels=replies.map(a=>a.kind==='move'?'move:'+a.move:'switch:'+a.slot);
      const multiplicity=Object.fromEntries(labels.map(label=>[label,labels.filter(x=>x===label).length]));
      const probabilities=labels.map(label=>Math.max(.0001,q.reply_probabilities[label]||0)/multiplicity[label]);
      const rng=new PRNG(world.seed);
      const sampled=Array.from({length:q.replies},()=>weighted(replies,probabilities,rng));
      const candidates=q.candidates.map(a=>probe.legal(0).find(x=>x.id===a));
      if(candidates.some(a=>!a))throw Error('Public and hydrated legal choices disagree');
      for(let replyIndex=0;replyIndex<sampled.length;replyIndex++)for(const action of candidates){
        const b=clone(root,world.rollout_seeds[replyIndex]);
        const g=wrap(b,q.observation,world.teams,Game,PublicState);
        try{
          let state=g.step([action.id,sampled[replyIndex].id]);
          let continuations=0;
          while(!state.ended&&b.turn<=q.observation.turn){
            if(++continuations>8)throw Error('Could not resolve one full turn');
            // Complete pivot/faint replacements with the healthiest legal bench.
            // This explicit prototype continuation is shared across candidates.
            const choices=[0,1].map(side=>{
              const legal=g.legal(side);if(!legal.length)return null;
              if(legal.some(a=>a.kind!=='switch'))throw Error('Unsupported interrupted move request');
              return legal.sort((a,z)=>b.sides[side].pokemon[z.party_slot].hp/b.sides[side].pokemon[z.party_slot].maxhp-
                b.sides[side].pokemon[a.party_slot].hp/b.sides[side].pokemon[a.party_slot].maxhp)[0].id;
            });
            state=g.step(choices);
          }
          rows.push({action:action.id,world:worldIndex,reply:replyIndex,observation:state.views[0],
            ended:state.ended,winner:state.winner,continuations,assumptions:prepared.assumptions});
        }finally{b.destroy();}
      }
    }catch(error){errors.push({world:worldIndex,error:String(error.message)});}
    finally{if(base)base.destroy();}
  }
  return {rows,errors};
}
module.exports={lookahead};
