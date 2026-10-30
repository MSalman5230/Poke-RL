'use strict';
// Local-server-only private player channel. Reuse the exact training view;
// never reconstruct the enemy from its actual hidden set.
const path=require('path');
const {Game,PublicState}=require('./worker.cjs');
const clone=x=>JSON.parse(JSON.stringify(x));
const views=new WeakMap();
function playerObservation(battle,side) {
  let shell=views.get(battle);
  if(!shell){
    if(battle.sides.length!==2||battle.sides.some(s=>!s||s.pokemon.length!==6))return null;
    shell=Object.create(Game.prototype);
    Object.assign(shell,{battle,teams:battle.sides.map(s=>s.pokemon.map(p=>clone(p.set))),
      public:[new PublicState(0),new PublicState(1)],logPos:0,trace:[],maxTurns:Infinity,truncated:false});
    views.set(battle,shell);
  }
  shell.sync();
  const observation=shell.view(side);
  for(const mon of observation.opponent){
    for(const key of ['set','stats','selected','current_hp','max_hp'])
      if(key in mon)throw Error('Private opponent field in local observation: '+key);
  }
  return observation;
}
function install(Battle,botName){
  const original=Battle.prototype.sendUpdates;
  const previous=new WeakMap();
  Battle.prototype.sendUpdates=function(...args){
    const result=original.apply(this,args);
    if(this.format.id!=='gen9championsbssregmc'||this.ended)return result;
    const index=this.sides.findIndex(s=>s?.name===botName);
    if(index<0)return result;
    const obs=playerObservation(this,index);
    if(!obs?.legal.length)return result;
    const encoded=JSON.stringify(obs);
    if(previous.get(this)===encoded)return result;
    previous.set(this,encoded);
    this.send('sideupdate',`${this.sides[index].id}\n|rlobserve|${Buffer.from(encoded).toString('base64')}`);
    return result;
  };
}
// NODE_OPTIONS applies only to the isolated local-server process tree.
if(process.env.CHAMPIONS_PS_DIR&&process.env.CHAMPIONS_LOCAL_BOT){
  const {Battle}=require(path.join(process.env.CHAMPIONS_PS_DIR,'dist/sim/battle.js'));
  install(Battle,process.env.CHAMPIONS_LOCAL_BOT);
}
module.exports={playerObservation,install};
