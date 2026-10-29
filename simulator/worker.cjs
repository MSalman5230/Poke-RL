'use strict';
// Headless local simulator. No server, sockets, accounts, OCR, or tablet control.
const readline = require('readline');
const {Battle, Dex, TeamValidator} = require('pokemon-showdown');
const {extractChannelMessages} = require('pokemon-showdown/dist/sim/battle');
const FORMAT = 'gen9championsbssregmc';
const dex = Dex.mod('champions');
const validator = new TeamValidator(FORMAT);
const permutations = [];
for(let a=0;a<6;a++)for(let b=0;b<6;b++)for(let c=0;c<6;c++)if(a!==b&&a!==c&&b!==c)permutations.push([a,b,c]);
const id = x => Dex.toID(x);
const clone = x => JSON.parse(JSON.stringify(x));
function freshMon(species) {
  return {base_species:species,species,hp:1,status:'',boosts:{},moves:[],item:null,ability:null,
    seen:false,active:false,fainted:false,types:[...dex.species.get(species).types],
    volatiles:{},last_move:null,times_attacked:0,entry_turn:0,
    moves_this_stint:[],pp_used:{},first_native_move:null,status_progress:{}};
}
class PublicState {
  constructor(viewer) {this.viewer=viewer;this.turn=0;this.teams=[[],[]];this.active=[null,null];
    this.conditions=[{},{}];this.weather='';this.terrain='';this.pseudo={};this.mega=[false,false];
    this.history=[];this.lastAttacker=null;this.faints=[[],[]];this.stints=[null,null];}
  mon(ident) {
    const side=Number(ident?.[1])-1;
    if(side!==0&&side!==1)return null;
    const nickname=ident.slice(ident.indexOf(':')+2);
    return this.teams[side].find(p=>p.nickname===nickname) ||
      (ident[2]!==':'&&this.active[side]!==null?this.teams[side][this.active[side]]:null);
  }
  hp(mon,condition) {
    if(!mon || !condition)return;
    const [fraction,status='']=condition.split(' ');
    if(fraction==='0'||status==='fnt'){mon.hp=0;mon.fainted=true;return;}
    const [hp,max]=fraction.split('/').map(Number);
    if(Number.isFinite(hp)&&max){mon.hp=hp/max;mon.fainted=hp===0;}
    mon.status=status;
  }
  feed(lines) {for(const line of lines)this.line(line);}
  line(line) {
    const p=line.split('|'), event=p[1];
    if(!event)return;
    const side=Number(p[2]?.[1])-1;
    const m=this.mon(p[2]);
    const effect=x=>id((x||'').replace(/^(move|ability|item): /,''));
    if(event==='poke') {
      const name=p[3].split(',')[0];this.teams[side].push(freshMon(name));
    } else if(['switch','drag','replace'].includes(event)) {
      const name=p[3].split(',')[0];
      let index=this.teams[side].findIndex(x=>x.base_species===name||x.species===name);
      if(index<0)index=this.teams[side].findIndex(x=>dex.species.get(x.base_species).baseSpecies===dex.species.get(name).baseSpecies);
      if(index<0)throw new Error('Public species not found in preview: '+name);
      const old=this.active[side]===null?null:this.teams[side][this.active[side]];
      const oldState=old?clone(old):null;
      if(old){old.active=false;if(event!=='replace'){old.boosts={};old.volatiles={};old.last_move=null;old.moves_this_stint=[];if(old.status==='tox')old.status_progress.toxicStage=0;}}
      const target=this.teams[side][index];this.active[side]=index;
      // Illusion is only resolved when the public protocol reveals the replacement.
      if(event==='replace' && old && old!==target) {
        const stint=this.stints[side];
        Object.assign(target,{hp:oldState.hp,status:oldState.status,boosts:oldState.boosts,volatiles:oldState.volatiles,
          last_move:oldState.last_move,times_attacked:oldState.times_attacked,
          moves:[...new Set([...target.moves,...(stint?.moves||[])])]});
        target.moves_this_stint=[...(stint?.moves||[])];
        for(const [move,cost] of Object.entries(stint?.pp_used||{}))target.pp_used[move]=(target.pp_used[move]||0)+cost;
        target.first_native_move ||= stint?.first_native_move;
        target.status_progress=oldState.status_progress;
        for(const key of ['item','ability'])if(stint?.revealed?.[key])target[key]=oldState[key];
        // Undo the mistaken identity: the disguised Pokémon did not thereby
        // prove that the apparent species was selected or reveal its moves.
        if(stint?.previous)Object.assign(old,stint.previous,{active:false});
      }
      if(event!=='replace')this.stints[side]={previous:clone(target),moves:[],revealed:{},pp_used:{}};
      target.nickname=p[2].slice(p[2].indexOf(':')+2);
      target.active=true;target.seen=true;target.species=name;
      target.types=[...dex.species.get(name).types];target.entry_turn=this.turn;
      this.hp(target,p[4]);this.lastAttacker=null;
    } else if(event==='turn') {this.turn=Number(p[2]);this.lastAttacker=null;
      for(const team of this.teams)for(const mon of team)for(const [key,value] of Object.entries(mon.volatiles))
        if(value.single&&value.turn<this.turn)delete mon.volatiles[key];
    } else if(event==='move' && m) {
      const move=id(p[3]);if(!p.some(x=>x.startsWith('[from]'))&&!m.moves.includes(move))m.moves.push(move);
      if(!p.some(x=>x.startsWith('[from]'))){
        m.moves_this_stint.push(move);
        const target=this.mon(p[4]);
        const cost=1+(target&&target!==m&&target.ability==='pressure'?1:0);
        m.pp_used[move]=(m.pp_used[move]||0)+cost;
        const first={turn:this.turn,without_mega:!this.mega[side]};
        m.first_native_move ||= first;
        if(this.stints[side]){const stint=this.stints[side];stint.moves.push(move);
          stint.pp_used[move]=(stint.pp_used[move]||0)+cost;stint.first_native_move ||= first;}
      }else if(p.some(x=>x==='[from]move: Sleep Talk'||x==='[from] move: Sleep Talk')){
        if(!m.moves.includes(move))m.moves.push(move);
      }
      m.last_move=move;this.lastAttacker=side;
      this.history.push({turn:this.turn,side,move,actor:this.teams[side].indexOf(m),species:m.species,
        order:this.history.filter(e=>e.turn===this.turn).length});this.history=this.history.slice(-12);
    } else if(['-damage','-heal','-sethp'].includes(event)) {
      this.hp(m,p[3]);
      if(event==='-damage'&&m?.status==='tox'&&p.includes('[from] psn'))m.status_progress.toxicStage=Math.min(15,(m.status_progress.toxicStage||0)+1);
      if(event==='-damage'&&m&&this.lastAttacker!==null&&this.lastAttacker!==side&&!p.some(x=>x.startsWith('[from]')))m.times_attacked++;
      if(event==='-sethp'&&p[4])this.hp(this.mon(p[4]),p[5]);
    } else if(event==='faint'&&m){m.hp=0;m.fainted=true;this.faints[side].push({species:m.base_species,turn:this.turn});
    } else if(event==='-status'&&m){m.status=p[3];m.status_since=this.turn;m.status_progress={};
      if(m.status==='slp')m.status_progress={sleepSource:m.last_move==='rest'?'rest':'sleep',sleepElapsed:0};
    } else if(event==='cant'&&m){
      if(p[3]==='slp')m.status_progress.sleepElapsed=(m.status_progress.sleepElapsed||0)+1;
      if(p[3]==='frz')m.status_progress.freezeElapsed=(m.status_progress.freezeElapsed||0)+1;
    } else if(event==='-curestatus'&&m){m.status='';m.status_progress={};
    } else if(event==='-cureteam'&&side>=0){for(const mon of this.teams[side]){mon.status='';mon.status_progress={};}
    } else if(['-boost','-unboost','-setboost'].includes(event)&&m){
      m.boosts[p[3]]=event==='-setboost'?Number(p[4]):Math.max(-6,Math.min(6,(m.boosts[p[3]]||0)+(event==='-boost'?1:-1)*Number(p[4])));
    } else if(event==='-clearboost'&&m){m.boosts={};
    } else if(event==='-clearallboost'){for(const s of this.teams)for(const mon of s)mon.boosts={};
    } else if(event==='-clearnegativeboost'&&m){for(const k in m.boosts)m.boosts[k]=Math.max(0,m.boosts[k]);
    } else if(event==='-clearpositiveboost'&&m){for(const k in m.boosts)m.boosts[k]=Math.min(0,m.boosts[k]);
    } else if(event==='-invertboost'&&m){for(const k in m.boosts)m.boosts[k]*=-1;
    } else if(event==='-copyboost'&&m){const other=this.mon(p[3]);if(other)m.boosts={...other.boosts};
    } else if(event==='-swapboost'&&m){const other=this.mon(p[3]);if(other){for(const k of (p[4]?p[4].split(', '):['atk','def','spa','spd','spe','accuracy','evasion']))[m.boosts[k],other.boosts[k]]=[other.boosts[k]||0,m.boosts[k]||0];}
    } else if(event==='-item'&&m){m.item=id(p[3]);if(this.stints[side])this.stints[side].revealed.item=true;
    } else if(event==='-enditem'&&m){m.item='';if(this.stints[side])this.stints[side].revealed.item=true;
    } else if(event==='-ability'&&m){m.ability=id(p[3]);if(this.stints[side])this.stints[side].revealed.ability=true;
    } else if(event==='-mega'&&m){this.mega[side]=true;m.item=id(p[4]);
    } else if(['detailschange','-formechange'].includes(event)&&m){m.species=p[3].split(',')[0];m.types=[...dex.species.get(m.species).types];
    } else if(event==='-transform'&&m){const other=this.mon(p[3]);if(other){m.species=other.species;m.types=[...other.types];m.boosts={...other.boosts};m.volatiles.transform={turn:this.turn};}
    } else if(event==='-start'&&m){
      if(p[3]==='typechange')m.types=p[4].split('/');
      else if(p[3]==='typeadd')m.types=[...new Set([...m.types,p[4]])];
      else m.volatiles[effect(p[3])]={turn:this.turn,move:p[4]?id(p[4]):null};
    } else if(event==='-end'&&m){delete m.volatiles[effect(p[3])];
    } else if(event==='-singleturn'&&m){m.volatiles[effect(p[3])]={turn:this.turn,single:true};
    } else if(event==='-weather'){this.weather=p[2]==='none'?'':id(p[2]);if(!p.includes('[upkeep]'))this.weather_since=this.turn;
    } else if(event==='-fieldstart'){const e=effect(p[2]);if(e.endsWith('terrain')){this.terrain=e;this.terrain_since=this.turn;}else this.pseudo[e]={turn:this.turn};
    } else if(event==='-fieldend'){const e=effect(p[2]);if(e.endsWith('terrain'))this.terrain='';else delete this.pseudo[e];
    } else if(event==='-sidestart'&&side>=0){const e=effect(p[3]);const old=this.conditions[side][e];this.conditions[side][e]={turn:this.turn,layers:Math.min(e==='spikes'?3:e==='toxicspikes'?2:1,(old?.layers||0)+1)};
    } else if(event==='-sideend'&&side>=0){delete this.conditions[side][effect(p[3])];
    } else if(event==='-swapsideconditions'){[this.conditions[0],this.conditions[1]]=[this.conditions[1],this.conditions[0]];}
    // Passive ability/item announcements are public too.
    const from=p.find(x=>x.startsWith('[from] ability: ')||x.startsWith('[from] item: '));
    if(from){const of=p.find(x=>x.startsWith('[of] '));const source=of?this.mon(of.slice(5)):m;
      if(source){const key=from.startsWith('[from] ability: ')?'ability':'item';
        source[key]=id(from.slice(key==='ability'?16:13));
        for(let s=0;s<2;s++)if(this.active[s]!==null&&this.teams[s][this.active[s]]===source&&this.stints[s])this.stints[s].revealed[key]=true;}}
  }
}
class Game {
  constructor(teams,seed,maxTurns=200,training=false) {
    this.training=training;
    this.teams=clone(teams);this.maxTurns=maxTurns;this.truncated=false;
    // Showdown shortens species-as-nicknames (Arcanine-Hisui -> Arcanine).
    // Stable legal nicknames keep our private six roster slots unambiguous.
    this.teams.forEach(team=>team.forEach((set,index)=>{set.name='unit'+index;}));
    this.public=[new PublicState(0),new PublicState(1)];this.logPos=0;this.trace=[];
    for(const t of teams){const errors=validator.validateTeam(clone(t));if(errors)throw new Error(errors.join('; '));}
    this.battle=new Battle({formatid:FORMAT,seed,send:()=>{}});
    this.teams.forEach((team,i)=>this.battle.setPlayer('p'+(i+1),{name:'player'+i,team:clone(team)}));
    this.sync();
  }
  sync(){const lines=this.battle.log.slice(this.logPos);this.logPos=this.battle.log.length;
    for(let i=0;i<2;i++)this.public[i].feed(extractChannelMessages(lines.join('\n'),[i+1])[i+1]);
    this.trace.push(...extractChannelMessages(lines.join('\n'),[0])[0]);
    if(this.battle.turn>this.maxTurns&&!this.battle.ended)this.truncated=true;
  }
  legal(i) {
    const side=this.battle.sides[i],req=side.activeRequest;
    if(!req||req.wait||side.isChoiceDone()||this.battle.ended||this.truncated)return [];
    if(req.teamPreview)return permutations.map((p,k)=>({id:14+k,kind:'preview',picks:p,choice:'team '+p.map(x=>x+1).join(',')}));
    const result=[],active=req.active?.[0];
    if(active){for(let j=0;j<active.moves.length;j++)if(!active.moves[j].disabled && (active.moves[j].pp===undefined||active.moves[j].pp>0||active.moves[j].id==='struggle')){
      result.push({id:j,kind:'move',slot:j,move:active.moves[j].id,mega:false,choice:'move '+(j+1)});
      if(active.canMegaEvo)result.push({id:4+j,kind:'move',slot:j,move:active.moves[j].id,mega:true,choice:'move '+(j+1)+' mega'});
    }}
    const reviving=!!req.side.pokemon.some(p=>p.reviving);
    if(req.forceSwitch?.[0] || (active&&!active.trapped))req.side.pokemon.forEach((p,k)=>{
      if(p.active||(reviving?!p.condition.endsWith('fnt'):p.condition.endsWith('fnt')))return;
      const name=p.ident.slice(4),index=this.teams[i].findIndex(s=>s.name===name||s.species===name);
      if(index<0)throw new Error('Missing own roster slot '+name);
      result.push({id:8+index,kind:'switch',slot:index,party_slot:k,revive:reviving,species:p.details.split(',')[0],choice:'switch '+(k+1)});
    });
    return result;
  }
  view(i){
    const pub=this.public[i],side=this.battle.sides[i],req=side.activeRequest;
    const own=this.teams[i].map((set,index)=>{
      const actual=side.pokemon.find(p=>p.name===(set.name||set.species));
      const visible=pub.teams[i].find(p=>p.base_species===set.species)||freshMon(set.species);
      const current=actual?{...visible,species:actual.species.name,hp:actual.hp/actual.maxhp,
        current_hp:actual.hp,max_hp:actual.maxhp,stats:clone(actual.storedStats),status:actual.status,
        boosts:clone(actual.boosts),types:actual.getTypes(),item:actual.item,ability:actual.ability,
        moves:actual.moveSlots.map(m=>({id:m.id,pp:m.pp,maxpp:m.maxpp,disabled:!!m.disabled})),
        active:side.active.includes(actual),fainted:actual.fainted,times_attacked:actual.timesAttacked||0,
        volatile_ids:Object.keys(actual.volatiles)}:{};
      return {...visible,item:id(set.item),ability:id(set.ability),
        moves:set.moves.map(mid=>{const m=dex.moves.get(mid);const pp=m.noPPBoosts?m.pp:(m.pp/5+1)*4;return {id:m.id,pp,maxpp:pp,disabled:false};}),
        ...current,index,set:clone(set),selected:this.battle.requestState==='teampreview'?null:!!actual};
    });
    const opponent=clone(pub.teams[1-i]);
    for(const mon of opponent)delete mon.nickname; // internal slot aliases are not model features
    return {format:FORMAT,turn:pub.turn,side:i,phase:(req?.wait||side.isChoiceDone())?'wait':req?.teamPreview?'preview':req?.forceSwitch?.[0]?'switch':'move',
      own,opponent,legal:this.legal(i),weather:pub.weather,terrain:pub.terrain,pseudo:clone(pub.pseudo),
      weather_since:pub.weather_since??pub.turn,terrain_since:pub.terrain_since??pub.turn,
      conditions:[clone(pub.conditions[i]),clone(pub.conditions[1-i])],mega:[pub.mega[i],pub.mega[1-i]],
      history:pub.history.map(e=>({...e,side:e.side===i?0:1})),
      faints:[clone(pub.faints[i]),clone(pub.faints[1-i])],
      ended:this.battle.ended,truncated:this.truncated};
  }
  result(){
    const views=[this.view(0),this.view(1)];
    const result={views,ended:this.battle.ended,truncated:this.truncated,
      winner:this.battle.ended?(this.battle.winner==='player0'?0:this.battle.winner==='player1'?1:null):null};
    // Never put privileged data inside a player's observation. The actor is
    // passed views only; an opt-in training channel serves a separate critic.
    if(this.training)result.training={critic_views:views.map((v,i)=>({...clone(v),opponent:clone(views[1-i].own)}))};
    return result;
  }
  step(actions){
    // Validate BOTH choices before submitting either. Policies never see a committed enemy choice.
    const choices=[0,1].map(i=>{const legal=this.legal(i);if(!legal.length)return null;
      const action=legal.find(a=>a.id===actions[i]);if(!action)throw new Error('Illegal action for player '+i+': '+actions[i]);return action;});
    if(!choices.some(Boolean) && !this.battle.ended && !this.truncated)throw new Error('No actionable request in nonterminal battle');
    for(let i=0;i<2;i++)if(choices[i]){
      const previous=JSON.stringify(this.battle.sides[i].activeRequest);
      const ok=this.battle.choose('p'+(i+1),choices[i].choice);
      // A hidden trap/disable may legally be discovered by attempting a choice.
      // Showdown then updates that player's request. Keep the opponent's already
      // committed choice private and request a replacement only from this side.
      if(!ok && JSON.stringify(this.battle.sides[i].activeRequest)===previous)
        throw new Error('Engine rejected '+choices[i].choice+': '+this.battle.sides[i].choice.error);
    }
    this.sync();return this.result();
  }
}
const games=new Map();
function dispatch(q){
  if(q.op==='ping')return {format:FORMAT,revision:'c0845f33d381402dfcb18ce324c20a94f18af075',node:process.version};
  if(q.op==='calculate')return require('./tactics.cjs').calculateMany(q.queries);
  if(q.op==='tactics_metadata')return require('./tactics.cjs').metadata();
  if(q.op==='lookahead')return require('./lookahead.cjs').lookahead(q,Game,PublicState);
  if(q.op==='validate')return q.teams?q.teams.map(t=>validator.validateTeam(clone(t))):q.sets.map(s=>validator.validateSet(clone(s)));
  if(q.op==='new'){if(games.has(q.game))games.get(q.game).battle.destroy();const g=new Game(q.teams,q.seed,q.max_turns,!!q.training);games.set(q.game,g);return g.result();}
  if(q.op==='step')return games.get(q.game).step(q.actions);
  if(q.op==='trace')return games.get(q.game).trace;
  if(q.op==='close'){const g=games.get(q.game);if(g)g.battle.destroy();games.delete(q.game);return true;}
  throw new Error('Unknown op '+q.op);
}
if(require.main===module){const rl=readline.createInterface({input:process.stdin});rl.on('line',line=>{
  let q;try{q=JSON.parse(line);const result=q.op==='batch'?q.requests.map(dispatch):dispatch(q);process.stdout.write(JSON.stringify({id:q.id,result})+'\n');}
  catch(e){process.stdout.write(JSON.stringify({id:q?.id,error:e.stack})+'\n');}
});}
module.exports={Game,PublicState,permutations};
