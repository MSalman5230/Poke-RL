'use strict';
// Offline public replay ingestion through the same protocol parser as live play.
const fs=require('fs');
const {Dex}=require('pokemon-showdown');
const {extractChannelMessages}=require('pokemon-showdown/dist/sim/battle');
const {PublicState}=require('./worker.cjs');
const copy=x=>JSON.parse(JSON.stringify(x));
function parseReplay(input){
  if(typeof input.log!=='string')throw Error('Replay JSON must contain a protocol log');
  const lines=extractChannelMessages(input.log,[0])[0];
  const tiers=lines.filter(x=>x.startsWith('|tier|')).map(x=>Dex.toID(x.split('|')[2]));
  const format=Dex.toID(input.formatid||input.format||tiers[0]);
  if(format!=='gen9championsbssregmc'||tiers.some(x=>x!=='gen9championsbssregmc'))throw Error('Replay format must be gen9championsbssregmc');
  const pub=new PublicState(0),players={},examples=[],warnings=[];
  let root=null,acted=new Set(),forced=new Set(),winner=null,ended=false,preview=null;
  const state=()=>({turn:pub.turn,teams:copy(pub.teams),conditions:copy(pub.conditions),weather:pub.weather,terrain:pub.terrain,
    weather_since:pub.weather_since??pub.turn,terrain_since:pub.terrain_since??pub.turn,pseudo:copy(pub.pseudo),
    mega:copy(pub.mega),history:copy(pub.history),faints:copy(pub.faints)});
  for(let index=0;index<lines.length;index++){
    const line=lines[index],p=line.split('|'),event=p[1],side=Number(p[2]?.[1])-1;
    if(event==='player')players[p[2]]=p[3];
    if(event==='win'){ended=true;winner=Object.entries(players).find(([s,n])=>n===p[2])?.[0]||null;}
    if(event==='tie'){ended=true;winner=null;}
    if((event==='-enditem'||event==='-activate')&&/Eject Button|Eject Pack|Red Card/.test(line)){forced.add(0);forced.add(1);}
    let label=null;
    if(root&&[0,1].includes(side)&&!acted.has(side)){
      if(event==='move'&&!p.some(x=>x.startsWith('[from]')))label={kind:'move',move:Dex.toID(p[3])};
      if(event==='switch'&&!forced.has(side)){
        const current=pub.active[side]===null?null:pub.teams[side][pub.active[side]];
        if(current&&!current.fainted)label={kind:'switch',species:p[3].split(',')[0]};
      }
    }
    // Snapshot is BEFORE either simultaneous choice is revealed. Forced
    // replacements and called moves are not mislabeled as ordinary actions.
    const before=label?copy(root):null;
    pub.line(line);
    if(label){
      if(label.kind==='switch')label.slot=pub.active[side];
      examples.push({turn:root.turn,actor_side:side,viewer_side:1-side,opponent_action:label,public_state:before});
      acted.add(side);
    }
    if(event==='cant'&&[0,1].includes(side))acted.add(side);
    if(event==='turn'){root=state();acted=new Set();forced=new Set();}
    if(event==='teampreview')preview=state();
    if(event==='-transform'||event==='replace')warnings.push({line:index,event,reason:'Identity/Transform evidence needs specialized imitation reconstruction'});
  }
  if(pub.teams.some(t=>t.length!==6))throw Error('A full six-versus-six team preview is required');
  const evidence=pub.teams.map(team=>team.map(m=>({species:m.base_species,moves:m.moves,item:m.item,ability:m.ability,seen:m.seen})));
  for(const example of examples){
    example.actor_outcome=ended?(winner===null?0:winner==='p'+(example.actor_side+1)?1:-1):null;
    example.viewer_outcome=ended?-example.actor_outcome:null;
    example.imitation_ready=false; // public logs contain no trustworthy own legal-action mask
  }
  return {format,ended,winner,preview,examples,team_evidence:evidence,warnings,
    limitations:['Public logs do not reveal unselected trio members, full hidden sets, investments, or private legal-action requests.',
      'Examples are public-evidence/opponent-action records. Policy imitation requires independently captured own observations and legal masks.']};
}
if(require.main===module){try{process.stdout.write(JSON.stringify(parseReplay(JSON.parse(fs.readFileSync(0,'utf8')))));}
  catch(e){process.stderr.write(String(e.stack));process.exitCode=1;}}
module.exports={parseReplay};
