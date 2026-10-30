'use strict';
// Explicit integration probe against a running local human-play launcher.
const {Teams,Dex}=require('../simulator/node_modules/pokemon-showdown');
const fs=require('fs'),path=require('path');
const port=Number(process.argv[2]);
if(!Number.isInteger(port)||port<1024||port>65535)throw Error('Supply the local test port');
const host=process.argv[3]||'127.0.0.1';
if(require('node:net').isIP(host)!==4)throw Error('Supply the server IPv4 address');
const team=JSON.parse(fs.readFileSync(path.join(__dirname,'../runtime_data/human_play/bot_team.json'),'utf8'));
const name='LocalTest'+Date.now()%1000000;
const ws=new WebSocket(`ws://${host}:${port}/showdown/websocket`);
let challenged=false,decisions=0,finished=false,errors=[];
const timer=setTimeout(()=>{console.error('Local battle timeout');ws.close();process.exitCode=1;},60000);
const send=(message,room='')=>ws.send(room+'|'+message);
ws.addEventListener('message',event=>{
  const lines=String(event.data).split('\n'),room=lines[0].startsWith('>')?lines.shift().slice(1):'';
  for(const line of lines){
    if(line.startsWith('|challstr|'))send(`/trn ${name},0,`);
    if(line.startsWith('|updateuser|')){
      const p=line.split('|');
      if(Dex.toID(p[2])===Dex.toID(name)&&p[3]==='1'&&!challenged){
        challenged=true;send('/utm '+Teams.pack(team));send('/challenge RLChampion, gen9championsbssregmc');
      }
    }
    if(line.startsWith('|rlobserve|'))throw Error('Bot observation leaked to human socket');
    if(line.startsWith('|request|')){
      const request=JSON.parse(line.slice(9));if(!request||request.wait)continue;
      let choice;
      if(request.teamPreview)choice='team 1,2,3';
      else if(request.forceSwitch?.[0]){
        const reviving=request.side.pokemon.some(p=>p.reviving);
        const index=request.side.pokemon.findIndex(p=>!p.active&&(reviving?p.condition.endsWith('fnt'):!p.condition.endsWith('fnt')));
        choice='switch '+(index+1);
      }else{
        const active=request.active[0];
        const index=active.moves.findIndex(m=>!m.disabled&&(m.pp===undefined||m.pp>0||m.id==='struggle'));
        choice='move '+(index+1)+(active.canMegaEvo?' mega':'');
      }
      send('/choose '+choice+'|'+request.rqid,room);decisions++;
    }
    if(line.startsWith('|error|')||line.startsWith('|popup|')||line.startsWith('|nametaken|'))errors.push(line);
    if(line.startsWith('|win|')||line==='|tie'){
      finished=true;console.log(JSON.stringify({event:'human_connection_smoke',room,result:line,decisions,errors}));
      if(errors.length)process.exitCode=1;
      clearTimeout(timer);ws.close();
    }
  }
});
ws.addEventListener('close',()=>{clearTimeout(timer);if(!finished)process.exitCode=1;});
