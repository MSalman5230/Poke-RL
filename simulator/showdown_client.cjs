'use strict';
// A loopback-only Showdown connection; Python supplies policy choices over stdio.
const fs=require('fs');
const readline=require('readline');
const {Teams,Dex}=require('pokemon-showdown');
const [port,name,teamFile]=process.argv.slice(2);
if(!/^\d+$/.test(port)||Number(port)<1024||Number(port)>65535)throw Error('Invalid local port');
if(!/^[A-Za-z][A-Za-z0-9 ]{2,17}$/.test(name))throw Error('Use a short local nickname');
const team=JSON.parse(fs.readFileSync(teamFile,'utf8'));
const socket=new WebSocket(`ws://127.0.0.1:${port}/showdown/websocket`);
const requests=new Map(),pending=new Map();
let accepted=false,playing=false;
const output=row=>process.stdout.write(JSON.stringify(row)+'\n');
const send=(text,room='')=>socket.send(`${room}|${text}`);
function accept(challenger,format){
  if(format!=='gen9championsbssregmc'||accepted||playing)return;
  accepted=true;send('/utm '+Teams.pack(team));send('/accept '+challenger);
  output({event:'challenge_accepted',challenger,format});
}
socket.addEventListener('message',event=>{
  const lines=String(event.data).split('\n');
  const room=lines[0].startsWith('>')?lines.shift().slice(1):'';
  for(const line of lines){
    if(line.startsWith('|challstr|'))send(`/trn ${name},0,`);
    else if(line.startsWith('|updateuser|')){
      const fields=line.split('|');
      if(Dex.toID(fields[2])===Dex.toID(name)&&fields[3]==='1'){
        send('/join lobby');output({event:'ready',name,port:Number(port)});
      }
    }else if(line.startsWith('|updatechallenges|')){
      const incoming=JSON.parse(line.slice('|updatechallenges|'.length)).challengesFrom||{};
      for(const [challenger,format] of Object.entries(incoming)){
        accept(challenger,format);
      }
    }else if(line.startsWith('|pm|')){
      // Current Showdown sends challenge controls as PM protocol messages.
      const fields=line.split('|');
      if(Dex.toID(fields[3])===Dex.toID(name)&&fields[4]?.startsWith('/challenge '))
        accept(Dex.toID(fields[2]),fields[4].slice('/challenge '.length));
    }else if(line.startsWith('|request|')){
      const request=JSON.parse(line.slice(9));
      if(request)requests.set(room,request);
    }else if(line.startsWith('|rlobserve|')){
      const obs=JSON.parse(Buffer.from(line.slice(11),'base64').toString());
      const request=requests.get(room);
      if(!request||request.wait)throw Error('Observation without a live player request');
      if(!room.startsWith('battle-gen9championsbssregmc-'))throw Error('Wrong local battle format');
      playing=true;pending.set(room,{rqid:request.rqid,obs});
      output({event:'decision',room,rqid:request.rqid,observation:obs});
    }else if(line.startsWith('|win|')||line==='|tie'){
      playing=false;accepted=false;pending.delete(room);requests.delete(room);
      output({event:'battle_finished',room,result:line});
    }else if(line.startsWith('|error|')||line.startsWith('|nametaken|')||line.startsWith('|popup|')){
      if(!playing)accepted=false;
      output({event:'server_notice',room,message:line});
    }
  }
});
socket.addEventListener('error',()=>{output({event:'connection_error'});process.exitCode=1;socket.close();});
socket.addEventListener('close',()=>process.exit(process.exitCode||0));
readline.createInterface({input:process.stdin}).on('line',line=>{
  const action=JSON.parse(line),current=pending.get(action.room);
  if(!current||current.rqid!==action.rqid)throw Error('Stale policy reply');
  const legal=current.obs.legal.find(a=>a.id===action.action);
  if(!legal)throw Error('Illegal policy reply');
  pending.delete(action.room);
  send(`/choose ${legal.choice}|${action.rqid}`,action.room);
}).on('close',()=>socket.close());
