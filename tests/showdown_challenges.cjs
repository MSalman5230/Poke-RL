'use strict';
// Exercise the actual socket client's protocol handling without a live game.
const assert=require('node:assert/strict'),fs=require('fs'),vm=require('vm');
const {EventEmitter}=require('node:events');
const source=fs.readFileSync(require.resolve('../simulator/showdown_client.cjs'),'utf8');
const sent=[],events=[],input=new EventEmitter();
let socket;
class FakeSocket extends EventEmitter {
  constructor(){super();socket=this;}
  addEventListener(event,callback){this.on(event,callback);}
  send(message){sent.push(message);}
  close(){}
}
vm.runInNewContext(source,{
  require:id=>id==='fs'?{readFileSync:()=> '[]'}:
    id==='readline'?{createInterface:()=>input}:
    require('../simulator/node_modules/'+id),
  process:{argv:['node','client','8765','RLChampion','team.json'],stdin:{},
    stdout:{write:line=>events.push(JSON.parse(line))},exit(){}},
  WebSocket:FakeSocket,Buffer,
});
const receive=data=>socket.emit('message',{data});
const challenge=name=>receive(`|pm| ${name}| RLChampion|/challenge gen9championsbssregmc|gen9championsbssregmc|||`);
const accepts=()=>sent.filter(s=>s.startsWith('|/accept '));
const room='battle-gen9championsbssregmc-1';
challenge('First');
challenge('First'); // Duplicate PM must not accept twice.
receive(`>${room}\n|request|{"rqid":1,"teamPreview":true}\n|rlobserve|`+
  Buffer.from(JSON.stringify({legal:[{id:17,choice:'team 1,2,3'}]})).toString('base64'));
input.emit('line',JSON.stringify({room,rqid:1,action:17}));
assert(sent.includes(`${room}|/choose team 1,2,3|1`));
challenge('Cancelled');
challenge('Second');
receive('|pm| RLChampion| Cancelled|/challenge');
receive('|pm| Unsupported| RLChampion|/challenge gen9randombattle||||');
assert.deepEqual(accepts(),['|/accept first']);
// Completing the first game must accept the already-waiting second player;
// no fresh challenge event is required. Cancelled and wrong-format requests go away.
receive(`>${room}\n|win|RLChampion`);
assert.deepEqual(accepts(),['|/accept first','|/accept second']);
assert(events.some(e=>e.event==='challenge_queued'&&e.challenger==='second'));
// Older clients send complete challenge snapshots instead of PM controls.
receive('|updatechallenges|{"challengesFrom":{"Withdrawn":"gen9championsbssregmc","Third":"gen9championsbssregmc"}}');
receive('|updatechallenges|{"challengesFrom":{"Third":"gen9championsbssregmc"}}');
receive('>battle-gen9championsbssregmc-2\n|tie');
assert.deepEqual(accepts(),['|/accept first','|/accept second','|/accept third']);
console.log('Queued challenges survive busy games and respect cancellations.');
