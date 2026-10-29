// Independent oracle: execute the simulator's actual hit-resolution pipeline.
const fs=require('fs');
const {build}=require('../simulator/tactics.cjs');
const queries=JSON.parse(fs.readFileSync(0,'utf8'));
const result=queries.map(q=>[85,100].map(roll=>{
  const b=build(q),a=b.p1.active[0],d=b.p2.active[0],initial=d.hp;
  try{
    const move=b.dex.getActiveMove(q.move);
    if(move.willCrit===undefined)move.willCrit=false;
    move.accuracy=true;
    b.randomizer=x=>Math.floor(x*roll/100);
    b.actions.useMove(move,a,{target:d});
    b.eachEvent('Update');
    return {loss:initial-d.hp,maxhp:d.maxhp,initial,hp:d.hp};
  }finally{b.destroy();}
}));
process.stdout.write(JSON.stringify(result));
