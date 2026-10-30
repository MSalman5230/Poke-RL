'use strict';
// Serve the official client on the same loopback origin as the battle server.
// No public Showdown account or login-server credential is used for local play.
const fs=require('fs'),path=require('path');
function localHTML(root,port){
  let html=fs.readFileSync(path.join(root,'testclient-new.html'),'utf8');
  const config={version:'local',bannedHosts:[],whitelist:[],customcolors:{},
    routes:{root:'pokemonshowdown.com',client:'play.pokemonshowdown.com',dex:'dex.pokemonshowdown.com',
      replays:'replay.pokemonshowdown.com',users:'pokemonshowdown.com/users'},
    defaultserver:{id:'localchampions',host:'127.0.0.1',port}};
  html=html.replace('<title>Showdown!</title>','<title>Champions RL — Local Showdown</title>')
    .replace('<script src="https://play.pokemonshowdown.com/config/config.js"></script>',
      '<script>window.Config='+JSON.stringify(config)+'</script>')
    .replace('<script src="../config/testclient-key.js"></script>','')
    .replace('<script src="js/client-connection.js"></script>',`<script src="js/client-connection.js"></script>
<script>
// Guest nicknames on this isolated server require no external authentication.
PSLoginServer.rawQuery = function (act) {
  if (act === 'getassertion') return Promise.resolve('local');
  if (act === 'upkeep') return Promise.resolve(']{}');
  if (act === 'getteams') return Promise.resolve(']{"teams":[]}');
  return Promise.resolve(']{"actionerror":"Local play uses guest nicknames; no account is needed."}');
};
PS.user.handleAssertion = function (name, assertion) {
  this.loggingIn = null;
  if (assertion === 'local') {
    PS.send('/trn ' + this.validateName(name) + ',0,');
    this.update({success:true});
  }
};
</script>`)
    .replace('<h4>Test client</h4>','<h4>Play the latest checkpoint</h4>')
    .replace('<p>&mdash;<strong>Zarel</strong> <small class="date">on Sep 25, 2015</small></p>','')
    .replace('Welcome to the test client! You can test client changes here!',
      'Local match against RLChampion. Import a team in Teambuilder, then use Find a user to challenge RLChampion in [Gen 9 Champions] BSS Reg M-C. No account is needed.');
  return html;
}
function handler(root,port){
  const {StaticServer}=require(path.join(process.env.CHAMPIONS_PS_DIR,'dist/lib/static-server.js'));
  const files=new StaticServer(root,{cacheTime:0});
  const html=localHTML(root,port);
  return (req,res)=>{
    if(/^\/(?:[a-zA-Z0-9][a-zA-Z0-9-]*)?\/?(?:\?.*)?$/.test(req.url)){
      res.writeHead(200,{'Content-Type':'text/html; charset=utf-8','Cache-Control':'no-store'});res.end(html);
    }else{void files.serve(req,res);}
    return true;
  };
}
module.exports={handler,localHTML};
