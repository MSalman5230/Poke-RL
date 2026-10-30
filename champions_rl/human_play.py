"""Run a local Showdown server and challenge a frozen checkpoint as RLChampion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
import torch
from .engine import ROOT,Engine
from .model import BattleModel,NeuralBot
from .features import Encoder,FEATURE_VERSION
from .legacy import load_frozen_policy
from .persistence import atomic_json
from .teams import TeamPool


def stop_process(process):
    if process is None or process.poll() is not None:return
    if os.name=='nt':
        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    else:process.kill()
    process.wait(timeout=15)


def prepare_server(port,bot_name,host='0.0.0.0'):
    source=ROOT/'simulator/node_modules/pokemon-showdown'
    target=ROOT/'runtime_data/human_showdown'
    if not (source/'dist/server/index.js').exists():raise RuntimeError('Install the pinned simulator first')
    # Keep the installed simulator and the user's other Showdown checkout intact.
    marker=target/'CHAMPIONS_LOCAL_SERVER.json'
    signature=hashlib.sha256((source/'dist/sim/battle.js').read_bytes()).hexdigest()
    if target.exists():
        if not marker.exists() or json.loads(marker.read_text())['battle_sha256']!=signature:
            raise RuntimeError('Existing local server is not this pinned build: '+str(target))
    else:
        shutil.copytree(source,target,ignore=shutil.ignore_patterns('node_modules','.git','logs'))
        atomic_json(marker,{'battle_sha256':signature,'source':str(source)})
    (target/'logs/repl').mkdir(parents=True,exist_ok=True)
    client_root=ROOT/'runtime_data/showdown_client/play.pokemonshowdown.com'
    if not (client_root/'js/client-main.js').exists():
        raise RuntimeError('Build the official local client first: .\\tools\\setup_human_play.ps1')
    subprocess.run([shutil.which('node') or 'node','-e',
        "require('./simulator/showdown_http.cjs').prepareAssets(process.argv[1])",str(client_root)],
        cwd=ROOT,check=True,capture_output=True,text=True,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    config="""'use strict';
exports.bindaddress = BIND_HOST;
exports.port = PORT;
exports.noguestsecurity = true;
exports.noipchecks = true;
exports.nothrottle = true;
exports.usesqlite = false;
exports.logchat = false;
exports.logchallenges = false;
exports.watchconfig = false;
exports.repl = false;
exports.reportjoins = false;
exports.subprocesses = {network:1, simulator:1, validator:1, verifier:0,
  localartemis:0, remoteartemis:0, battlesearch:0, datasearch:0,
  friends:0, chatdb:0, pm:0, modlog:0};
""".replace('PORT',str(port)).replace('BIND_HOST',json.dumps(host))
    config+='\nexports.customhttpresponse = require('+json.dumps((ROOT/'simulator/showdown_http.cjs').as_posix())+').handler('+json.dumps(client_root.as_posix())+','+str(port)+');\n'
    (target/'config/config.js').write_text(config,encoding='utf-8')
    env=dict(os.environ,NODE_PATH=str(ROOT/'simulator/node_modules'),CHAMPIONS_PS_DIR=str(target),
             CHAMPIONS_LOCAL_BOT=bot_name)
    hook=ROOT/'simulator/showdown_observer.cjs'
    env['NODE_OPTIONS']='--require "'+hook.as_posix()+'"'
    return target,env


def load_policy(checkpoint,legacy_runtime,engine,sampled=False):
    data=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if legacy_runtime is None and data.get('feature_version')!=FEATURE_VERSION:
        legacy_runtime={7:ROOT/'runtime_data/baselines/v7_e3e2d8a',6:ROOT/'runtime_data/baselines/v6_3cc5468'}.get(data.get('feature_version'))
        if legacy_runtime is None:raise ValueError('This older checkpoint needs an explicit --legacy-runtime')
    if legacy_runtime:
        model,metadata,encoder,_=load_frozen_policy(checkpoint,legacy_runtime,engine=engine)
    else:
        model,metadata=BattleModel.load(checkpoint);encoder=Encoder(engine=engine)
    return NeuralBot(model,encoder,deterministic=not sampled),metadata,legacy_runtime


def run(checkpoint=None,legacy_runtime=None,port=8765,name='RLChampion',sampled=False,host='0.0.0.0'):
    if not 1024<=port<=65535:raise ValueError('Choose an unprivileged local port')
    if host not in ('0.0.0.0','127.0.0.1'):raise ValueError('Use 0.0.0.0 for LAN play or 127.0.0.1 for this PC only')
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9 ]{2,17}',name):raise ValueError('Use a 3-18 character local nickname')
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1',port))==0:raise RuntimeError(f'Port {port} is already in use; choose --port')
    checkpoint=Path(checkpoint or ROOT/'runtime_data/ppo_v7_generalization/latest.pt').resolve()
    torch.set_num_threads(1)
    output=ROOT/'runtime_data/human_play';output.mkdir(parents=True,exist_ok=True)
    node=shutil.which('node') or 'node'
    team=TeamPool().data['reference_team']
    for index,mon in enumerate(team):mon['name']='unit'+str(index)
    team_path=output/'bot_team.json';atomic_json(team_path,team)
    exported=subprocess.run([node,'-e',"const {Teams}=require('pokemon-showdown'); const fs=require('fs'); console.log(Teams.export(JSON.parse(fs.readFileSync(process.argv[1],'utf8'))));",str(team_path)],
        cwd=ROOT/'simulator',capture_output=True,text=True,encoding='utf-8',check=True,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).stdout
    (output/'example_team.txt').write_text(exported,encoding='utf-8')
    server=client=None
    with Engine() as engine,(output/'server.log').open('w',encoding='utf-8') as log:
        try:
            bot,metadata,runtime=load_policy(checkpoint,legacy_runtime,engine,sampled)
            target,env=prepare_server(port,name,host)
            server=subprocess.Popen([node,str(target/'dist/server/index.js')],cwd=target,env=env,
                stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            started=time.monotonic()
            while True:
                if server.poll() is not None:raise RuntimeError('Local server failed; see '+str(output/'server.log'))
                with socket.socket() as probe:
                    if probe.connect_ex(('127.0.0.1',port))==0:break
                if time.monotonic()-started>45:raise TimeoutError('Local server startup timed out; see server.log')
                time.sleep(.2)
            # Do not pass the server-only preload to the policy's socket process.
            client=subprocess.Popen([node,str(ROOT/'simulator/showdown_client.cjs'),str(port),name,str(team_path)],
                cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',bufsize=1,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            session={'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                     'metadata':metadata,'legacy_runtime':str(runtime) if runtime else None,'port':port,'bot':name,'bind_host':host,
                     'url':f'http://127.0.0.1:{port}','pid':os.getpid(),'server_pid':server.pid,'training':False}
            atomic_json(output/'session.json',session)
            for line in client.stdout:
                row=json.loads(line)
                if row['event']=='decision':
                    obs=row['observation']
                    if any(any(k in m for k in ('set','stats','selected','current_hp','max_hp')) for m in obs['opponent']):
                        raise ValueError('Private opposing information in player view')
                    action=bot.choose(obs)
                    client.stdin.write(json.dumps({'room':row['room'],'rqid':row['rqid'],'action':action})+'\n');client.stdin.flush()
                    print(json.dumps({'event':'played','room':row['room'],'turn':obs['turn'],'phase':obs['phase'],'action':action}),flush=True)
                else:
                    print(json.dumps(row),flush=True)
                    if row['event']=='ready':
                        print(f"Listening on {host}:{port}.",flush=True)
                        print(f"Open {session['url']}, choose a nickname, and challenge {name} in [Gen 9 Champions] BSS Reg M-C.",flush=True)
                        if host=='0.0.0.0':
                            print(f"Other computers on your LAN: open http://<this-PC-LAN-IP>:{port}/.",flush=True)
                        print(f"Checkpoint: {metadata.get('steps','unknown')} decisions. Example team: {output/'example_team.txt'}. Ctrl+C stops local play.",flush=True)
            if client.wait():raise RuntimeError('Showdown connection failed; see server.log')
        except KeyboardInterrupt:print('Stopping local play.',flush=True)
        finally:
            stop_process(client);stop_process(server)
            if (output/'session.json').exists():
                session=json.loads((output/'session.json').read_text());session['stopped']=True;atomic_json(output/'session.json',session)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--legacy-runtime',type=Path)
    parser.add_argument('--port',type=int,default=8765);parser.add_argument('--name',default='RLChampion')
    parser.add_argument('--host',choices=['0.0.0.0','127.0.0.1'],default='0.0.0.0',
                        help='Bind all IPv4 interfaces for LAN play (default), or only this PC')
    parser.add_argument('--sampled',action='store_true',help='Sample actions instead of using the evaluated deterministic mode')
    run(**vars(parser.parse_args()))
