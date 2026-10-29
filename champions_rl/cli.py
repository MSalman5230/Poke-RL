"""Offline build, data preparation, training, evaluation, and play commands."""
import argparse
import json
from pathlib import Path
import subprocess
from .engine import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    prepare=sub.add_parser("prepare",help="Export restricted facts and build/validate teams")
    collect=sub.add_parser("collect",help="Record behavior-cloning demonstrations")
    collect.add_argument("--output",type=Path,default=ROOT/"runtime_data/demos")
    collect.add_argument("--games",type=int,default=100)
    collect.add_argument("--teacher",choices=["reference","balanced","damage","setup","defensive"],default="balanced")
    collect.add_argument("--seed",type=int,default=41)
    clone=sub.add_parser("pretrain",help="Domain pretraining followed by behavior cloning")
    clone.add_argument("--data",type=Path,default=ROOT/"runtime_data/demos")
    clone.add_argument("--output",type=Path,default=ROOT/"runtime_data/bc.pt")
    clone.add_argument("--steps",type=int,default=1500)
    clone.add_argument("--device",default="cuda")
    clone.add_argument("--width",type=int,default=128)
    clone.add_argument("--layers",type=int,default=2)
    train=sub.add_parser("train",help="Masked PPO against varied opponents and snapshots")
    train.add_argument("--initial",type=Path)
    train.add_argument("--resume",type=Path,help="Continue optimizer/weights; starts fresh seeded battles")
    train.add_argument("--output",type=Path,default=ROOT/"runtime_data/training")
    train.add_argument("--steps",type=int,default=200000)
    train.add_argument("--envs",type=int,default=16)
    train.add_argument("--horizon",type=int,default=64)
    train.add_argument("--device",default="cuda")
    train.add_argument("--seed",type=int,default=47)
    train.add_argument("--reference-fraction",type=float,default=0.)
    train.add_argument("--snapshot-fraction",type=float,default=.3)
    train.add_argument("--checkpoint-every",type=int,default=10)
    train.add_argument("--fixed-teams",action="store_true",help="Use only the training manifest instead of fresh team compositions")
    train.add_argument("--learning-rate",type=float,default=2e-4)
    train.add_argument("--heuristic-weight",type=float,help="Optional fixed tactical logit prior; the network learns a residual policy")
    train.add_argument("--history-position",action=argparse.BooleanOptionalAction,default=None,help="Preserve move order in attention; older checkpoints default off")
    train.add_argument("--stop-file",type=Path,default=ROOT/"runtime_data/STOP")
    fast=sub.add_parser("train-fast",help="Parallel CPU battles, central GPU batches, dual-seat PPO, and asynchronous reference replay")
    fast.add_argument("--initial",type=Path)
    fast.add_argument("--resume",type=Path)
    fast.add_argument("--output",type=Path,default=ROOT/"runtime_data/parallel_training")
    fast.add_argument("--steps",type=int,default=200000)
    fast.add_argument("--workers",type=int,default=4)
    fast.add_argument("--envs",type=int,default=64)
    fast.add_argument("--horizon",type=int,default=32)
    fast.add_argument("--device",default="cuda")
    fast.add_argument("--seed",type=int,default=101)
    fast.add_argument("--selfplay-fraction",type=float,default=.3)
    fast.add_argument("--snapshot-fraction",type=float,default=.3)
    fast.add_argument("--reference-workers",type=int,default=1)
    fast.add_argument("--reference-weight",type=float,default=.1)
    fast.add_argument("--learning-rate",type=float,default=1e-4)
    fast.add_argument("--batch-size",type=int,default=128)
    fast.add_argument("--checkpoint-every",type=int,default=10)
    fast.add_argument("--stop-file",type=Path,default=ROOT/"runtime_data/STOP")
    fast.add_argument("--history-position",action=argparse.BooleanOptionalAction,default=None)
    fast.add_argument("--heuristic-weight",type=float)
    evaluate=sub.add_parser("evaluate",help="Evaluate; report whether the 60% acceptance gate passes")
    evaluate.add_argument("--checkpoint",type=Path)
    evaluate.add_argument("--learner-style",default=None)
    evaluate.add_argument("--opponents",default="reference,damage,balanced,setup,defensive,stochastic")
    evaluate.add_argument("--games",type=int,default=200)
    evaluate.add_argument("--split",choices=["validation","validation_competitive","test"],default="validation")
    evaluate.add_argument("--output",type=Path)
    evaluate.add_argument("--seed",type=int,default=92341)
    evaluate.add_argument("--device",default="cpu")
    evaluate.add_argument("--deterministic",action="store_true",help="Use the highest-probability legal action")
    evaluate.add_argument("--reference-teams",choices=["saved","varied"],default="saved")
    evaluate.add_argument("--workers",type=int,default=1,help="Independent local evaluation processes (up to 8)")
    evaluate.add_argument("--shard-index",type=int,default=0,help=argparse.SUPPRESS)
    evaluate.add_argument("--shard-count",type=int,default=1,help=argparse.SUPPRESS)
    sub.add_parser("status",help="Show local GPU and training status")
    play=sub.add_parser("play",help="Play and save a complete offline battle")
    play.add_argument("--checkpoint",type=Path,required=True)
    play.add_argument("--opponent",choices=["reference","random","damage","balanced","setup","defensive"],default="balanced")
    play.add_argument("--seed",type=int,default=101)
    play.add_argument('--deterministic',action=argparse.BooleanOptionalAction,default=True)
    play.add_argument("--output",type=Path,default=ROOT/"runtime_data/play")
    args=parser.parse_args()
    if args.command=="prepare":
        subprocess.run(["node",str(ROOT/"simulator/export_knowledge.cjs")],check=True)
        from .teams import build_sets,TeamPool,fingerprint
        from .features import Encoder
        print(json.dumps(build_sets()),flush=True)
        pool=TeamPool()
        # Preserve existing fixed benchmarks when refreshing training sets.
        forbidden=set()
        for split,count,seed in [('validation',64,91731),('test',128,191731),('validation_competitive',64,31731)]:
            path=ROOT/f'data/teams_{split}.json'
            manifest=json.loads(path.read_text()) if path.exists() else pool.manifest(count,seed,split,forbidden)
            forbidden.update(fingerprint(t['team']) for t in manifest['teams'])
        pool.manifest(128,1731,'train',forbidden)
        (ROOT/"data/feature_schema.json").write_text(json.dumps(Encoder().schema(),indent=2)+"\n",encoding="utf-8")
    elif args.command=="collect":
        from .training import collect_demonstrations
        collect_demonstrations(args.output,args.games,args.teacher,args.seed)
    elif args.command=="pretrain":
        from .training import behavior_clone
        behavior_clone(args.data,args.output,args.steps,device=args.device,width=args.width,layers=args.layers)
    elif args.command=="train":
        from .training import train_ppo
        train_ppo(args.output,args.initial,args.steps,args.envs,args.horizon,device=args.device,seed=args.seed,
                  reference_fraction=args.reference_fraction,snapshot_fraction=args.snapshot_fraction,
                  checkpoint_every=args.checkpoint_every,stop_file=args.stop_file,resume=args.resume,
                  procedural_teams=not args.fixed_teams,learning_rate=args.learning_rate,heuristic_weight=args.heuristic_weight,
                  history_position=args.history_position)
    elif args.command=="train-fast":
        from .parallel_training import train_parallel
        train_parallel(args.output,initial=args.initial,resume=args.resume,steps=args.steps,workers=args.workers,
                       envs=args.envs,horizon=args.horizon,device=args.device,seed=args.seed,
                       selfplay_fraction=args.selfplay_fraction,snapshot_fraction=args.snapshot_fraction,
                       reference_workers=args.reference_workers,reference_weight=args.reference_weight,
                       learning_rate=args.learning_rate,batch_size=args.batch_size,checkpoint_every=args.checkpoint_every,
                       stop_file=args.stop_file,history_position=args.history_position,heuristic_weight=args.heuristic_weight)
    elif args.command=="play":
        from .evaluation import evaluate
        report=evaluate(args.checkpoint,[args.opponent],1,args.output,"validation",args.seed,deterministic=args.deterministic)
        print(json.dumps(report["results"],indent=2))
    elif args.command=="evaluate":
        from .evaluation import evaluate,evaluate_parallel
        options={"workers":args.workers} if args.workers>1 else {"shard_index":args.shard_index,"shard_count":args.shard_count}
        runner=evaluate_parallel if args.workers>1 else evaluate
        report=runner(args.checkpoint,args.opponents.split(","),args.games,args.output,args.split,args.seed,
                      args.device,args.learner_style,args.reference_teams,args.deterministic,**options)
        print(json.dumps(report,indent=2))
    elif args.command=="status":
        import torch
        checkpoints=sorted((ROOT/"runtime_data").rglob("latest.pt"),key=lambda p:p.stat().st_mtime,reverse=True)
        print(json.dumps({"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),
                          "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                          "latest_checkpoints":[str(p) for p in checkpoints[:5]]},indent=2))


if __name__=="__main__":
    main()
