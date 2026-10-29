"""Held-out evaluation and an explicit, conservative 60% acceptance gate."""
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import random
import time
from .arena import load_teams, play_match
from .engine import Engine, ROOT
from .features import Encoder
from .knowledge import knowledge
from .model import BattleModel, NeuralBot
from .policies import ScriptedBot
from .reference import ReferenceBot
from .teams import TeamPool, fingerprint
from .training import emit, seed_all
from .persistence import atomic_json
from .provenance import feature_signature


def wilson(wins, total, z=1.96):
    if not total:
        return [0.,1.]
    p=wins/total
    denominator=1+z*z/total
    center=(p+z*z/(2*total))/denominator
    margin=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [max(0.,center-margin),min(1.,center+margin)]


def acceptance(rows, split):
    reference=rows.get("reference",{})
    other=[r for n,r in rows.items() if n!="reference"]
    requirements={
        "held_out_test_split":split=="test",
        "at_least_200_reference_games":reference.get("games",0)>=200,
        "reference_win_rate_at_least_60_percent":reference.get("win_rate",0)>=.60,
        "reference_95_percent_lower_bound_at_least_60_percent":reference.get("confidence",[0])[0]>=.60,
        "at_least_four_other_styles":len(other)>=4,
        "at_least_200_games_per_other_style":bool(other) and all(r.get("games",0)>=200 for r in other),
        "other_styles_at_least_55_percent":bool(other) and all(r.get("win_rate",0)>=.55 for r in other),
        "no_reference_fallbacks":reference.get("fallbacks",1)==0,
        "no_failed_games":all(r.get("errors",0)==0 for r in rows.values()),
        "truncation_rate_at_most_one_percent":bool(rows) and all(r.get("truncations",0)/max(1,r.get("games",0))<=.01 for r in rows.values()),
    }
    return {"passed":all(requirements.values()),"requirements":requirements,
            "jaxcalibur_comparison":"Not measured: different format; no public executable/weights supplied."}


def evaluate(checkpoint=None, opponents=("damage","balanced","setup","defensive"),games=100,
             output=None, split="validation",seed=92341,device="cpu",learner_style=None,reference_teams="saved",deterministic=False,
             shard_index=0,shard_count=1,search=False):
    if games<1 or not 0<=shard_index<shard_count:
        raise ValueError("Invalid game count or shard range")
    if len(set(opponents))!=len(opponents):
        raise ValueError("Opponent styles must be distinct")
    if search and (not checkpoint or not deterministic):raise ValueError('Search requires a checkpoint and deterministic evaluation')
    seed_all(seed)
    import torch
    torch.set_num_threads(1)  # inference uses single positions; reserve CPU for simulators
    encoder=Encoder()
    checkpoint_bytes=Path(checkpoint).read_bytes() if checkpoint else None
    if checkpoint:
        model,metadata=BattleModel.load(io.BytesIO(checkpoint_bytes),device)
        learner=NeuralBot(model,encoder,deterministic=deterministic)
        if search:
            from .search import SearchBot
            learner=SearchBot(model,encoder,seed=seed)
    else:
        learner=ScriptedBot(learner_style or "balanced",seed,encoder)
        metadata={}
    teams=load_teams(split)
    reference_team=TeamPool().data["reference_team"]
    output=Path(output or ROOT/"runtime_data/evaluation"/time.strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True,exist_ok=True)
    if checkpoint_bytes:
        (output/"evaluated_model.pt").write_bytes(checkpoint_bytes)
    code_paths=sorted((ROOT/"champions_rl").glob("*.py"))+sorted((ROOT/"simulator").glob("*.cjs"))
    code_digest=hashlib.sha256()
    for path in code_paths:
        code_digest.update(path.relative_to(ROOT).as_posix().encode())
        code_digest.update(path.read_bytes())
        snapshot=output/"code_snapshot"/path.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
    report={"format":knowledge().data["format"],"knowledge_sha256":knowledge().sha256,
            'feature_signature':feature_signature(),
            "checkpoint_metadata":metadata,
            "model_config":model.config if checkpoint else None,
            "checkpoint":str(checkpoint) if checkpoint else None,
            "checkpoint_sha256":hashlib.sha256(checkpoint_bytes).hexdigest() if checkpoint_bytes else None,
            "implementation_sha256":code_digest.hexdigest(),
            "team_manifest_sha256":hashlib.sha256((ROOT/f"data/teams_{split}.json").read_bytes()).hexdigest(),
            "reference_manifest_sha256":hashlib.sha256((ROOT/"opponents/autoplay_reference/SOURCE_MANIFEST.json").read_bytes()).hexdigest(),
            "split":split,"seed":seed,"reference_teams":reference_teams,"deterministic":deterministic,
            'search':search,
            "shard_index":shard_index,"shard_count":shard_count,"requested_games_per_opponent":games,
            "reference_policy":"Frozen 32-trial original scoring/prediction; public-state adapter",
            "results":{}}
    with Engine() as engine:
        for style in opponents:
            rng=random.Random(seed)  # identical composition pairs across styles
            counts=Counter()
            records=[]
            pair=None
            for game in range(games):
                if game%2==0:
                    pair=[rng.choice(teams),rng.choice(teams)]
                    if style=="reference" and reference_teams=="saved":
                        # Identical original six on both sides isolates policy strength
                        # from team-building/counter-team advantages.
                        pair=[reference_team,reference_team]
                if game%shard_count!=shard_index:
                    continue
                # Independent per-game RNG streams make worker count irrelevant.
                torch.manual_seed(seed+104729*game)
                if search:learner.rng.seed(seed+104729*game)
                before_search=dict(learner.statistics) if search else {}
                seat=game%2
                # Hold the teams/engine seed fixed while swapping the policies.
                # Reversing teams here as well would keep the learner on the
                # same roster twice and fail to cancel team-strength advantage.
                party=pair
                opponent=ReferenceBot(seed+game,strict=True) if style=="reference" else ScriptedBot(style,seed+game,encoder)
                bots=[opponent,opponent]
                bots[seat]=learner
                t0=time.monotonic()
                try:
                    last=[t0]
                    def progress(row):
                        if time.monotonic()-last[0]>15:
                            emit("evaluation_progress",opponent=style,game=game+1,turn=row["turn"],seconds=round(time.monotonic()-t0,1))
                            last[0]=time.monotonic()
                    result=play_match(engine,party,bots,seed+game//2,progress=progress)
                    win=result["winner"]==seat
                    counts["wins" if win else "draws" if result["winner"] is None else "losses"]+=1
                    counts["truncations"]+=result["truncated"]
                    counts["fallbacks"]+=getattr(opponent,"fallbacks",0)
                    record={k:v for k,v in result.items() if k!="trace"}
                    record.update(game=game,seat=seat,won=win,team_ids=list(map(fingerprint,party)))
                    if search:record['search']={key:value-before_search.get(key,0) for key,value in learner.statistics.items()}
                    if style=="reference":
                        record["reference_decisions"]=opponent.audit
                    if game<4 or result["truncated"]:
                        (output/f"{style}_{game:04d}.log").write_text("\n".join(result["trace"])+"\n",encoding="utf-8")
                except Exception as exc:
                    counts["errors"]+=1
                    record={"game":game,"error":repr(exc),"reference_diagnostics":getattr(opponent,"last_diagnostics",{})}
                    # Invalid games are retained in denominator and cannot certify.
                    emit("evaluation_error",opponent=style,**record)
                    if style=="reference":
                        counts["fallbacks"]+=getattr(opponent,"fallbacks",0)
                records.append(record)
                n=len(records)
                summary={"games":n,"wins":counts["wins"],"losses":counts["losses"],"draws":counts["draws"],
                         "errors":counts["errors"],"truncations":counts["truncations"],"fallbacks":counts["fallbacks"],
                         "win_rate":counts["wins"]/n,"confidence":wilson(counts["wins"],n)}
                if search:
                    combined=Counter()
                    for record in records:combined.update(record.get('search',{}))
                    summary['search']=dict(combined)
                report["results"][style]=summary
                report["acceptance"]=acceptance(report["results"],split)
                atomic_json(output/'report.json',report)
                atomic_json(output/f'{style}_games.json',records)
                emit("evaluation_game",opponent=style,**summary)
                if counts["errors"]>=3:
                    emit("evaluation_aborted",opponent=style,reason="Three invalid games; fix adapter before benchmarking")
                    break
    return report


def evaluate_parallel(checkpoint=None, opponents=("reference","damage","balanced","setup","defensive","stochastic"),
                      games=200, output=None,split="validation",seed=92341,device="cpu",learner_style=None,
                      reference_teams="saved",deterministic=False,workers=4,search=False):
    """Separate processes preserve reference-policy isolation and exact trials.

    Each worker takes a disjoint subset of the same paired schedule. Reports
    are merged by game ID, never by averaging shard percentages. Checkpoint,
    implementation, team and reference hashes must agree before certification.
    """
    import subprocess
    import sys
    if workers<1 or workers>min(8,games):
        raise ValueError("Use 1 to min(8, games) evaluation workers")
    output=Path(output or ROOT/"runtime_data/evaluation"/time.strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True,exist_ok=True)
    frozen=None
    if checkpoint:
        frozen=output/"evaluated_model.pt"
        frozen.write_bytes(Path(checkpoint).read_bytes())
    jobs=[]
    try:
        for index in range(workers):
            folder=output/f"worker_{index}"
            command=[sys.executable,"-u","-m","champions_rl","evaluate","--games",str(games),
                     "--output",str(folder),"--split",split,"--seed",str(seed),"--device",device,
                     "--opponents",",".join(opponents),"--reference-teams",reference_teams,
                     "--shard-index",str(index),"--shard-count",str(workers)]
            if frozen:
                command.extend(["--checkpoint",str(frozen)])
            if learner_style:
                command.extend(["--learner-style",learner_style])
            if deterministic:
                command.append("--deterministic")
            if search:command.append('--search')
            log=(output/f"worker_{index}.log").open("w",encoding="utf-8")
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            jobs.append((process,log,folder))
        last_progress=0
        while True:
            reports=[]
            for process,log,folder in jobs:
                report_file=folder/"report.json"
                if report_file.exists():
                    try:
                        reports.append(json.loads(report_file.read_text(encoding="utf-8")))
                    except (json.JSONDecodeError,PermissionError):
                        pass  # next poll will read the completed write
                if process.poll() not in (None,0):
                    raise RuntimeError(f"Evaluation worker failed; see {folder.parent/(folder.name+'.log')}")
            if reports:
                merged=dict(reports[0])
                for key in ("checkpoint_sha256","implementation_sha256","knowledge_sha256","feature_signature","team_manifest_sha256","reference_manifest_sha256"):
                    if len({r.get(key) for r in reports})!=1:
                        raise ValueError(f"Incompatible evaluation shards: {key}")
                merged.update(checkpoint=str(checkpoint) if checkpoint else None,workers=workers,
                              shard_index=None,shard_count=workers,complete=False)
                merged["results"]={}
                for style in opponents:
                    rows=[r["results"][style] for r in reports if style in r["results"]]
                    if not rows:
                        continue
                    row={key:sum(r[key] for r in rows) for key in ("games","wins","losses","draws","errors","truncations","fallbacks")}
                    row.update(win_rate=row["wins"]/row["games"],confidence=wilson(row["wins"],row["games"]))
                    if search:
                        combined=Counter()
                        for r in rows:combined.update(r.get('search',{}))
                        row['search']=dict(combined)
                    merged["results"][style]=row
                merged["complete"]=len(reports)==workers and all(p.poll()==0 for p,_,_ in jobs)
                merged["acceptance"]=acceptance(merged["results"],split)
                if not merged["complete"]:
                    merged["acceptance"]["passed"]=False
                atomic_json(output/'report.json',merged)
                if time.monotonic()-last_progress>15:
                    emit("parallel_evaluation_progress",workers=workers,
                         completed_games={s:r["games"] for s,r in merged["results"].items()})
                    last_progress=time.monotonic()
                if merged["complete"]:
                    for style in opponents:
                        records=[]
                        for _,_,folder in jobs:
                            file=folder/f"{style}_games.json"
                            if file.exists():
                                records.extend(json.loads(file.read_text(encoding="utf-8")))
                        records.sort(key=lambda r:r["game"])
                        if len({r["game"] for r in records})!=len(records):
                            raise ValueError("Duplicate game IDs across evaluation shards")
                        atomic_json(output/f'{style}_games.json',records)
                    return merged
            if all(p.poll()==0 for p,_,_ in jobs) and len(reports)!=workers:
                raise RuntimeError("Evaluation worker exited without a readable report")
            time.sleep(2)
    finally:
        for process,log,_ in jobs:
            if process.poll() is None:
                import os
                if os.name=="nt":
                    subprocess.run(["taskkill","/PID",str(process.pid),"/T","/F"],stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    process.terminate()
                process.wait(timeout=10)
            log.close()
