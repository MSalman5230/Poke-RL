import json
import pytest
from pathlib import Path
from tools.recover_reference_evaluation import recover


@pytest.mark.parametrize('workers',[1,2])
def test_recovery_keeps_valid_losses_and_resumes_only_missing_or_invalid_games(tmp_path,monkeypatch,workers):
    source=tmp_path/'source';source.mkdir();output=tmp_path/'recovered'
    report={'results':{'reference':{}},'requested_games_per_opponent':4,'implementation_sha256':'old-code',
            'checkpoint_sha256':'weights','feature_signature':'features','knowledge_sha256':'facts',
            'team_manifest_sha256':'teams','reference_manifest_sha256':'reference','seed':7,
            'split':'validation','roster':'manifest','reference_teams':'saved','deterministic':True}
    def record(game,won):return {'game':game,'won':won,'winner':game%2 if won else 1-game%2,'truncated':False}
    original=[record(0,False),{'game':2,'error':'worker exited'},record(3,True)]
    (source/'report.json').write_text(json.dumps(report));(source/'evaluated_model.pt').write_bytes(b'weights')
    worker=source/'worker_0';worker.mkdir();(worker/'reference_games.json').write_text(json.dumps(original))
    calls=[]
    def run(command,**kwargs):
        game=int(command[command.index('--shard-index')+1]);calls.append(game)
        folder=Path(command[command.index('--output')+1]);folder.mkdir()
        (folder/'report.json').write_text(json.dumps({**report,'implementation_sha256':'new-code'}))
        (folder/'reference_games.json').write_text(json.dumps([record(game,True)]))
    monkeypatch.setattr('tools.recover_reference_evaluation.subprocess.run',run)
    recover(source,output,workers=workers)
    assert sorted(calls)==[1,2]
    final=json.loads((output/'reference_games.json').read_text())
    assert final[0]==original[0] and final[3]==original[2]
    assert json.loads((worker/'reference_games.json').read_text())==original
    assert json.loads((output/'report.json').read_text())['acceptance']['passed'] is False
    recover(source,output,workers=workers)
    assert sorted(calls)==[1,2]  # a second restart does not replay already recovered games
