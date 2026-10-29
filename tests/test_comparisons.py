import json
import pytest
from tools.compare_evaluations import compare


def test_comparison_preserves_seed_blocks_and_rejects_missing_seats(tmp_path):
    paths=[tmp_path/'baseline',tmp_path/'candidate']
    for index,path in enumerate(paths):
        path.mkdir()
        report={'complete':True,'requested_games_per_opponent':4,'checkpoint_sha256':str(index),
                'implementation_sha256':'same','feature_signature':'same','deterministic':True,
                'results':{'balanced':{'games':4,'wins':index*2,'errors':0,'fallbacks':0,'truncations':0}}}
        records=[{'game':game,'team_ids':['a','b'],'seat':game%2,'won':bool(index and game<2)} for game in range(4)]
        (path/'report.json').write_text(json.dumps(report))
        (path/'balanced_games.json').write_text(json.dumps(records))
    result=compare(*paths)
    assert result['aggregate']['paired_seed_blocks']==2
    assert result['aggregate']['candidate_minus_baseline']==.5
    (paths[1]/'balanced_games.json').write_text(json.dumps(records[:-1]))
    with pytest.raises(ValueError,match='Incomplete paired'):compare(*paths)
