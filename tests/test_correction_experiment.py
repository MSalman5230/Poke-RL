import hashlib
import json
import numpy as np
import pytest
from tools import correction_experiment as experiment
from champions_rl.provenance import feature_signature
from champions_rl.opponent_labels import switch_offset


def test_experiment_is_one_fixed_budget_ablation_with_untouched_test(tmp_path):
    jobs=experiment.commands(tmp_path)
    training=[argv for name,argv in jobs if name.endswith('_train')]
    assert len(training)==2
    for argv in training:
        assert argv[argv.index('--steps')+1]=='400'
        assert argv[argv.index('--identity-dropout')+1]=='0.25'
    assert training[0][training[0].index('--correction-mass')+1]=='0'
    assert training[1][training[1].index('--correction-mass')+1]=='0.05'
    assert all('test_v7' not in argv and '--search' not in argv for _,argv in jobs)
    primary=dict(jobs)['head_to_head']
    assert primary[primary.index('--games')+1]=='2048'


def test_explicit_pause_blocks_every_training_command(tmp_path,monkeypatch):
    (tmp_path/'runtime_data').mkdir();(tmp_path/'runtime_data/STOP').write_text('paused')
    monkeypatch.setattr(experiment,'ROOT',tmp_path)
    with pytest.raises(RuntimeError,match='explicitly paused'):experiment.run(tmp_path/'experiment')


def test_dataset_preflight_catches_unrepresentable_opponent_targets(tmp_path):
    path=tmp_path/'example.npz'
    np.savez(path,labels=np.array([0]),mask=np.array([[True]]),
             enemy_labels=np.array([switch_offset()]),opponent_switch_mask=np.zeros((1,6),dtype=bool))
    path.with_suffix('.json').write_text(json.dumps({'feature_signature':feature_signature()}))
    with pytest.raises(ValueError,match='Masked opponent supervision'):experiment.preflight(tmp_path)
