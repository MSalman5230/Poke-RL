import pytest
from champions_rl import power


def test_power_request_restores_prior_state_after_nested_job_failure(monkeypatch):
    state=power.CONTINUOUS
    calls=[]
    def setter(flags):
        nonlocal state
        previous=state;state=flags;calls.append(flags)
        return previous
    monkeypatch.setattr(power,'_execution_state',setter)
    with pytest.raises(RuntimeError,match='job failed'):
        with power.keep_awake():
            with power.keep_awake():
                raise RuntimeError('job failed')
    assert calls==[power.CONTINUOUS|power.SYSTEM_REQUIRED]*3+[power.CONTINUOUS]
    assert state==power.CONTINUOUS


def test_failed_power_request_does_not_prevent_job_or_clear_other_requests(monkeypatch):
    calls=[]
    monkeypatch.setattr(power,'_execution_state',lambda flags:calls.append(flags) or 0)
    with pytest.warns(RuntimeWarning,match='keep-awake'):
        with power.keep_awake():
            assert len(calls)==1
    assert calls==[power.CONTINUOUS|power.SYSTEM_REQUIRED]
