from pathlib import Path
import json
from champions_rl import persistence


def test_atomic_report_survives_a_transient_windows_reader_lock(tmp_path,monkeypatch):
    path=tmp_path/'report.json'
    persistence.atomic_json(path,{'generation':1})
    original=Path.replace
    attempts=[]
    def transient_lock(self,target):
        attempts.append(target)
        if len(attempts)<3:
            assert json.loads(path.read_text())=={'generation':1}
            raise PermissionError('temporary reader without delete sharing')
        return original(self,target)
    monkeypatch.setattr(Path,'replace',transient_lock)
    monkeypatch.setattr(persistence.time,'sleep',lambda _:None)
    persistence.atomic_json(path,{'generation':2})
    assert len(attempts)==3 and json.loads(path.read_text())=={'generation':2}


def test_validation_queue_recovers_interrupted_work_without_erasing_reports(tmp_path,monkeypatch):
    from champions_rl import validation_queue as module
    previous=tmp_path/'step_000000100';previous.mkdir()
    (previous/'partial.json').write_text('{"completed":12}')
    job={'checkpoint':'actor.pt','step':100,'output':str(previous),'pid':98765}
    persistence.atomic_json(tmp_path/'queue.json',{'pending':[{'checkpoint':'next.pt','step':200}],
                                                  'active':job,'finished':[]})
    monkeypatch.setattr(module,'process_running',lambda _:False)
    queue=module.ValidationQueue(tmp_path,tmp_path/'initial.pt')
    assert [j['step'] for j in queue.pending]==[100,200]
    assert queue.pending[0]['attempt']==1 and (previous/'partial.json').exists()
    queue.schedule('actor.pt',100)
    assert len(queue.pending)==2  # an already scheduled checkpoint is never duplicated


def test_validation_queue_adopts_surviving_process_and_recovers_completion(tmp_path,monkeypatch):
    from champions_rl import validation_queue as module
    previous=tmp_path/'step_000000100';previous.mkdir()
    job={'checkpoint':'actor.pt','step':100,'output':str(previous),'pid':98765}
    persistence.atomic_json(tmp_path/'queue.json',{'pending':[],'active':job,'finished':[]})
    monkeypatch.setattr(module,'process_running',lambda _:True)
    queue=module.ValidationQueue(tmp_path,tmp_path/'initial.pt')
    assert queue.active and not queue.pending
    queue.poll()
    monkeypatch.setattr(module,'process_running',lambda _:False)
    persistence.atomic_json(previous/'summary.json',{'complete':True})
    queue.poll()
    assert not queue.active and queue.finished[0]['exit_code']==0
    restored=module.ValidationQueue(tmp_path,tmp_path/'initial.pt')
    assert restored.finished==queue.finished and not restored.pending


def test_validation_process_liveness_detects_current_process():
    import os
    from champions_rl.validation_queue import process_running
    assert process_running(os.getpid())
