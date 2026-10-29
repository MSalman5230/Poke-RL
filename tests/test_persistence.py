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
