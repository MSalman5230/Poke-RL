import copy
import hashlib
import json
from pathlib import Path
import pytest
from champions_rl.mistake_data import verify_teacher


def test_teacher_corrections_require_matching_stronger_validation(tmp_path):
    settings={'seed':682917}
    for name,wins,interval in [('teacher',126,[.945,.996]),('student',110,[.789,.909])]:
        model=tmp_path/(name+'.pt');model.write_bytes(name.encode())
        report={'checkpoint_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),'complete':True,
                'deterministic':True,'reference_teams':'saved','seed':92341,
                'reference_manifest_sha256':'same-reference','knowledge_sha256':'same-knowledge',
                'results':{'reference':{'games':128,'wins':wins,'errors':0,'fallbacks':0,'truncations':0,'confidence':interval}}}
        source=tmp_path/(name+'.json');source.write_text(json.dumps(report))
        settings[name]=str(model);settings[name+'_validation']=str(source)
    assert 'Saved-roster' in verify_teacher(settings)['scope']
    report=json.loads(Path(settings['teacher_validation']).read_text())
    report['results']['reference']['confidence']=[.8,.95]
    Path(settings['teacher_validation']).write_text(json.dumps(report))
    with pytest.raises(ValueError,match='advantage'):verify_teacher(settings)
    Path(settings['teacher']).write_bytes(b'different weights')
    with pytest.raises(ValueError,match='weights'):verify_teacher(settings)


def test_failed_reference_game_is_durable_but_never_a_successful_shard(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from champions_rl import mistake_data as collector
    (tmp_path/'raw').mkdir()
    calls=[]
    def request(operation,**payload):
        calls.append(operation)
        if operation=='new':return {'ended':False,'truncated':False,'views':[{'legal':[0]},{'legal':[0]}]}
    def fail(_):raise RuntimeError('Reference deadline passed')
    monkeypatch.setattr(collector,'ReferenceBot',lambda *args,**kwargs:SimpleNamespace(choose=fail,last_diagnostics={'source':'heuristic_recovery'}))
    for name,value in {'_settings':{'seed':100,'output':str(tmp_path)},'_team':[],
                       '_engine':SimpleNamespace(request=request),'_student':SimpleNamespace(choose=lambda obs:0)}.items():
        monkeypatch.setattr(collector,name,value,raising=False)
    result=collector.game(0)
    assert result['positions']==0 and not result['loss'] and 'deadline' in result['error']
    assert not (tmp_path/'raw/loss_100_000000.json').exists()
    assert (tmp_path/'raw/loss_100_000000_error.json').exists()
    assert calls==['new','close']
