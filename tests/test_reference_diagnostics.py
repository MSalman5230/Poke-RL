from collections import deque
import json
import pytest
from champions_rl.reference_diagnostics import install


def test_diagnostics_preserve_reference_result_and_failure(tmp_path):
    class Process:
        pid=123
        def poll(self):return -1073741819
    class Transport:
        def __init__(self):self.process=Process();self.stderr=deque(['native diagnostic'])
        def close(self):self.process=None
        def request(self,op,**payload):
            if op=='valid':return payload
            try:raise RuntimeError('Simulator process exited')
            except RuntimeError:
                self.close()
                raise
    install(Transport,tmp_path)
    client=Transport()
    assert client.request('valid',trials=32)=={'trials':32}
    with pytest.raises(RuntimeError,match='Simulator process exited'):
        client.request('probe',root={'public':'state'},seeds=[1,2])
    report=json.loads(next(tmp_path.glob('*.json')).read_text())
    assert report['process']['exit_code_before_close']==-1073741819
    assert report['payload']['seeds']==[1,2] and report['operation']=='probe'
    assert report['process']['stderr']==['native diagnostic']
