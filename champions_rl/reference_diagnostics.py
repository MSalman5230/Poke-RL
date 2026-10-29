"""Failure evidence around the immutable reference transport; no policy changes."""
from functools import wraps
from pathlib import Path
import subprocess
import sys
import uuid
from .engine import ROOT
from .persistence import atomic_json


def install(transport=None,directory=None):
    if transport is None:
        from opponents.autoplay_reference.bot.simulator.bridge import Showdown
        transport=Showdown
    if getattr(transport,'_champions_failure_diagnostics',False):return
    directory=Path(directory or ROOT/'runtime_data/reference_failures')
    original_close=transport.close;original_request=transport.request

    @wraps(original_close)
    def close(self):
        process=self.process;error=sys.exc_info()[1]
        details=None
        if process is not None and error is not None:
            # EOF can reach the reader before Windows exposes the exit code.
            # The request has already failed; this wait cannot change a decision.
            if 'Simulator process exited' in str(error) and process.poll() is None:
                try:process.wait(timeout=.25)
                except subprocess.TimeoutExpired:pass
            details={'exit_code_before_close':process.poll(),'stderr':list(self.stderr),
                     'error_context':repr(error),'pid':process.pid}
        try:return original_close(self)
        finally:
            if details is not None:
                details['exit_code_after_close']=process.poll()
                self._champions_close_diagnostics=details

    @wraps(original_request)
    def request(self,op,**payload):
        self._champions_close_diagnostics=None
        try:return original_request(self,op,**payload)
        except Exception as error:
            path=directory/(uuid.uuid4().hex+'.json')
            details={'operation':op,'payload':payload,'error':repr(error),
                     'process':getattr(self,'_champions_close_diagnostics',None)}
            try:
                atomic_json(path,details)
                self.champions_last_failure={'path':str(path),'error':repr(error),'process':details['process']}
            except OSError as failure:
                self.champions_last_failure={'error':repr(error),'diagnostic_write_error':repr(failure)}
            raise
    transport.close=close;transport.request=request;transport._champions_failure_diagnostics=True
