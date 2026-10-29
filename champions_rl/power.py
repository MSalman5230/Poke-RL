"""Temporary Windows power requests for active jobs; no power-plan changes."""
from contextlib import contextmanager
from functools import lru_cache
import os
import time
import warnings

CONTINUOUS=0x80000000
SYSTEM_REQUIRED=0x00000001


@lru_cache(maxsize=1)
def _kernel():
    if os.name!='nt':return None
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.SetThreadExecutionState.argtypes=[wintypes.DWORD]
    kernel.SetThreadExecutionState.restype=wintypes.DWORD
    kernel.QueryUnbiasedInterruptTime.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)]
    kernel.QueryUnbiasedInterruptTime.restype=wintypes.BOOL
    return kernel


def _execution_state(flags):
    kernel=_kernel()
    return kernel.SetThreadExecutionState(flags) if kernel else None


@contextmanager
def keep_awake():
    # A thread-scoped request permits display sleep and explicit user sleep.
    # Restore the caller's state, including when jobs fail or contexts nest.
    previous=_execution_state(CONTINUOUS|SYSTEM_REQUIRED)
    if previous==0:warnings.warn('Windows rejected the temporary keep-awake request',RuntimeWarning)
    try:
        yield
    finally:
        if previous:_execution_state(previous|CONTINUOUS)


def awake_seconds():
    """Elapsed system-awake time, excluding Windows sleep and hibernation."""
    kernel=_kernel()
    if kernel is None:return time.monotonic()
    import ctypes
    ticks=ctypes.c_ulonglong()
    if not kernel.QueryUnbiasedInterruptTime(ctypes.byref(ticks)):
        raise ctypes.WinError(ctypes.get_last_error())
    return ticks.value/10000000
