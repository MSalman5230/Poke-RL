"""Fingerprint the immutable facts and set priors that define model inputs."""
from functools import lru_cache
import hashlib
from .engine import ROOT


@lru_cache(maxsize=1)
def feature_signature():
    from .features import FEATURE_VERSION
    digest=hashlib.sha256(str(FEATURE_VERSION).encode())
    # Cached vectors depend on mechanics and public-state parsing as well as
    # their dimensions. A future semantic fix must invalidate them automatically.
    for relative in ('data/knowledge.json','data/sets.json','champions_rl/features.py','champions_rl/knowledge.py',
                     'champions_rl/beliefs.py','champions_rl/tactics.py','simulator/tactics.cjs','simulator/worker.cjs'):
        digest.update(relative.encode())
        digest.update((ROOT/relative).read_bytes())
    return digest.hexdigest()
