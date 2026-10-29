"""Fingerprint the immutable facts and set priors that define model inputs."""
from functools import lru_cache
import hashlib
from .engine import ROOT


@lru_cache(maxsize=1)
def feature_signature():
    from .features import FEATURE_VERSION
    digest=hashlib.sha256(str(FEATURE_VERSION).encode())
    for relative in ('data/knowledge.json','data/sets.json'):
        digest.update(relative.encode())
        digest.update((ROOT/relative).read_bytes())
    return digest.hexdigest()
