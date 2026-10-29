"""Verify the isolated reference before accepting it as a benchmark opponent."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from .engine import ROOT


@lru_cache(maxsize=4)
def _verify(folder,signature):
    root=Path(folder)
    manifest=json.loads((root/"SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    expected={path:row["snapshot_sha256"] for path,row in manifest["files"].items()}
    expected.update(manifest.get("generated_files",{}))
    for relative,digest in expected.items():
        path=root/relative
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f"Frozen reference changed: {relative}. Restore the snapshot before benchmarking.")
    return hashlib.sha256((root/"SOURCE_MANIFEST.json").read_bytes()).hexdigest()


def verify_reference(root=None):
    root=Path(root or ROOT/"opponents/autoplay_reference")
    manifest=json.loads((root/"SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    paths=[root/"SOURCE_MANIFEST.json",*[root/r for r in manifest["files"]],*[root/r for r in manifest.get("generated_files",{})]]
    signature=tuple((str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in paths)
    return _verify(str(root),signature)
