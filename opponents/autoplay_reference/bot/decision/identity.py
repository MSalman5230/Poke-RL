"""Source identity of the isolated reference policy."""
import hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
def current():
    return {"implementation": "autoplay-offline-snapshot-v1", "files": {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "bot").rglob("*.py")) if "node_modules" not in p.parts}}
