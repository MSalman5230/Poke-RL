"""Offline snapshot paths; no device configuration."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RUNTIME_DATA = ROOT.parents[1] / "runtime_data/reference"
def runtime_path(*parts):
    return str(RUNTIME_DATA.joinpath(*parts))
