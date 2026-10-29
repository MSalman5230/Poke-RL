"""Fixed monthly Smogon snapshots. All reads occur before matchmaking."""
import hashlib
import json
from pathlib import Path
import re
from opponents.autoplay_reference.bot.domain import catalog
from opponents.autoplay_reference.bot.runtime import config

FORMAT = "gen9championsbssregmc"
MONTH = "2026-09"


class Datasets:
    def __init__(self, root=None, *, verify_manifest=True):
        self.root = Path(root or Path(config.DATA) / "smogon" / MONTH)
        self.tables, self.leads, self.identities = {}, {}, {}
        manifest = Path(config.DATA) / "smogon_manifest.json"
        checks = json.loads(manifest.read_text()).get("files", {}) if verify_manifest and manifest.is_file() else {}
        for cutoff in (1500, 0):
            rel = f"chaos/{FORMAT}-{cutoff}.json"
            path = self.root / rel
            if not path.is_file():
                raise RuntimeError(f"Missing Smogon dataset {path}; run python tools/fetch_smogon.py")
            raw_bytes = path.read_bytes()
            sha = hashlib.sha256(raw_bytes).hexdigest()
            if rel in checks and checks[rel] != sha:
                raise ValueError(f"Smogon checksum mismatch: {path}")
            raw = json.loads(raw_bytes)
            if raw["info"]["metagame"] != FORMAT or int(raw["info"]["cutoff"]) != cutoff:
                raise ValueError(f"Wrong Smogon format/cutoff: {path}")
            self.identities[rel] = {"sha256": sha, "cutoff": cutoff, "month": MONTH, "path": str(path.resolve())}
            rows = {}
            for name, row in raw["data"].items():
                key = catalog.canonical(name)
                # Duplicate typography aliases must not replace complete data with an empty row.
                if sum(row.get("Abilities", {}).values()) >= sum(rows.get(key, {}).get("Abilities", {}).values()):
                    rows[key] = row
            self.tables[cutoff] = rows
            lead_path = self.root / "leads" / f"{FORMAT}-{cutoff}.txt"
            self.leads[cutoff] = {}
            if lead_path.is_file():
                content = lead_path.read_bytes()
                self.identities[f"leads/{lead_path.name}"] = {"sha256": hashlib.sha256(content).hexdigest()}
                for line in content.decode("utf-8").splitlines():
                    match = re.match(r"\s*\|\s*\d+\s*\|\s*(.+?)\s*\|\s*([\d.]+)%", line)
                    if match:
                        self.leads[cutoff][catalog.canonical(match[1])] = float(match[2])

    def entries(self, name):
        key = catalog.canonical(name)
        return [(cutoff, rows[key]) for cutoff, rows in self.tables.items() if key in rows]

    def ranked(self, name, trait):
        for cutoff, row in self.entries(name):
            values = [(key, weight) for key, weight in row.get(trait, {}).items()
                      if isinstance(weight, (int, float)) and weight > 0]
            total = sum(w for _, w in values)
            for key, weight in sorted(values, key=lambda pair: (-pair[1], pair[0])):
                yield key, {"source": "smogon", "cutoff": cutoff, "weight": weight, "share": weight / total,
                            "species": name, "assumption": "compatible usage marginal; not a joint set"}

    def lead(self, name):
        return next((self.leads[c].get(catalog.canonical(name)) for c in (1500, 0)
                     if self.leads[c].get(catalog.canonical(name), 0) > 0), 0)

    def family_lead(self, name):
        c = catalog.get()
        base = next((base for base, stones in c.stones.items() if name in stones.values()), name)
        family = list(dict.fromkeys([base, *c.stones.get(base, {}).values()]))
        for cutoff in (1500, 0):
            total = sum(self.leads[cutoff].get(catalog.canonical(form), 0) for form in family)
            if total > 0:
                return total
        return 0


_datasets = None


def datasets():
    global _datasets
    if _datasets is None:
        _datasets = Datasets()
    return _datasets
