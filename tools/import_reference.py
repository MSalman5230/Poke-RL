"""Snapshot the existing bot's strategy only. Never modifies the source project."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    src = args.source.resolve()
    dst = ROOT / "opponents/autoplay_reference"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "__init__.py").write_text('"""Frozen strategy snapshot with programmatic observation adapters."""\n')
    files = [Path("bot/__init__.py"), Path("bot/runtime/__init__.py"),
             Path("bot/runtime/budget.py"), Path("bot/runtime/context.py")]
    for folder in ("decision", "domain", "opponent", "team_selection", "simulator"):
        files.extend(p.relative_to(src) for p in (src / "bot" / folder).rglob("*.py") if "node_modules" not in p.parts)
    files.extend(Path("bot/simulator/node") / name for name in
                 ("worker.cjs", "preview.cjs", "export_catalog.cjs", "package.json", "package-lock.json"))
    files.extend(Path("data") / name for name in
                 ("catalog.json", "display_names.json", "team.json", "smogon_manifest.json", "champions_moves.json"))
    for cutoff in (0, 1500):
        files.extend([Path(f"data/smogon/2026-09/chaos/gen9championsbssregmc-{cutoff}.json"),
                      Path(f"data/smogon/2026-09/leads/gen9championsbssregmc-{cutoff}.txt")])
    files.extend(p.relative_to(src) for p in (src / "docs/rules").glob("*.md"))
    manifest = {"source": str(src), "source_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=src, text=True).strip(), "files": {}}
    for rel in files:
        original = (src / rel).read_bytes()
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel.suffix == ".py":
            content = original.decode("utf-8-sig").replace("from bot.", "from opponents.autoplay_reference.bot.")
            content = content.replace("from bot import", "from opponents.autoplay_reference.bot import")
            content = content.replace(".bot.perception", ".bot.state_support")
            target.write_text(content, encoding="utf-8")
        else:
            target.write_bytes(original)
        manifest["files"][rel.as_posix()] = {"original_sha256": hashlib.sha256(original).hexdigest(),
            "snapshot_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    # Only pure state conversion helpers are retained. No OCR, images, tablet,
    # device control, caption parsers, live databases or original runtime loop.
    support = dst / "bot/state_support"
    support.mkdir(exist_ok=True)
    (support / "__init__.py").write_text("")
    (support / "mechanics.py").write_text('CHOICE_ITEMS = {"Choice Band", "Choice Specs", "Choice Scarf"}\n')
    tree = ast.parse((src / "bot/perception/tracker.py").read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("remaining_pp", "volatiles")]
    (support / "tracker.py").write_text(ast.unparse(ast.Module(body=functions, type_ignores=[])) + "\n", encoding="utf-8")
    tree = ast.parse((src / "bot/perception/field.py").read_text(encoding="utf-8"))
    field = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Field")
    field.body = [node for node in field.body if isinstance(node, ast.FunctionDef) and node.name in ("__init__", "predicted_duration")]
    (support / "field.py").write_text('from opponents.autoplay_reference.bot.domain import catalog\n\n' + ast.unparse(field) + "\n", encoding="utf-8")
    (dst / "bot/runtime/config.py").write_text('''"""Offline snapshot paths; no device configuration."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RUNTIME_DATA = ROOT.parents[1] / "runtime_data/reference"
def runtime_path(*parts):
    return str(RUNTIME_DATA.joinpath(*parts))
''', encoding="utf-8")
    (dst / "bot/decision/identity.py").write_text('''"""Source identity of the isolated reference policy."""
import hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
def current():
    return {"implementation": "autoplay-offline-snapshot-v1", "files": {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "bot").rglob("*.py")) if "node_modules" not in p.parts}}
''', encoding="utf-8")
    manifest["adaptations"] = ["Python imports namespaced", "Offline paths", "Source-identity scope",
        "Only remaining_pp/volatiles and Field constructor/duration helpers extracted; parsing removed"]
    for relative,record in manifest["files"].items():
        record["snapshot_sha256"]=hashlib.sha256((dst/relative).read_bytes()).hexdigest()
    generated=["__init__.py","bot/runtime/config.py",*[p.relative_to(dst).as_posix() for p in support.glob("*.py")]]
    manifest["generated_files"]={r:hashlib.sha256((dst/r).read_bytes()).hexdigest() for r in generated}
    (dst / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    engine = ROOT / "simulator"
    engine.mkdir(exist_ok=True)
    for name in ("package.json", "package-lock.json"):
        shutil.copy2(src / "bot/simulator/node" / name, engine / name)
    print(f"Imported {len(files)} strategy/data files into {dst}")


if __name__ == "__main__":
    main()
