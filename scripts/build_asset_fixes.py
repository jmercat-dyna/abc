"""Regenerate abc_sim/models/asset_fixes/ from an installed, unfixed asset tree.

    python scripts/build_asset_fixes.py --assets abc_sim/models/assets

Each fix is a whole replacement file, derived here from the released original so the edit is
reviewable and reproducible. prepare.py overlays them after unpacking a package (see
``apply_asset_fixes``); the original's sha256 is recorded so a re-published package that already
carries the fix, or a different file, is detected instead of overwritten blindly.

Fixes:
- tomato_2 (task_conveyor_pick, task_put_relative, task_grab_clutter): drop the black
  "inverted hull" outline -- mesh ``model_normalized_0_vis`` (visual/model_normalized_0.obj),
  material ``material.121`` (rgba 0 0 0) and the visual geom using them. The shell is wound
  inward around the red body, so a culling renderer never drew it and a two-sided one draws the
  tomato black. Visual geoms are density 0 in every task that spawns tomatoes, so mass, inertia
  and contacts are unchanged. The OBJ stays in the package: recorded scene_assembled.xml files
  still reference it.
- worn_eraser, desk_calculator (task_multi_drawer_search): the visual OBJ is wound inward and
  its authored normals point inward. Reverse every face (v1 v2 v3 -> v1 v3 v2, keeping each
  corner's v/vt/vn together) and negate every ``vn``. Vertices and UVs are untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FIXES_DIR = Path(__file__).resolve().parent.parent / "abc_sim" / "models" / "asset_fixes"
FORMAT = "abc_sim_asset_fixes/v1"

TOMATO_2 = "objects/tomato/tomato_2/model.xml"
TOMATO_SHELL_LINES = ('name="model_normalized_0_vis"', 'name="material.121"', 'mesh="model_normalized_0_vis"')
INSIDE_OUT_OBJS = {
    "task_multi_drawer_search/objects/eraser/worn_eraser/visual/model.obj": "worn_eraser",
    "task_multi_drawer_search/objects/calculator/desk_calculator/visual/model.obj": "desk_calculator",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def drop_tomato_shell(text: str) -> str:
    kept = [line for line in text.splitlines(keepends=True) if not any(t in line for t in TOMATO_SHELL_LINES)]
    removed = len(text.splitlines()) - len(kept)
    if removed != 3:
        raise ValueError(f"expected to drop 3 lines (mesh, material, geom), dropped {removed}")
    return "".join(kept)


def negate(number: str) -> str:
    """Textual negation, so the parsed value is exactly -x (zeros stay as written)."""
    if float(number) == 0:
        return number
    return number[1:] if number.startswith("-") else "-" + number


def flip_obj(text: str) -> str:
    out = []
    for line in text.splitlines(keepends=True):
        parts = line.split()
        if parts and parts[0] == "f":
            corners = parts[1:]
            out.append("f " + " ".join([corners[0], *reversed(corners[1:])]) + "\n")
        elif parts and parts[0] == "vn":
            out.append("vn " + " ".join(negate(x) for x in parts[1:]) + "\n")
        else:
            out.append(line)
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assets", type=Path, required=True, help="installed abc_sim/models/assets tree")
    args = parser.parse_args()
    entries = []
    jobs = [(f"{pkg}/{TOMATO_2}", pkg, drop_tomato_shell, "Drop tomato_2's black inverted-hull outline shell (mesh, material.121, geom).")
            for pkg in ("task_conveyor_pick", "task_grab_clutter", "task_put_relative")]
    jobs += [(path, path.split("/")[0], flip_obj, f"Re-wind the inside-out {name} visual mesh outward and negate its normals.")
             for path, name in INSIDE_OUT_OBJS.items()]
    for path, package, edit, reason in sorted(jobs):
        original = (args.assets / path).read_bytes()
        fixed = edit(original.decode()).encode()
        target = FIXES_DIR / "files" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(fixed)
        entries.append({"path": path, "package": package, "original_sha256": sha256(original),
                        "fixed_sha256": sha256(fixed), "reason": reason})
    manifest = {"format": FORMAT, "root": "abc_sim/models/assets", "fixes": entries}
    (FIXES_DIR / "fixes.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(entries)} fixes to {FIXES_DIR}")


if __name__ == "__main__":
    main()
