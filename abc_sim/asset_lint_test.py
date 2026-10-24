"""Asset lint: every visual surface faces outward, so culling and two-sided renderers agree.

MJWarp culls back faces; the classic OpenGL renderer can draw both sides. An asset that is right
under one and wrong under the other is a bug: an inward-wound outline shell (tomato_2 rendered
black two-sided), or an object whose real surface is wound inward (the count prisms were
invisible culled; worn_eraser and desk_calculator hollow). See models/asset_fixes/README.md.

The inline-mesh and fix-manifest tests need only numpy. The installed-asset test needs mujoco and
an unpacked asset tree (``abc_sim/models/assets``, or ``ABC_SIM_ASSETS``) with the fixes applied
(``python prepare.py --sim-task ...`` applies them); it skips without one.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

MODELS = Path(__file__).resolve().parent / "models"
FIXES = MODELS / "asset_fixes"
ASSETS = Path(os.environ.get("ABC_SIM_ASSETS", MODELS / "assets"))

# Darker than this (max over RGB, untextured) counts as black.
BLACK_RGB_MAX = 0.05
# Share of outside-in ray hits on a visual geom that must land on front faces.
MIN_FRONT_FRACTION = 0.9
MIN_HITS = 20
# Visual meshes that legitimately show back faces from outside: inner liners and open shells
# whose inside is meant to be seen through an opening. Keyed by "<asset dir>:<mesh name>".
FRONT_FACE_ALLOWLIST: dict[str, str] = {
    # Single-sided plate surface: the underside is the same sheet seen from behind.
    "task_dishrack/plate/plate_10:model_normalized_0_vis": "open single-sided plate (front 0.51)",
    # One flipped component among many; shading-only difference, < 1 % pixels change with culling.
    "task_dishrack/dish_rack/dish_rack_3:DishRack028_vis": "partly flipped rack component (front 0.58)",
    # Open bottle shell whose inside is visible through the neck.
    "task_water_bottles/bottle/bottle_7:model_normalized_0_vis": "open bottle shell (front 0.87)",
}


def signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


def is_closed(faces: np.ndarray) -> bool:
    """Every directed edge has its reverse exactly once (a closed, consistently wound surface)."""
    edges: dict[tuple[int, int], int] = {}
    for a, b, c in faces.tolist():
        for edge in ((a, b), (b, c), (c, a)):
            edges[edge] = edges.get(edge, 0) + 1
    return all(count == 1 and edges.get((b, a)) == 1 for (a, b), count in edges.items())


def inline_meshes():
    for path in sorted(MODELS.glob("*.xml")):
        for mesh in ET.parse(path).getroot().iter("mesh"):
            if mesh.get("vertex") and mesh.get("face"):
                vertices = np.array(mesh.get("vertex").split(), dtype=float).reshape(-1, 3)
                faces = np.array(mesh.get("face").split(), dtype=int).reshape(-1, 3)
                yield f"{path.name}:{mesh.get('name')}", vertices, faces


def test_closed_inline_meshes_wind_outward():
    checked, inward = 0, []
    for label, vertices, faces in inline_meshes():
        if not is_closed(faces):
            continue  # open shells (chess squares) have no inside
        checked += 1
        if signed_volume(vertices, faces) <= 0:
            inward.append(label)
    assert checked >= 2  # the count prisms at least
    assert inward == [], f"closed inline meshes wound inward: {inward}"


def test_asset_fix_files_match_their_manifest():
    manifest = json.loads((FIXES / "fixes.json").read_text())
    assert manifest["format"] == "abc_sim_asset_fixes/v1"
    packages = {pkg["name"] for pkg in json.loads((MODELS / "assets_manifest.json").read_text())["packages"]}
    for fix in manifest["fixes"]:
        assert fix["package"] in packages
        assert fix["path"].split("/")[0] == fix["package"]
        data = (FIXES / "files" / fix["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == fix["fixed_sha256"], fix["path"]
        assert fix["original_sha256"] != fix["fixed_sha256"]


def fibonacci_sphere(count: int) -> np.ndarray:
    index = np.arange(count) + 0.5
    polar = np.arccos(1 - 2 * index / count)
    azimuth = np.pi * (1 + 5**0.5) * index
    return np.stack([np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)], -1)


def visual_mesh_report(model, label_prefix: str, rays: int = 600) -> list[dict]:
    """Per rendered visual mesh geom: share of outside-in ray hits that land on a front face.

    Rays start on a sphere around the rendered geoms and aim at random points inside their
    bounds; mj_ray reports the first geom hit and its surface normal, which for a mesh is the
    hit triangle's winding normal. Only geom groups 0-2 (the rendered ones) are considered.
    """
    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rendered = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
    geoms = []
    for geom in range(model.ngeom):
        material = int(model.geom_matid[geom])
        rgba = model.mat_rgba[material] if material >= 0 else model.geom_rgba[geom]
        if model.geom_group[geom] <= 2 and rgba[3] > 0:
            geoms.append(geom)
    if not geoms:
        return []
    low = np.min([data.geom_xpos[g] - model.geom_rbound[g] for g in geoms], axis=0)
    high = np.max([data.geom_xpos[g] + model.geom_rbound[g] for g in geoms], axis=0)
    centre, extent = (low + high) / 2, high - low
    rng = np.random.default_rng(0)
    origins = centre + fibonacci_sphere(rays) * (np.linalg.norm(extent) * 0.75 + 1e-3)
    targets = centre + (rng.random((rays, 3)) - 0.5) * extent * 0.8
    hits: dict[int, list[bool]] = {}
    geom_id, normal = np.zeros(1, dtype=np.int32), np.zeros(3)
    for origin, target in zip(origins, targets):
        direction = (target - origin) / np.linalg.norm(target - origin)
        if mujoco.mj_ray(model, data, origin, direction, rendered, 1, -1, geom_id, normal) < 0:
            continue
        hits.setdefault(int(geom_id[0]), []).append(float(normal @ direction) < 0)
    report = []
    for geom, facing in hits.items():
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        material = int(model.geom_matid[geom])
        textured = material >= 0 and bool((model.mat_texid[material] >= 0).any())
        rgba = model.mat_rgba[material] if material >= 0 else model.geom_rgba[geom]
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[geom]))
        report.append({"label": f"{label_prefix}:{mesh_name}", "hits": len(facing), "front": float(np.mean(facing)),
                       "black": not textured and float(max(rgba[:3])) < BLACK_RGB_MAX})
    return report


def test_installed_visual_meshes_face_outward():
    mujoco = pytest.importorskip("mujoco")
    models = sorted(ASSETS.rglob("model.xml")) if ASSETS.is_dir() else []
    if not models:
        pytest.skip(f"no unpacked assets under {ASSETS} (set ABC_SIM_ASSETS)")
    compiled, bad = 0, []
    for path in models:
        try:
            with contextlib.chdir(path.parent):
                model = mujoco.MjModel.from_xml_string(path.read_text())
        except ValueError:
            continue  # fragments that only compile inside their scene
        compiled += 1
        for row in visual_mesh_report(model, str(path.parent.relative_to(ASSETS))):
            if row["hits"] < MIN_HITS or row["front"] >= MIN_FRONT_FRACTION:
                continue
            if row["black"] or row["label"] not in FRONT_FACE_ALLOWLIST:
                bad.append(f"{row['label']} front={row['front']:.2f} hits={row['hits']} black={row['black']}")
    assert compiled > 100
    assert bad == [], "visual meshes seen from outside as back faces:\n" + "\n".join(bad)
