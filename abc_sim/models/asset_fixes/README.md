# Asset fixes

Some files in the published asset tarballs (`assets_manifest.json`) are wrong. These tarballs live on the upstream data host and cannot be re-published from this fork, so the corrected files are tracked here instead. `prepare.py` overlays them right after unpacking a package, and it also applies them to packages that are already installed.

To fix an asset tree installed some other way (for example a shared copy), run:

```bash
python prepare.py --sim-fix-assets <assets dir>
```

`fixes.json` records each file's released `original_sha256` and its `fixed_sha256`. A file that matches neither is never overwritten. That case means the package was re-published or edited, and the fix must be regenerated.

To regenerate the fixes from an unfixed tree:

```bash
python scripts/build_asset_fixes.py --assets <assets dir>
```

| file | defect | fix |
|---|---|---|
| `task_{conveyor_pick,grab_clutter,put_relative}/objects/tomato/tomato_2/model.xml` | Visual mesh `visual/model_normalized_0.obj` is an inward-wound black outline shell around the red body (material `material.121`, rgba 0 0 0; signed volume −0.424 vs +0.420). Culling renderers never draw it. Two-sided rendering draws the tomato black. | Drop the mesh, the material and the geom. Visual geoms are density 0 where tomatoes spawn, so physics is unchanged. The OBJ stays, because recorded scenes reference it. |
| `task_multi_drawer_search/objects/{eraser/worn_eraser,calculator/desk_calculator}/visual/model.obj` | Wound inward, and the authored normals also point inward. With culling the object is hollow or dark; two-sided rendering lights it from the wrong side. | Reverse every face and negate every `vn`. Vertices, UVs and collision boxes are unchanged. |

The inline count prisms (`count_triangle_prism_*` in `yam_count_into_opaque_box_scene.xml`) had the same inside-out winding. They are fixed in that XML directly.

`abc_sim/asset_lint_test.py` keeps all of this from coming back.
