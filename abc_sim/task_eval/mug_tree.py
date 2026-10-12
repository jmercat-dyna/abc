"""Mug-on-rack task evaluator."""

from __future__ import annotations

import copy
import re
from typing import Any

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_specs import SimTaskSpec


class MugOnRackEvaluator:
    """Require released mugs supported by rack pegs through their handle loops.

    Collision geometry determines the vessel envelope and handle opening;
    center proximity alone is only a diagnostic. A threaded peg must bear
    upward load, and table, gripper, mug-stack or trunk/base contacts reject
    hanging. Live and qpos-batch paths share the same contact/geometry checks.
    """

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        tree_body_name: str = "mug_tree",
        mug_joint_regex: str = r"mug_(\d+)_jnt",
        hang_radius_m: float = 0.115,
        base_clearance_m: float = 0.005,
        top_clearance_m: float = 0.05,
        support_weight_fraction: float = 0.5,
        bottom_slice_fraction: float = 0.1,
        geometry_tolerance_m: float = 1e-5,
        max_penetration_m: float = 0.002,
    ) -> None:
        self.spec = spec
        self._tree_body_name = tree_body_name
        self._mug_joint_regex = mug_joint_regex
        self.hang_radius_m = float(hang_radius_m)
        self.base_clearance_m = float(base_clearance_m)
        self.top_clearance_m = float(top_clearance_m)
        self.support_weight_fraction = float(support_weight_fraction)
        self.bottom_slice_fraction = float(bottom_slice_fraction)
        self.geometry_tolerance_m = float(geometry_tolerance_m)
        self.max_penetration_m = float(max_penetration_m)
        if (
            not np.isfinite(self.support_weight_fraction)
            or not 0 < self.support_weight_fraction <= 1
        ):
            raise ValueError("support_weight_fraction must be finite in (0, 1]")
        if (
            not np.isfinite(self.bottom_slice_fraction)
            or not 0 < self.bottom_slice_fraction < 1
        ):
            raise ValueError("bottom_slice_fraction must be finite in (0, 1)")
        if not np.isfinite(self.geometry_tolerance_m) or self.geometry_tolerance_m <= 0:
            raise ValueError("geometry_tolerance_m must be positive and finite")
        if not np.isfinite(self.max_penetration_m) or self.max_penetration_m <= 0:
            raise ValueError("max_penetration_m must be positive and finite")
        self._bind_model(model)
        self.reset(nworld=1)

    def _bind_model(self, model: mujoco.MjModel) -> None:
        self.model = model
        self._qpos_model: mujoco.MjModel | None = None
        self._qpos_scratch_model: mujoco.MjModel | None = None
        self._qpos_scratch_data: mujoco.MjData | None = None
        self.mug_names, self._mug_qpos_addrs = self._resolve_mug_qpos_addrs(
            model, self._mug_joint_regex
        )
        self._tree_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self._tree_body_name
        )
        if self._tree_body_id < 0:
            raise ValueError(f"Body {self._tree_body_name!r} not found in model")
        if int(model.body_jntnum[self._tree_body_id]) != 0:
            raise ValueError(
                f"Body {self._tree_body_name!r} must be a static body; a jointed "
                "tree would need its pose read from qpos"
            )
        # Collision-mesh vertices in the tree's own frame, measured once; the
        # world pose composes in per reset.
        self._tree_local_vertices = self._collect_local_vertices(
            model, self._tree_body_id
        )
        self._measure_tree(model)
        self._bind_support_geometry(model)

    def _measure_tree(self, model: mujoco.MjModel) -> None:
        """Compose the tree's current world pose into the hang band."""

        pos = np.asarray(model.body_pos[self._tree_body_id], dtype=np.float64)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.body_quat[self._tree_body_id])
        world = self._tree_local_vertices @ rot.reshape(3, 3).T + pos
        self._axis_xy = world[:, :2].mean(axis=0)
        base_z = float(world[:, 2].min())
        top_z = float(world[:, 2].max())
        self._hang_z_lo = base_z + self.base_clearance_m
        self._hang_z_hi = top_z + self.top_clearance_m
        if self._hang_z_lo >= self._hang_z_hi:
            raise ValueError(
                f"Tree {self._tree_body_name!r} measured an empty hang band"
            )

    @staticmethod
    def _collect_local_vertices(model: mujoco.MjModel, body_id: int) -> np.ndarray:
        vertices = []
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) != body_id:
                continue
            if int(model.geom_group[geom_id]) != 3:
                continue
            mesh_id = int(model.geom_dataid[geom_id])
            if mesh_id < 0:
                continue
            start = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            mesh_vertices = np.asarray(
                model.mesh_vert[start : start + count], dtype=np.float64
            )
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, model.geom_quat[geom_id])
            vertices.append(
                mesh_vertices @ rot.reshape(3, 3).T
                + np.asarray(model.geom_pos[geom_id], dtype=np.float64)
            )
        if not vertices:
            raise ValueError(
                f"Body id {body_id} has no collision-mesh vertices to measure"
            )
        return np.concatenate(vertices, axis=0)

    @staticmethod
    def _resolve_mug_qpos_addrs(
        model: mujoco.MjModel,
        mug_joint_regex: str,
    ) -> tuple[list[str], np.ndarray]:
        pattern = re.compile(mug_joint_regex)
        entries: list[tuple[int, str, int]] = []
        for joint_id in range(model.njnt):
            name = model.jnt(joint_id).name
            if not name:
                continue
            match = pattern.fullmatch(name)
            if match is None:
                continue
            if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError(f"Joint {name!r} must be a free joint")
            entries.append(
                (
                    int(match.group(1)),
                    name.removesuffix("_jnt"),
                    int(model.jnt_qposadr[joint_id]),
                )
            )
        if not entries:
            raise ValueError(f"No joints matching {mug_joint_regex!r} found in model")
        entries.sort(key=lambda item: item[0])
        names = [name for _, name, _ in entries]
        addrs = np.asarray([addr for _, _, addr in entries], dtype=np.int32)
        return names, addrs

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._max_hanging = np.zeros((self._nworld,), dtype=np.int32)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)

    def configure_from_randomization(
        self,
        model: mujoco.MjModel,
        randomization: object,
    ) -> None:
        """Re-measure the tree: its static pose is edited in place per reset."""

        if model is not self.model:
            self._bind_model(model)
            self.reset(nworld=self._nworld)
        else:
            self._measure_tree(model)
            self._bind_support_geometry(model)

        # GPU worlds share mesh topology but randomize the static tree pose.
        # Preserve each episode's pose when reconstructing its CPU contacts.
        states = getattr(randomization, "object_states", None)
        if states is None and isinstance(randomization, dict):
            states = randomization.get("object_states")
        tree = states.get(self._tree_body_name) if isinstance(states, dict) else None
        self._qpos_model = None
        self._qpos_scratch_model = None
        self._qpos_scratch_data = None
        if isinstance(tree, dict):
            self._qpos_model = copy.deepcopy(model)
            self._qpos_model.body_pos[self._tree_body_id] = tree["pos"]
            self._qpos_model.body_quat[self._tree_body_id] = tree["quat"]

    def debug_spec(self) -> dict[str, object] | None:
        return None

    @staticmethod
    def _subtree_bodies(model: mujoco.MjModel, root: int) -> set[int]:
        bodies = {root}
        for body in range(root + 1, model.nbody):
            if int(model.body_parentid[body]) in bodies:
                bodies.add(body)
        return bodies

    @staticmethod
    def _geom_vertices(model: mujoco.MjModel, geom: int) -> np.ndarray:
        kind = int(model.geom_type[geom])
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(model.geom_dataid[geom])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            return np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)
        size = model.geom_size[geom]
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            return np.asarray(
                [
                    [x, y, z]
                    for x in (-size[0], size[0])
                    for y in (-size[1], size[1])
                    for z in (-size[2], size[2])
                ]
            )
        if kind in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            height = size[1] + (size[0] if kind == mujoco.mjtGeom.mjGEOM_CAPSULE else 0)
            return np.asarray(
                [
                    [size[0] * np.cos(a), size[0] * np.sin(a), z]
                    for a in np.linspace(0, 2 * np.pi, 32, endpoint=False)
                    for z in (-height, height)
                ]
            )
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            return np.asarray(
                [
                    [sign * size[0] if k == axis else 0 for k in range(3)]
                    for axis in range(3)
                    for sign in (-1, 1)
                ]
            )
        raise ValueError(f"Unsupported mug/rack collision geom type {kind}")

    def _bind_support_geometry(self, model: mujoco.MjModel) -> None:
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        tree_bodies = self._subtree_bodies(model, self._tree_body_id)
        self._peg_segments: dict[int, np.ndarray] = {}
        # These six pieces are the pegs in the bundled eleven-piece rack asset.
        peg_meshes = {f"mug_tree_col_{i}" for i in (1, 2, 5, 6, 8, 9)}
        for geom in range(model.ngeom):
            if int(model.geom_bodyid[geom]) not in tree_bodies:
                continue
            mesh_id = int(model.geom_dataid[geom])
            mesh_name = (
                model.mesh(mesh_id).name
                if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
                else ""
            )
            name = model.geom(geom).name or ""
            if mesh_name not in peg_meshes and not name.startswith("mug_tree_peg_"):
                continue
            points = self._geom_vertices(model, geom)
            center = points.mean(axis=0)
            _, _, axes = np.linalg.svd(points - center, full_matrices=False)
            axis = axes[0]
            distances = (points - center) @ axis
            self._peg_segments[geom] = center + np.outer(
                [distances.min(), distances.max()], axis
            )
        if not self._peg_segments:
            raise ValueError("Mug rack has no identified peg collision geometry")

        self._solid_geometry = {}
        self._mug_bodies = []
        self._mug_geoms = []
        self._vessel_geometry = []
        self._mug_weights = []
        gravity = float(np.linalg.norm(model.opt.gravity))
        if gravity <= 0:
            raise ValueError("Mug hanging scoring requires nonzero gravity")
        for address in self._mug_qpos_addrs:
            joint = int(np.flatnonzero(model.jnt_qposadr == address)[0])
            body = int(model.jnt_bodyid[joint])
            bodies = self._subtree_bodies(model, body)
            geoms = {
                g
                for g in range(model.ngeom)
                if int(model.geom_bodyid[g]) in bodies
                and (model.geom_contype[g] or model.geom_conaffinity[g])
            }
            points = []
            for geom in geoms:
                if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                    mesh = int(model.geom_dataid[geom])
                    vertices = self._geom_vertices(model, geom)
                    start = int(model.mesh_faceadr[mesh])
                    count = int(model.mesh_facenum[mesh])
                    faces = model.mesh_face[start : start + count]
                    triangles = vertices[faces]
                    normals = np.cross(
                        triangles[:, 1] - triangles[:, 0],
                        triangles[:, 2] - triangles[:, 0],
                    )
                    length = np.linalg.norm(normals, axis=1)
                    valid = length > np.finfo(float).eps
                    normals = normals[valid] / length[valid, None]
                    offsets = (normals * triangles[valid, 0]).sum(axis=1)
                    reverse = normals @ vertices.mean(axis=0) > offsets
                    normals[reverse] *= -1
                    offsets[reverse] *= -1
                    self._solid_geometry[geom] = (normals, offsets)
                world = (
                    self._geom_vertices(model, geom)
                    @ data.geom_xmat[geom].reshape(3, 3).T
                    + data.geom_xpos[geom]
                )
                points.append((world - data.xpos[body]) @ data.xmat[body].reshape(3, 3))
            if not points:
                raise ValueError("Mug has no collision geometry")
            points = np.concatenate(points)
            low, high = points[:, 2].min(), points[:, 2].max()
            bottom = points[
                points[:, 2] <= low + self.bottom_slice_fraction * (high - low)
            ]
            center = (bottom[:, :2].min(axis=0) + bottom[:, :2].max(axis=0)) / 2
            # Every shipped mug's handle protrudes along local -y and is thin
            # in x, so its x envelope measures the vessel, not the handle.
            radius = float(np.max(np.abs(points[:, 0] - center[0])))
            self._mug_bodies.append(body)
            self._mug_geoms.append(geoms)
            self._vessel_geometry.append((center, radius))
            self._mug_weights.append(
                sum(float(model.body_mass[b]) for b in bodies) * gravity
            )

    def _ray_mug(
        self,
        data: mujoco.MjData,
        geoms: set[int],
        point: np.ndarray,
        direction: np.ndarray,
        model: mujoco.MjModel,
    ) -> float:
        distances = []
        for geom in geoms:
            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                distance = mujoco.mj_rayMesh(model, data, geom, point, direction)
            else:
                distance = mujoco.mju_rayGeom(
                    data.geom_xpos[geom],
                    data.geom_xmat[geom],
                    model.geom_size[geom],
                    point,
                    direction,
                    int(model.geom_type[geom]),
                )
            if distance >= 0:
                distances.append(distance)
        return min(distances) if distances else -1.0

    def _inside_mug_solid(
        self,
        data: mujoco.MjData,
        geoms: set[int],
        point: np.ndarray,
        model: mujoco.MjModel,
    ) -> bool:
        for geom in geoms:
            local = data.geom_xmat[geom].reshape(3, 3).T @ (
                point - data.geom_xpos[geom]
            )
            kind = int(model.geom_type[geom])
            size = model.geom_size[geom]
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                normals, offsets = self._solid_geometry[geom]
                inside = np.all(normals @ local <= offsets + self.geometry_tolerance_m)
            elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                inside = np.all(np.abs(local) <= size + self.geometry_tolerance_m)
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                inside = (
                    np.linalg.norm(local[:2]) <= size[0] and abs(local[2]) <= size[1]
                )
            elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                delta = local.copy()
                delta[2] -= np.clip(local[2], -size[1], size[1])
                inside = np.linalg.norm(delta) <= size[0] + self.geometry_tolerance_m
            elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                inside = np.linalg.norm(local) <= size[0] + self.geometry_tolerance_m
            else:
                inside = False
            if inside:
                return True
        return False

    def _peg_threaded(
        self, data: mujoco.MjData, mug_index: int, peg: int, model: mujoco.MjModel
    ) -> bool:
        body = self._mug_bodies[mug_index]
        rotation = data.xmat[body].reshape(3, 3)
        segment_world = (
            self._peg_segments[peg] @ data.geom_xmat[peg].reshape(3, 3).T
            + data.geom_xpos[peg]
        )
        segment = (segment_world - data.xpos[body]) @ rotation
        center, radius = self._vessel_geometry[mug_index]
        delta = segment[1] - segment[0]
        if abs(delta[0]) <= self.geometry_tolerance_m:
            return False
        t = (center[0] - segment[0, 0]) / delta[0]
        if not 0 < t < 1:
            return False
        crossing = segment[0] + t * delta
        if np.linalg.norm(crossing[:2] - center) <= radius + self.geometry_tolerance_m:
            return False
        point_world = rotation @ crossing + data.xpos[body]
        if self._inside_mug_solid(data, self._mug_geoms[mug_index], point_world, model):
            return False
        for axis in (1, 2):
            for sign in (-1, 1):
                direction = sign * rotation[:, axis]
                distance = self._ray_mug(
                    data, self._mug_geoms[mug_index], point_world, direction, model
                )
                if distance <= self.geometry_tolerance_m:
                    return False
        return True

    def _sample_support(
        self, data: mujoco.MjData, model: mujoco.MjModel
    ) -> dict[str, np.ndarray]:
        count = len(self.mug_names)
        forces = np.zeros(count)
        other_contact = np.zeros(count, dtype=bool)
        threaded = np.zeros(count, dtype=bool)
        upward = -np.asarray(model.opt.gravity) / np.linalg.norm(model.opt.gravity)
        force = np.zeros(6)
        threaded_checks: dict[tuple[int, int], bool] = {}
        for i, geoms in enumerate(self._mug_geoms):
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                g1, g2 = int(contact.geom1), int(contact.geom2)
                if g1 not in geoms and g2 not in geoms:
                    continue
                other = g2 if g1 in geoms else g1
                if other in geoms or contact.dist > self.geometry_tolerance_m:
                    continue
                if (
                    other not in self._peg_segments
                    or contact.dist < -self.max_penetration_m
                ):
                    other_contact[i] = True
                    continue
                pair = (i, other)
                if pair not in threaded_checks:
                    threaded_checks[pair] = self._peg_threaded(data, i, other, model)
                if not threaded_checks[pair]:
                    other_contact[i] = True
                    continue
                body = self._mug_bodies[i]
                local_point = data.xmat[body].reshape(3, 3).T @ (
                    contact.pos - data.xpos[body]
                )
                center, radius = self._vessel_geometry[i]
                if (
                    np.linalg.norm(local_point[:2] - center)
                    <= radius + self.geometry_tolerance_m
                ):
                    other_contact[i] = True
                    continue
                threaded[i] = True
                mujoco.mj_contactForce(model, data, contact_index, force)
                world_force = np.asarray(contact.frame).reshape(3, 3).T @ force[:3]
                if g1 in geoms:
                    world_force = -world_force
                forces[i] += float(world_force @ upward)
        weight = np.asarray(self._mug_weights)
        supported = threaded & (forces >= self.support_weight_fraction * weight)
        return {
            "mug_threaded_peg": threaded,
            "mug_peg_support_force_n": forces,
            "mug_other_contact": other_contact,
            "mug_hanging": supported & ~other_contact,
        }

    def evaluate_model_data(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> TaskEvalResult:
        if model is not self.model:
            self._bind_model(model)
            self.reset(nworld=1)
        if self._nworld != 1:
            self.reset(nworld=1)
        return self._result_from_samples([self._sample_support(data, model)])

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float64)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])
        model = self._qpos_model if self._qpos_model is not None else self.model
        if self._qpos_scratch_model is not model or self._qpos_scratch_data is None:
            self._qpos_scratch_model = model
            self._qpos_scratch_data = mujoco.MjData(model)
        else:
            # A batch formerly started with a fresh MjData. Clear velocities,
            # controls, external forces and solver warm-start state at that
            # same boundary, while preserving reuse between rows in a batch.
            mujoco.mj_resetData(model, self._qpos_scratch_data)
        data = self._qpos_scratch_data
        samples = []
        for qpos in qpos_batch:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            samples.append(self._sample_support(data, model))
        return self._result_from_samples(samples)

    def _result_from_samples(
        self, samples: list[dict[str, np.ndarray]]
    ) -> TaskEvalResult:
        metrics: dict[str, Any] = {
            key: np.stack([row[key] for row in samples]) for key in samples[0]
        }
        hanging = metrics["mug_hanging"]
        num_hanging = hanging.sum(axis=1).astype(np.int32)
        total = len(self.mug_names)
        self._max_hanging = np.maximum(self._max_hanging, num_hanging)
        success = num_hanging == total
        self._ever_success |= success
        return TaskEvalResult(
            reward=num_hanging.astype(np.float32) / float(total),
            success=success,
            metrics={
                **metrics,
                "num_mugs_hanging": num_hanging,
                "num_active_mugs": np.full_like(num_hanging, total),
                "max_mugs_hanging_so_far": self._max_hanging.copy(),
                "mugs_hanging": [
                    [name for name, ok in zip(self.mug_names, world) if ok]
                    for world in hanging
                ],
                "mug_names": list(self.mug_names),
                "ever_success": self._ever_success.copy(),
                "success_semantics": "released_handle_supported_by_peg",
                "support_weight_fraction": self.support_weight_fraction,
            },
        )
