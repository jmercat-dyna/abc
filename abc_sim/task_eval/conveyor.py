"""Conveyor-pick task evaluator."""

from __future__ import annotations

import re

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_specs import SimTaskSpec


class ConveyorPickGeometry:
    """Live collision geometry and two-finger lift detection shared with the runtime."""

    tolerance_m = 1e-5

    def __init__(self, model: mujoco.MjModel, addresses: np.ndarray):
        self.model = model
        self.belt = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "conveyor_belt_surface"
        )
        if self.belt < 0 or model.geom_type[self.belt] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError("conveyor belt surface must be a named collision box")
        self.object_bodies = [
            int(model.jnt_bodyid[np.flatnonzero(model.jnt_qposadr == address)[0]])
            for address in addresses
        ]
        self.object_geoms = [
            self._geoms(self._subtree(body)) for body in self.object_bodies
        ]
        if any(not geoms for geoms in self.object_geoms):
            raise ValueError("conveyor object has no collision geometry")
        self.grippers = []
        for side in ("left", "right"):
            fingers = []
            for finger in ("left", "right"):
                body = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_link_{finger}_finger"
                )
                fingers.append(self._geoms(self._subtree(body)) if body >= 0 else set())
            self.grippers.append(fingers)
        self.mesh_points = {}
        for geoms in self.object_geoms:
            for geom in geoms:
                if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                    mesh = int(model.geom_dataid[geom])
                    start, count = (
                        int(model.mesh_vertadr[mesh]),
                        int(model.mesh_vertnum[mesh]),
                    )
                    self.mesh_points[geom] = np.asarray(
                        model.mesh_vert[start : start + count], dtype=np.float64
                    ).copy()

    def _subtree(self, body: int) -> set[int]:
        result = {body}
        for child in range(body + 1, self.model.nbody):
            if int(self.model.body_parentid[child]) in result:
                result.add(child)
        return result

    def _geoms(self, bodies: set[int]) -> set[int]:
        return {
            g
            for g in range(self.model.ngeom)
            if int(self.model.geom_bodyid[g]) in bodies
            and (self.model.geom_contype[g] or self.model.geom_conaffinity[g])
        }

    def bounds_on_belt(
        self, data: mujoco.MjData, geoms: set[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._bounds_in_frame(
            data,
            geoms,
            data.geom_xpos[self.belt],
            data.geom_xmat[self.belt].reshape(3, 3).T,
        )

    def _bounds_in_frame(
        self,
        data: mujoco.MjData,
        geoms: set[int],
        origin: np.ndarray,
        frame_from_world: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        lows, highs = [], []
        for geom in geoms:
            rotation = frame_from_world @ data.geom_xmat[geom].reshape(3, 3)
            center = frame_from_world @ (data.geom_xpos[geom] - origin)
            size, kind = self.model.geom_size[geom], self.model.geom_type[geom]
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                points = self.mesh_points[geom] @ rotation.T + center
                lows.append(points.min(axis=0))
                highs.append(points.max(axis=0))
                continue
            if kind == mujoco.mjtGeom.mjGEOM_BOX:
                half = np.abs(rotation) @ size
            elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                half = np.full(3, size[0])
            elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                half = size[0] + size[1] * np.abs(rotation[:, 2])
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                half = size[0] * np.linalg.norm(rotation[:, :2], axis=1) + size[
                    1
                ] * np.abs(rotation[:, 2])
            elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                half = np.linalg.norm(rotation * size[None, :], axis=1)
            else:
                raise ValueError(f"unsupported conveyor object collision type {kind}")
            lows.append(center - half)
            highs.append(center + half)
        return np.min(lows, axis=0), np.max(highs, axis=0)

    def _grasped(self, data: mujoco.MjData) -> np.ndarray:
        if not hasattr(self, "_contact_lookup"):
            objects = {
                geom: i for i, geoms in enumerate(self.object_geoms) for geom in geoms
            }
            fingers = {
                geom: 1 << (2 * hand + finger)
                for hand, pair in enumerate(self.grippers)
                for finger, geoms in enumerate(pair)
                for geom in geoms
            }
            self._contact_lookup = objects, fingers
        objects, fingers = self._contact_lookup
        flags = np.zeros(len(self.object_geoms), dtype=np.uint8)
        for contact in data.contact[: data.ncon]:
            if contact.dist > self.tolerance_m:
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            i = objects.get(a)
            if i is not None:
                flags[i] |= fingers.get(b, 0)
            i = objects.get(b)
            if i is not None:
                flags[i] |= fingers.get(a, 0)
        return ((flags & 3) == 3) | ((flags & 12) == 12)

    def _proxy_safe_on_belt(self, data: mujoco.MjData) -> np.ndarray:
        if not hasattr(self, "_body_proxies"):
            initial = mujoco.MjData(self.model)
            mujoco.mj_forward(self.model, initial)
            centers, halves, material_points, rigid = [], [], [], []
            for body, geoms in zip(self.object_bodies, self.object_geoms):
                origin = initial.xpos[body]
                rotation = initial.xmat[body].reshape(3, 3).T
                low, high = self._bounds_in_frame(initial, geoms, origin, rotation)
                centers.append((low + high) / 2)
                halves.append((high - low) / 2)
                geom = next(iter(geoms))
                local = (
                    self.mesh_points[geom].mean(axis=0)
                    if geom in self.mesh_points
                    else np.zeros(3)
                )
                world = (
                    initial.geom_xmat[geom].reshape(3, 3) @ local
                    + initial.geom_xpos[geom]
                )
                material_points.append(rotation @ (world - origin))
                rigid.append(
                    all(
                        child == body or not self.model.body_jntnum[child]
                        for child in self._subtree(body)
                    )
                )
            self._body_proxies = tuple(
                np.asarray(values)
                for values in (centers, halves, material_points, rigid)
            )
        centers, halves, material_points, rigid = self._body_proxies
        belt_from_world = data.geom_xmat[self.belt].reshape(3, 3).T
        rotations = np.einsum(
            "ij,njk->nik",
            belt_from_world,
            data.xmat[self.object_bodies].reshape(-1, 3, 3),
        )
        origins = (
            data.xpos[self.object_bodies] - data.geom_xpos[self.belt]
        ) @ belt_from_world.T
        position = origins + np.einsum("nij,nj->ni", rotations, centers)
        half = np.einsum("nij,nj->ni", np.abs(rotations), halves)
        material = origins + np.einsum("nij,nj->ni", rotations, material_points)
        size = self.model.geom_size[self.belt]
        # A conservative containing box wholly inside the belt footprint proves
        # overlap. A REAL material point above its top proves the object is not
        # wholly below; the proxy's upper corner alone would not prove this.
        return (
            rigid
            & np.all(
                position[:, :2] - half[:, :2] >= -size[:2] - self.tolerance_m, axis=1
            )
            & np.all(
                position[:, :2] + half[:, :2] <= size[:2] + self.tolerance_m, axis=1
            )
            & (material[:, 2] >= size[2] - self.tolerance_m)
        )

    def sample(
        self,
        data: mujoco.MjData,
        *,
        picked_history: set[str] | None = None,
        object_names: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        count = len(self.object_geoms)
        on_belt = np.zeros(count, dtype=bool)
        lifted = np.zeros(count, dtype=bool)
        grasped = self._grasped(data)
        size = self.model.geom_size[self.belt]
        safe = (
            self._proxy_safe_on_belt(data)
            if picked_history is not None
            else np.zeros(count, dtype=bool)
        )
        for i, geoms in enumerate(self.object_geoms):
            if picked_history is not None:
                already_picked = (
                    object_names is not None and object_names[i] in picked_history
                )
                if already_picked or (safe[i] and not grasped[i]):
                    # Runtime needs only new picks/misses. Previously picked
                    # objects cannot become a new unpicked-object loss.
                    on_belt[i] = True
                    continue
            low, high = self.bounds_on_belt(data, geoms)
            overlap = np.all(high[:2] >= -size[:2] - self.tolerance_m) and np.all(
                low[:2] <= size[:2] + self.tolerance_m
            )
            on_belt[i] = overlap and high[2] >= size[2] - self.tolerance_m
            lifted[i] = low[2] > size[2] + self.tolerance_m
        return {
            "object_on_belt_mask": on_belt,
            "object_lifted_mask": lifted,
            "object_two_finger_grasp_mask": grasped,
            "object_picked_now_mask": lifted & grasped,
        }


class ConveyorPickObjectsInBinEvaluator:
    """Score success when every conveyor object is inside the static bin.

    Leaving the belt before a genuine two-finger lift or reaching the bin is
    terminal failure. Pick and miss histories are independent per object/world.
    Bin geometry is measured in its body frame, preserving rotated/scaled scenes.
    """

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        bin_body_name: str = "conveyor_bin",
        object_joint_regex: str = r"conveyor_object_(\d+)_joint",
        floor_tolerance_m: float = 0.01,
    ) -> None:
        self.spec = spec
        self._bin_body_name = bin_body_name
        self._object_joint_regex = object_joint_regex
        self.floor_tolerance_m = float(floor_tolerance_m)
        self._runtime = None
        self._bind_model(model)
        self.reset(nworld=1)

    def _bind_model(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.object_names, self._object_qpos_addrs = self._resolve_object_qpos_addrs(
            model,
            self._object_joint_regex,
        )
        (
            self._bin_x_bounds,
            self._bin_y_bounds,
            self._bin_floor_z,
            self._bin_rim_z,
        ) = self._resolve_bin_interior(model, self._bin_body_name)
        self._bin_body_id = model.body(self._bin_body_name).id
        self._pick_geometry = ConveyorPickGeometry(model, self._object_qpos_addrs)

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._max_objects_in_bin = np.zeros((self._nworld,), dtype=np.int32)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)
        self._ever_failed = np.zeros((self._nworld,), dtype=bool)
        self._picked = np.zeros((self._nworld, len(self.object_names)), dtype=bool)
        self._missed = np.zeros_like(self._picked)

    def bind_runtime(self, runtime: object) -> None:
        self._runtime = runtime

    def configure_from_randomization(
        self,
        model: mujoco.MjModel,
        randomization: object,
    ) -> None:
        """Rebind to a model the object randomizer swapped in under us."""

        if model is not self.model:
            self._bind_model(model)
            self.reset(nworld=self._nworld)

    def debug_spec(self) -> dict[str, object] | None:
        return None

    @staticmethod
    def _resolve_object_qpos_addrs(
        model: mujoco.MjModel,
        object_joint_regex: str,
    ) -> tuple[list[str], np.ndarray]:
        pattern = re.compile(object_joint_regex)
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
                    name.removesuffix("_joint"),
                    int(model.jnt_qposadr[joint_id]),
                )
            )

        if not entries:
            raise ValueError(
                f"No joints matching {object_joint_regex!r} found in model"
            )

        entries.sort(key=lambda item: item[0])
        names = [name for _, name, _ in entries]
        addrs = np.asarray([addr for _, _, addr in entries], dtype=np.int32)
        return names, addrs

    @staticmethod
    def _resolve_bin_interior(
        model: mujoco.MjModel,
        bin_body_name: str,
    ) -> tuple[tuple[float, float], tuple[float, float], float, float]:
        """Measure the static bin's interior as (x bounds, y bounds, floor, rim).

        The bin body must be welded to the world through joint-free bodies.
        Bounds are measured in its own frame; its live world rotation composes in
        while sampling. Its box
        geoms split into the floor plate (the one whose top face sits lowest)
        and four walls; each wall bounds the axis its centre is displaced
        along, by its inner face.
        """

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bin_body_name)
        if body_id < 0:
            raise ValueError(f"Body {bin_body_name!r} not found in model")

        # Require a static bin, but permit rotated parent bodies: samples use
        # the complete live world pose rather than summing unrotated offsets.
        current = body_id
        while current != 0:
            if int(model.body_jntnum[current]) != 0:
                raise ValueError(
                    f"Body {bin_body_name!r} must be welded to the world, but "
                    f"{model.body(current).name!r} has a joint"
                )
            current = int(model.body_parentid[current])

        boxes: list[tuple[np.ndarray, np.ndarray]] = []
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) != body_id:
                continue
            if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                continue
            if not np.allclose(model.geom_quat[geom_id], (1.0, 0.0, 0.0, 0.0)):
                raise ValueError(
                    f"Geom {model.geom(geom_id).name!r} of {bin_body_name!r} is "
                    "rotated; the axis-aligned interior test assumes identity "
                    "geom orientation"
                )
            boxes.append(
                (
                    np.asarray(model.geom_pos[geom_id], dtype=np.float64),
                    np.asarray(model.geom_size[geom_id], dtype=np.float64),
                )
            )

        if len(boxes) < 5:
            raise ValueError(
                f"Body {bin_body_name!r} has {len(boxes)} box geoms; expected a "
                "floor plate and four walls"
            )

        floor_pos, floor_size = min(boxes, key=lambda box: box[0][2] + box[1][2])
        floor_top_z = float(floor_pos[2] + floor_size[2])

        x_faces: list[float] = []
        y_faces: list[float] = []
        rim_z = floor_top_z
        for pos, size in boxes:
            if pos is floor_pos:
                continue
            rim_z = max(rim_z, float(pos[2] + size[2]))
            dx = pos[0] - floor_pos[0]
            dy = pos[1] - floor_pos[1]
            if abs(dx) >= abs(dy):
                x_faces.append(float(abs(dx) - size[0]))
            else:
                y_faces.append(float(abs(dy) - size[1]))

        if not x_faces or not y_faces or rim_z <= floor_top_z:
            raise ValueError(
                f"Body {bin_body_name!r} walls do not enclose the floor plate"
            )

        x_half = min(x_faces)
        y_half = min(y_faces)
        if x_half <= 0.0 or y_half <= 0.0:
            raise ValueError(
                f"Body {bin_body_name!r} measured a non-positive interior extent"
            )
        return (
            (float(floor_pos[0] - x_half), float(floor_pos[0] + x_half)),
            (float(floor_pos[1] - y_half), float(floor_pos[1] + y_half)),
            floor_top_z,
            rim_z,
        )

    def _sample(
        self, data: mujoco.MjData, *, picked_history: set[str] | None = None
    ) -> dict[str, np.ndarray]:
        geometry = self._pick_geometry.sample(
            data, picked_history=picked_history, object_names=self.object_names
        )
        position = np.asarray([data.xpos[b] for b in self._pick_geometry.object_bodies])
        local = (position - data.xpos[self._bin_body_id]) @ data.xmat[
            self._bin_body_id
        ].reshape(3, 3)
        x_lo, x_hi = self._bin_x_bounds
        y_lo, y_hi = self._bin_y_bounds
        geometry["object_in_bin_mask"] = (
            (local[:, 0] >= x_lo)
            & (local[:, 0] <= x_hi)
            & (local[:, 1] >= y_lo)
            & (local[:, 1] <= y_hi)
            & (local[:, 2] >= self._bin_floor_z - self.floor_tolerance_m)
            & (local[:, 2] <= self._bin_rim_z)
        )
        return geometry

    def evaluate_model_data(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> TaskEvalResult:
        if model is not self.model:
            self._bind_model(model)
            self.reset(nworld=1)
        if self._nworld != 1:
            self.reset(nworld=1)
        return self._result([self._sample(data)], use_runtime=True)

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float64)
        if (
            qpos_batch.ndim != 2
            or qpos_batch.shape[1] != self.model.nq
            or qpos_batch.shape[0] < 1
        ):
            raise ValueError(
                f"Expected nonempty qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])
        data = mujoco.MjData(self.model)
        samples = []
        for qpos in qpos_batch:
            mujoco.mj_resetData(self.model, data)
            data.qpos[:] = qpos
            mujoco.mj_forward(self.model, data)
            samples.append(self._sample(data))
        return self._result(samples)

    def _result(
        self, samples: list[dict[str, np.ndarray]], *, use_runtime: bool = False
    ) -> TaskEvalResult:
        metrics = {
            key: np.stack([sample[key] for sample in samples]) for key in samples[0]
        }
        self._picked |= metrics["object_picked_now_mask"]
        if use_runtime and self._runtime is not None:
            state = self._runtime.debug_state()
            picked = set(state.get("picked_objects", ()))
            missed = set(state.get("missed_objects", ()))
            self._picked[0] |= [name in picked for name in self.object_names]
            self._missed[0] |= [name in missed for name in self.object_names]
        in_bin = metrics["object_in_bin_mask"]
        self._missed |= ~metrics["object_on_belt_mask"] & ~self._picked & ~in_bin
        self._ever_failed |= self._missed.any(axis=1)
        num_objects = in_bin.sum(axis=1).astype(np.int32)
        total = len(self.object_names)
        self._max_objects_in_bin = np.maximum(self._max_objects_in_bin, num_objects)
        success = (num_objects == total) & ~self._ever_failed
        self._ever_success |= success
        self._ever_success &= ~self._ever_failed
        return TaskEvalResult(
            reward=num_objects.astype(np.float32) / float(total),
            success=success,
            metrics={
                **metrics,
                "num_objects_in_bin": num_objects,
                "num_active_objects": np.full_like(num_objects, total),
                "max_objects_in_bin_so_far": self._max_objects_in_bin.copy(),
                "ever_success": self._ever_success.copy(),
                "ever_failed": self._ever_failed.copy(),
                "terminal_failure": self._ever_failed.copy(),
                "failure_reason": np.where(
                    self._ever_failed, "unpicked_object_left_belt", ""
                ),
                "object_picked_history_mask": self._picked.copy(),
                "failed_objects": [
                    [name for name, bad in zip(self.object_names, row) if bad]
                    for row in self._missed
                ],
                "objects_in_bin": [
                    [name for name, inside in zip(self.object_names, row) if inside]
                    for row in in_bin
                ],
                "object_names": list(self.object_names),
                "bin_x_bounds": list(self._bin_x_bounds),
                "bin_y_bounds": list(self._bin_y_bounds),
                "bin_floor_z": self._bin_floor_z,
                "bin_rim_z": self._bin_rim_z,
                "bin_bounds_frame": "bin_body_local",
                "success_semantics": "latched_success_until_unpicked_object_lost",
            },
        )
