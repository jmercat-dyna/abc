"""Grab-target-from-clutter scoring with irreversible wrong-object insertion."""

from __future__ import annotations

import re
from collections.abc import Mapping

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_specs import SimTaskSpec


class GrabClutterTargetInBoxEvaluator:
    """Require a released target placement followed by a real-time settling window.

    Any wrong object entering the crate below its rim is terminal failure. The
    first released, floor-supported correct placement starts a fixed deadline;
    the target must remain in the crate when that deadline arrives. Pose-only
    scoring never manufactures a clock: callers supply actual per-world time.
    """

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        goal_site_name: str = "grab_clutter_target_box_goal_region",
        settling_s: float = 3.0,
    ) -> None:
        self.spec = spec
        self.settling_s = float(settling_s)
        if not np.isfinite(self.settling_s) or self.settling_s < 0:
            raise ValueError("settling_s must be finite and nonnegative")
        # Kept as a compatible constructor option; the collision crate defines
        # containment, rather than the historical oversized visualization site.
        self.goal_site_name = goal_site_name
        self._bind_model(model)
        self.reset(nworld=1)

    @staticmethod
    def _subtree(model: mujoco.MjModel, root: int) -> set[int]:
        bodies = {root}
        for body in range(root + 1, model.nbody):
            if int(model.body_parentid[body]) in bodies:
                bodies.add(body)
        return bodies

    def _bind_model(self, model: mujoco.MjModel) -> None:
        self.model = model
        self._scratch = None
        self._box_body = model.body("grab_clutter_target_box").id
        self._floor = model.geom("grab_clutter_target_box_floor").id
        if model.geom_type[self._floor] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError("grab-clutter crate floor must be a collision box")
        boxes = []
        for geom in range(model.ngeom):
            if int(model.geom_bodyid[geom]) != self._box_body:
                continue
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            if not np.allclose(model.geom_quat[geom], [1, 0, 0, 0]):
                raise ValueError(
                    "crate collision boxes must be axis aligned in the crate frame"
                )
            boxes.append(geom)
        floor_pos = model.geom_pos[self._floor]
        floor_size = model.geom_size[self._floor]
        self._floor_z = float(floor_pos[2] + floor_size[2])
        walls = [geom for geom in boxes if geom != self._floor]
        faces = [[], []]
        for geom in walls:
            delta = model.geom_pos[geom] - floor_pos
            axis = int(abs(delta[1]) > abs(delta[0]))
            faces[axis].append(float(abs(delta[axis]) - model.geom_size[geom, axis]))
        if len(walls) < 4 or any(not values for values in faces):
            raise ValueError(
                "grab-clutter crate requires a floor and four collision walls"
            )
        self._xy_center = np.asarray(floor_pos[:2], dtype=np.float64).copy()
        self._half_size = np.asarray([min(values) for values in faces])
        self._rim_z = min(
            float(model.geom_pos[g, 2] + model.geom_size[g, 2]) for g in walls
        )
        if np.any(self._half_size <= 0) or self._rim_z <= self._floor_z:
            raise ValueError("grab-clutter crate has an invalid interior")
        joints = [
            joint
            for joint in range(model.njnt)
            if (
                model.jnt(joint).name == "target_object_joint"
                or re.fullmatch(r"clutter_obj_\d+_joint", model.jnt(joint).name or "")
            )
        ]
        if not any(model.jnt(j).name == "target_object_joint" for j in joints):
            raise ValueError("target_object_joint is missing")
        if any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE for j in joints):
            raise ValueError("grab-clutter object joints must be free")
        joints.sort(key=lambda joint: model.jnt(joint).name != "target_object_joint")
        self.object_names = [model.jnt(j).name.removesuffix("_joint") for j in joints]
        self._addresses = np.asarray([model.jnt_qposadr[j] for j in joints], dtype=int)
        self._object_geoms = []
        self._vertices = {}
        for joint in joints:
            bodies = self._subtree(model, int(model.jnt_bodyid[joint]))
            geoms = {
                geom
                for geom in range(model.ngeom)
                if int(model.geom_bodyid[geom]) in bodies
                and (model.geom_contype[geom] or model.geom_conaffinity[geom])
            }
            if not geoms:
                raise ValueError("grab-clutter object has no collision geometry")
            self._object_geoms.append(geoms)
            for geom in geoms:
                self._vertices[geom] = self._geom_points(model, geom)
        fingers = set()
        for side in ("left", "right"):
            for finger in ("left", "right"):
                body = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_link_{finger}_finger"
                )
                if body >= 0:
                    fingers |= self._subtree(model, body)
        self._gripper_geoms = {
            g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in fingers
        }

    @staticmethod
    def _geom_points(model: mujoco.MjModel, geom: int) -> np.ndarray:
        kind = int(model.geom_type[geom])
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(model.geom_dataid[geom])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            return np.asarray(
                model.mesh_vert[start : start + count], dtype=np.float64
            ).copy()
        # Box corners are exact. Round primitives use their bounding envelope;
        # imported task objects use convex collision meshes or box shapes.
        size = model.geom_size[geom].copy()
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            size[:] = size[0]
        elif kind in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            size = np.asarray(
                [
                    size[0],
                    size[0],
                    size[1] + (size[0] if kind == mujoco.mjtGeom.mjGEOM_CAPSULE else 0),
                ]
            )
        elif kind != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError(f"unsupported grab-clutter object collision shape {kind}")
        return np.asarray(
            [
                [x, y, z]
                for x in (-size[0], size[0])
                for y in (-size[1], size[1])
                for z in (-size[2], size[2])
            ]
        )

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._ever_failed = np.zeros(self._nworld, dtype=bool)
        self._ever_success = np.zeros(self._nworld, dtype=bool)
        self._first_placement_s = np.full(self._nworld, np.nan)
        self._last_time_s = np.full(self._nworld, np.nan)
        self._failure_reason = np.full(self._nworld, "", dtype="<U32")
        self._wrong_history = np.zeros(
            (self._nworld, len(self.object_names) - 1), dtype=bool
        )

    def configure_from_randomization(
        self, model: mujoco.MjModel, randomization: object
    ) -> None:
        # Reset may apply scale_states to an existing model in place.
        self._bind_model(model)
        self.reset(nworld=self._nworld)
        metadata = getattr(randomization, "metadata", None)
        if metadata is None and isinstance(randomization, Mapping):
            metadata = randomization.get("metadata")
        if isinstance(metadata, Mapping) and metadata.get("objects"):
            names = {str(item["name"]) for item in metadata["objects"]}
            if (
                names != set(self.object_names)
                or metadata.get("target_joint", "target_object_joint")
                != "target_object_joint"
            ):
                raise ValueError(
                    "grab-clutter object metadata differs from the compiled scene"
                )

    def _sample(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        box_position = data.xpos[self._box_body]
        box_rotation = data.xmat[self._box_body].reshape(3, 3)
        bottoms = []
        tops = []
        for geoms in self._object_geoms:
            lows, highs = [], []
            for geom in geoms:
                rotation = box_rotation.T @ data.geom_xmat[geom].reshape(3, 3)
                center = (data.geom_xpos[geom] - box_position) @ box_rotation
                kind = int(self.model.geom_type[geom])
                radius, half_length = self.model.geom_size[geom, :2]
                if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                    extent = np.full(3, radius)
                elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    extent = radius + half_length * np.abs(rotation[:, 2])
                elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    extent = radius * np.linalg.norm(
                        rotation[:, :2], axis=1
                    ) + half_length * np.abs(rotation[:, 2])
                else:
                    points = self._vertices[geom] @ rotation.T + center
                    lows.append(points.min(axis=0))
                    highs.append(points.max(axis=0))
                    continue
                lows.append(center - extent)
                highs.append(center + extent)
            low, high = np.min(lows, axis=0), np.max(highs, axis=0)
            bottoms.append([*(0.5 * (low[:2] + high[:2])), low[2]])
            tops.append(high[2])
        bottoms = np.asarray(bottoms)
        floor_contact = np.zeros(len(self.object_names), dtype=bool)
        for index in range(data.ncon):
            contact = data.contact[index]
            a, b = int(contact.geom1), int(contact.geom2)
            if contact.dist > 1e-5 or self._floor not in (a, b):
                continue
            other = b if a == self._floor else a
            for object_index, geoms in enumerate(self._object_geoms):
                floor_contact[object_index] |= other in geoms
        # MuJoCo's compliant contacts can place collision bottoms slightly
        # below the surface. Actual floor contact establishes placement there;
        # an object wholly under the floor does not count.
        above_floor = (bottoms[:, 2] >= self._floor_z - 1e-5) | (
            floor_contact & (np.asarray(tops) > self._floor_z)
        )
        inside = (
            np.all(np.abs(bottoms[:, :2] - self._xy_center) <= self._half_size, axis=1)
            & above_floor
            & (bottoms[:, 2] < self._rim_z)
        )
        target_floor_contact = False
        target_gripper_contact = False
        target_geoms = self._object_geoms[0]
        for index in range(data.ncon):
            contact = data.contact[index]
            a, b = int(contact.geom1), int(contact.geom2)
            if contact.dist > 1e-5 or (a not in target_geoms and b not in target_geoms):
                continue
            other = b if a in target_geoms else a
            target_floor_contact |= other == self._floor
            target_gripper_contact |= other in self._gripper_geoms
        return {
            "object_in_box_mask": inside,
            "object_bottom_box_frame": bottoms,
            "target_floor_supported": np.asarray(target_floor_contact),
            "target_released": np.asarray(not target_gripper_contact),
        }

    def evaluate_model_data(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> TaskEvalResult:
        if model is not self.model:
            self._bind_model(model)
            self.reset(nworld=1)
        if self._nworld != 1:
            self.reset(nworld=1)
        return self._result([self._sample(data)], np.asarray([float(data.time)]))

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        return self._evaluate_poses(qpos_batch, None)

    def evaluate_qpos_batch_at_time(
        self, qpos_batch: np.ndarray, time_s: np.ndarray | float
    ) -> TaskEvalResult:
        return self._evaluate_poses(qpos_batch, time_s)

    def _evaluate_poses(
        self, qpos_batch: np.ndarray, time_s: np.ndarray | float | None
    ) -> TaskEvalResult:
        qpos = np.asarray(qpos_batch, dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != self.model.nq or not len(qpos):
            raise ValueError(
                f"expected nonempty qpos batch (B, {self.model.nq}), got {qpos.shape}"
            )
        if len(qpos) != self._nworld:
            self.reset(nworld=len(qpos))
        if self._scratch is None:
            self._scratch = mujoco.MjData(self.model)
        samples = []
        for pose in qpos:
            mujoco.mj_resetData(self.model, self._scratch)
            self._scratch.qpos[:] = pose
            mujoco.mj_forward(self.model, self._scratch)
            samples.append(self._sample(self._scratch))
        times = (
            None
            if time_s is None
            else np.broadcast_to(
                np.asarray(time_s, dtype=np.float64), (len(qpos),)
            ).copy()
        )
        return self._result(samples, times)

    def _result(
        self, samples: list[dict[str, np.ndarray]], times: np.ndarray | None
    ) -> TaskEvalResult:
        if times is not None:
            if not np.isfinite(times).all() or np.any(times < 0):
                raise ValueError(
                    "grab-clutter simulation times must be finite and nonnegative"
                )
            if np.any(times < self._last_time_s):
                raise ValueError(
                    "grab-clutter simulation clock moved backward; reset evaluator"
                )
        metrics = {key: np.stack([row[key] for row in samples]) for key in samples[0]}
        inside = metrics["object_in_box_mask"]
        self._wrong_history |= inside[:, 1:]
        wrong = self._wrong_history.any(axis=1)
        first_wrong = wrong & ~self._ever_failed
        self._failure_reason[first_wrong] = "wrong_object_in_box"
        self._ever_failed |= wrong
        placed = (
            inside[:, 0]
            & metrics["target_released"]
            & metrics["target_floor_supported"]
        )
        deadline = self._first_placement_s + self.settling_s
        complete = np.zeros(self._nworld, dtype=bool)
        if times is not None:
            self._last_time_s[:] = times
            start = placed & np.isnan(self._first_placement_s) & ~self._ever_failed
            self._first_placement_s[start] = times[start]
            deadline = self._first_placement_s + self.settling_s
            complete = np.isfinite(deadline) & (times >= deadline)
            removed = complete & ~inside[:, 0] & ~self._ever_failed
            self._failure_reason[removed] = "target_missing_after_settling"
            self._ever_failed |= removed
        success = complete & inside[:, 0] & ~self._ever_failed
        self._ever_success |= success
        self._ever_success &= ~self._ever_failed
        reward = np.where(
            self._ever_failed,
            0.0,
            np.where(success, 1.0, np.where(inside[:, 0], 0.5, 0.0)),
        ).astype(np.float32)
        return TaskEvalResult(
            reward=reward,
            success=success,
            metrics={
                **metrics,
                "target_in_box": inside[:, 0].copy(),
                "target_placed": placed,
                "wrong_objects_in_box": [
                    [
                        name
                        for name, present in zip(self.object_names[1:], row)
                        if present
                    ]
                    for row in inside[:, 1:]
                ],
                "wrong_objects_ever_in_box": [
                    [
                        name
                        for name, present in zip(self.object_names[1:], row)
                        if present
                    ]
                    for row in self._wrong_history
                ],
                "ever_failed": self._ever_failed.copy(),
                "terminal_failure": self._ever_failed.copy(),
                "failure_reason": self._failure_reason.copy(),
                "ever_success": self._ever_success.copy(),
                "settling_started_s": self._first_placement_s.copy(),
                "settling_deadline_s": deadline,
                "settling_s": self.settling_s,
                "settle_complete": complete,
                "clock_available": times is not None,
                "success_semantics": "placed_target_settle_until_terminal_failure",
                "box_inner_half_size": self._half_size.tolist(),
                "box_floor_z": self._floor_z,
                "box_rim_z": self._rim_z,
                "object_names": self.object_names,
            },
        )
