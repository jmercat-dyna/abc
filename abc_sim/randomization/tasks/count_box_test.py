"""Counting goals are task data, independent of task names and scene sampling."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from abc_sim import make_env
from abc_sim.env import MuJoCoYAMEnv
from abc_sim.task_specs import SimTaskSpec, get_task_spec, list_task_specs


FIXED_SPECS = tuple(spec for spec in list_task_specs() if spec.fixed_count is not None)
HAS_ASSETS = (Path(__file__).parents[2] / "models/assets/i2rt_yam/assets/model2.stl").is_file()
requires_assets = pytest.mark.skipif(not HAS_ASSETS, reason="YAM simulator assets required")


@pytest.mark.parametrize("spec", FIXED_SPECS, ids=lambda spec: spec.name)
def test_fixed_count_aliases_resolve_to_the_same_goal(spec: SimTaskSpec) -> None:
    for name in (spec.name, spec.prompt, *spec.aliases):
        assert get_task_spec(name).fixed_count == spec.fixed_count


def assert_exact_success(env: MuJoCoYAMEnv, count: int) -> None:
    result = env.evaluate_task().to_info(squeeze=True)
    assert result["target_count"] == count
    objects = env._last_randomization.metadata["objects"]
    initial = env.data.qpos.copy()
    for inside_count in (count, count + 1):
        env.data.qpos[:] = initial
        for obj in objects[:inside_count]:
            address = int(env.model.joint(obj["joint"]).qposadr[0])
            env.data.qpos[address : address + 3] = [.70, 0, .82]
        result = env.evaluate_task().to_info(squeeze=True)
        assert bool(result["success"]) == (inside_count == count)
        assert result["num_eligible_in_box"] == inside_count
        assert result["num_ineligible_in_box"] == 0


@requires_assets
@pytest.mark.parametrize("spec", FIXED_SPECS, ids=lambda spec: spec.name)
def test_direct_factory_keeps_the_selected_count_on_every_reset(spec: SimTaskSpec) -> None:
    assert spec.fixed_count is not None
    for task in (spec.name, spec.prompt, *spec.aliases):
        env = make_env(task=task, render_cameras=False)
        try:
            # Construction prepares the randomizer before the first reset.
            assert env.prompt == spec.prompt
            for seed in (0, 42):
                observation, info = env.reset(seed=seed, randomize=True)
                metadata = info["randomization"].metadata
                assert observation["prompt"] == spec.prompt
                assert metadata["prompt"] == spec.prompt
                assert metadata["target_count"] == spec.fixed_count
                assert metadata["prompt_type"] == "exact_count"
                assert metadata["eligible_objects"] == [obj["name"] for obj in metadata["objects"]]
                assert metadata["attributes"] == {}
                assert_exact_success(env, spec.fixed_count)
        finally:
            env.close()


@requires_assets
def test_new_task_and_count_need_no_randomizer_name_mapping() -> None:
    spec = replace(
        FIXED_SPECS[0], name="new_count_task", fixed_count=5,
        prompt="put exactly five objects in the opaque box", aliases=(),
    )
    env = make_env(task=spec, render_cameras=False)
    try:
        observation, _ = env.reset(seed=42, randomize=True)
        assert observation["prompt"] == spec.prompt
        assert_exact_success(env, 5)
    finally:
        env.close()


@requires_assets
def test_fixed_and_general_tasks_sample_the_same_objects_without_sharing_goals() -> None:
    general = make_env(task="count_into_opaque_box", render_cameras=False)
    fixed = make_env(task=FIXED_SPECS[0].name, prompt="sim " + FIXED_SPECS[0].prompt, render_cameras=False)
    try:
        prompts = []
        for seed in (0, 42):
            general_obs, general_info = general.reset(seed=seed, randomize=True)
            fixed_obs, fixed_info = fixed.reset(seed=seed, randomize=True)
            prompts.append(general_obs["prompt"])
            assert fixed_obs["prompt"] == "sim " + FIXED_SPECS[0].prompt
            assert general_info["randomization"].object_states == fixed_info["randomization"].object_states
            assert general_info["randomization"].metadata["objects"] == fixed_info["randomization"].metadata["objects"]
            np.testing.assert_allclose(general.data.qpos, fixed.data.qpos)
        assert prompts == ["put all orange cubes in the box", "put exactly 2 objects in the box"]
    finally:
        fixed.close()
        general.close()


@requires_assets
def test_switching_count_tasks_updates_the_instruction_and_goal_together() -> None:
    env = make_env(
        task=FIXED_SPECS[0].name,
        prompt="sim " + FIXED_SPECS[0].prompt,
        render_cameras=False,
    )
    try:
        # Resolving a different alias of the same task preserves an explicit override.
        env.set_task(FIXED_SPECS[0].aliases[0])
        observation, _ = env.reset(seed=42, randomize=True)
        assert observation["prompt"] == "sim " + FIXED_SPECS[0].prompt
        for task in (
            FIXED_SPECS[1].name,
            "count_into_opaque_box",
            FIXED_SPECS[2].name,
        ):
            env.set_task(task)
            observation, info = env.reset(seed=42, randomize=True)
            spec = get_task_spec(task)
            if spec.fixed_count is None:
                assert observation["prompt"] == "put exactly 2 objects in the box"
            else:
                assert observation["prompt"] == spec.prompt
                assert info["randomization"].metadata["target_count"] == spec.fixed_count
                assert_exact_success(env, spec.fixed_count)
    finally:
        env.close()
