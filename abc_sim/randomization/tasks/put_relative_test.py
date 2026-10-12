"""Prompt sampling must use the scorer's whole accepted directional band."""

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from abc_sim.randomization.core import RandomizationSamplingError
from abc_sim.randomization.tasks import put_relative as sampling
from abc_sim.task_eval.put_relative import PutRelativeEvaluator
from abc_sim.task_specs import get_task_spec


def objects():
    return [
        {
            "name": f"relative_obj_{i:02d}",
            "joint": f"relative_obj_{i:02d}_joint",
            "category": "mug",
            "label": f"mug {i}",
            "variant": "tiny",
            "x": x,
            "y": y,
            "z": 0.77,
            "footprint_radius": 0.02,
            "qw": 1.0,
            "qz": 0.0,
        }
        for i, (x, y) in enumerate(
            ((0.43, 0.36), (0.67, 0.36), (0.43, -0.36), (0.67, -0.36)), 1
        )
    ]


def evaluator_for(selections, prompt):
    bodies = "".join(
        f'<body name="{o["name"]}" pos="{o["x"]:.4f} {o["y"]:.4f} {o["z"]:.4f}">'
        f'<freejoint name="{o["joint"]}"/><geom type="sphere" size=".02"/></body>'
        for o in selections
    )
    model = mujoco.MjModel.from_xml_string(
        f"<mujoco><worldbody>{bodies}</worldbody></mujoco>"
    )
    evaluator = PutRelativeEvaluator(model=model, spec=get_task_spec("put_relative"))
    state = SimpleNamespace(
        metadata=prompt,
        object_states={
            o["joint"]: {"pos": [o["x"], o["y"], o["z"]]} for o in selections
        },
    )
    evaluator.configure_from_randomization(model, state)
    return evaluator, mujoco.MjData(model)


def test_old_exact_goal_rejection_misses_already_solved_band():
    selections = objects()
    mover, reference = selections[:2]
    goal_xy = np.asarray([reference["x"] - 0.14, reference["y"]])
    assert sampling._put_relative_goal_ok(
        goal_xy, mover_name=mover["name"], mover_radius=0.02, objects=selections
    )
    # This fixture used to survive the sampler's exact-goal radius rejection.
    assert np.linalg.norm(np.asarray([mover["x"], mover["y"]]) - goal_xy) > 0.055 + 0.02
    evaluator, data = evaluator_for(
        selections,
        {
            "mover_object": mover["name"],
            "reference_object": reference["name"],
            "direction_offset_xy": [-0.14, 0],
        },
    )
    assert evaluator.evaluate_qpos_batch(data.qpos[None]).scalar_success()


def test_no_sampled_prompt_passes_the_real_scorer_at_reset():
    selections = objects()
    prompts = [
        sampling._put_relative_sample_prompt(selections, np.random.default_rng(seed))
        for seed in range(200)
    ]
    assert len({p["prompt"] for p in prompts}) > 1
    for prompt in prompts:
        evaluator, data = evaluator_for(selections, prompt)
        assert not evaluator.evaluate_qpos_batch(data.qpos[None]).scalar_success(), (
            prompt
        )


def test_xml_rounding_at_goal_band_boundary_is_rejected():
    selections = objects()
    selections[0].update(x=0.730049, y=0.0)
    selections[1].update(x=0.43, y=0.0)
    already_solved = {
        "mover_object": selections[0]["name"],
        "reference_object": selections[1]["name"],
        "direction_offset_xy": [0.14, 0],
    }
    evaluator, data = evaluator_for(selections, already_solved)
    # The unrounded displacement is > .30; actual serialized qpos passes.
    assert selections[0]["x"] - selections[1]["x"] > 0.30
    assert evaluator.evaluate_qpos_batch(data.qpos[None]).scalar_success()
    for seed in range(200):
        prompt = sampling._put_relative_sample_prompt(
            selections, np.random.default_rng(seed)
        )
        evaluator, data = evaluator_for(selections, prompt)
        assert not evaluator.evaluate_qpos_batch(data.qpos[None]).scalar_success(), (
            prompt
        )


@pytest.mark.parametrize("seed", [None, 42])
def test_transient_layout_failure_retries_inside_randomize(monkeypatch, seed):
    randomizer = sampling.PutRelativeRandomizer()
    attempts = []

    def sample(rng):
        attempts.append(rng)
        if len(attempts) == 1:
            raise RandomizationSamplingError(
                "put_relative could not place non-overlapping objects"
            )
        return objects()

    monkeypatch.setattr(randomizer, "_sample_selections", sample)
    monkeypatch.setattr(sampling, "_put_relative_build_xml", lambda _: "unused")
    state = randomizer.randomize(None, None, seed=seed)
    assert len(attempts) == 2
    assert (
        attempts[0] is attempts[1]
    )  # Continue RNG instead of repeating the failed scene.
    evaluator, data = evaluator_for(objects(), state.metadata)
    assert not evaluator.evaluate_qpos_batch(data.qpos[None]).scalar_success()


def test_layout_exhaustion_itself_is_retryable(monkeypatch):
    randomizer = sampling.PutRelativeRandomizer()
    monkeypatch.setattr(randomizer, "_get_variants", lambda _: [Path("fake")])
    monkeypatch.setattr(
        sampling,
        "_put_relative_variant_metrics",
        lambda _: {
            "spawn_z": 0.77,
            "footprint_radius": 0.5,
        },
    )
    with pytest.raises(RandomizationSamplingError, match="non-overlapping objects"):
        randomizer._sample_selections(np.random.default_rng(42))


def test_exhausted_layout_sampling_raises_retryable_error(monkeypatch):
    randomizer = sampling.PutRelativeRandomizer()
    attempts = []

    def sample(rng):
        attempts.append(rng)
        raise RandomizationSamplingError("cannot place objects")

    monkeypatch.setattr(randomizer, "_sample_selections", sample)
    with pytest.raises(
        RandomizationSamplingError, match="could not sample a valid relative prompt"
    ):
        randomizer.randomize(None, None, seed=42)
    assert len(attempts) == sampling._PUT_RELATIVE_MAX_PROMPT_TRIES


def test_asset_errors_are_not_retried(monkeypatch):
    randomizer = sampling.PutRelativeRandomizer()
    attempts = []

    def sample(rng):
        attempts.append(rng)
        raise RuntimeError("invalid asset collision proxy")

    monkeypatch.setattr(randomizer, "_sample_selections", sample)
    with pytest.raises(RuntimeError, match="invalid asset"):
        randomizer.randomize(None, None, seed=42)
    assert len(attempts) == 1
