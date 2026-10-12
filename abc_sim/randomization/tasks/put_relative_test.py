"""Prompt sampling must use the scorer's whole accepted directional band."""

from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from abc_sim.put_relative_config import PutRelativeConfig
from abc_sim.randomization.core import RandomizationSamplingError
from abc_sim.randomization.tasks import put_relative as sampling
from abc_sim.task_eval.put_relative import PutRelativeEvaluator
from abc_sim.task_eval.registry import make_task_evaluator
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


def evaluator_for(selections, prompt, **options):
    bodies = "".join(
        f'<body name="{o["name"]}" pos="{o["x"]:.4f} {o["y"]:.4f} {o["z"]:.4f}">'
        f'<freejoint name="{o["joint"]}"/><geom type="sphere" size=".02"/></body>'
        for o in selections
    )
    model = mujoco.MjModel.from_xml_string(
        f"<mujoco><worldbody>{bodies}</worldbody></mujoco>"
    )
    evaluator = PutRelativeEvaluator(
        model=model, spec=get_task_spec("put_relative"), **options
    )
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


@pytest.mark.parametrize(
    "options",
    [
        {"along_min_m": 0.03, "along_max_m": 0.8, "perp_max_m": 0.15},
        {"along_min_m": 0.08, "along_max_m": 0.2, "perp_max_m": 0.04},
    ],
)
def test_custom_scoring_config_keeps_reset_sampling_and_evaluation_consistent(options):
    config = PutRelativeConfig(**options)
    for seed in range(100):
        prompt = sampling._put_relative_sample_prompt(
            objects(), np.random.default_rng(seed), scoring_config=config
        )
        evaluator, data = evaluator_for(objects(), prompt, **asdict(config))
        assert not evaluator.evaluate_qpos_batch(data.qpos[None]).scalar_success()


def test_bound_randomizer_uses_effective_task_spec_scoring_config(monkeypatch):
    config = PutRelativeConfig(along_min_m=0.08, along_max_m=0.2, perp_max_m=0.04)
    evaluator, data = evaluator_for(
        objects(),
        sampling._put_relative_sample_prompt(objects(), np.random.default_rng(42)),
    )
    spec = replace(
        get_task_spec("put_relative"), evaluator_options=tuple(asdict(config).items())
    )
    evaluator = make_task_evaluator(evaluator.model, spec)
    assert evaluator.config == config
    randomizer = sampling.PutRelativeRandomizer()
    randomizer._env_ref = SimpleNamespace(
        _task_evaluator=evaluator,
        model=evaluator.model,
        data=data,
        _get_reset_arm_state=lambda: None,
        reload_from_xml=lambda _: None,
        _set_qpos_from_state=lambda _: None,
    )
    monkeypatch.setattr(randomizer, "_sample_selections", lambda _: objects())
    monkeypatch.setattr(sampling, "_put_relative_build_xml", lambda _: "unused")
    state = randomizer.randomize(None, None, seed=42)
    assert state.metadata["scoring_config"] == asdict(config)
    evaluator.configure_from_randomization(evaluator.model, state)
    assert not evaluator.evaluate_qpos_batch(data.qpos[None]).scalar_success()


def test_replay_rejects_mismatching_sampler_and_evaluator_config():
    prompt = sampling._put_relative_sample_prompt(objects(), np.random.default_rng(42))
    prompt["scoring_config"] = asdict(PutRelativeConfig(along_max_m=0.2))
    with pytest.raises(ValueError, match="scoring_config disagree"):
        evaluator_for(objects(), prompt)


@pytest.mark.parametrize(
    "options",
    [
        {"along_min_m": 0.3, "along_max_m": 0.2},
        {"perp_max_m": -1},
        {"along_min_m": float("nan")},
        {"placed_z_tolerance_m": float("inf")},
    ],
)
def test_invalid_scoring_config_is_rejected(options):
    with pytest.raises(ValueError):
        PutRelativeConfig(**options)


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
