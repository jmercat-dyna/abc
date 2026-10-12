"""Both native eval loops honor completed countdowns and legacy semantics."""

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from abc_minimal.batched_eval import over
from abc_minimal.config import SimEvalConfig
from abc_minimal.eval_policy import (
    build_summary,
    rollout_over,
    rollout_worlds,
    world_success_flags,
)
from abc_sim.env import MuJoCoYAMEnv
from abc_sim.task_eval.base import TaskEvalResult


@pytest.mark.parametrize(
    "complete, failed, expected",
    [
        (False, False, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ],
)
@pytest.mark.parametrize(
    "semantics",
    [
        "held_countdown_complete",
        "latched_success_until_unpicked_object_lost",
        "placed_target_settle_until_terminal_failure",
    ],
)
def test_countdown_termination_in_scalar_and_batched_loops(
    complete, failed, expected, semantics
):
    metrics = {
        "success_semantics": semantics,
        "ever_success": complete,
        "ever_failed": failed,
    }
    assert rollout_over(metrics) is expected
    metrics = {
        **metrics,
        "ever_success": np.asarray([complete]),
        "ever_failed": np.asarray([failed]),
    }
    result = TaskEvalResult(np.zeros(1), np.asarray([complete]), metrics)
    np.testing.assert_array_equal(over(result, 1), [expected])


def test_legacy_maintenance_does_not_terminate_at_reset():
    assert not rollout_over({"ever_success": True, "ever_failed": False})
    result = TaskEvalResult(
        np.ones(2),
        np.ones(2, dtype=bool),
        {
            "ever_success": np.ones(2, dtype=bool),
            "ever_failed": np.asarray([False, True]),
        },
    )
    np.testing.assert_array_equal(over(result, 2), [False, True])


@pytest.mark.parametrize(
    "qualified, failed, timed_out, expected",
    [
        (False, False, False, False),
        (True, False, False, False),
        (True, True, False, True),
        (False, False, True, True),
        (True, False, True, True),
    ],
)
def test_lifted_countdown_continues_until_timeout_or_loss(
    qualified, failed, timed_out, expected
):
    metrics = {
        "success_semantics": "held_lifted_countdown_then_survive_timeout",
        "ever_success": qualified,
        "ever_failed": failed,
        "timed_out": timed_out,
    }
    assert rollout_over(metrics) is expected
    result = TaskEvalResult(
        np.ones(1),
        np.asarray([qualified]),
        {
            **metrics,
            "ever_success": np.asarray([qualified]),
            "ever_failed": np.asarray([failed]),
            "timed_out": np.asarray([timed_out]),
        },
    )
    np.testing.assert_array_equal(over(result, 1), [expected])


def test_regular_task_still_terminates_on_success():
    assert rollout_over({"ever_success": True})
    assert not rollout_over({"ever_success": False})


@pytest.mark.parametrize(
    "evaluator_failed, diverged", [(True, False), (False, True), (True, True)]
)
def test_world_reports_mask_conflicting_success_on_terminal_failure(
    evaluator_failed, diverged
):
    assert world_success_flags(
        {"success": True, "ever_success": True, "ever_failed": evaluator_failed},
        failed=diverged,
    ) == (False, False)


def test_world_reports_preserve_success_without_terminal_failure():
    assert world_success_flags({"success": True, "ever_success": True}) == (True, True)


@pytest.mark.parametrize("timed_out", [False, True])
def test_ball_final_world_flags_require_observed_timeout(timed_out):
    flags = world_success_flags(
        {
            "success": True,
            "ever_success": True,
            "timed_out": timed_out,
            "success_semantics": "held_lifted_countdown_then_survive_timeout",
        }
    )
    assert flags == (timed_out, timed_out)


def test_native_summary_excludes_simulator_errors_from_model_denominator(tmp_path):
    worlds = [
        {
            "success": True,
            "reward": 1,
            "max_reward": 1,
            "final_task_eval": {},
            "sim_error": None,
        },
        {
            "success": False,
            "reward": 0.25,
            "max_reward": 0.5,
            "final_task_eval": {"ever_failed": True},
            "sim_error": None,
        },
        {
            "success": True,
            "reward": 100,
            "max_reward": 100,
            "final_task_eval": {},
            "sim_error": "QACC auto-reset",
        },
    ]
    summary = build_summary(
        config=SimEvalConfig(checkpoint="unused"),
        ckpt_path=Path("unused"),
        device="cpu",
        worlds=worlds,
        out_dir=tmp_path,
    )
    assert summary["num_valid_worlds"] == 2
    assert summary["num_worlds"] == 3
    assert summary["num_attempted_worlds"] == 3
    assert summary["num_sim_errors"] == summary["num_failure"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["mean_reward"] == 0.625
    assert summary["mean_max_progress"] == 0.75
    assert len(summary["worlds"]) == 3
    invalid_only = build_summary(
        config=SimEvalConfig(checkpoint="unused"),
        ckpt_path=Path("unused"),
        device="cpu",
        worlds=worlds[2:],
        out_dir=tmp_path,
    )
    assert invalid_only["success_rate"] is None
    assert invalid_only["mean_reward"] is None
    assert invalid_only["num_valid_worlds"] == invalid_only["num_failure"] == 0
    assert invalid_only["num_worlds"] == 1


def test_native_real_autoreset_is_invalid_and_video_closes_before_reset_frame(
    tmp_path, monkeypatch
):
    import imageio.v2 as imageio

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body pos="0 0 1">'
        '<freejoint/><geom type="sphere" size=".01" mass=".01"/></body></worldbody></mujoco>'
    )
    physics = object.__new__(MuJoCoYAMEnv)
    physics.model, physics.data = model, mujoco.MjData(model)
    physics._task_runtime = None
    physics._control_decimation = 17
    frames = []
    closed = []
    monkeypatch.setattr(
        imageio,
        "get_writer",
        lambda *_, **__: SimpleNamespace(
            append_data=lambda frame: frames.append(frame),
            close=lambda: closed.append(True),
        ),
    )
    obs = {
        "state": np.zeros(14),
        "images": {"cam": np.zeros((3, 4, 4), dtype=np.uint8)},
    }

    def step_one(_action):
        physics.data.xfrc_applied[1, 3] = 1e30
        physics.advance_physics()

    env = SimpleNamespace(
        reset=lambda **_: obs,
        obs=lambda: obs,
        evaluate=lambda: {"reward": 1, "success": True, "ever_success": True},
        step_one=step_one,
        render_cameras=lambda: obs["images"],
        randomization=None,
        close=lambda: None,
    )
    config = SimEvalConfig(
        checkpoint="unused",
        num_worlds=1,
        num_chunks=2,
        execute_chunk_dim=2,
        fast_inference=False,
        rtc=False,
        save_video=True,
    )
    policy = SimpleNamespace(infer=lambda *_, **__: np.zeros((2, 14)))
    worlds = rollout_worlds(
        config,
        policy,
        env,
        0,
        None,
        tmp_path,
        SimpleNamespace(chunk_length=2, action_dim=14, camera_keys=("cam",)),
    )
    assert len(worlds) == 1
    world = worlds[0]
    assert world["status"] == "simulation_error"
    assert "mjWARN_BADQACC" in world["sim_error"]
    assert world["success"] is None and world["final_success"] is None
    assert world["steps"] == 1
    assert len(frames) == 1 and closed == [True]
    assert world["video_path"] == str(tmp_path / "world_000.mp4")


def test_nonfinite_diagnostics_use_strict_json_null():
    from abc_minimal.eval_policy import jsonable
    import json

    output = jsonable({"unset": np.array([np.nan, np.inf]), "scalar": np.float32(np.nan)})
    assert output == {"unset": [None, None], "scalar": None}
    assert json.loads(json.dumps(output, allow_nan=False)) == output
