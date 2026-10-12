"""Both native eval loops honor completed countdowns and legacy semantics."""

import numpy as np
import pytest

from abc_minimal.batched_eval import over
from abc_minimal.eval_policy import rollout_over, world_success_flags
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
def test_countdown_termination_in_scalar_and_batched_loops(complete, failed, expected):
    metrics = {
        "success_semantics": "held_countdown_complete",
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
