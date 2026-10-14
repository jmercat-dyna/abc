"""The simulator protocol version is separate and persists in native summaries."""

import json
import re
from pathlib import Path

from abc_minimal.config import SimEvalConfig
from abc_minimal.eval_policy import build_summary
from abc_sim.version import get_sim_version


def test_simulator_version_is_valid_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", get_sim_version())
    assert get_sim_version() == "3.1.2"


def test_native_summary_records_simulator_version(tmp_path):
    summary = build_summary(
        config=SimEvalConfig(checkpoint="unused"),
        ckpt_path=Path("unused"),
        device="cpu",
        out_dir=tmp_path,
        worlds=[],
    )
    assert summary["sim_version"] == get_sim_version()
    assert (
        json.loads((tmp_path / "summary.json").read_text())["sim_version"]
        == get_sim_version()
    )
