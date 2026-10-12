"""Simulator task/protocol version, independent of policy package releases."""

from importlib.resources import files


def get_sim_version() -> str:
    """Read the simulator protocol version bundled with the installed package."""
    return files("abc_sim").joinpath("VERSION").read_text(encoding="utf-8").strip()
