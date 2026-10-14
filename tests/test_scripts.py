"""Smoke tests: every script a user is told to run must actually run."""

import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run(*args, cwd=ROOT):
    result = subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"},
                            timeout=300)
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.parametrize("script,expected", [
    ("test.py", "All good"),
    ("workload_generator.py", "Generated 10 processes"),
    ("simulator.py", "avg_waiting_time"),
    ("fcfs.py", "FCFS metrics"),
    ("sjf.py", "SJF metrics"),
    ("round_robin.py", "Round Robin metrics"),
    ("priority.py", "Priority scheduling metrics"),
    ("gym_env.py", "Episode finished"),
    ("q_learning.py", "Final Q-table size"),
    ("evaluate.py", "Hypothesis"),
])
def test_module_demo_runs(script, expected):
    assert expected in _run(script)


def test_main_demo_shows_hand_checkable_fcfs_result():
    out = _run("main.py", "demo", "--episodes", "200")
    assert "avg_waiting_time=8.20" in out          # FCFS on Appendix A
    assert "RL (Q-learning)" in out


def test_main_rejects_bad_arguments():
    result = subprocess.run([sys.executable, "main.py", "--episodes", "0"], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode != 0
