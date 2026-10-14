"""Robustness: bad input is rejected with a clear message (never a crash or a
silently wrong answer), plus performance limits so the simulator scales."""

import os
import subprocess
import sys
import time

import pytest

from simulator import CPUSimulator, QueueView
from workload_generator import Process, generate_workload, load_workload_csv
from fcfs import run_fcfs
from round_robin import run_round_robin
from sjf import run_sjf
from gym_env import CPUSchedulingEnv
from q_learning import train, run_agent, QLearningAgent, ProcessQLearningAgent

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLES = os.path.join(ROOT, "examples")


# ------------------------------------------------------------ input validation
@pytest.mark.parametrize("bad, message", [
    ([Process("P1", 0, 0, 1)], "burst_time must be >= 1"),          # B12: used to crash (ZeroDivisionError)
    ([Process("P1", 0, -3, 1)], "burst_time must be >= 1"),
    ([Process("P1", -5, 3, 1)], "arrival_time must be >= 0"),       # B13: used to inflate waiting silently
    ([Process("P1", 0, 3, 1), Process("P1", 1, 2, 1)], "duplicate"),  # B14: broke switch counting
    ([Process("P1", 0, 2.5, 1)], "whole number"),
    ([Process("P1", "0", 2, 1)], "whole number"),
    ([Process("P1", 0, True, 1)], "whole number"),
    ([Process("P1", 0, 2, None)], "whole number"),
])
def test_invalid_workloads_are_rejected_with_clear_message(bad, message):
    with pytest.raises(ValueError, match=message):
        CPUSimulator(bad)


def test_numpy_integers_are_accepted():
    import numpy as np
    sim = CPUSimulator([Process("P1", np.int64(0), np.int32(3), np.int8(1))])
    assert run_fcfs(sim)["avg_turnaround_time"] == 3


def test_empty_workload_is_handled():
    m = run_fcfs(CPUSimulator([]))
    assert m["avg_waiting_time"] == 0 and m["context_switches"] == 0


def test_step_returns_read_only_view():
    sim = CPUSimulator([Process("A", 0, 2, 1), Process("B", 0, 2, 1)])
    view, _, _ = sim.step(0)
    assert isinstance(view, QueueView) and len(view) == 1 and view[0].id == "B"
    with pytest.raises((TypeError, AttributeError)):
        view.append(None)


# ------------------------------------------------------------------ CSV loader
def test_example_csv_matches_report_sample():
    from workload_generator import REPORT_SAMPLE_WORKLOAD
    assert load_workload_csv(os.path.join(EXAMPLES, "report_appendix_a.csv")) == REPORT_SAMPLE_WORKLOAD


@pytest.mark.parametrize("name", ["convoy_effect.csv", "textbook_simultaneous.csv"])
def test_all_example_csvs_load_and_run(name):
    w = load_workload_csv(os.path.join(EXAMPLES, name))
    assert run_sjf(CPUSimulator(w))["avg_waiting_time"] <= run_fcfs(CPUSimulator(w))["avg_waiting_time"]


def test_textbook_example_answer():
    w = load_workload_csv(os.path.join(EXAMPLES, "textbook_simultaneous.csv"))
    assert run_sjf(CPUSimulator(w))["avg_waiting_time"] == pytest.approx(7.0)


def test_csv_aliases_optional_priority_bom_and_blank_lines(tmp_path):
    f = tmp_path / "w.csv"
    f.write_text("﻿PID, Arrival_Time ,BT\nB,3,4\n\nA,0,2\n", encoding="utf-8")
    w = load_workload_csv(f)
    assert w == [Process("A", 0, 2, 1), Process("B", 3, 4, 1)]


@pytest.mark.parametrize("content, message", [
    ("", "empty"),
    ("id,arrival\nP1,0\n", "missing column 'burst"),
    ("id,arrival,burst\nP1,zero,3\n", "line 2: arrival_time must be a whole number"),
    ("id,arrival,burst\nP1,0,3\nP1,1,2\n", "line 3: duplicate"),
    ("id,arrival,burst\nP1,0,0\n", "line 2: burst must be >= 1"),
    ("id,arrival,burst\nP1,-1,2\n", "line 2: arrival must be >= 0"),
    ("id,arrival,burst\n,0,2\n", "empty process id"),
    ("id,arrival,burst\n", "no processes"),
])
def test_bad_csv_files_give_line_numbered_errors(tmp_path, content, message):
    f = tmp_path / "bad.csv"
    f.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_workload_csv(f)


# ------------------------------------------------------------ agent/env checks
def test_agent_and_environment_size_mismatch_is_caught():
    agent = QLearningAgent(num_actions=10)
    with pytest.raises(ValueError, match="max_queue_size=10"):
        run_agent(agent, CPUSchedulingEnv(max_queue_size=20))


@pytest.mark.parametrize("kwargs", [
    {"num_episodes": 0}, {"num_processes": 0}, {"num_processes": (5, 2)}, {"seed": -1},
])
def test_train_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        train(verbose_every=0, **kwargs)


def test_training_uses_a_range_of_workload_sizes(monkeypatch):
    sizes = []
    original = CPUSchedulingEnv.reset

    def spy(self, *, seed=None, options=None):
        sizes.append(self.num_processes)
        return original(self, seed=seed, options=options)

    monkeypatch.setattr(CPUSchedulingEnv, "reset", spy)
    train(num_episodes=300, num_processes=(4, 20), verbose_every=0)
    assert min(sizes) == 4 and max(sizes) == 20


def test_per_process_agent_handles_queues_longer_than_it_can_see():
    agent, _ = train(num_episodes=200, verbose_every=0)
    env = CPUSchedulingEnv(num_processes=60, max_queue_size=agent.num_actions,
                           workload_kwargs={"mean_inter_arrival": 0.3})
    m = run_agent(agent, env, seed=5)
    assert len(env.sim.completed) == 60 and m["avg_waiting_time"] > 0
    assert isinstance(agent, ProcessQLearningAgent)


# ------------------------------------------------------------------ command line
def _cli(*args):
    return subprocess.run([sys.executable, "main.py", *args], cwd=ROOT, capture_output=True,
                          text=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})


@pytest.mark.parametrize("args, message", [
    (["--seed", "-1"], "--seed must be >= 0"),                 # B15: used to print a NumPy traceback
    (["--episodes", "0"], "must be >= 1"),
    (["--workloads", "-4"], "must be >= 1"),
    (["train", "--workload", "x.csv"], "only be used with the demo"),
    (["demo", "--workload", "does_not_exist.csv"], "Could not load workload"),
])
def test_cli_rejects_bad_arguments_without_traceback(args, message):
    result = _cli(*args)
    assert result.returncode != 0
    assert message in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_demo_with_custom_workload(tmp_path):
    result = _cli("demo", "--workload", os.path.join("examples", "convoy_effect.csv"),
                  "--episodes", "300", "--out", str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "Ranking by average waiting time" in result.stdout
    assert (tmp_path / "gantt_demo.png").exists()


# ------------------------------------------------------------------ performance
@pytest.mark.parametrize("run", [run_fcfs, lambda s: run_round_robin(s, 3), run_sjf])
def test_ten_thousand_processes_simulate_quickly(run):
    """B16: the simulator used to touch every queued process on every step
    (and copy the queue), so 30 000 processes took 15 s. Now ~0.8 s."""
    w = generate_workload(10_000, seed=0, mean_inter_arrival=6)
    t0 = time.perf_counter()
    m = run(CPUSimulator(w))
    assert time.perf_counter() - t0 < 5.0
    assert m["avg_turnaround_time"] > 0


def test_overloaded_system_stays_fast():
    w = generate_workload(20_000, seed=0, mean_inter_arrival=1)   # queue grows to ~16 000
    t0 = time.perf_counter()
    run_fcfs(CPUSimulator(w))
    assert time.perf_counter() - t0 < 5.0
