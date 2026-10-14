"""Property-based tests (Hypothesis).

Instead of hand-picked examples, Hypothesis generates hundreds of arbitrary
workloads and scheduling decisions per test (tiny, huge, simultaneous
arrivals, idle gaps, extreme bursts...) and, if a property ever fails,
automatically shrinks the input to the smallest failing example.
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

from simulator import CPUSimulator
from workload_generator import Process
from fcfs import run_fcfs
from sjf import run_sjf
from priority import run_priority
from round_robin import run_round_robin
from gym_env import CPUSchedulingEnv
import evaluate as ev

SETTINGS = settings(max_examples=300, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])


@st.composite
def workloads(draw, max_size=25):
    n = draw(st.integers(1, max_size))
    arrivals = draw(st.lists(st.integers(0, 60), min_size=n, max_size=n))
    bursts = draw(st.lists(st.integers(1, 40), min_size=n, max_size=n))
    prios = draw(st.lists(st.integers(1, 10), min_size=n, max_size=n))
    return [Process(f"P{i + 1}", a, b, p)
            for i, (a, b, p) in enumerate(zip(arrivals, bursts, prios))]


POLICIES = {
    "FCFS": run_fcfs,
    "SJF": run_sjf,
    "Priority": run_priority,
    "RR q=1": lambda sim: run_round_robin(sim, 1),
    "RR q=4": lambda sim: run_round_robin(sim, 4),
}


@SETTINGS
@given(workload=workloads(), policy=st.sampled_from(sorted(POLICIES)))
def test_conservation_laws_hold_for_any_workload(workload, policy):
    sim = CPUSimulator(workload)
    m = POLICIES[policy](sim)
    bursts = {p.id: p.burst_time for p in workload}

    assert sorted(p.id for p in sim.completed) == sorted(bursts)
    for sp in sim.completed:
        assert sp.waiting_time == sp.turnaround_time - bursts[sp.id] >= 0
    # The CPU does exactly sum(bursts) units of work and never runs two things at once.
    assert sum(e - s for _, s, e in sim.timeline) == sum(bursts.values())
    assert all(a[2] <= b[1] for a, b in zip(sim.timeline, sim.timeline[1:]))
    # Little's-law style identity: avg turnaround = avg waiting + avg burst.
    assert m["avg_turnaround_time"] == pytest.approx(
        m["avg_waiting_time"] + sum(bursts.values()) / len(bursts))
    assert 0 < m["fairness_index"] <= 1 + 1e-9
    assert m["max_waiting_time"] >= m["avg_waiting_time"] - 1e-9


@SETTINGS
@given(workload=workloads())
def test_makespan_is_identical_for_every_work_conserving_scheduler(workload):
    """All these schedulers keep the CPU busy whenever work exists, so they
    must all finish the whole workload at exactly the same time."""
    finish = set()
    for run in POLICIES.values():
        sim = CPUSimulator(workload)
        run(sim)
        finish.add(sim.time)
    assert len(finish) == 1


@SETTINGS
@given(workload=workloads(), choices=st.lists(st.integers(0, 10 ** 6), min_size=200, max_size=200))
def test_any_sequence_of_decisions_keeps_the_simulator_consistent(workload, choices):
    sim = CPUSimulator(workload)
    i = 0
    while not sim.is_done():
        if not sim.ready_queue:
            sim.advance_to_next_arrival()
            continue
        c = choices[i % len(choices)]
        i += 1
        sim.step(c % len(sim.ready_queue), time_slice=[None, 1, 3][c % 3])
    for sp in sim.completed:
        assert sp.waiting_time == sp.turnaround_time - sp.burst_time


@SETTINGS
@given(workload=workloads(max_size=8))
def test_sjf_never_worse_than_fcfs_when_all_jobs_arrive_together(workload):
    same_time = [Process(p.id, 0, p.burst_time, p.priority) for p in workload]
    assert (run_sjf(CPUSimulator(same_time))["avg_waiting_time"]
            <= run_fcfs(CPUSimulator(same_time))["avg_waiting_time"] + 1e-9)


@SETTINGS
@given(workload=workloads(), shift=st.integers(1, 1000))
def test_shifting_all_arrivals_in_time_changes_nothing(workload, shift):
    shifted = [Process(p.id, p.arrival_time + shift, p.burst_time, p.priority) for p in workload]
    for run in POLICIES.values():
        a, b = run(CPUSimulator(workload)), run(CPUSimulator(shifted))
        assert a == pytest.approx(b)


@SETTINGS
@given(workload=workloads(max_size=30), actions=st.lists(st.integers(-5, 50), min_size=1))
def test_environment_survives_arbitrary_actions(workload, actions):
    env = CPUSchedulingEnv(workload=workload)
    obs, info = env.reset()
    done, i = False, 0
    while not done:
        assert env.observation_space.contains(obs)
        obs, r, done, _, info = env.step(actions[i % len(actions)])
        assert np.isfinite(r)
        i += 1
    assert len(env.sim.completed) == len(workload)


@settings(max_examples=100, deadline=None)
@given(a=st.lists(st.floats(-100, 100), min_size=2, max_size=40))
def test_permutation_test_p_value_is_valid_and_symmetric(a):
    b = [x + 1.0 for x in a]
    p_ab = ev.paired_permutation_test(a, b)
    assert 0 < p_ab <= 1
    assert p_ab == pytest.approx(ev.paired_permutation_test(b, a))
