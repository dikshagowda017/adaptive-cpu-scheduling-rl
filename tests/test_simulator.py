"""Simulator correctness: hand-computed golden values, plus invariants that
must hold for ANY scheduling decision on ANY workload."""

import random

import pytest

from conftest import SCHEDULERS, random_workloads
from simulator import CPUSimulator, run_policy
from workload_generator import Process, generate_workload


# ---------------------------------------------------------------- golden values
# Appendix A workload: P1(arr 0, burst 5, pr 2)  P2(1, 3, 1)  P3(2, 8, 4)
#                      P4(3, 6, 3)              P5(4, 2, 5)
# Worked out by hand; the full working is in TESTING.md.
GOLDEN = {
    # name: (timeline, waiting time of P1..P5, avg turnaround, context switches)
    "fcfs": ([("P1", 0, 5), ("P2", 5, 8), ("P3", 8, 16), ("P4", 16, 22), ("P5", 22, 24)],
             [0, 4, 6, 13, 18], 13.0, 4),
    "sjf": ([("P1", 0, 5), ("P5", 5, 7), ("P2", 7, 10), ("P4", 10, 16), ("P3", 16, 24)],
            [0, 6, 14, 7, 1], 10.4, 4),
    "priority": ([("P1", 0, 5), ("P2", 5, 8), ("P4", 8, 14), ("P3", 14, 22), ("P5", 22, 24)],
                 [0, 4, 12, 5, 18], 12.6, 4),
    "rr_q3": ([("P1", 0, 3), ("P2", 3, 6), ("P3", 6, 9), ("P4", 9, 12), ("P1", 12, 14),
               ("P5", 14, 16), ("P3", 16, 19), ("P4", 19, 22), ("P3", 22, 24)],
              [9, 2, 14, 13, 10], 14.4, 8),
}


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_report_sample_workload_golden_values(name, sample_workload):
    timeline, waits, avg_tat, switches = GOLDEN[name]
    sim = CPUSimulator(sample_workload)
    metrics = SCHEDULERS[name](sim)

    assert sim.timeline == timeline
    by_id = {p.id: p.waiting_time for p in sim.completed}
    assert [by_id[f"P{i}"] for i in range(1, 6)] == waits
    assert metrics["avg_waiting_time"] == pytest.approx(sum(waits) / 5)
    assert metrics["avg_turnaround_time"] == pytest.approx(avg_tat)
    assert metrics["context_switches"] == switches
    assert metrics["max_waiting_time"] == max(waits)


def test_round_robin_response_time(sample_workload):
    # First CPU access: P1 at 0, P2 at 3, P3 at 6, P4 at 9, P5 at 14
    # -> response times 0, 2, 4, 6, 10 -> average 4.4
    m = SCHEDULERS["rr_q3"](CPUSimulator(sample_workload))
    assert m["avg_response_time"] == pytest.approx(4.4)


def test_mid_run_arrival_waiting_time_is_counted():
    """Regression for bug B1: a process that arrives while another is running
    used to start with waiting_time = 0 instead of (now - arrival)."""
    sim = CPUSimulator([Process("A", 0, 10, 1), Process("B", 3, 1, 1)])
    sim.step(0)                 # A runs 0 -> 10; B arrived at 3
    assert sim.ready_queue[0].waiting_time == 7
    sim.step(0)
    assert sim.get_metrics()["avg_waiting_time"] == pytest.approx(3.5)


def test_waiting_added_reports_all_new_waiting(sample_workload):
    sim = CPUSimulator(sample_workload)
    _, _, info = sim.step(0)    # P1 runs 0 -> 5 alone; P2..P5 arrive at 1..4
    assert info["waiting_added"] == (5 - 1) + (5 - 2) + (5 - 3) + (5 - 4)


# ------------------------------------------------------------------ invariants
def _check_invariants(workload, sim, metrics):
    by_id = {p.id: p for p in workload}
    assert len(sim.completed) == len(workload), "every process must finish"
    assert sim.is_done()
    for sp in sim.completed:
        p = by_id[sp.id]
        assert sp.remaining_burst_time == 0
        assert sp.turnaround_time == sp.completion_time - p.arrival_time
        # The key invariant: waiting time is exactly turnaround minus service time.
        assert sp.waiting_time == sp.turnaround_time - p.burst_time
        assert sp.waiting_time >= 0
        assert p.arrival_time <= sp.start_time <= sp.completion_time

    # Timeline: no overlaps, time only moves forward, each process gets exactly its burst.
    for (_, _, end), (_, start, _) in zip(sim.timeline, sim.timeline[1:]):
        assert start >= end
    service = {}
    for pid, start, end in sim.timeline:
        assert end > start
        assert start >= by_id[pid].arrival_time
        service[pid] = service.get(pid, 0) + end - start
    assert service == {p.id: p.burst_time for p in workload}

    # Context switches = number of times the running process changes.
    changes = sum(a[0] != b[0] for a, b in zip(sim.timeline, sim.timeline[1:]))
    assert metrics["context_switches"] == changes

    # Work conserving: the CPU never sits idle while a process is ready, so in
    # an idle gap every process that had arrived must already be finished.
    completion = {sp.id: sp.completion_time for sp in sim.completed}
    for (_, _, end), (_, start, _) in zip(sim.timeline, sim.timeline[1:]):
        if start > end:
            assert all(completion[p.id] <= end for p in workload if p.arrival_time <= end)
            assert start == min(p.arrival_time for p in workload if p.arrival_time > end)

    assert 0 < metrics["fairness_index"] <= 1 + 1e-9


@pytest.mark.parametrize("workload", random_workloads(40), ids=lambda w: f"n{len(w)}")
def test_invariants_hold_for_every_scheduler(workload, scheduler):
    _, run = scheduler
    sim = CPUSimulator(workload)
    _check_invariants(workload, sim, run(sim))


@pytest.mark.parametrize("seed", range(25))
def test_invariants_hold_for_random_decisions(seed):
    """Fuzz test: pick ANY ready process at random, with random preemption."""
    rng = random.Random(seed)
    dist = ["exponential", "uniform", "bimodal"][seed % 3]
    workload = generate_workload(1 + seed % 15, seed=seed, burst_distribution=dist)
    sim = CPUSimulator(workload)
    while not sim.is_done():
        if not sim.ready_queue:
            sim.advance_to_next_arrival()
            continue
        sim.step(rng.randrange(len(sim.ready_queue)), time_slice=rng.choice([None, 1, 2, 5]))
    _check_invariants(workload, sim, sim.get_metrics())


# ----------------------------------------------------------------- edge cases
def test_single_process_with_late_arrival():
    sim = CPUSimulator([Process("P1", 5, 4, 3)])
    m = run_policy(sim, lambda q: 0)
    assert sim.timeline == [("P1", 5, 9)]
    assert m["avg_waiting_time"] == 0
    assert m["avg_turnaround_time"] == 4
    assert m["context_switches"] == 0
    assert m["fairness_index"] == pytest.approx(1.0)


def test_idle_gap_between_processes():
    sim = CPUSimulator([Process("P1", 0, 2, 1), Process("P2", 10, 3, 1)])
    m = run_policy(sim, lambda q: 0)
    assert sim.timeline == [("P1", 0, 2), ("P2", 10, 13)]
    assert m["avg_waiting_time"] == 0
    assert sim.time == 13


def test_simultaneous_arrivals_keep_id_order():
    procs = [Process("P1", 0, 3, 1), Process("P2", 0, 3, 1), Process("P3", 0, 3, 1)]
    assert [p.id for p in CPUSimulator(procs).ready_queue] == ["P1", "P2", "P3"]


def test_unsorted_input_is_handled():
    sim = CPUSimulator([Process("P2", 4, 1, 1), Process("P1", 0, 2, 1)])
    run_policy(sim, lambda q: 0)
    assert [pid for pid, _, _ in sim.timeline] == ["P1", "P2"]


def test_preempted_process_queues_behind_new_arrivals():
    sim = CPUSimulator([Process("A", 0, 4, 1), Process("B", 1, 1, 1)])
    sim.step(0, time_slice=2)
    assert [p.id for p in sim.ready_queue] == ["B", "A"]


def test_same_process_rescheduled_is_not_a_context_switch():
    sim = CPUSimulator([Process("A", 0, 4, 1)])
    sim.step(0, time_slice=1)
    sim.step(0, time_slice=1)
    assert sim.context_switches == 0


def test_step_on_empty_queue_raises():
    with pytest.raises(ValueError):
        CPUSimulator([Process("P1", 5, 1, 1)]).step(0)


@pytest.mark.parametrize("index", [-1, 1, 99])
def test_step_with_bad_index_raises(index):
    with pytest.raises(IndexError):
        CPUSimulator([Process("P1", 0, 1, 1)]).step(index)


@pytest.mark.parametrize("slice_", [0, -2])
def test_non_positive_time_slice_raises(slice_):
    with pytest.raises(ValueError):
        CPUSimulator([Process("P1", 0, 3, 1)]).step(0, time_slice=slice_)


def test_reset_restores_initial_state_and_does_not_mutate_workload(sample_workload):
    original = [Process(p.id, p.arrival_time, p.burst_time, p.priority) for p in sample_workload]
    sim = CPUSimulator(sample_workload)
    first = run_policy(sim, lambda q: 0)
    second = run_policy(sim, lambda q: 0)       # run_policy resets first
    assert first == second
    assert sample_workload == original
    sim.reset()
    assert sim.time == 0 and sim.completed == [] and sim.timeline == []
    assert sim.context_switches == 0


def test_metrics_before_anything_finishes():
    m = CPUSimulator([Process("P1", 0, 3, 1)]).get_metrics()
    assert m["avg_waiting_time"] == 0.0 and m["fairness_index"] == 1.0


def test_fairness_index_is_one_when_every_job_has_equal_slowdown():
    sim = CPUSimulator([Process("A", 0, 2, 1), Process("B", 2, 6, 1)])
    assert run_policy(sim, lambda q: 0)["fairness_index"] == pytest.approx(1.0)


def test_fairness_index_is_not_degenerate_for_near_zero_waits():
    """Regression for bug B3: Jain's index over raw waiting times reported 0.2
    ("very unfair") when one of five processes waited 1 unit and the rest 0."""
    procs = [Process("P1", 0, 1, 1), Process("P2", 0, 5, 1), Process("P3", 10, 5, 1),
             Process("P4", 20, 5, 1), Process("P5", 30, 5, 1)]
    m = run_policy(CPUSimulator(procs), lambda q: 0)
    assert m["avg_waiting_time"] == pytest.approx(0.2)
    assert m["fairness_index"] > 0.95


def test_fairness_index_drops_when_a_short_job_starves():
    procs = [Process("Long", 0, 20, 1), Process("Short", 0, 1, 1)]
    unfair = run_policy(CPUSimulator(procs), lambda q: 0)       # short job waits 20
    fair = run_policy(CPUSimulator(procs), lambda q: len(q) - 1)
    assert unfair["fairness_index"] < fair["fairness_index"]
