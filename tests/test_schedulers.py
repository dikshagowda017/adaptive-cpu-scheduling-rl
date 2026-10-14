from itertools import permutations

import pytest

from conftest import random_workloads
from simulator import CPUSimulator, run_policy
from workload_generator import Process, generate_workload
from fcfs import run_fcfs
from sjf import run_sjf
from priority import run_priority
from round_robin import run_round_robin


def _order(run, workload):
    sim = CPUSimulator(workload)
    run(sim)
    return [pid for pid, _, _ in sim.timeline]


def test_fcfs_runs_in_arrival_order():
    w = generate_workload(15, seed=5)
    assert _order(run_fcfs, w) == [p.id for p in w]


def test_priority_prefers_lower_number_and_breaks_ties_by_arrival():
    w = [Process("P1", 0, 1, 5), Process("P2", 0, 1, 1), Process("P3", 0, 1, 3),
         Process("P4", 0, 1, 1)]
    assert _order(run_priority, w) == ["P2", "P4", "P3", "P1"]


def test_sjf_prefers_shortest_burst():
    w = [Process("P1", 0, 7, 1), Process("P2", 0, 2, 1), Process("P3", 0, 4, 1)]
    assert _order(run_sjf, w) == ["P2", "P3", "P1"]


@pytest.mark.parametrize("run", [run_fcfs, run_sjf, run_priority])
def test_non_preemptive_schedulers_run_each_process_once(run):
    w = generate_workload(10, seed=11)
    order = _order(run, w)
    assert len(order) == len(set(order)) == len(w)


def test_round_robin_with_huge_quantum_equals_fcfs():
    for w in random_workloads(20):
        assert run_round_robin(CPUSimulator(w), quantum=1000) == run_fcfs(CPUSimulator(w))


def test_round_robin_no_slice_exceeds_quantum():
    sim = CPUSimulator(generate_workload(12, seed=2, mean_burst_time=8))
    run_round_robin(sim, quantum=2)
    assert all(end - start <= 2 for _, start, end in sim.timeline)


@pytest.mark.parametrize("q", [0, -1, None])
def test_round_robin_rejects_bad_quantum(q):
    with pytest.raises(ValueError):
        run_round_robin(CPUSimulator([Process("P1", 0, 1, 1)]), quantum=q)


def test_round_robin_trades_switches_for_response_time():
    w = [Process(f"P{i}", 0, 10, 1) for i in range(1, 6)]
    rr = run_round_robin(CPUSimulator(w), quantum=2)
    fcfs = run_fcfs(CPUSimulator(w))
    assert rr["avg_response_time"] < fcfs["avg_response_time"]
    assert rr["context_switches"] > fcfs["context_switches"]


def test_sjf_is_optimal_when_all_jobs_arrive_together():
    # Textbook theorem: with simultaneous arrivals, SJF gives the minimum
    # average waiting time of every possible non-preemptive order.
    w = [Process("P1", 0, 6, 1), Process("P2", 0, 8, 1), Process("P3", 0, 7, 1),
         Process("P4", 0, 3, 1)]
    best = run_sjf(CPUSimulator(w))["avg_waiting_time"]
    assert best == pytest.approx(7.0)    # (0 + 3 + 9 + 16) / 4
    for perm in permutations([p.id for p in w]):
        order = list(perm)

        def pick(queue, order=order):
            ids = [p.id for p in queue]
            return ids.index(next(pid for pid in order if pid in ids))

        assert run_policy(CPUSimulator(w), pick)["avg_waiting_time"] >= best - 1e-9


def test_sjf_beats_fcfs_on_average():
    ws = random_workloads(60)
    sjf = sum(run_sjf(CPUSimulator(w))["avg_waiting_time"] for w in ws)
    fcfs = sum(run_fcfs(CPUSimulator(w))["avg_waiting_time"] for w in ws)
    assert sjf < fcfs


def test_all_schedulers_are_deterministic(scheduler):
    _, run = scheduler
    w = generate_workload(10, seed=99)
    assert run(CPUSimulator(w)) == run(CPUSimulator(w))
