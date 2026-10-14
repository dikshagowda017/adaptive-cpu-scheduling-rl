import pytest

from workload_generator import generate_workload, Process, BURST_DISTRIBUTIONS


def test_same_seed_gives_identical_workload():
    assert generate_workload(20, seed=7) == generate_workload(20, seed=7)


def test_different_seeds_give_different_workloads():
    assert generate_workload(20, seed=1) != generate_workload(20, seed=2)


@pytest.mark.parametrize("dist", BURST_DISTRIBUTIONS)
def test_process_attributes_are_valid(dist):
    procs = generate_workload(200, seed=3, burst_distribution=dist, priority_range=(1, 10))
    assert len(procs) == 200
    assert [p.id for p in procs] == [f"P{i}" for i in range(1, 201)]
    assert all(p.burst_time >= 1 for p in procs)
    assert all(1 <= p.priority <= 10 for p in procs)
    assert all(p.arrival_time >= 0 for p in procs)
    arrivals = [p.arrival_time for p in procs]
    assert arrivals == sorted(arrivals)
    assert all(type(v) is int for p in procs
               for v in (p.arrival_time, p.burst_time, p.priority))


def test_mean_inter_arrival_controls_load():
    # Regression for bug B2: the old `arrival_rate` docstring claimed a HIGHER
    # value packs arrivals closer together. It is the mean gap: higher = sparser.
    dense = generate_workload(500, seed=0, mean_inter_arrival=1.0)
    sparse = generate_workload(500, seed=0, mean_inter_arrival=4.0)
    assert dense[-1].arrival_time < sparse[-1].arrival_time
    assert sparse[-1].arrival_time / 500 == pytest.approx(4.0, rel=0.15)


def test_mean_burst_time_is_respected():
    procs = generate_workload(5000, seed=0, mean_burst_time=5.0)
    mean = sum(p.burst_time for p in procs) / len(procs)
    assert mean == pytest.approx(5.0, rel=0.1)


def test_bimodal_has_long_and_short_jobs():
    procs = generate_workload(1000, seed=0, burst_distribution="bimodal", mean_burst_time=5)
    long_share = sum(p.burst_time >= 10 for p in procs) / len(procs)
    assert 0.12 < long_share < 0.3


def test_min_burst_floor():
    assert all(p.burst_time >= 3 for p in generate_workload(300, seed=0, min_burst_time=3))


def test_single_priority_value():
    assert {p.priority for p in generate_workload(50, seed=0, priority_range=(4, 4))} == {4}


@pytest.mark.parametrize("kwargs", [
    {"num_processes": 0},
    {"num_processes": -3},
    {"mean_inter_arrival": -1},
    {"mean_burst_time": 0},
    {"min_burst_time": 0},
    {"priority_range": (5, 1)},
    {"burst_distribution": "gaussian"},
])
def test_invalid_arguments_raise(kwargs):
    args = {"num_processes": 5, "seed": 0, **kwargs}
    with pytest.raises(ValueError):
        generate_workload(**args)


def test_process_repr():
    assert repr(Process("P1", 0, 5, 2)) == "Process(P1, arrival=0, burst=5, priority=2)"
