"""Shared pytest fixtures. Makes the project modules importable from tests/."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from workload_generator import REPORT_SAMPLE_WORKLOAD, generate_workload  # noqa: E402
from fcfs import run_fcfs  # noqa: E402
from sjf import run_sjf  # noqa: E402
from priority import run_priority  # noqa: E402
from round_robin import run_round_robin  # noqa: E402

SCHEDULERS = {
    "fcfs": run_fcfs,
    "sjf": run_sjf,
    "priority": run_priority,
    "rr_q1": lambda sim: run_round_robin(sim, quantum=1),
    "rr_q3": lambda sim: run_round_robin(sim, quantum=3),
}


@pytest.fixture
def sample_workload():
    """Appendix A of the Phase-1 report (hand-checkable)."""
    return list(REPORT_SAMPLE_WORKLOAD)


@pytest.fixture(params=sorted(SCHEDULERS))
def scheduler(request):
    return request.param, SCHEDULERS[request.param]


def random_workloads(count=30, **kwargs):
    """A spread of random workloads (1-12 processes), including idle gaps
    and simultaneous arrivals."""
    return [generate_workload(num_processes=1 + seed % 12, seed=seed, **kwargs)
            for seed in range(count)]
