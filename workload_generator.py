"""
Workload Generator
-------------------
Generates synthetic CPU process workloads for the scheduling simulator.

Each process has:
- id: unique identifier (P1, P2, ...)
- arrival_time: when the process enters the ready queue
  (gaps between arrivals are Poisson-sampled)
- burst_time: how much CPU time the process needs
  (exponential, uniform, or bimodal)
- priority: integer priority level (1 = highest, 10 = lowest)

Using a fixed random seed makes workloads reproducible, so FCFS, RR,
Priority, SJF, and the RL agent can all be evaluated on identical workloads.
"""

from dataclasses import dataclass
import numpy as np

BURST_DISTRIBUTIONS = ("exponential", "uniform", "bimodal")


@dataclass
class Process:
    id: str
    arrival_time: int
    burst_time: int
    priority: int

    def __repr__(self):
        return (f"Process({self.id}, arrival={self.arrival_time}, "
                f"burst={self.burst_time}, priority={self.priority})")


def generate_workload(num_processes: int = 10, seed: int = 42,
                      mean_inter_arrival: float = 2.0,
                      mean_burst_time: float = 5.0,
                      min_burst_time: int = 1,
                      priority_range: tuple = (1, 10),
                      burst_distribution: str = "exponential") -> list:
    """
    Generate a synthetic workload of processes.

    Parameters
    ----------
    num_processes : int
        Number of processes to generate (must be >= 1).
    seed : int
        Random seed for reproducibility. Same seed -> same workload.
    mean_inter_arrival : float
        Average gap (in time units) between consecutive arrivals; the gap
        is sampled from a Poisson distribution with this mean.
        LOWER = processes arrive closer together = heavier load.
        (This parameter used to be called `arrival_rate`, but it was
        never a rate: a higher value spread the arrivals further apart.)
    mean_burst_time : float
        Mean CPU burst time.
    min_burst_time : int
        Minimum burst time, to avoid zero-length bursts after rounding.
    priority_range : tuple
        (min_priority, max_priority) inclusive range for random priority assignment.
    burst_distribution : str
        "exponential" (default): many short jobs, a few long ones.
        "uniform": bursts uniformly spread over [min_burst_time, 2*mean - min].
        "bimodal": 80% short interactive jobs (~mean/2), 20% long CPU-bound
                   jobs (~3*mean) — the mix where a scheduler's choices matter most.

    Returns
    -------
    list[Process]
        List of Process objects sorted by arrival_time.
    """
    if num_processes < 1:
        raise ValueError(f"num_processes must be >= 1, got {num_processes}")
    if mean_inter_arrival < 0:
        raise ValueError(f"mean_inter_arrival must be >= 0, got {mean_inter_arrival}")
    if mean_burst_time <= 0:
        raise ValueError(f"mean_burst_time must be > 0, got {mean_burst_time}")
    if min_burst_time < 1:
        raise ValueError(f"min_burst_time must be >= 1, got {min_burst_time}")
    if priority_range[0] > priority_range[1]:
        raise ValueError(f"invalid priority_range {priority_range}")
    if burst_distribution not in BURST_DISTRIBUTIONS:
        raise ValueError(f"burst_distribution must be one of {BURST_DISTRIBUTIONS}, "
                         f"got {burst_distribution!r}")

    rng = np.random.default_rng(seed)

    # Inter-arrival gaps sampled from a Poisson distribution.
    # Cumulative sum gives non-decreasing arrival times.
    inter_arrival_times = rng.poisson(lam=mean_inter_arrival, size=num_processes)
    arrival_times = np.cumsum(inter_arrival_times)

    if burst_distribution == "exponential":
        burst_times = rng.exponential(scale=mean_burst_time, size=num_processes)
    elif burst_distribution == "uniform":
        high = max(2 * mean_burst_time - min_burst_time, min_burst_time)
        burst_times = rng.uniform(min_burst_time, high, size=num_processes)
    else:  # bimodal
        is_long = rng.random(num_processes) < 0.2
        short = rng.exponential(scale=mean_burst_time / 2, size=num_processes)
        long_ = rng.normal(loc=3 * mean_burst_time, scale=mean_burst_time / 2,
                           size=num_processes)
        burst_times = np.where(is_long, long_, short)

    # Round to whole time units, with a minimum floor so no process has 0 burst time.
    burst_times = np.maximum(min_burst_time, np.round(burst_times)).astype(int)

    # Priorities sampled uniformly from the given range (inclusive).
    priorities = rng.integers(priority_range[0], priority_range[1] + 1, size=num_processes)

    processes = [
        Process(
            id=f"P{i+1}",
            arrival_time=int(arrival_times[i]),
            burst_time=int(burst_times[i]),
            priority=int(priorities[i]),
        )
        for i in range(num_processes)
    ]

    # Sort by arrival time so the simulator can process them in order.
    processes.sort(key=lambda p: p.arrival_time)
    return processes


CSV_COLUMNS = {
    "id": ("id", "pid", "process", "process_id", "name"),
    "arrival_time": ("arrival", "arrival_time", "at"),
    "burst_time": ("burst", "burst_time", "bt", "cpu_time"),
    "priority": ("priority", "prio", "pr"),
}


def load_workload_csv(path) -> list:
    """Load a workload from a CSV file so anyone can schedule their own processes.

    Expected header (case-insensitive; common aliases accepted):
        id, arrival, burst, priority
    The priority column is optional (defaults to 1). Every problem is reported
    with its line number instead of a crash further down."""
    import csv

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path}: file is empty")
        header = {h.strip().lower(): h for h in reader.fieldnames if h}
        columns = {}
        for key, aliases in CSV_COLUMNS.items():
            match = next((header[a] for a in aliases if a in header), None)
            if match is None and key != "priority":
                raise ValueError(f"{path}: missing column '{aliases[1] if key != 'id' else 'id'}' "
                                 f"(found: {', '.join(reader.fieldnames)})")
            columns[key] = match

        processes, seen = [], set()
        for line, row in enumerate(reader, start=2):
            if not any((v or "").strip() for v in row.values()):
                continue  # skip blank lines
            pid = (row[columns["id"]] or "").strip()
            if not pid:
                raise ValueError(f"{path}, line {line}: empty process id")
            if pid in seen:
                raise ValueError(f"{path}, line {line}: duplicate process id {pid!r}")
            seen.add(pid)
            values = {}
            for key in ("arrival_time", "burst_time", "priority"):
                raw = (row[columns[key]] or "").strip() if columns[key] else "1"
                try:
                    values[key] = int(raw)
                except ValueError:
                    raise ValueError(f"{path}, line {line}: {key} must be a whole number, "
                                     f"got {raw!r}") from None
            if values["arrival_time"] < 0:
                raise ValueError(f"{path}, line {line}: arrival must be >= 0")
            if values["burst_time"] < 1:
                raise ValueError(f"{path}, line {line}: burst must be >= 1")
            processes.append(Process(pid, **values))

    if not processes:
        raise ValueError(f"{path}: no processes found")
    processes.sort(key=lambda p: p.arrival_time)
    return processes


# Appendix A of the Phase-1 report: small hand-checkable workload used in
# unit tests and in the demo, so results can be verified with pen and paper.
REPORT_SAMPLE_WORKLOAD = [
    Process("P1", 0, 5, 2),
    Process("P2", 1, 3, 1),
    Process("P3", 2, 8, 4),
    Process("P4", 3, 6, 3),
    Process("P5", 4, 2, 5),
]


if __name__ == "__main__":
    # Quick manual test: generate and print a small workload.
    workload = generate_workload(num_processes=10, seed=42)
    print(f"Generated {len(workload)} processes:\n")
    for p in workload:
        print(p)
