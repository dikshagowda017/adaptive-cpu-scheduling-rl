"""
CPU Scheduling Simulator
-------------------------
This is the core engine. It does NOT decide which process to run —
that decision is made by whichever scheduler (FCFS, Round Robin,
Priority, SJF, or the RL agent) is plugged in from outside.

The simulator's job is only to:
  - Track simulated time
  - Admit processes into the ready queue as they arrive
  - Run whichever process the scheduler picks
  - Update waiting time / remaining burst time / completion time
  - Count context switches
  - Record a timeline (for Gantt charts)
  - Report final performance metrics

Non-preemptive by default: once a process is picked, it runs until
either it finishes or a given time_slice runs out (time_slice is used
for Round Robin; leave it as None for FCFS/Priority to run to completion).

Definitions used for the metrics (standard textbook definitions):
  turnaround_time = completion_time - arrival_time
  waiting_time    = turnaround_time - burst_time   (time spent in the ready queue)
  response_time   = first_start_time - arrival_time
  context switch  = the CPU moves from one process to a DIFFERENT process
                    (the very first dispatch is not a switch)
"""

import numbers
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

from workload_generator import Process


class QueueView(Sequence):
    """Read-only view of the ready queue. Returned by step() instead of a copy,
    so a step costs O(1) even when thousands of processes are waiting."""

    def __init__(self, queue):
        self._queue = queue

    def __getitem__(self, index):
        return self._queue[index]

    def __len__(self):
        return len(self._queue)

    def __repr__(self):
        return f"QueueView({self._queue!r})"


@dataclass
class SimProcess:
    """Wraps a Process with the extra bookkeeping the simulator needs
    while the process moves through the system."""
    process: Process
    remaining_burst_time: int = field(init=False)
    start_time: int = None
    completion_time: int = None
    turnaround_time: int = None
    # Waiting time is tracked lazily: time already waited, plus (if the
    # process is in the ready queue now) the time since it joined the queue.
    # This keeps every simulation step O(1) instead of touching the whole queue.
    _waited: int = field(default=0, repr=False)
    _queued_since: int = field(default=None, repr=False)
    _sim: "CPUSimulator" = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self.remaining_burst_time = self.process.burst_time

    @property
    def waiting_time(self) -> int:
        """Total time spent in the ready queue so far."""
        if self._queued_since is None:
            return self._waited
        return self._waited + (self._sim.time - self._queued_since)

    @property
    def id(self):
        return self.process.id

    @property
    def arrival_time(self):
        return self.process.arrival_time

    @property
    def burst_time(self):
        return self.process.burst_time

    @property
    def priority(self):
        return self.process.priority


class CPUSimulator:
    def __init__(self, processes: list):
        # Keep the original workload untouched; reset() builds fresh
        # SimProcess wrappers from it every time.
        self._original_processes = list(processes)
        validate_workload(self._original_processes)
        self.reset()

    def reset(self):
        """Reset the simulator to time 0 with a fresh copy of the workload.
        Call this at the start of every scheduling episode."""
        self.time = 0
        self.ready_queue = []
        self.completed = []
        self.context_switches = 0
        self.timeline = []          # list of (process_id, start, end)
        self._last_run_id = None

        # Fresh SimProcess wrappers, sorted by arrival time (stable sort,
        # so processes arriving at the same instant keep their ID order).
        self._pending = deque(sorted(
            (SimProcess(process=p, _sim=self) for p in self._original_processes),
            key=lambda sp: sp.arrival_time,
        ))

        self._admit_arrivals()
        return self.get_ready_queue_snapshot()

    def _admit_arrivals(self) -> int:
        """Move any process whose arrival_time has passed from the
        pending list into the ready queue.

        A process that arrived while another one was running has already
        been waiting since its arrival, so its waiting time starts at
        (now - arrival_time), not 0.

        Returns the total waiting time credited to the newly admitted
        processes (used for the RL reward)."""
        credited = 0
        while self._pending and self._pending[0].arrival_time <= self.time:
            proc = self._pending.popleft()
            proc._queued_since = proc.arrival_time
            credited += self.time - proc.arrival_time
            self.ready_queue.append(proc)
        return credited

    def advance_to_next_arrival(self):
        """CPU is idle: jump the clock forward to the next arrival and
        admit it. No-op if something is already in the ready queue."""
        if self.ready_queue or not self._pending:
            return
        self.time = max(self.time, self._pending[0].arrival_time)
        self._admit_arrivals()

    def is_done(self) -> bool:
        """True once every process has finished executing."""
        return not self.ready_queue and not self._pending

    def get_ready_queue_snapshot(self):
        """Returns the current ready queue as a plain list, safe to
        inspect without accidentally mutating simulator state."""
        return list(self.ready_queue)

    def step(self, action_index: int, time_slice: int = None):
        """
        Advance the simulation by running one process from the ready queue.

        Parameters
        ----------
        action_index : int
            Index of the process to run, within the CURRENT ready queue.
        time_slice : int or None
            If None, the process runs to completion (FCFS / Priority style).
            If set, the process runs for at most `time_slice` time units,
            then goes back to the end of the ready queue if unfinished
            (Round Robin style). Processes that arrived during the slice
            are queued before the preempted process.

        Returns
        -------
        next_queue : QueueView
            Read-only view of the ready queue after this step.
        done : bool
        info : dict
            Details about what happened this step (used by the Gym
            wrapper's reward function and for debugging).
        """
        if not self.ready_queue:
            raise ValueError("step() called with an empty ready queue. "
                             "Check is_done() / advance_to_next_arrival() first.")

        if not (0 <= action_index < len(self.ready_queue)):
            raise IndexError(
                f"action_index {action_index} out of range for ready "
                f"queue of length {len(self.ready_queue)}"
            )

        if time_slice is not None and time_slice <= 0:
            raise ValueError(f"time_slice must be positive, got {time_slice}")

        proc = self.ready_queue.pop(action_index)
        proc._waited += self.time - proc._queued_since
        proc._queued_since = None

        context_switch = (self._last_run_id is not None
                          and proc.id != self._last_run_id)
        if context_switch:
            self.context_switches += 1
        self._last_run_id = proc.id

        start = self.time
        if proc.start_time is None:
            proc.start_time = start

        run_time = proc.remaining_burst_time
        if time_slice is not None:
            run_time = min(run_time, time_slice)

        # Every other process still waiting accumulates waiting time for the
        # duration this process occupies the CPU (applied lazily, see SimProcess).
        waiting_added = run_time * len(self.ready_queue)

        self.time += run_time
        proc.remaining_burst_time -= run_time
        self.timeline.append((proc.id, start, self.time))

        # New processes may have arrived while this one was running.
        waiting_added += self._admit_arrivals()

        finished = proc.remaining_burst_time <= 0
        if finished:
            proc.completion_time = self.time
            proc.turnaround_time = proc.completion_time - proc.arrival_time
            self.completed.append(proc)
        else:
            # Not done yet (Round Robin case) — goes to the back of the queue.
            proc._queued_since = self.time
            self.ready_queue.append(proc)

        done = self.is_done()

        info = {
            "ran_process_id": proc.id,
            "run_time": run_time,
            "context_switch": context_switch,
            "finished": finished,
            "current_time": self.time,
            "waiting_added": waiting_added,
        }

        return QueueView(self.ready_queue), done, info

    def get_metrics(self) -> dict:
        """Compute summary performance metrics after an episode finishes.
        Call this once is_done() is True.

        fairness_index is Jain's Fairness Index computed over each
        process's slowdown (turnaround_time / burst_time). Slowdown is
        always >= 1, so the index is well defined, and it is 1.0 when
        every process is delayed in proportion to its size."""
        if not self.completed:
            return {
                "avg_waiting_time": 0.0,
                "avg_turnaround_time": 0.0,
                "avg_response_time": 0.0,
                "max_waiting_time": 0,
                "context_switches": self.context_switches,
                "fairness_index": 1.0,
            }

        n = len(self.completed)
        waiting_times = [p.waiting_time for p in self.completed]
        turnaround_times = [p.turnaround_time for p in self.completed]
        response_times = [p.start_time - p.arrival_time for p in self.completed]
        slowdowns = [p.turnaround_time / p.burst_time for p in self.completed]

        # Jain's Fairness Index: (sum(x))^2 / (n * sum(x^2))
        sum_x = sum(slowdowns)
        sum_x2 = sum(x ** 2 for x in slowdowns)
        fairness = (sum_x ** 2) / (n * sum_x2)

        return {
            "avg_waiting_time": sum(waiting_times) / n,
            "avg_turnaround_time": sum(turnaround_times) / n,
            "avg_response_time": sum(response_times) / n,
            "max_waiting_time": max(waiting_times),
            "context_switches": self.context_switches,
            "fairness_index": fairness,
        }


def validate_workload(processes) -> None:
    """Reject workloads the simulator can't run correctly, with a clear message.

    Without this, a zero burst crashed get_metrics() (division by zero), a
    negative arrival silently inflated waiting time, and duplicate IDs
    silently broke context-switch counting."""
    seen = set()
    for p in processes:
        for name in ("arrival_time", "burst_time", "priority"):
            value = getattr(p, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise ValueError(f"{p.id}: {name} must be a whole number, got {value!r}")
        if p.arrival_time < 0:
            raise ValueError(f"{p.id}: arrival_time must be >= 0, got {p.arrival_time}")
        if p.burst_time < 1:
            raise ValueError(f"{p.id}: burst_time must be >= 1, got {p.burst_time}")
        if p.id in seen:
            raise ValueError(f"duplicate process id {p.id!r}")
        seen.add(p.id)


def run_policy(sim: CPUSimulator, choose_index, time_slice: int = None) -> dict:
    """Run a simple scheduling policy to completion.

    choose_index(ready_queue) -> index of the process to run next.
    Shared by all the classical schedulers so the idle-time handling
    lives in exactly one place."""
    sim.reset()
    while not sim.is_done():
        if not sim.ready_queue:
            sim.advance_to_next_arrival()
            continue
        sim.step(action_index=choose_index(sim.ready_queue), time_slice=time_slice)
    return sim.get_metrics()


if __name__ == "__main__":
    # Run all classical schedulers on the SAME workload, so the
    # comparison is fair -- every algorithm faces identical processes.
    from workload_generator import generate_workload
    from fcfs import run_fcfs
    from sjf import run_sjf
    from round_robin import run_round_robin
    from priority import run_priority

    workload = generate_workload(num_processes=10, seed=42)

    print(f"Workload (seed=42, {len(workload)} processes):")
    for p in workload:
        print(f"  {p}")
    print()

    results = {
        "FCFS": run_fcfs(CPUSimulator(workload)),
        "SJF": run_sjf(CPUSimulator(workload)),
        "Round Robin": run_round_robin(CPUSimulator(workload), quantum=3),
        "Priority": run_priority(CPUSimulator(workload)),
    }

    metric_names = list(next(iter(results.values())).keys())

    header = f"{'Metric':<22}" + "".join(f"{name:<15}" for name in results)
    print(header)
    print("-" * len(header))

    for metric in metric_names:
        row = f"{metric:<22}"
        for scheduler_name in results:
            value = results[scheduler_name][metric]
            row += f"{value:<15.3f}"
        print(row)
