"""
Round Robin Scheduler
----------------------
Run the process at the front of the ready queue for at most `quantum`
time units. If it is not finished it goes to the back of the queue
(behind any processes that arrived while it was running).
"""

from simulator import CPUSimulator, run_policy
from fcfs import fcfs_choice

DEFAULT_QUANTUM = 3


def run_round_robin(sim: CPUSimulator, quantum: int = DEFAULT_QUANTUM) -> dict:
    """
    Run the Round Robin policy on a (freshly reset) simulator until
    all processes finish, then return the performance metrics.

    Parameters
    ----------
    quantum : int
        Maximum time units a process may run before being preempted
        and sent to the back of the queue if unfinished.
    """
    if quantum is None or quantum <= 0:
        raise ValueError(f"quantum must be a positive integer, got {quantum}")
    return run_policy(sim, fcfs_choice, time_slice=quantum)


if __name__ == "__main__":
    from workload_generator import generate_workload

    workload = generate_workload(num_processes=5, seed=42)
    sim = CPUSimulator(workload)

    metrics = run_round_robin(sim, quantum=DEFAULT_QUANTUM)
    print(f"Round Robin metrics (quantum={DEFAULT_QUANTUM}):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
