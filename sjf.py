"""
Shortest Job First (SJF) Scheduler (non-preemptive)
----------------------------------------------------
Always run the waiting process with the smallest remaining burst time.
Ties are broken by arrival order.

SJF is included as a REFERENCE baseline, not a competitor in the
report's original comparison: it needs to know burst times in advance
(which a real OS usually can't), and it is known to minimise average
waiting time among non-preemptive policies when all jobs are present.
The RL agent observes the same burst-time feature, so SJF tells us how
close the learned policy gets to a strong hand-designed rule.
"""

from simulator import CPUSimulator, run_policy


def sjf_choice(ready_queue) -> int:
    """Index of the process with the shortest remaining burst time."""
    return min(range(len(ready_queue)),
               key=lambda i: ready_queue[i].remaining_burst_time)


def run_sjf(sim: CPUSimulator) -> dict:
    """Run non-preemptive SJF to completion and return the metrics."""
    return run_policy(sim, sjf_choice)


if __name__ == "__main__":
    from workload_generator import generate_workload

    workload = generate_workload(num_processes=5, seed=42)
    metrics = run_sjf(CPUSimulator(workload))
    print("SJF metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
