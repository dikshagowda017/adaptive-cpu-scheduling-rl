"""
Priority Scheduler (non-preemptive)
------------------------------------
Always run the waiting process with the highest priority, where a
LOWER priority number means HIGHER priority (1 = most important).
Ties are broken by arrival order (FCFS), because min() returns the
first index with the smallest value.
"""

from simulator import CPUSimulator, run_policy


def priority_choice(ready_queue) -> int:
    """Index of the process with the lowest priority number."""
    return min(range(len(ready_queue)), key=lambda i: ready_queue[i].priority)


def run_priority(sim: CPUSimulator) -> dict:
    """
    Run the Priority scheduling policy on a (freshly reset) simulator
    until all processes finish, then return the performance metrics.
    """
    return run_policy(sim, priority_choice)


if __name__ == "__main__":
    from workload_generator import generate_workload

    workload = generate_workload(num_processes=5, seed=42)
    sim = CPUSimulator(workload)

    metrics = run_priority(sim)
    print("Priority scheduling metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
