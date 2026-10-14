"""
First Come First Serve (FCFS) Scheduler
-----------------------------------------
The simplest policy: always run whichever process arrived first and
is still waiting. Since the simulator's ready queue already holds
processes in arrival order (new arrivals are appended to the back),
"pick the first one" is exactly FCFS behavior. Each process runs to
completion (time_slice=None) — no preemption.
"""

from simulator import CPUSimulator, run_policy


def fcfs_choice(ready_queue) -> int:
    """FCFS: always pick the process at the front of the ready queue."""
    return 0


def run_fcfs(sim: CPUSimulator) -> dict:
    """
    Run the FCFS policy on a (freshly reset) simulator until all
    processes finish, then return the performance metrics.
    """
    return run_policy(sim, fcfs_choice)


if __name__ == "__main__":
    from workload_generator import generate_workload

    workload = generate_workload(num_processes=5, seed=42)
    sim = CPUSimulator(workload)

    metrics = run_fcfs(sim)
    print("FCFS metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
