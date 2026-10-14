# Example workloads

Run any of these with `python main.py demo --workload examples/<file>.csv`,
or write your own CSV with the same columns (`id, arrival, burst, priority`;
priority is optional, 1 = highest).

| File | What it shows | Avg waiting time |
|---|---|---|
| `report_appendix_a.csv` | The Phase-1 report's sample workload, checkable by hand | FCFS 8.2 · RR 9.6 · Priority 7.8 · **RL 5.6** · SJF 5.6 |
| `convoy_effect.csv` | A long job and short jobs arrive together. FCFS runs the long job first and every short job waits behind it ("convoy effect"); RL learned to run the short jobs first | FCFS 18.2 · RR 6.2 · Priority 3.8 · **RL 3.2** · SJF 3.2 |
| `long_job_alone_first.csv` | Honest limitation: the long job arrives *alone*, so any non-preemptive scheduler (including RL and SJF) must start it; only preemptive Round Robin escapes | RR 5.2 · RL / SJF 16.4 · FCFS 17.2 |
| `textbook_simultaneous.csv` | Classic textbook case: all jobs at time 0, SJF is optimal | SJF 7.0 |
