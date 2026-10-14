# Testing & Debugging Report

**Project:** Adaptive CPU Scheduling using Reinforcement Learning (Team 5, MITE CSE-AI&ML)

| Quality measure | Result |
|---|---|
| Automated tests | **401**, all passing (about 45 s), including 7 property-based tests (~2 100 generated cases) and 6 tests that check the README numbers against the code |
| Line coverage | **95 %** (`pytest --cov`; demo `__main__` blocks and tool scripts excluded) |
| Mutation score | **28 / 28** planted bugs caught (`python mutation_test.py`) |
| Bugs found & fixed | **16** (4 high severity), each with a regression test |
| Reproducibility | Every number is seeded; the same command gives the same result |
| CI | GitHub Actions: Ubuntu + Windows × Python 3.10 / 3.12 / 3.13, plus mutation testing |

```powershell
python verify_all.py                  # EVERYTHING + results/dashboard.html (or double-click run_everything.bat)
python -m pytest                      # all tests
python -m pytest --cov=. --cov-report=term-missing   # with coverage
python mutation_test.py               # test the tests
python mutation_test.py --show 1      # plant bug #1 live and see which tests catch it
```

---

## 1. Bugs found and fixed

Severity: **High** = wrong results or invalid conclusions · **Medium** = misleading
behaviour · **Low** = robustness / hygiene.

| ID | Sev. | Module | Bug | How it was found | Regression test |
|---|---|---|---|---|---|
| B1 | High | `simulator.py` | **Waiting time under-counted.** A process that arrived while another one was running joined the queue with `waiting_time = 0`, so the time it waited before the next dispatch was lost. On the report's Appendix-A workload FCFS reported an average wait of **6.2 instead of 8.2**. Every metric, every comparison and the RL reward were affected. | Hand calculation of Appendix A didn't match; the invariant `waiting = turnaround − burst` failed | `test_mid_run_arrival_waiting_time_is_counted`, golden-value tests, invariant tests |
| B2 | Medium | `workload_generator.py` | **`arrival_rate` documented backwards.** Its docstring said "higher = processes arrive closer together", but it was the Poisson *mean gap*, so higher = *further apart*. Anyone tuning "heavier load" would get the opposite. Renamed to `mean_inter_arrival` and documented correctly. | Statistical test of the generated arrival spacing | `test_mean_inter_arrival_controls_load` |
| B3 | Medium | `simulator.py` | **Degenerate fairness metric.** Jain's index over raw waiting times breaks down when most waits are 0: one process waiting 1 unit and four waiting 0 scored **0.20 ("very unfair")**. Now computed over each process's *slowdown* (turnaround / burst ≥ 1), the standard choice in the scheduling literature. | Sanity check: FCFS had avg wait 0.2 but "fairness" 0.2 | `test_fairness_index_is_not_degenerate_for_near_zero_waits` |
| B4 | High | `q_learning.py`, `gym_env.py` | **Results not reproducible.** Exploration and workload generation used NumPy's global unseeded RNG, so each run gave different numbers (the progress slides report −4.56; a re-run gave −13.14). This contradicted the report's "fixed random seeds" claim. | Ran the training script twice | `test_training_is_reproducible`, `test_seeded_env_without_explicit_seed_is_reproducible` |
| B5 | High | `q_learning.py` | **Original agent memorises its training workloads.** The full 30-feature state is the Q-table key, so on new workloads **61 % of states were never seen in training**, and the agent falls back to action 0, i.e. it *becomes FCFS* (9.99 vs FCFS 10.05). Fixed with a per-process tabular agent (see README); the original agent is kept for comparison. | Evaluation on held-out workloads (B7) plus an unseen-state counter | `test_per_process_agent_sees_few_unseen_states`, `test_trained_agent_beats_classical_schedulers_on_unseen_workloads` |
| B6 | Medium | `gym_env.py`, `q_learning.py` | **No action masking.** The agent could choose an empty queue slot, which was silently clipped to the last process. Several Q-values aliased the same decision and exploration was wasted. The env now returns `info["action_mask"]` and agents only choose valid actions. Invalid actions are still clipped, but flagged with `info["action_was_invalid"]`. | Code review of `step()` | `test_actions_always_respect_mask`, `test_out_of_range_action_is_clipped_and_flagged` |
| B7 | High | methodology | **No held-out test set.** Training cycled through workload seeds 0–9 and the planned evaluation used the same generator seeds, so it would have measured memorisation. Training now draws seeds below 100 000; evaluation uses seeds ≥ 1 000 000. | Review of the train/evaluate plan | `test_training_never_sees_test_workloads` |
| B8 | Low | `simulator.py` | **First dispatch counted as a context switch.** FCFS on 5 processes reported 5 switches; the textbook answer is 4 (the CPU changes process 4 times). | Golden-value test | `test_report_sample_workload_golden_values`, `test_single_process_with_late_arrival` |
| B9 | Low | `requirements.txt` | **File saved as UTF-16** (from PowerShell `pip freeze >`), which many tools can't read, and it pinned NumPy 2.5 / pandas 3.0, which need Python ≥ 3.11, while the report promises Python 3.9+. Now plain UTF-8 with minimum versions; exact versions are in `requirements-lock.txt`. | Inspecting the file encoding | CI installs it on 3 Python versions |
| B10 | Low | `gym_env.py` | **Reward ignored part of the waiting.** The waiting penalty was `run_time × (queue length − 1)` and missed processes that arrived mid-run (same root cause as B1). The completion bonus was a constant +1 every step, so it carried no signal. The reward now uses the simulator's exact `waiting_added`, so the waiting term sums to exactly −(average waiting time). | Unit test linking reward to metric | `test_waiting_reward_sums_to_minus_average_waiting_time` |
| B11 | Low | `simulator.py` | Duplicate module docstring; the demo printed "5 processes" while generating 10; idle-time handling was copy-pasted into every scheduler and reached into private fields. It now lives in one helper (`run_policy` / `advance_to_next_arrival`). | Code review | `test_idle_gap_between_processes`, script smoke tests |
| B12 | Medium | `simulator.py` | **Zero-length burst crashed the program.** A process with `burst_time = 0` (e.g. typed into a CSV) caused `ZeroDivisionError` in `get_metrics()`. Workloads are now validated up front with a clear message. | Adversarial input testing ("what would a judge type?") | `test_invalid_workloads_are_rejected_with_clear_message` |
| B13 | Medium | `simulator.py` | **Negative arrival time gave silently wrong answers.** A lone process arriving at -5 was reported as waiting 5 units. Now rejected. | Adversarial input testing | same test, `bad2` case |
| B14 | Medium | `simulator.py` | **Duplicate process IDs broke context-switch counting.** Two different processes both named `P1` running back to back counted as *zero* switches. Now rejected (the CSV loader reports the line number). | Adversarial input testing | same test, `bad3` case; `test_bad_csv_files_give_line_numbered_errors` |
| B15 | Low | `main.py` | **Bad command-line values printed raw tracebacks.** For example `--seed -1` gave a NumPy `ValueError` traceback. All arguments are now validated with friendly messages. | Adversarial CLI testing | `test_cli_rejects_bad_arguments_without_traceback` |
| B16 | Medium | `simulator.py` | **Simulator slowed down quadratically.** Every step added waiting time to every queued process *and* copied the whole queue: 30 000 processes took **15.6 s**. Waiting time is now computed lazily (time waited before + time since joining the queue) and `step()` returns a read-only view, so the same run takes **0.76 s (20x faster)**; 100 000 processes take 0.4 s. | Performance / stress testing | `test_ten_thousand_processes_simulate_quickly`, `test_overloaded_system_stays_fast` |

### Found by our own testing during development
* `save_agent` crashed with *"int64 is not JSON serializable"* because `action_space.n` is a NumPy integer. Found by the end-to-end run and fixed.
* **Weak tests found by mutation testing.** Planting "Q-learning ignores γ" did *not* fail any test at first: a fresh Q-table is all zeros, so γ·max Q(s′) = 0 either way. Likewise, nothing checked the *values* inside the observation, and the train/test-separation test was tautological. The Holm-correction check (`p_holm >= p_value`) also passed when the correction was skipped entirely. All four tests were strengthened; every mutant is now caught.
* A new safety check (an agent trained for N queue slots must run in an environment with N slots) immediately caught one of our *own* tests that silently mixed a 10-slot agent with a 20-slot environment.

---

## 2. Test strategy

| Level | What it checks | File |
|---|---|---|
| **Golden values** | Every scheduler on the report's Appendix-A workload against hand-computed timelines, waits, turnaround and switches | `test_simulator.py` |
| **Invariants (property tests)** | For every scheduler × 40 random workloads: all processes finish; `waiting = turnaround − burst`; the timeline never overlaps; each process gets exactly its burst; switches = changes in the timeline; the CPU never idles while work is ready; 0 < fairness ≤ 1 | `test_simulator.py` |
| **Fuzzing** | 25 runs of a random policy with random preemption, checked against the same invariants | `test_simulator.py` |
| **Edge cases** | Single process, late first arrival, idle gaps, simultaneous arrivals, unsorted input, bad index, empty queue, non-positive quantum, reset, input not mutated | `test_simulator.py`, `test_schedulers.py` |
| **Algorithm properties** | FCFS order; Priority tie-break; SJF optimal over *all 24 permutations*; RR with a huge quantum equals FCFS; RR slices ≤ quantum; RR trades switches for response time | `test_schedulers.py` |
| **Generator statistics** | Seeds reproducible; ranges; mean burst and mean gap match their parameters; bimodal mix share; invalid arguments rejected | `test_workload_generator.py` |
| **Environment contract** | Official `gymnasium` `check_env`; exact observation values; masks; padding; determinism; reward parts add up; reward ↔ metric identity; errors before reset / after done | `test_gym_env.py` |
| **Learning algorithm** | Update rule equals the formula; terminal states don't bootstrap; masks respected; ε decay floor; hyperparameter validation; reproducibility; save/load round trip; no train/test leakage | `test_q_learning.py` |
| **End-to-end ML** | A trained agent beats FCFS, RR and Priority by > 15 % on 60 unseen workloads | `test_q_learning.py` |
| **Statistics** | Permutation test: p = 1 for identical samples, small for a real effect, no false alarm on noise; Holm correction | `test_evaluate.py` |
| **Pipeline / smoke** | `main.py` writes all 13 output files; every module's demo script runs; bad CLI arguments fail | `test_evaluate.py`, `test_scripts.py` |
| **Property-based (Hypothesis)** | ~2 100 generated workloads and decision sequences: conservation laws, identical makespan for every work-conserving scheduler, invariance to shifting all arrivals in time, SJF <= FCFS for simultaneous arrivals, environment survives any action, p-values valid and symmetric | `test_properties.py` |
| **Robustness** | Invalid workloads (zero/negative burst, negative arrival, duplicate IDs, non-integers) rejected with clear messages; CSV loader errors carry line numbers; CLI rejects bad values without tracebacks | `test_robustness.py` |
| **Performance** | 10 000 processes per scheduler and a 20 000-process overloaded system must each finish in < 5 s | `test_robustness.py` |
| **Documentation tests** | The README headline table, improvement percentages and example-workload numbers are recomputed from the code; every file the README mentions must exist | `test_docs.py` |
| **Cross-version reproducibility** | Identical results (to 1e-16) on Python 3.10, 3.12 and 3.14, checked manually; CI runs 3.10/3.12/3.13 on Ubuntu and Windows | CI, `requirements-lock.txt` |
| **Mutation testing** | 28 realistic bugs planted one at a time (run in parallel); the suite must fail for each | `mutation_test.py` |

---

## 3. Hand calculation for the golden tests (Appendix A)

| Process | Arrival | Burst | Priority |
|---|---|---|---|
| P1 | 0 | 5 | 2 |
| P2 | 1 | 3 | 1 |
| P3 | 2 | 8 | 4 |
| P4 | 3 | 6 | 3 |
| P5 | 4 | 2 | 5 |

Waiting = start − arrival for non-preemptive schedulers; in general waiting = turnaround − burst.

**FCFS:** P1 0–5, P2 5–8, P3 8–16, P4 16–22, P5 22–24 → waits 0, 4, 6, 13, 18 → **avg 8.2**; turnaround 5, 7, 14, 19, 20 → **avg 13.0**; 4 switches.

**SJF** (non-preemptive): at t = 5 the queue holds P2(3), P3(8), P4(6), P5(2) → P5 5–7, P2 7–10, P4 10–16, P3 16–24 → waits 0, 6, 14, 7, 1 → **avg 5.6**; turnaround **avg 10.4**.

**Priority** (1 = highest): P1 0–5, then P2(pr 1) 5–8, P4(pr 3) 8–14, P3(pr 4) 14–22, P5(pr 5) 22–24 → waits 0, 4, 12, 5, 18 → **avg 7.8**; turnaround **avg 12.6**.

**Round Robin, q = 3** (new arrivals queue ahead of the preempted process): P1 0–3, P2 3–6, P3 6–9, P4 9–12, P1 12–14, P5 14–16, P3 16–19, P4 19–22, P3 22–24 → completions P1 14, P2 6, P3 24, P4 22, P5 16 → turnaround 14, 5, 22, 19, 12 (**avg 14.4**) → waits 9, 2, 14, 13, 10 (**avg 9.6**); response **avg 4.4**; 8 switches.

---

## 4. Optimisation log (hyperparameter-tuning phase)

Avg waiting time on 200 unseen workloads, mean of 3 training seeds (lower is better).
The report's Gantt chart plans a hyperparameter-tuning phase; these are the experiments behind the final settings.

| Setting | standard | heavy load | bimodal | large (20 procs) |
|---|---|---|---|---|
| SJF reference (ceiling) | 6.42 | 8.09 | 7.43 | 14.09 |
| FCFS baseline | 10.62 | 14.54 | 12.29 | 29.45 |
| Initial agent: trained on 8 processes, 10 slots, 1000 episodes | 7.38 | 9.95 | 8.72 | 26.17 |
| + 20 visible queue slots only | 7.38 | 9.95 | 8.72 | 26.29 |
| + train on mixed workload sizes (4-20 processes) | 7.26 | 9.14 | 8.04 | 18.54 |
| + extra queue-length bin | 7.22 | 9.08 | 8.04 | 19.03 |
| **+ 3000 episodes (final)** | **7.13** | **8.87** | **7.99** | **17.08** |
| Reward ablation: waiting term only | 7.09 | 8.83 | 8.02 | 17.17 |
| Feature ablation: add burst-rank feature (8 procs) | 8.09 | 10.79 | 9.28 | - |

Takeaways: training on varied queue sizes is what fixed the large-workload weakness (-29 %).
Reward weights barely matter (within noise), so the report's multi-objective reward is kept.
Adding features made the table bigger and the results worse.

## 5. Known limitations (honest results)

* The RL agent is non-preemptive. It approaches but does not beat **SJF** (about 8-11 % behind; 21 % on 20-process workloads), which is shown as a *reference* because it needs exact burst times in advance. The agent sees the same burst feature, so SJF marks the ceiling for a non-preemptive policy.
* RL's worst-case wait matches the best baselines (max wait 17.9 vs Round Robin 17.8 and SJF 17.9), but Round Robin has better *response* time, because it preempts.
* Context switches are identical for all non-preemptive schedulers (n - 1 when there is no idle time), so RL can only "win" that metric against Round Robin. The environment supports `time_slice=` for preemptive RL experiments.
* Tabular Q-learning, as the report's scope specifies. A function approximator (DQN/PPO, report Appendix E) is the natural next step for much larger state spaces.
* With 10 000 permutations and 180 simultaneous comparisons, the smallest Holm-corrected p-value possible is 0.018. That's why many entries show exactly that value.
