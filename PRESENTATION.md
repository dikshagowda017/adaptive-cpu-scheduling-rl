# Testathon presentation guide

For the team: what to run, what to say, and how to answer the judges.

## Before you go (the night before)

1. Double-click **`run_everything.bat`** and wait about 3 minutes. It must end with
   `OVERALL: ALL CHECKS PASSED`, and the dashboard opens in your browser.
2. Keep that browser tab open. `results/dashboard.html` works **offline**, so venue Wi-Fi doesn't matter.
3. Open a terminal in the project folder with the venv active (`venv\Scripts\Activate.ps1`)
   and run `python main.py demo` once, so everything is warmed up.
4. Optional: push the repo to GitHub so the green CI badge / Actions run is visible.

## The 5-minute demo (in this order)

| # | Show | Say |
|---|---|---|
| 1 | Dashboard, top tiles | "401 automated tests, 95 % coverage, 16 real bugs found and fixed, and our tests catch 28 out of 28 bugs we planted on purpose. The RL scheduler waits 29-34 % less than FCFS, Round Robin and Priority, and is significantly better in all 6 workload scenarios." |
| 2 | `TESTING.md` → bug B1 | "The original simulator under-counted waiting time. On our own report's example, FCFS said 6.2. We worked it out by hand: the right answer is 8.2." |
| 3 | `python main.py demo` | "Here's the report's example again: FCFS waits are 0, 4, 6, 13, 18, so the average is 8.2. You can check it on paper. Our RL agent finds the optimal order here (5.6, same as SJF)." |
| 4 | `python main.py demo --workload examples/convoy_effect.csv` | "The convoy effect: FCFS makes the short jobs wait behind a long one, 18.2 average. RL learned to run the short jobs first: 3.2." Invite a judge to give you processes and type them into a CSV. |
| 5 | `python mutation_test.py --show 1` (~40 s) | "Who tests the tests? This puts the original bug back. 164 tests fail immediately, and the first one names the exact function." |
| 6 | Dashboard → generalisation chart | "Testing on workloads the agent never saw revealed the original agent was memorising: 61 % of states were new to it, so it secretly behaved like FCFS. We redesigned it, and it now generalises." |
| 7 | Dashboard → "Known limitations" in `TESTING.md` | "We're honest about limits: RL doesn't beat SJF, which needs to know the future, and when a long job arrives alone, only preemptive Round Robin helps." |

## Testing techniques you can name (all in this repo)

| Technique | Where | One-line explanation |
|---|---|---|
| Golden / oracle tests | `test_simulator.py` | Expected answers worked out by hand from the report's Appendix A |
| Invariant (metamorphic) testing | `test_simulator.py`, `test_properties.py` | Laws that must always hold, e.g. waiting = turnaround − burst; shifting every arrival by k changes nothing |
| Property-based testing | `test_properties.py` | The Hypothesis library generates ~2 100 random workloads and shrinks any failure to the smallest example |
| Fuzzing | `test_simulator.py`, `test_properties.py` | Random scheduling decisions and random actions; nothing may crash or break the invariants |
| Boundary / negative testing | `test_robustness.py` | Zero burst, negative arrival, duplicate IDs, bad CSV, bad CLI values → clear errors |
| Regression tests | Every bug B1–B16 | Each bug has a test that fails if it ever comes back |
| Integration / contract tests | `test_gym_env.py` | Official Gymnasium `check_env`, exact observation values |
| End-to-end ML test | `test_q_learning.py` | A trained agent must beat FCFS/RR/Priority by > 15 % on unseen workloads |
| Data-leakage test | `test_q_learning.py` | Training seeds and test seeds may never overlap |
| Statistical testing | `evaluate.py` | Paired permutation test + Holm correction for 180 comparisons |
| Performance / stress tests | `test_robustness.py` | 10 000 processes in < 5 s (a 20× speed-up was found this way) |
| Smoke tests | `test_scripts.py` | Every documented command actually runs |
| Mutation testing | `mutation_test.py` | 28 planted bugs, all caught; it also exposed 4 weak tests, which we fixed |
| Continuous integration | `.github/workflows/tests.yml` | Ubuntu + Windows × Python 3.10/3.12/3.13 on every push |

## Likely judge questions

**"How do you know your simulator is correct?"**
Three independent checks. Hand-computed golden values for every scheduler on the report's example.
The invariant waiting = turnaround − burst, checked on thousands of random workloads. And the textbook
theorem that SJF is optimal when all jobs arrive together, checked against all 24 possible orders.

**"Isn't the RL agent just memorising?"**
That was bug B5, and we test for it. Training uses workload seeds below 100 000 and testing uses seeds from 1 000 000,
which a test enforces. The original design saw 61 % unseen states on new workloads and fell back to FCFS. The new agent
sees under 1 % unseen states and still wins.

**"Why doesn't RL beat SJF?"**
SJF needs to know exact burst times in advance, which a real OS usually can't. It's the theoretical ceiling for
non-preemptive scheduling, so we show it as a reference. RL gets within 8-11 % of it, and matches its worst-case waiting time.

**"Is the improvement statistically significant, or just luck?"**
We used 100 unseen workloads per scenario and a paired permutation test, Holm-corrected for 180 comparisons.
All 18 RL-vs-FCFS/RR/Priority comparisons are significant (p = 0.018, the smallest value that test can give).

**"What happens if I give it bad input?"**
Try it: `python main.py demo --workload` with a CSV that has a zero burst or duplicate IDs gives a
clear error with the line number. Before our fixes, a zero burst crashed the program (B12).

**"How do you know your tests are any good?"**
Mutation testing: we plant 28 realistic bugs and all 28 are caught. When we first ran it, 4 bugs slipped through,
which showed us 4 weak tests. We strengthened them; that story is in `TESTING.md`.

**"Can you reproduce your numbers?"**
Yes. Everything is seeded: `python main.py` gives identical results on every run, and a test checks that two
training runs produce byte-identical agents (bug B4 was that this wasn't true).

**"Why tabular Q-learning, not deep RL?"**
That's the report's scope (§3.2). It's lightweight, explainable (the Q-table has under 600 entries)
and trains in seconds. DQN/PPO is listed as future work (Appendix E).

**"What did you change from the report's design?"**
Same algorithm and hyperparameters (α = 0.1, γ = 0.95, ε 1.0 → 0.01). We changed two things during the
planned tuning phase: the Q-table key is per process, so the agent generalises, and training covers
varied workload sizes for 3000 episodes. `TESTING.md` §4 has the experiments; `--episodes 1000` reproduces Appendix B.

## If something goes wrong on the day

| Problem | Fix |
|---|---|
| `Activate.ps1 cannot be loaded` | Skip activation: type `.\venv\Scripts\python` instead of `python` (e.g. `.\venv\Scripts\python main.py demo`), or use `run_everything.bat` |
| `python` not found | Install Python 3.10+ from python.org and tick "Add to PATH" |
| No internet at the venue | Not needed once `run_everything.bat` has run once on the presenting laptop (that creates `venv`). Do that at home. A `venv` folder only works on the laptop that created it, so a USB backup of the folder only helps on that same laptop |
| "The filename or extension is too long" | The folder is too deeply nested for Windows. Move the project folder to a short path such as `C:\Projects\` or the Desktop, delete `venv`, and run `run_everything.bat` again |
| Out of time | `python verify_all.py --fast` (about 1 minute; skips mutation testing) |
| A judge asks to see one test | `python -m pytest tests/test_simulator.py -k golden -v` |
