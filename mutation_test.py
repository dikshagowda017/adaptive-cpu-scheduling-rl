"""
Mutation testing: "who tests the tests?"
-----------------------------------------
Each mutation below deliberately puts a realistic bug back into a copy of
the code (several are the exact bugs we found and fixed, see TESTING.md).
A good test suite must FAIL on every one of them ("mutant killed").
A mutant that survives means there's a gap in the tests.

    python mutation_test.py            # run all mutants
    python mutation_test.py --list     # list them
    python mutation_test.py --show 1   # plant bug #1 and show which tests catch it

Runs in a temporary copy, so the real source files are never touched.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))

# (description, file, original text, mutated text)
MUTATIONS = [
    ("B1: mid-run arrivals start with 0 waiting time", "simulator.py",
     "proc._queued_since = proc.arrival_time", "proc._queued_since = self.time"),
    ("preempted process forgets it is waiting again", "simulator.py",
     "            proc._queued_since = self.time\n            self.ready_queue.append(proc)",
     "            self.ready_queue.append(proc)"),
    ("B12: zero-length bursts accepted (crash later)", "simulator.py",
     "if p.burst_time < 1:", "if p.burst_time < 0:"),
    ("B13: negative arrival times accepted", "simulator.py",
     "if p.arrival_time < 0:", "if p.arrival_time < -10**9:"),
    ("B14: duplicate process IDs accepted", "simulator.py",
     "        if p.id in seen:\n", "        if False:\n"),
    ("first dispatch counted as a context switch", "simulator.py",
     "context_switch = (self._last_run_id is not None", "context_switch = (True"),
    ("B3: fairness computed on raw waiting times", "simulator.py",
     "slowdowns = [p.turnaround_time / p.burst_time for p in self.completed]",
     "slowdowns = [p.waiting_time + 1e-9 for p in self.completed]"),
    ("arrival waiting missing from reward signal", "simulator.py",
     "        waiting_added += self._admit_arrivals()", "        self._admit_arrivals()"),
    ("preempted process re-queued before new arrivals", "simulator.py",
     "        waiting_added += self._admit_arrivals()\n",
     "        self.ready_queue.append(proc) if proc.remaining_burst_time > 0 else None\n"
     "        waiting_added += self._admit_arrivals()\n"),
    ("off-by-one: time slice runs one unit too long", "simulator.py",
     "run_time = min(run_time, time_slice)", "run_time = min(run_time, time_slice + 1)"),
    ("Priority scheduler picks the LOWEST priority", "priority.py",
     "return min(range(len(ready_queue)), key=lambda i: ready_queue[i].priority)",
     "return max(range(len(ready_queue)), key=lambda i: ready_queue[i].priority)"),
    ("SJF picks the longest job", "sjf.py",
     "    return min(range(len(ready_queue)),", "    return max(range(len(ready_queue)),"),
    ("B2: inter-arrival gap ignored (always 1)", "workload_generator.py",
     "rng.poisson(lam=mean_inter_arrival, size=num_processes)",
     "rng.poisson(lam=1.0, size=num_processes)"),
    ("CSV loader ignores the priority column", "workload_generator.py",
     'raw = (row[columns[key]] or "").strip() if columns[key] else "1"', 'raw = "1" if key == "priority" else (row[columns[key]] or "").strip()'),
    ("B4: environment uses unseeded global RNG", "gym_env.py",
     "use_seed = int(self.np_random.integers(0, 2**31 - 1))",
     "use_seed = int(np.random.randint(0, 2**31 - 1))"),
    ("observation drops the priority feature", "gym_env.py",
     "obs[base + 2] = min(proc.priority / MAX_PRIORITY, 1.0)", "pass"),
    ("waiting reward not normalised per process", "gym_env.py",
     'waiting = info["waiting_added"] / n', 'waiting = info["waiting_added"]'),
    ("Q-learning ignores the discount factor", "q_learning.py",
     "td_target = reward + self.gamma * best_next_value", "td_target = reward + best_next_value"),
    ("Q-learning ignores the learning rate", "q_learning.py",
     "self.q_table[key] += self.alpha * td_error", "self.q_table[key] += td_error"),
    ("Q-update moves AWAY from the target", "q_learning.py",
     "self.q_table[key] += self.alpha * td_error", "self.q_table[key] -= self.alpha * td_error"),
    ("B6: no action masking (agent may pick empty slots)", "q_learning.py",
     "        valid = _valid_actions(mask)\n", "        valid = np.arange(len(mask))\n"),
    ("terminal state still bootstraps from next state", "q_learning.py",
     "        if done:\n            best_next_value = 0.0\n        else:\n"
     "            best_next_value = float(np.max(self.action_values(next_obs, next_mask)))",
     "        best_next_value = float(np.max(self.action_values(next_obs, next_mask)))"),
    ("training always uses the smallest workload size", "q_learning.py",
     "env.num_processes = int(seed_rng.integers(low, high + 1)) if high > low else low",
     "env.num_processes = low"),
    ("agent/environment size mismatch not detected", "q_learning.py",
     "    if env.action_space.n != agent.num_actions:", "    if False:"),
    ("evaluation leaks training seeds into the test set", "q_learning.py",
     "TEST_SEED_START = 1_000_000", "TEST_SEED_START = 0"),
    ("permutation test is one-sided", "evaluate.py",
     "observed = abs(diffs.mean())", "observed = diffs.mean()"),
    ("Holm correction skipped (raw p-values used)", "evaluate.py",
     'result["p_holm"] = holm_correction(result["p_value"])', 'result["p_holm"] = result["p_value"]'),
    ("B15: CLI accepts negative seeds", "main.py",
     "    if args.seed < 0:", "    if False:"),
]


def _copy_project(tmp):
    for name in os.listdir(HERE):
        if name.endswith((".py", ".md")) or name == "pytest.ini":
            shutil.copy(os.path.join(HERE, name), tmp)
    for folder in ("tests", "examples"):
        shutil.copytree(os.path.join(HERE, folder), os.path.join(tmp, folder),
                        ignore=shutil.ignore_patterns("__pycache__", ".hypothesis"))


def run_mutation(description, filename, original, mutated, show=False):
    """Returns (status, names of the tests that failed)."""
    with tempfile.TemporaryDirectory() as tmp:
        _copy_project(tmp)
        path = os.path.join(tmp, filename)
        with open(path, encoding="utf-8") as f:
            source = f.read()
        if original not in source:
            return "NOT APPLIED", []
        with open(path, "w", encoding="utf-8") as f:
            f.write(source.replace(original, mutated, 1))
        args = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                "--no-header", "-rfE", "--tb=line" if show else "--tb=no"]
        if not show:
            args.append("-x")      # one failing test is enough to kill the mutant
        result = subprocess.run(args, cwd=tmp, capture_output=True, text=True,
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        failed = [line.split(" ", 1)[1].split(" - ")[0]
                  for line in result.stdout.splitlines() if line.startswith(("FAILED ", "ERROR "))]
        return ("KILLED" if result.returncode != 0 else "SURVIVED"), failed


def main(argv=None):
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="list the mutants and exit")
    parser.add_argument("--show", type=int, metavar="N",
                        help="plant only mutant N and show every test that catches it")
    parser.add_argument("--json", metavar="FILE", help="also write the results as JSON")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="mutants to run in parallel (default: half the CPU cores)")
    args = parser.parse_args(argv)

    if args.list:
        for i, (description, filename, _, _) in enumerate(MUTATIONS, 1):
            print(f"  {i:2d}. {filename:<22} {description}")
        return 0
    if args.show:
        if not 1 <= args.show <= len(MUTATIONS):
            parser.error(f"--show must be between 1 and {len(MUTATIONS)}")
        description, filename, original, mutated = MUTATIONS[args.show - 1]
        print(f"Planting mutant {args.show}: {description} ({filename})")
        print(f"  - {original.strip()[:100]}")
        print(f"  + {mutated.strip()[:100]}\n")
        status, failed = run_mutation(description, filename, original, mutated, show=True)
        if failed:
            from collections import Counter
            print("Tests that failed (= caught the bug):")
            for test in failed[:12]:
                print(f"  x {test}")
            if len(failed) > 12:
                print(f"  ... and {len(failed) - 12} more")
            print("\nBy test file:")
            for name, count in Counter(t.split("::")[0] for t in failed).most_common():
                print(f"  {count:4d}  {name}")
        print(f"\n{status}: {len(failed)} test(s) caught this bug.")
        return 0 if status == "KILLED" else 1

    jobs = max(1, args.jobs)
    print(f"Running {len(MUTATIONS)} mutants, {jobs} at a time "
          f"(each runs the full test suite)...", flush=True)
    rows = [None] * len(MUTATIONS)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(run_mutation, *m): i for i, m in enumerate(MUTATIONS)}
        for done_count, future in enumerate(as_completed(futures), 1):
            i = futures[future]
            status, failed = future.result()
            description, filename, _, _ = MUTATIONS[i]
            rows[i] = {"id": i + 1, "file": filename, "mutation": description,
                       "status": status, "caught_by": failed[:1]}
            print(f"  ...{done_count}/{len(MUTATIONS)} done", end=chr(13), flush=True)
    print(" " * 30)
    for r in rows:
        caught = f"  <- {r['caught_by'][0]}" if r["caught_by"] else ""
        print(f"  {r['id']:2d}. [{r['status']:^11}] {r['file']:<22} {r['mutation']}{caught}")
    killed = sum(r["status"] == "KILLED" for r in rows)
    print(f"\nMutation score: {killed}/{len(rows)} mutants killed ({killed / len(rows):.0%})")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"killed": killed, "total": len(rows), "mutants": rows}, f, indent=2)
    return 0 if killed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
