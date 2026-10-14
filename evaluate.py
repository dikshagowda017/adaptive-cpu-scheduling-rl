"""
Evaluation Module
------------------
Runs every scheduler on IDENTICAL held-out workloads and compares them.

  - Schedulers: FCFS, Round Robin, Priority (the report's baselines),
    SJF (reference: needs burst times in advance), and the RL agent(s).
  - Scenarios: the training distribution plus harder / different loads,
    to check the agent adapts instead of memorising.
  - Workload seeds start at TEST_SEED_START, which training never uses.
  - Statistics: mean, std, 95% confidence interval, and a paired
    sign-flip permutation test (with Holm correction) of RL vs each
    baseline — this is how hypothesis H1 of report section 4.9.1 is tested.
"""

import numpy as np
import pandas as pd

from simulator import CPUSimulator
from workload_generator import generate_workload
from fcfs import run_fcfs
from sjf import run_sjf
from round_robin import run_round_robin, DEFAULT_QUANTUM
from priority import run_priority
from gym_env import CPUSchedulingEnv
from q_learning import run_agent, TEST_SEED_START

RL_NAME = "RL (Q-learning)"
CLASSICAL = ["FCFS", "Round Robin", "Priority"]
REFERENCE = ["SJF (reference)"]

BASELINES = {
    "FCFS": run_fcfs,
    "Round Robin": lambda sim: run_round_robin(sim, quantum=DEFAULT_QUANTUM),
    "Priority": run_priority,
    "SJF (reference)": run_sjf,
}

# name -> (num_processes, workload_kwargs, description)
SCENARIOS = {
    "standard": (8, {}, "Training distribution: 8 processes, mean gap 2, exponential bursts (mean 5)"),
    "heavy_load": (8, {"mean_inter_arrival": 1.0}, "Arrivals twice as dense -> longer queues"),
    "light_load": (8, {"mean_inter_arrival": 4.0}, "Sparse arrivals -> short queues, little choice"),
    "bimodal": (8, {"burst_distribution": "bimodal"}, "80% short interactive + 20% long CPU-bound jobs"),
    "uniform": (8, {"burst_distribution": "uniform"}, "Uniform bursts in [1, 9]"),
    "large": (20, {}, "20 processes per workload (agent was trained on 8)"),
}

# For each metric: True if lower is better.
METRICS = {
    "avg_waiting_time": True,
    "avg_turnaround_time": True,
    "avg_response_time": True,
    "max_waiting_time": True,
    "context_switches": True,
    "fairness_index": False,
}


def evaluate(agents: dict, scenarios: dict = None, num_workloads: int = 100,
             seed_start: int = TEST_SEED_START) -> pd.DataFrame:
    """Run every baseline and every agent on the same workloads.

    agents: {display_name: trained agent}
    Returns one row per (scenario, workload, scheduler)."""
    scenarios = SCENARIOS if scenarios is None else scenarios
    rows = []
    for scenario, (num_processes, kwargs, _) in scenarios.items():
        envs = {name: CPUSchedulingEnv(num_processes=num_processes, workload_kwargs=kwargs,
                                       max_queue_size=agent.num_actions)
                for name, agent in agents.items()}
        for seed in range(seed_start, seed_start + num_workloads):
            workload = generate_workload(num_processes=num_processes, seed=seed, **kwargs)
            for name, run in BASELINES.items():
                rows.append({"scenario": scenario, "workload_seed": seed, "scheduler": name,
                             **run(CPUSimulator(workload))})
            for name, agent in agents.items():
                rows.append({"scenario": scenario, "workload_seed": seed, "scheduler": name,
                             **run_agent(agent, envs[name], seed=seed)})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, std and 95% confidence half-width per scenario/scheduler/metric."""
    out = []
    for (scenario, scheduler), group in df.groupby(["scenario", "scheduler"], sort=False):
        row = {"scenario": scenario, "scheduler": scheduler, "n": len(group)}
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
            row[f"{metric}_ci95"] = 1.96 * row[f"{metric}_std"] / np.sqrt(len(values))
        out.append(row)
    return pd.DataFrame(out)


def paired_permutation_test(a, b, num_permutations: int = 10_000, seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation test on mean(a - b).

    Under H0 (no difference) each paired difference is equally likely to
    be positive or negative, so we randomly flip signs and count how often
    the shuffled mean is at least as extreme as the observed one. Makes no
    normality assumption and needs no SciPy."""
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if diffs.size == 0:
        raise ValueError("need at least one paired observation")
    if np.allclose(diffs, 0):
        return 1.0
    observed = abs(diffs.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(num_permutations, diffs.size))
    permuted = np.abs((signs * diffs).mean(axis=1))
    return float((np.sum(permuted >= observed - 1e-12) + 1) / (num_permutations + 1))


def holm_correction(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (controls family-wise error)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running_max
    return adjusted


def significance(df: pd.DataFrame, target: str = RL_NAME, alpha: float = 0.05) -> pd.DataFrame:
    """Compare `target` against every other scheduler, per scenario and metric."""
    rows = []
    for scenario, sdf in df.groupby("scenario", sort=False):
        pivot = {name: g.set_index("workload_seed").sort_index()
                 for name, g in sdf.groupby("scheduler", sort=False)}
        if target not in pivot:
            continue
        for baseline, bdf in pivot.items():
            if baseline == target:
                continue
            for metric, lower_is_better in METRICS.items():
                t = pivot[target][metric].to_numpy(dtype=float)
                b = bdf[metric].to_numpy(dtype=float)
                better = (t < b) if lower_is_better else (t > b)
                worse = (t > b) if lower_is_better else (t < b)
                diff = t.mean() - b.mean()
                improvement = (-diff if lower_is_better else diff) / abs(b.mean()) * 100 \
                    if b.mean() != 0 else 0.0
                rows.append({
                    "scenario": scenario, "baseline": baseline, "metric": metric,
                    "target_mean": t.mean(), "baseline_mean": b.mean(),
                    "improvement_pct": improvement,
                    "wins": int(better.sum()), "ties": int((~better & ~worse).sum()),
                    "losses": int(worse.sum()),
                    "p_value": paired_permutation_test(t, b),
                })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_holm"] = holm_correction(result["p_value"])
        result["significant_better"] = (result["p_holm"] < alpha) & (result["improvement_pct"] > 0)
        result["significant_worse"] = (result["p_holm"] < alpha) & (result["improvement_pct"] < 0)
    return result


def hypothesis_verdict(sig: pd.DataFrame, scenario: str = "standard") -> dict:
    """Report section 4.9.1: H1 holds if RL is significantly better than ALL
    three classical schedulers on at least one metric."""
    s = sig[(sig.scenario == scenario) & (sig.baseline.isin(CLASSICAL))]
    metrics_won = [m for m in METRICS
                   if s[(s.metric == m)]["significant_better"].sum() == len(CLASSICAL)]
    return {"scenario": scenario, "metrics_beating_all_classical": metrics_won,
            "reject_H0": bool(metrics_won)}


def generalisation_check(agents: dict, train_seeds, num_unseen: int = 100,
                         num_processes: int = 8) -> pd.DataFrame:
    """Avg waiting time of each agent (and FCFS) on the workloads an agent was
    trained on vs workloads it has never seen. A big gap = memorisation."""
    rows = []
    sets = {"training workloads": list(train_seeds),
            "unseen workloads": list(range(TEST_SEED_START, TEST_SEED_START + num_unseen))}
    for set_name, seeds in sets.items():
        fcfs = [run_fcfs(CPUSimulator(generate_workload(num_processes, seed=s)))["avg_waiting_time"]
                for s in seeds]
        rows.append({"workload_set": set_name, "scheduler": "FCFS",
                     "avg_waiting_time": float(np.mean(fcfs)), "unseen_state_rate": np.nan})
        for name, agent in agents.items():
            env = CPUSchedulingEnv(num_processes=num_processes, max_queue_size=agent.num_actions)
            agent.lookups = agent.unseen_lookups = 0
            waits = [run_agent(agent, env, seed=s)["avg_waiting_time"] for s in seeds]
            rows.append({"workload_set": set_name, "scheduler": name,
                         "avg_waiting_time": float(np.mean(waits)),
                         "unseen_state_rate": agent.unseen_state_rate()})
    return pd.DataFrame(rows)


def format_table(summary: pd.DataFrame, scenario: str, metrics=None) -> str:
    """Markdown table of mean ± 95% CI for one scenario."""
    metrics = metrics or list(METRICS)
    s = summary[summary.scenario == scenario]
    header = "| Scheduler | " + " | ".join(metrics) + " |"
    lines = [header, "|" + "---|" * (len(metrics) + 1)]
    for _, r in s.iterrows():
        cells = [f"{r[f'{m}_mean']:.2f} ± {r[f'{m}_ci95']:.2f}" for m in metrics]
        lines.append(f"| {r.scheduler} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    from q_learning import train

    agent, _ = train(verbose_every=0)
    df = evaluate({RL_NAME: agent}, num_workloads=50)
    summary = summarize(df)
    for scenario in SCENARIOS:
        print(f"\n### {scenario}")
        print(format_table(summary, scenario, ["avg_waiting_time", "avg_turnaround_time",
                                               "max_waiting_time", "fairness_index"]))
    print("\nHypothesis:", hypothesis_verdict(significance(df)))
