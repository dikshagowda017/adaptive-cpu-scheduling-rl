"""
Adaptive CPU Scheduling using Reinforcement Learning — main entry point
=========================================================================

    python main.py                 # full pipeline: train -> evaluate -> plots -> results/RESULTS.md
    python main.py --quick         # same pipeline, small numbers (~10 s) for a smoke test
    python main.py demo            # step-by-step walk-through on the report's sample workload
    python main.py demo --workload examples/convoy_effect.csv   # ...or on YOUR OWN processes
    python main.py train           # only train + save the Q-table and learning curve
    python main.py evaluate        # evaluate a saved Q-table (results/q_table.json)

Everything is seeded (--seed), so the same command gives the same numbers.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from workload_generator import REPORT_SAMPLE_WORKLOAD, load_workload_csv
from simulator import CPUSimulator
from round_robin import DEFAULT_QUANTUM
from gym_env import CPUSchedulingEnv
from q_learning import (train, save_agent, load_agent, run_agent, NUM_EPISODES,
                        REPORT_NUM_EPISODES, TRAIN_PROCESS_RANGE,
                        ALPHA, GAMMA, EPSILON_START, EPSILON_MIN, EPSILON_DECAY)
import evaluate as ev
import visualize as viz

FULL_STATE_NAME = "RL (full-state, original)"
# The original design, reproduced exactly: 8 processes, 10 queue slots,
# a fixed pool of 10 training workloads, Appendix-B episode count.
FULL_STATE_POOL = 10
FULL_STATE_PROCESSES = 8
FULL_STATE_QUEUE = 10


def cmd_train(args):
    low, high = TRAIN_PROCESS_RANGE
    print(f"Training per-process Q-learning agent: {args.episodes} episodes, "
          f"{low}-{high} processes per workload, seed {args.seed}")
    t0 = time.time()
    agent, history = train(num_episodes=args.episodes, seed=args.seed,
                           verbose_every=max(args.episodes // 10, 1))
    print(f"  done in {time.time() - t0:.1f}s - Q-table entries: {len(agent.q_table)}")

    os.makedirs(args.out, exist_ok=True)
    save_agent(agent, os.path.join(args.out, "q_table.json"))
    pd.DataFrame(history).rename_axis("episode").to_csv(
        os.path.join(args.out, "training_history.csv"))
    if not args.no_plots:
        viz.plot_learning_curve(history, args.out, window=max(min(50, args.episodes // 10), 1))
    return agent, history


def cmd_evaluate(args, agent=None, history=None):
    if agent is None:
        path = os.path.join(args.out, "q_table.json")
        if not os.path.exists(path):
            raise SystemExit(f"No trained agent at {path}. Run `python main.py train` first.")
        agent = load_agent(path)

    full_episodes = min(args.episodes, REPORT_NUM_EPISODES)
    print(f"Training the original full-state agent for comparison "
          f"({full_episodes} episodes on a fixed pool of {FULL_STATE_POOL} workloads)")
    full_agent, _ = train(num_episodes=full_episodes, num_processes=FULL_STATE_PROCESSES,
                          agent_kind="full_state", num_training_workloads=FULL_STATE_POOL,
                          seed=args.seed, verbose_every=0,
                          env_kwargs={"max_queue_size": FULL_STATE_QUEUE})

    agents = {ev.RL_NAME: agent, FULL_STATE_NAME: full_agent}
    print(f"Evaluating {len(ev.BASELINES) + len(agents)} schedulers on "
          f"{args.workloads} unseen workloads x {len(ev.SCENARIOS)} scenarios ...")
    t0 = time.time()
    df = ev.evaluate(agents, num_workloads=args.workloads)
    summary = ev.summarize(df)
    sig = ev.significance(df)
    gen = ev.generalisation_check(agents, train_seeds=range(FULL_STATE_POOL),
                                  num_unseen=args.workloads, num_processes=FULL_STATE_PROCESSES)
    verdict = ev.hypothesis_verdict(sig)
    print(f"  done in {time.time() - t0:.1f}s")

    df.to_csv(os.path.join(args.out, "per_workload_results.csv"), index=False)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    sig.to_csv(os.path.join(args.out, "significance.csv"), index=False)
    gen.to_csv(os.path.join(args.out, "generalisation.csv"), index=False)

    timelines = sample_timelines(agent)
    if not args.no_plots:
        viz.plot_comparison(summary, args.out)
        viz.plot_scenarios(summary, args.out)
        viz.plot_distribution(df, args.out)
        viz.plot_generalisation(gen, args.out)
        viz.plot_gantt(timelines, args.out)

    write_results_md(args, summary, sig, gen, verdict, history)
    print_headline(summary, sig, verdict)


def sample_timelines(agent) -> dict:
    """CPU timeline of every scheduler on the report's Appendix-A workload."""
    timelines = {}
    for name, run in ev.BASELINES.items():
        sim = CPUSimulator(REPORT_SAMPLE_WORKLOAD)
        run(sim)
        timelines[name] = sim.timeline
    env = CPUSchedulingEnv(workload=REPORT_SAMPLE_WORKLOAD)
    run_agent(agent, env)
    timelines[ev.RL_NAME] = env.sim.timeline
    return timelines


def print_headline(summary, sig, verdict):
    std = summary[summary.scenario == "standard"].set_index("scheduler")
    print("\n=== Standard scenario (unseen workloads) ===")
    cols = ["avg_waiting_time", "avg_turnaround_time", "max_waiting_time",
            "context_switches", "fairness_index"]
    table = std[[f"{c}_mean" for c in cols]].round(3)
    table.columns = cols
    print(table.to_string())
    s = sig[(sig.scenario == "standard") & (sig.metric == "avg_waiting_time")]
    print("\nRL improvement in avg waiting time (positive = RL better):")
    for _, r in s.iterrows():
        print(f"  vs {r.baseline:<26} {r.improvement_pct:+6.1f}%  "
              f"(wins {r.wins}/{r.wins + r.ties + r.losses}, Holm p = {r.p_holm:.4f})")
    print(f"\nHypothesis H1 (RL significantly better than FCFS, RR and Priority on >= 1 metric): "
          f"{'SUPPORTED' if verdict['reject_H0'] else 'NOT supported'} "
          f"-> metrics: {', '.join(verdict['metrics_beating_all_classical']) or 'none'}")


def write_results_md(args, summary, sig, gen, verdict, history):
    lines = ["# Results", "",
             f"Generated by `python main.py` — seed {args.seed}, {args.episodes} training episodes, "
             f"{args.workloads} unseen test workloads per scenario "
             f"(workload seeds >= {ev.TEST_SEED_START}, never used in training).", "",
             f"Hyperparameters: alpha={ALPHA}, gamma={GAMMA}, epsilon {EPSILON_START} -> "
             f"{EPSILON_MIN} (x{EPSILON_DECAY}/episode), Round Robin quantum={DEFAULT_QUANTUM}.", ""]
    if history:
        r = history["reward"]
        k = max(len(r) // 20, 1)
        lines += [f"Training: mean reward of first {k} episodes {np.mean(r[:k]):.2f} -> "
                  f"last {k} episodes {np.mean(r[-k:]):.2f}.", ""]
    lines += ["## Hypothesis (report section 4.9.1)", "",
              f"**H1 {'supported — H0 rejected' if verdict['reject_H0'] else 'not supported'}**: "
              f"in the standard scenario the RL scheduler is significantly better "
              f"(paired permutation test, Holm-corrected p < 0.05) than all of FCFS, Round Robin "
              f"and Priority on: {', '.join(verdict['metrics_beating_all_classical']) or 'none'}.", ""]
    for scenario, (_, _, desc) in ev.SCENARIOS.items():
        lines += [f"## Scenario `{scenario}` — {desc}", "",
                  ev.format_table(summary, scenario), ""]
        s = sig[(sig.scenario == scenario) & (sig.metric == "avg_waiting_time")]
        lines += ["RL improvement in avg waiting time (positive = RL better, Holm-corrected p): " + "; ".join(
            f"{r.baseline} {r.improvement_pct:+.1f}% (p={r.p_holm:.3g})" for _, r in s.iterrows()), ""]
    lines += ["## Generalisation check", "",
              "| Workload set | Scheduler | Avg waiting time | Unseen-state rate |", "|---|---|---|---|"]
    for _, r in gen.iterrows():
        rate = "—" if np.isnan(r.unseen_state_rate) else f"{r.unseen_state_rate:.0%}"
        lines.append(f"| {r.workload_set} | {r.scheduler} | {r.avg_waiting_time:.2f} | {rate} |")
    lines += ["", "## Charts", ""] + [f"![{p}]({p})" for p in (
        "learning_curve.png", "comparison_standard.png", "scenarios_waiting_time.png",
        "waiting_time_distribution.png", "generalisation.png", "gantt_sample.png")]
    with open(os.path.join(args.out, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def cmd_demo(args):
    """Run every scheduler on one workload: the report's Appendix A by default,
    or the user's own processes from --workload FILE.csv."""
    if args.workload:
        try:
            workload = load_workload_csv(args.workload)
        except (OSError, ValueError) as e:
            raise SystemExit(f"Could not load workload: {e}")
        title = f"Workload from {args.workload}"
    else:
        workload = list(REPORT_SAMPLE_WORKLOAD)
        title = "Report Appendix A workload"
    print(f"{title}:")
    for p in workload:
        print(f"  {p}")

    saved = os.path.join(args.out, "q_table.json")
    if os.path.exists(saved) and args.episodes == NUM_EPISODES:
        agent = load_agent(saved)
        print(f"\nUsing the trained agent saved in {saved}")
    else:
        print(f"\nTraining the RL agent ({args.episodes} episodes, seed {args.seed}) ...")
        agent, _ = train(num_episodes=args.episodes, seed=args.seed, verbose_every=0)

    results, timelines = {}, {}
    print()
    for name, run in ev.BASELINES.items():
        sim = CPUSimulator(workload)
        results[name] = run(sim)
        timelines[name] = sim.timeline
    env = CPUSchedulingEnv(workload=workload, max_queue_size=agent.num_actions)
    results[ev.RL_NAME] = run_agent(agent, env)
    timelines[ev.RL_NAME] = env.sim.timeline
    for name in results:
        _print_run(name, timelines[name], results[name])

    print("Ranking by average waiting time (lower is better):")
    ranked = sorted(results.items(), key=lambda kv: kv[1]["avg_waiting_time"])
    for rank, (name, m) in enumerate(ranked, 1):
        print(f"  {rank}. {name:<18} avg wait {m['avg_waiting_time']:6.2f}   "
              f"avg turnaround {m['avg_turnaround_time']:6.2f}   "
              f"fairness {m['fairness_index']:.2f}")
    if not args.no_plots:
        path = viz.plot_gantt(timelines, args.out, title=f"CPU timeline - {title}",
                              filename="gantt_demo.png")
        print(f"\nGantt chart saved to {path}")
    if not args.workload:
        print("\nCheck by hand: waiting = turnaround - burst for every process; "
              "FCFS waits are 0, 4, 6, 13, 18 -> average 8.2.")


def _print_run(name, timeline, metrics):
    bar = " ".join(f"[{pid} {s}-{e}]" for pid, s, e in timeline)
    print(f"{name}\n  {bar}")
    print("  " + ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in metrics.items()) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Adaptive CPU scheduling with Q-learning")
    parser.add_argument("command", nargs="?", default="all",
                        choices=["all", "train", "evaluate", "demo"])
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--workloads", type=int, default=100,
                        help="unseen test workloads per scenario")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workload", metavar="CSV",
                        help="demo only: schedule your own processes from a CSV file "
                             "(columns: id, arrival, burst, priority)")
    parser.add_argument("--out", default="results")
    parser.add_argument("--quick", action="store_true",
                        help="small run for smoke testing (300 episodes, 20 workloads)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    if args.quick:
        args.episodes, args.workloads = 300, 20
    if args.episodes < 1 or args.workloads < 1:
        parser.error("--episodes and --workloads must be >= 1")
    if args.seed < 0:
        parser.error("--seed must be >= 0")
    if args.workload and args.command != "demo":
        parser.error("--workload can only be used with the demo command")

    if args.command == "demo":
        cmd_demo(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    else:
        agent, history = cmd_train(args)
        cmd_evaluate(args, agent, history)
        print(f"\nAll outputs written to ./{args.out}/ (CSV, PNG, RESULTS.md)")


if __name__ == "__main__":
    main()
