"""
Visualisation Module
---------------------
Turns training history and evaluation results into PNG charts:

  learning_curve.png       reward + avg waiting time per training episode
  comparison_standard.png  all schedulers, key metrics, standard scenario (mean ± 95% CI)
  scenarios_waiting.png    avg waiting time per scheduler across all scenarios
  waiting_distribution.png spread of avg waiting time over the test workloads
  generalisation.png       full-state vs per-process agent on seen vs unseen workloads
  gantt_sample.png         CPU timeline of every scheduler on the report's sample workload

Each scheduler keeps the same colour in every chart.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # render to files; no display needed
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Fixed categorical order (validated reference palette): colour follows the scheduler.
SCHEDULER_COLORS = {
    "RL (Q-learning)": "#2a78d6",
    "FCFS": "#eb6834",
    "Round Robin": "#1baf7a",
    "Priority": "#eda100",
    "SJF (reference)": "#e87ba4",
    "RL (full-state, original)": "#4a3aa7",
}
SCHEDULER_ORDER = ["FCFS", "Round Robin", "Priority", "SJF (reference)",
                   "RL (full-state, original)", "RL (Q-learning)"]
SHORT_LABELS = {
    "RL (Q-learning)": "RL\n(ours)",
    "RL (full-state, original)": "RL\n(original)",
    "SJF (reference)": "SJF\n(ref.)",
    "Round Robin": "Round\nRobin",
}
PROCESS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                  "#008300", "#4a3aa7", "#e34948"]

METRIC_LABELS = {
    "avg_waiting_time": "Avg waiting time",
    "avg_turnaround_time": "Avg turnaround time",
    "avg_response_time": "Avg response time",
    "max_waiting_time": "Max waiting time",
    "context_switches": "Context switches",
    "fairness_index": "Fairness index (Jain)",
}


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2)


def _figure(*args, **kwargs):
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def _ordered(names):
    return [n for n in SCHEDULER_ORDER if n in names] + \
           [n for n in names if n not in SCHEDULER_ORDER]


def _moving_average(values, window):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_learning_curve(history: dict, out_dir: str, window: int = 50):
    """Two stacked panels sharing the episode axis (never a dual y-axis).

    Every training episode is a different random workload, so raw per-episode
    numbers mostly reflect how hard that workload was. The bottom panel puts
    FCFS on the *same* workloads next to the agent, so the gap that opens up
    as epsilon decays is the learning."""
    fig, (ax1, ax2) = _figure(2, 1, figsize=(9, 6.5), sharex=True)
    episodes = np.arange(1, len(history["reward"]) + 1)
    rl, fcfs = SCHEDULER_COLORS["RL (Q-learning)"], SCHEDULER_COLORS["FCFS"]

    def smooth_line(ax, values, color, label):
        sm = _moving_average(values, window)
        ax.plot(episodes[len(episodes) - len(sm):], sm, color=color, linewidth=2, label=label)
        ax.annotate(label, (episodes[-1], sm[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_2)

    ax1.plot(episodes, history["reward"], color=rl, alpha=0.15, linewidth=1)
    smooth_line(ax1, history["reward"], rl, "RL agent")
    _style(ax1, ylabel="Episode reward")
    ax1.set_title(f"Q-learning training ({window}-episode moving averages; faint line = raw)",
                  color=INK, fontsize=12, loc="left", pad=10)

    if "fcfs_avg_waiting_time" in history:
        smooth_line(ax2, history["fcfs_avg_waiting_time"], fcfs, "FCFS, same workloads")
    smooth_line(ax2, history["avg_waiting_time"], rl, "RL agent")
    _style(ax2, ylabel="Avg waiting time")
    ax2.legend(frameon=False, labelcolor=INK_2, loc="lower left")
    ax2.set_xlabel("Episode (epsilon decays from 1.0 to 0.01; each episode is a new random workload)",
                   color=INK_2)
    for ax in (ax1, ax2):
        ax.set_xlim(0, episodes[-1] * 1.18)
    return _save(fig, out_dir, "learning_curve.png")


def plot_comparison(summary, out_dir: str, scenario: str = "standard",
                    metrics=("avg_waiting_time", "avg_turnaround_time",
                             "max_waiting_time", "fairness_index")):
    s = summary[summary.scenario == scenario].set_index("scheduler")
    names = _ordered(list(s.index))
    fig, axes = _figure(1, len(metrics), figsize=(4.2 * len(metrics), 4.6))
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        means = [s.loc[n, f"{metric}_mean"] for n in names]
        cis = [s.loc[n, f"{metric}_ci95"] for n in names]
        x = np.arange(len(names))
        ax.bar(x, means, width=0.62, yerr=cis, capsize=3,
               color=[SCHEDULER_COLORS.get(n, MUTED) for n in names],
               edgecolor=SURFACE, linewidth=2, error_kw={"ecolor": INK_2, "lw": 1})
        for xi, m, c in zip(x, means, cis):
            ax.text(xi, m + c, f"{m:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_LABELS.get(n, n) for n in names], fontsize=8)
        better = "higher is better" if metric == "fairness_index" else "lower is better"
        _style(ax, title=f"{METRIC_LABELS[metric]}\n({better})")
    fig.suptitle(f"Scheduler comparison — '{scenario}' scenario (mean ± 95% CI over "
                 f"{int(s['n'].iloc[0])} unseen workloads)", color=INK, fontsize=12, x=0.01,
                 ha="left")
    return _save(fig, out_dir, f"comparison_{scenario}.png")


def plot_scenarios(summary, out_dir: str, metric: str = "avg_waiting_time"):
    scenarios = list(dict.fromkeys(summary.scenario))
    names = _ordered(list(dict.fromkeys(summary.scheduler)))
    fig, ax = _figure(figsize=(12, 5))
    width = 0.8 / len(names)
    x = np.arange(len(scenarios))
    for i, name in enumerate(names):
        vals = [summary[(summary.scenario == sc) & (summary.scheduler == name)][f"{metric}_mean"].iloc[0]
                for sc in scenarios]
        ax.bar(x + (i - (len(names) - 1) / 2) * width, vals, width=width * 0.92,
               color=SCHEDULER_COLORS.get(name, MUTED), label=name,
               edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    _style(ax, title=f"{METRIC_LABELS[metric]} across workload scenarios (lower is better)",
           ylabel=METRIC_LABELS[metric])
    ax.legend(frameon=False, labelcolor=INK_2, ncol=len(names), loc="upper center",
              bbox_to_anchor=(0.5, -0.08), fontsize=9)
    return _save(fig, out_dir, f"scenarios_{metric.replace('avg_', '')}.png")


def plot_distribution(df, out_dir: str, scenario: str = "standard",
                      metric: str = "avg_waiting_time"):
    d = df[df.scenario == scenario]
    names = _ordered(list(dict.fromkeys(d.scheduler)))
    data = [d[d.scheduler == n][metric].to_numpy() for n in names]
    fig, ax = _figure(figsize=(10, 4.8))
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops={"color": INK, "linewidth": 1.5},
                    whiskerprops={"color": MUTED}, capprops={"color": MUTED},
                    flierprops={"marker": "o", "markersize": 3, "markerfacecolor": MUTED,
                                "markeredgecolor": "none"})
    for patch, n in zip(bp["boxes"], names):
        patch.set_facecolor(SCHEDULER_COLORS.get(n, MUTED))
        patch.set_edgecolor(SURFACE)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels([SHORT_LABELS.get(n, n) for n in names], fontsize=9)
    _style(ax, title=f"{METRIC_LABELS[metric]} per workload — '{scenario}' scenario "
                     f"({len(data[0])} unseen workloads)", ylabel=METRIC_LABELS[metric])
    return _save(fig, out_dir, f"{metric.replace('avg_', '')}_distribution.png")


def plot_generalisation(gen_df, out_dir: str):
    sets = list(dict.fromkeys(gen_df.workload_set))
    names = _ordered(list(dict.fromkeys(gen_df.scheduler)))
    fig, ax = _figure(figsize=(9, 4.6))
    width = 0.8 / len(names)
    x = np.arange(len(sets))
    for i, name in enumerate(names):
        rows = [gen_df[(gen_df.workload_set == s) & (gen_df.scheduler == name)].iloc[0] for s in sets]
        xs = x + (i - (len(names) - 1) / 2) * width
        ax.bar(xs, [r.avg_waiting_time for r in rows], width=width * 0.92,
               color=SCHEDULER_COLORS.get(name, MUTED), label=name, edgecolor=SURFACE, linewidth=1)
        for xi, r in zip(xs, rows):
            note = f"{r.avg_waiting_time:.2f}"
            if not np.isnan(r.unseen_state_rate):
                note += f"\n{r.unseen_state_rate:.0%} unseen"
            ax.text(xi, r.avg_waiting_time, note, ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(sets)
    _style(ax, title="Does the agent generalise? Avg waiting time on training vs unseen workloads",
           ylabel="Avg waiting time")
    ax.legend(frameon=False, labelcolor=INK_2, loc="upper center",
              bbox_to_anchor=(0.5, -0.1), ncol=len(names), fontsize=9)
    ax.margins(y=0.2)
    return _save(fig, out_dir, "generalisation.png")


def plot_gantt(timelines: dict, out_dir: str, title: str = "CPU timeline on the report's "
               "sample workload (Appendix A)", filename: str = "gantt_sample.png"):
    """timelines: {scheduler_name: [(process_id, start, end), ...]}

    Works for any workload size: labels are only drawn on bars wide enough
    to hold them, and the time axis picks a readable tick spacing."""
    names = _ordered(list(timelines))
    pids = sorted({pid for tl in timelines.values() for pid, _, _ in tl},
                  key=lambda p: (0, int(p[1:]), "") if p[1:].isdigit() else (1, 0, p))
    pcolor = {pid: PROCESS_COLORS[i % len(PROCESS_COLORS)] for i, pid in enumerate(pids)}
    start0 = min(s for tl in timelines.values() for _, s, _ in tl)
    end = max(e for tl in timelines.values() for _, _, e in tl)
    span = max(end - start0, 1)
    min_label_width = span / 40

    fig, ax = _figure(figsize=(11, 0.75 * len(names) + 1.5))
    for row, name in enumerate(reversed(names)):
        for pid, start, stop in timelines[name]:
            ax.barh(row, stop - start, left=start, height=0.6, color=pcolor[pid],
                    edgecolor=SURFACE, linewidth=1.5 if span <= 60 else 0.5)
            if stop - start >= max(min_label_width, len(pid) * span / 110):
                ax.text((start + stop) / 2, row, pid, ha="center", va="center", fontsize=8,
                        color="#ffffff" if pcolor[pid] in ("#2a78d6", "#4a3aa7", "#008300") else INK)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(reversed(names)))
    ax.set_xlim(start0, end)
    step = next(s for s in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10**4, 10**5)
                if span / s <= 15)
    ax.set_xticks(range(start0 - start0 % step, end + 1, step))
    _style(ax, title=title, xlabel="Time")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    return _save(fig, out_dir, filename)
