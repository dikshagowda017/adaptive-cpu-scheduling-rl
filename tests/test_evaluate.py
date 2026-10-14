import numpy as np
import pytest

import evaluate as ev
import main as main_module
from q_learning import train


def test_permutation_test_identical_samples():
    assert ev.paired_permutation_test([1, 2, 3], [1, 2, 3]) == 1.0


def test_permutation_test_detects_clear_difference():
    rng = np.random.default_rng(0)
    b = rng.normal(10, 1, 50)
    assert ev.paired_permutation_test(b - 2, b) < 0.001


def test_permutation_test_no_false_alarm_on_noise():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 40), rng.normal(0, 1, 40)
    assert ev.paired_permutation_test(a, b) > 0.05


def test_permutation_test_rejects_empty():
    with pytest.raises(ValueError):
        ev.paired_permutation_test([], [])


def test_holm_correction():
    adj = ev.holm_correction([0.01, 0.04, 0.03])
    assert adj.tolist() == pytest.approx([0.03, 0.06, 0.06])
    assert np.all(ev.holm_correction([0.5, 0.9]) <= 1.0)


@pytest.fixture(scope="module")
def small_eval():
    agent, _ = train(num_episodes=300, seed=0, verbose_every=0)
    scenarios = {k: ev.SCENARIOS[k] for k in ("standard", "heavy_load")}
    df = ev.evaluate({ev.RL_NAME: agent}, scenarios=scenarios, num_workloads=15)
    return agent, df


def test_evaluate_runs_every_scheduler_on_identical_workloads(small_eval):
    _, df = small_eval
    assert set(df.scheduler) == set(ev.BASELINES) | {ev.RL_NAME}
    counts = df.groupby(["scenario", "scheduler"]).size()
    assert (counts == 15).all()
    seeds = df.groupby("scheduler").workload_seed.apply(lambda s: tuple(sorted(s)))
    assert seeds.nunique() == 1
    assert df.workload_seed.min() >= ev.TEST_SEED_START


def test_summary_and_significance_tables(small_eval):
    _, df = small_eval
    summary = ev.summarize(df)
    assert len(summary) == 2 * (len(ev.BASELINES) + 1)
    assert (summary["avg_waiting_time_ci95"] >= 0).all()
    sig = ev.significance(df)
    assert len(sig) == 2 * len(ev.BASELINES) * len(ev.METRICS)
    assert ((sig.wins + sig.ties + sig.losses) == 15).all()
    assert (sig.p_holm >= sig.p_value - 1e-12).all()
    # Multiple-comparison correction must actually be applied (mutation testing
    # showed the previous check also passed with raw p-values).
    assert np.allclose(sig.p_holm, ev.holm_correction(sig.p_value))
    assert (sig.p_holm > sig.p_value).any()
    verdict = ev.hypothesis_verdict(sig)
    assert set(verdict) == {"scenario", "metrics_beating_all_classical", "reject_H0"}


def test_generalisation_check_columns(small_eval):
    agent, _ = small_eval
    gen = ev.generalisation_check({ev.RL_NAME: agent}, train_seeds=range(5), num_unseen=5)
    assert set(gen.workload_set) == {"training workloads", "unseen workloads"}


def test_main_quick_pipeline_writes_all_outputs(tmp_path):
    main_module.main(["--quick", "--episodes", "60", "--workloads", "4", "--out", str(tmp_path)])
    for name in ("q_table.json", "training_history.csv", "per_workload_results.csv",
                 "summary.csv", "significance.csv", "generalisation.csv", "RESULTS.md",
                 "learning_curve.png", "comparison_standard.png", "gantt_sample.png",
                 "scenarios_waiting_time.png", "waiting_time_distribution.png",
                 "generalisation.png"):
        assert (tmp_path / name).exists(), name


def test_main_evaluate_without_trained_agent_fails_cleanly(tmp_path):
    with pytest.raises(SystemExit):
        main_module.main(["evaluate", "--out", str(tmp_path)])
