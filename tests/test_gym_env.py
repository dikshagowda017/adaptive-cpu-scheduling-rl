import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from gym_env import CPUSchedulingEnv, MAX_QUEUE_SIZE, FEATURES_PER_PROCESS
from simulator import CPUSimulator
from fcfs import run_fcfs
from workload_generator import Process, REPORT_SAMPLE_WORKLOAD, generate_workload

ONLY_WAITING = {"waiting": 1.0, "turnaround": 0.0, "context_switch": 0.0,
                "fairness": 0.0, "starvation": 0.0}


def _run_episode(env, policy, seed=None):
    obs, info = env.reset(seed=seed)
    total, steps, done = 0.0, 0, False
    while not done:
        obs, r, done, truncated, info = env.step(policy(obs, info))
        assert not truncated
        total += r
        steps += 1
    return total, steps


def test_passes_gymnasium_env_checker():
    check_env(CPUSchedulingEnv(num_processes=6), skip_render_check=True)


def test_spaces():
    env = CPUSchedulingEnv()
    assert env.action_space.n == MAX_QUEUE_SIZE
    assert env.observation_space.shape == (MAX_QUEUE_SIZE * FEATURES_PER_PROCESS,)


def test_observations_stay_in_bounds_and_padding_is_zero():
    env = CPUSchedulingEnv(num_processes=30, workload_kwargs={"mean_inter_arrival": 0.5})
    obs, info = env.reset(seed=1)
    done = False
    while not done:
        assert env.observation_space.contains(obs)
        n = int(info["action_mask"].sum())
        assert np.all(obs[n * FEATURES_PER_PROCESS:] == 0)
        assert np.all(obs[0:n * FEATURES_PER_PROCESS:FEATURES_PER_PROCESS] > 0)
        obs, _, done, _, info = env.step(0)


def test_observation_encodes_exact_normalised_values():
    # Appendix A: after P1 runs 0-5, the queue is P2..P5 with
    # remaining bursts 3, 8, 6, 2; waits 4, 3, 2, 1; priorities 1, 4, 3, 5.
    env = CPUSchedulingEnv(workload=REPORT_SAMPLE_WORKLOAD)
    obs, _ = env.reset()
    assert obs[:3] == pytest.approx([5 / 20, 0.0, 2 / 10])
    obs, _, _, _, _ = env.step(0)
    expected = []
    for burst, wait, prio in [(3, 4, 1), (8, 3, 4), (6, 2, 3), (2, 1, 5)]:
        expected += [burst / 20, wait / 50, prio / 10]
    assert obs[:12] == pytest.approx(expected)
    assert np.all(obs[12:] == 0)


def test_same_seed_same_episode():
    env = CPUSchedulingEnv(num_processes=8)
    a = _run_episode(env, lambda o, i: 0, seed=5)
    b = _run_episode(env, lambda o, i: 0, seed=5)
    assert a == b


def test_seeded_env_without_explicit_seed_is_reproducible():
    """Regression for bug B4: reset() drew workloads from the global NumPy RNG,
    so seeding the environment did not make it reproducible."""
    def workloads(env_seed):
        env = CPUSchedulingEnv(num_processes=5)
        env.reset(seed=env_seed)
        first = env.workload
        env.reset()
        return first, env.workload
    assert workloads(3) == workloads(3)


def test_non_preemptive_episode_takes_one_step_per_process():
    env = CPUSchedulingEnv(num_processes=9)
    _, steps = _run_episode(env, lambda o, i: 0, seed=2)
    assert steps == 9


def test_action_zero_reproduces_fcfs():
    w = generate_workload(10, seed=4)
    env = CPUSchedulingEnv(workload=w)
    _run_episode(env, lambda o, i: 0)
    assert env.sim.get_metrics() == run_fcfs(CPUSimulator(w))


def test_fixed_workload_is_used_every_episode():
    env = CPUSchedulingEnv(workload=REPORT_SAMPLE_WORKLOAD)
    env.reset(seed=1)
    assert env.workload == REPORT_SAMPLE_WORKLOAD
    env.reset(seed=2)
    assert env.workload == REPORT_SAMPLE_WORKLOAD


def test_out_of_range_action_is_clipped_and_flagged():
    env = CPUSchedulingEnv(workload=[Process("P1", 0, 2, 1), Process("P2", 0, 3, 1)])
    env.reset()
    _, _, _, _, info = env.step(MAX_QUEUE_SIZE - 1)
    assert info["action_was_invalid"]
    assert info["ran_process_id"] == "P2"   # clipped to the last real process


def test_action_mask_matches_queue():
    env = CPUSchedulingEnv(workload=REPORT_SAMPLE_WORKLOAD)
    _, info = env.reset()
    assert info["action_mask"].tolist() == [True] + [False] * (MAX_QUEUE_SIZE - 1)
    _, _, _, _, info = env.step(0)   # P1 runs 0-5, P2..P5 arrive
    assert info["action_mask"].sum() == 4


def test_step_before_reset_and_after_done_raise():
    env = CPUSchedulingEnv(num_processes=2)
    with pytest.raises(RuntimeError):
        env.step(0)
    _run_episode(env, lambda o, i: 0, seed=0)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_waiting_reward_sums_to_minus_average_waiting_time():
    """With only the waiting term switched on, the undiscounted episode return
    must equal -(average waiting time). This ties the reward to the metric."""
    for seed in range(20):
        env = CPUSchedulingEnv(num_processes=8, reward_weights=ONLY_WAITING)
        rng = np.random.default_rng(seed)
        total, _ = _run_episode(
            env, lambda o, i: int(rng.choice(np.flatnonzero(i["action_mask"]))), seed=seed)
        assert total == pytest.approx(-env.sim.get_metrics()["avg_waiting_time"])


def test_reward_parts_add_up_and_have_expected_signs():
    env = CPUSchedulingEnv(num_processes=8)
    env.reset(seed=3)
    done = False
    while not done:
        _, r, done, _, info = env.step(0)
        parts = info["reward_parts"]
        assert r == pytest.approx(sum(parts.values()))
        assert parts["waiting"] <= 0 and parts["turnaround"] <= 0
        assert parts["context_switch"] <= 0 and parts["starvation"] <= 0
        assert 0 < parts["fairness"] <= env.reward_weights["fairness"] + 1e-9


def test_preemptive_mode_with_time_slice():
    env = CPUSchedulingEnv(workload=[Process("A", 0, 5, 1)], time_slice=2)
    _, steps = _run_episode(env, lambda o, i: 0)
    assert steps == 3        # 2 + 2 + 1
    assert env.sim.context_switches == 0


@pytest.mark.parametrize("kwargs", [{"max_queue_size": 0}, {"time_slice": 0}])
def test_invalid_constructor_arguments(kwargs):
    with pytest.raises(ValueError):
        CPUSchedulingEnv(**kwargs)
