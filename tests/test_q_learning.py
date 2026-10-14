import numpy as np
import pytest

from q_learning import (QLearningAgent, ProcessQLearningAgent, discretize, train, run_agent,
                        save_agent, load_agent, TEST_SEED_START, EPSILON_MIN)
from gym_env import CPUSchedulingEnv, MAX_QUEUE_SIZE
from simulator import CPUSimulator
from fcfs import run_fcfs
from round_robin import run_round_robin
from priority import run_priority
from workload_generator import generate_workload


def test_discretize_bins():
    assert discretize(np.array([0.0, 0.24, 0.25, 0.42, 0.99, 1.0]), 4) == (0, 0, 1, 1, 3, 3)


@pytest.mark.parametrize("cls", [QLearningAgent, ProcessQLearningAgent])
def test_update_rule_matches_formula(cls):
    env = CPUSchedulingEnv(num_processes=6)
    agent = cls(num_actions=env.action_space.n, alpha=0.5, gamma=0.9, seed=0)
    obs, info = env.reset(seed=1)
    a = 0
    nxt, r, done, _, ninfo = env.step(a)
    # Give s' non-zero Q-values; otherwise gamma * max Q(s') = 0 and a bug that
    # ignores gamma would go unnoticed (found by mutation testing).
    if cls is QLearningAgent:
        agent.q_table[agent.state_key(nxt)][:] = -4.0
    else:
        for key in agent.features(nxt, ninfo["action_mask"]):
            if key is not None:
                agent.q_table[key] = -4.0
    before = agent.action_values(obs, info["action_mask"])[a]
    best_next = 0.0 if done else max(
        v for v in agent.action_values(nxt, ninfo["action_mask"]) if np.isfinite(v))
    assert best_next != 0.0
    agent.update(obs, a, r, nxt, ninfo["action_mask"], done)
    after = agent.action_values(obs, info["action_mask"])[a]
    assert after == pytest.approx(before + 0.5 * (r + 0.9 * best_next - before))


def test_terminal_update_ignores_next_state():
    agent = QLearningAgent(num_actions=3, alpha=1.0, gamma=0.9)
    s = np.full(3, 0.5)
    agent.q_table[discretize(s)][:] = 100.0
    agent.update(s, 1, reward=-2.0, next_obs=s, next_mask=np.array([1, 1, 1], bool), done=True)
    assert agent.q_table[discretize(s)][1] == pytest.approx(-2.0)


@pytest.mark.parametrize("cls", [QLearningAgent, ProcessQLearningAgent])
def test_actions_always_respect_mask(cls):
    env = CPUSchedulingEnv(num_processes=8)
    agent = cls(num_actions=env.action_space.n, seed=0)
    for epsilon in (1.0, 0.0):
        agent.epsilon = epsilon
        for seed in range(10):
            obs, info = env.reset(seed=seed)
            done = False
            while not done:
                a = agent.choose_action(obs, info["action_mask"], training=True)
                assert info["action_mask"][a]
                obs, _, done, _, info = env.step(a)
                assert not info["action_was_invalid"]


def test_choose_action_rejects_empty_mask():
    agent = ProcessQLearningAgent(num_actions=3)
    with pytest.raises(ValueError):
        agent.choose_action(np.zeros(9), np.zeros(3, bool))


def test_epsilon_decays_to_floor():
    agent = QLearningAgent(num_actions=2)
    for _ in range(5000):
        agent.decay_epsilon()
    assert agent.epsilon == EPSILON_MIN


@pytest.mark.parametrize("bad", [{"alpha": 0}, {"alpha": 1.5}, {"gamma": -0.1}, {"gamma": 2}])
def test_invalid_hyperparameters(bad):
    with pytest.raises(ValueError):
        QLearningAgent(num_actions=2, **bad)


@pytest.mark.parametrize("kind,pool", [("per_process", None), ("full_state", 10)])
def test_training_is_reproducible(kind, pool):
    """Regression for bug B4: training used unseeded global randomness, so two
    runs of the same script gave different agents and different numbers."""
    a1, h1 = train(num_episodes=60, agent_kind=kind, num_training_workloads=pool,
                   seed=7, verbose_every=0)
    a2, h2 = train(num_episodes=60, agent_kind=kind, num_training_workloads=pool,
                   seed=7, verbose_every=0)
    assert h1 == h2
    assert a1.to_dict() == a2.to_dict()


@pytest.mark.parametrize("pool", [None, 10])
def test_training_never_sees_test_workloads(monkeypatch, pool):
    """No data leakage: every workload seed used in training must be outside
    the range the evaluation uses."""
    import q_learning
    seen = []
    original_reset = CPUSchedulingEnv.reset

    def recording_reset(self, *, seed=None, options=None):
        seen.append(seed)
        return original_reset(self, seed=seed, options=options)

    monkeypatch.setattr(CPUSchedulingEnv, "reset", recording_reset)
    train(num_episodes=200, num_training_workloads=pool, seed=3, verbose_every=0)
    assert len(seen) == 200 and None not in seen
    assert max(seen) < q_learning.TRAIN_SEED_LIMIT <= q_learning.TEST_SEED_START


def test_history_has_one_entry_per_episode():
    _, h = train(num_episodes=25, seed=0, verbose_every=0)
    assert all(len(v) == 25 for v in h.values())
    assert h["epsilon"][0] == 1.0


@pytest.mark.parametrize("kind,pool", [("per_process", None), ("full_state", 5)])
def test_save_and_load_round_trip(tmp_path, kind, pool):
    agent, _ = train(num_episodes=40, agent_kind=kind, num_training_workloads=pool,
                     seed=1, verbose_every=0)
    path = tmp_path / "q.json"
    save_agent(agent, path)
    loaded = load_agent(path)
    assert type(loaded) is type(agent)
    env = CPUSchedulingEnv(num_processes=8)
    for seed in range(TEST_SEED_START, TEST_SEED_START + 5):
        assert run_agent(agent, env, seed) == run_agent(loaded, env, seed)


def test_greedy_lookup_does_not_grow_the_q_table():
    agent = QLearningAgent(num_actions=MAX_QUEUE_SIZE)
    env = CPUSchedulingEnv()
    run_agent(agent, env, seed=0)
    assert len(agent.q_table) == 0


@pytest.fixture(scope="module")
def trained_agent():
    agent, _ = train(num_episodes=1000, seed=0, verbose_every=0)
    return agent


def _avg_wait(run, n=60):
    seeds = range(TEST_SEED_START, TEST_SEED_START + n)
    return np.mean([run(s) for s in seeds])


def test_trained_agent_beats_classical_schedulers_on_unseen_workloads(trained_agent):
    """End-to-end: the learned policy must beat FCFS, RR and Priority on
    workloads it never saw during training."""
    env = CPUSchedulingEnv(num_processes=8)
    rl = _avg_wait(lambda s: run_agent(trained_agent, env, s)["avg_waiting_time"])
    for baseline in (run_fcfs, run_priority, lambda sim: run_round_robin(sim, 3)):
        base = _avg_wait(lambda s: baseline(CPUSimulator(generate_workload(8, seed=s)))
                         ["avg_waiting_time"])
        assert rl < 0.85 * base


def test_per_process_agent_sees_few_unseen_states(trained_agent):
    env = CPUSchedulingEnv(num_processes=8)
    trained_agent.lookups = trained_agent.unseen_lookups = 0
    for s in range(TEST_SEED_START, TEST_SEED_START + 30):
        run_agent(trained_agent, env, s)
    assert trained_agent.unseen_state_rate() < 0.1
