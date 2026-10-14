"""
Q-Learning Agents
-------------------
Two tabular Q-learning agents that share the same training loop,
hyperparameters (report Appendix B) and update rule:

    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

1. QLearningAgent  ("full-state", the original Phase-2 design)
   The whole 30-number observation is binned into a tuple and used as
   the Q-table key; each entry holds one Q-value per action. Because the
   key describes every process in the queue, a workload it has never
   seen produces states it has never seen, so this agent only works on
   the workloads it was trained on (see TESTING.md, bug B5).

2. ProcessQLearningAgent  ("per-process", the default)
   Still a tabular Q-table, but the key for Q(s, a) is the binned
   attributes of the process that action a would run, plus the queue
   length:  (burst_bin, waiting_bin, priority_bin, queue_len_bin).
   The same table entry is reused whenever ANY process with similar
   attributes is a candidate, in any workload and any queue position,
   so the agent generalises to unseen workloads with a table of at
   most 8*6*3*4 = 576 entries.

Training uses an epsilon-greedy policy (per report Appendix B):
  - Start with epsilon = 1.0 (act completely randomly -> explore)
  - Decay epsilon toward 0.01 (mostly act on learned knowledge -> exploit)
  - Learning rate (alpha) = 0.1
  - Discount factor (gamma) = 0.95
  - Episodes = 3000 (tuned; Appendix B used 1000, see NUM_EPISODES)
  - Workload size drawn from 4-20 processes per episode (see TRAIN_PROCESS_RANGE)
Both agents only ever choose actions that hold a real process (action
masking), and both take a seed so training is reproducible.
"""

import json
from collections import defaultdict

import numpy as np

from simulator import CPUSimulator
from fcfs import run_fcfs
from gym_env import (CPUSchedulingEnv, FEATURES_PER_PROCESS, MAX_BURST_TIME,
                     MAX_WAITING_TIME, MAX_PRIORITY)

# --- Hyperparameters (matches report Appendix B) ---
ALPHA = 0.1             # learning rate
GAMMA = 0.95            # discount factor
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995   # multiplied into epsilon after every episode
REPORT_NUM_EPISODES = 1000   # Appendix B value
# Tuned in the hyperparameter-tuning phase: epsilon reaches its floor after
# ~920 episodes, and the extra greedy episodes refine the Q-table further
# (avg waiting on unseen 20-process workloads: 18.5 -> 17.1). Training still
# takes only a few seconds. Use --episodes 1000 to reproduce Appendix B exactly.
NUM_EPISODES = 3000
# Each training episode draws its workload size from this range, so the agent
# sees short and long ready queues (training on 8 processes only left it
# weak on 20-process workloads: 86% behind SJF instead of ~25%).
TRAIN_PROCESS_RANGE = (4, 20)
NUM_BINS = 4            # full-state agent: buckets per continuous feature

# Workload seeds: training draws from [0, TRAIN_SEED_LIMIT); evaluation uses
# seeds from TEST_SEED_START upward, so the two never overlap.
TRAIN_SEED_LIMIT = 100_000
TEST_SEED_START = 1_000_000

# Per-process agent bin edges, in real units (time units / priority levels).
BURST_EDGES = [1.5, 2.5, 3.5, 5.5, 7.5, 10.5, 14.5]     # -> 8 burst bins
WAIT_EDGES = [2, 5, 10, 20, 35]                         # -> 6 waiting bins
PRIORITY_EDGES = [3.5, 6.5]                             # -> high / medium / low
QUEUE_EDGES = [1.5, 3.5, 6.5]                           # -> 1, 2-3, 4-6, 7+


def discretize(obs: np.ndarray, num_bins: int = NUM_BINS) -> tuple:
    """
    Convert the continuous observation vector (values in [0,1])
    into a tuple of small integers, so it can be used as a dictionary
    key in the Q-table.

    Example: a value of 0.42 with num_bins=4 becomes bucket 1 (of 0,1,2,3).
    """
    bins = np.clip((np.asarray(obs) * num_bins).astype(int), 0, num_bins - 1)
    return tuple(bins.tolist())


def _valid_actions(mask) -> np.ndarray:
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        raise ValueError("action mask has no valid actions")
    return valid


class _EpsilonGreedyMixin:
    def _init_common(self, alpha, gamma, epsilon, epsilon_min, epsilon_decay, seed):
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not 0 <= gamma <= 1:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        # Diagnostics: how often greedy decisions hit a state never trained on.
        self.lookups = 0
        self.unseen_lookups = 0

    def choose_action(self, obs, mask=None, training: bool = True) -> int:
        """Epsilon-greedy action selection over VALID actions only.
        training=True  -> may explore randomly (used during training)
        training=False -> always exploit the best known action (evaluation)
        """
        if mask is None:
            mask = self._mask_from_obs(obs)
        valid = _valid_actions(mask)
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid))          # explore
        values = self.action_values(obs, mask, count_unseen=not training)
        return int(valid[int(np.argmax(values[valid]))])  # exploit (ties -> lowest index = FCFS)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def unseen_state_rate(self) -> float:
        return self.unseen_lookups / self.lookups if self.lookups else 0.0

    @staticmethod
    def _mask_from_obs(obs) -> np.ndarray:
        # A real process always has remaining burst > 0, padding slots are all zero.
        return np.asarray(obs)[0::FEATURES_PER_PROCESS] > 0


class QLearningAgent(_EpsilonGreedyMixin):
    """Full-state tabular Q-learning (the original design)."""

    kind = "full_state"

    def __init__(self, num_actions: int, alpha=ALPHA, gamma=GAMMA,
                 epsilon=EPSILON_START, epsilon_min=EPSILON_MIN,
                 epsilon_decay=EPSILON_DECAY, num_bins: int = NUM_BINS, seed=None):
        self.num_actions = int(num_actions)
        self.num_bins = num_bins
        self._init_common(alpha, gamma, epsilon, epsilon_min, epsilon_decay, seed)
        # Q-table: discretized state (tuple) -> array of Q-values, one per action.
        self.q_table = defaultdict(lambda: np.zeros(self.num_actions))

    def state_key(self, obs) -> tuple:
        return discretize(obs, self.num_bins)

    def action_values(self, obs, mask, count_unseen=False) -> np.ndarray:
        key = self.state_key(obs)
        if count_unseen:
            self.lookups += 1
            self.unseen_lookups += key not in self.q_table
        # .get() so that merely looking at a state doesn't add it to the table.
        return self.q_table.get(key, np.zeros(self.num_actions))

    def update(self, obs, action: int, reward: float, next_obs, next_mask, done: bool):
        """Apply the Q-learning update rule after one step."""
        state = self.state_key(obs)
        if done:
            best_next_value = 0.0
        else:
            next_values = self.action_values(next_obs, next_mask)
            best_next_value = float(np.max(next_values[_valid_actions(next_mask)]))
        td_target = reward + self.gamma * best_next_value
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
        return td_error

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "num_actions": self.num_actions, "num_bins": self.num_bins,
            "alpha": self.alpha, "gamma": self.gamma, "epsilon": self.epsilon,
            "q_table": [[list(k), v.tolist()] for k, v in self.q_table.items()],
        }

    @classmethod
    def from_dict(cls, data: dict):
        agent = cls(num_actions=data["num_actions"], alpha=data["alpha"],
                    gamma=data["gamma"], epsilon=data["epsilon"], num_bins=data["num_bins"])
        for key, values in data["q_table"]:
            agent.q_table[tuple(key)] = np.array(values, dtype=float)
        return agent


class ProcessQLearningAgent(_EpsilonGreedyMixin):
    """Tabular Q-learning keyed on the candidate process's attributes."""

    kind = "per_process"

    def __init__(self, num_actions: int, alpha=ALPHA, gamma=GAMMA,
                 epsilon=EPSILON_START, epsilon_min=EPSILON_MIN,
                 epsilon_decay=EPSILON_DECAY, seed=None):
        self.num_actions = int(num_actions)
        self._init_common(alpha, gamma, epsilon, epsilon_min, epsilon_decay, seed)
        # Q-table: (burst_bin, wait_bin, priority_bin, queue_bin) -> Q-value.
        self.q_table = defaultdict(float)

    @staticmethod
    def features(obs, mask) -> list:
        """One discrete key per action slot (None for empty slots)."""
        obs = np.asarray(obs)
        queue_bin = int(np.digitize(int(np.sum(mask)), QUEUE_EDGES))
        keys = []
        for i in range(len(mask)):
            if not mask[i]:
                keys.append(None)
                continue
            base = i * FEATURES_PER_PROCESS
            burst = obs[base] * MAX_BURST_TIME
            wait = obs[base + 1] * MAX_WAITING_TIME
            prio = obs[base + 2] * MAX_PRIORITY
            keys.append((int(np.digitize(burst, BURST_EDGES)),
                         int(np.digitize(wait, WAIT_EDGES)),
                         int(np.digitize(prio, PRIORITY_EDGES)),
                         queue_bin))
        return keys

    def action_values(self, obs, mask, count_unseen=False) -> np.ndarray:
        values = np.full(len(mask), -np.inf)
        for i, key in enumerate(self.features(obs, mask)):
            if key is None:
                continue
            if count_unseen:
                self.lookups += 1
                self.unseen_lookups += key not in self.q_table
            values[i] = self.q_table.get(key, 0.0)
        return values

    def update(self, obs, action: int, reward: float, next_obs, next_mask, done: bool):
        """Apply the Q-learning update rule after one step."""
        mask = self._mask_from_obs(obs)
        key = self.features(obs, mask)[action]
        if key is None:
            raise ValueError(f"action {action} does not refer to a process")
        if done:
            best_next_value = 0.0
        else:
            best_next_value = float(np.max(self.action_values(next_obs, next_mask)))
        td_target = reward + self.gamma * best_next_value
        td_error = td_target - self.q_table[key]
        self.q_table[key] += self.alpha * td_error
        return td_error

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "num_actions": self.num_actions,
            "alpha": self.alpha, "gamma": self.gamma, "epsilon": self.epsilon,
            "q_table": [[list(k), v] for k, v in self.q_table.items()],
        }

    @classmethod
    def from_dict(cls, data: dict):
        agent = cls(num_actions=data["num_actions"], alpha=data["alpha"],
                    gamma=data["gamma"], epsilon=data["epsilon"])
        for key, value in data["q_table"]:
            agent.q_table[tuple(key)] = float(value)
        return agent


AGENT_TYPES = {cls.kind: cls for cls in (QLearningAgent, ProcessQLearningAgent)}


def save_agent(agent, path):
    """Save a trained agent's Q-table as JSON (the "D1: Q-table store")."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(agent.to_dict(), f)


def load_agent(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return AGENT_TYPES[data["kind"]].from_dict(data)


def train(num_episodes: int = NUM_EPISODES, num_processes=TRAIN_PROCESS_RANGE,
          verbose_every: int = 50, num_training_workloads: int = None,
          agent_kind: str = "per_process", seed: int = 0, env_kwargs: dict = None):
    """
    Train a Q-learning agent on the CPU scheduling environment.

    Parameters
    ----------
    num_processes : int or (low, high)
        Processes per training workload. A tuple draws the size uniformly
        from [low, high] for every episode.
    num_training_workloads : int or None
        None -> a fresh random workload every episode (default; fine for
        the per-process agent, which generalises).
        int  -> cycle through a fixed pool of this many workloads. This is
        what the full-state agent needs, because with fresh workloads its
        state space is too sparse to ever revisit the same situation.
        Workload seeds are always below TRAIN_SEED_LIMIT, so they never
        overlap with the evaluation seeds.
    agent_kind : "per_process" or "full_state"
    seed : int
        Seeds both the agent's exploration and the workload sequence, so
        the same call always produces the same agent.

    Returns
    -------
    agent : the trained agent (its Q-table holds the learned policy).
    history : dict of per-episode lists: reward, epsilon, avg_waiting_time,
              fcfs_avg_waiting_time (FCFS on the same workload), q_table_size — used to plot the learning curve.
    """
    if num_episodes < 1:
        raise ValueError("num_episodes must be >= 1")
    if isinstance(num_processes, (tuple, list)):
        low, high = num_processes
    else:
        low = high = num_processes
    if not 1 <= low <= high:
        raise ValueError(f"invalid num_processes {num_processes!r}")
    if seed is None or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    env = CPUSchedulingEnv(num_processes=low, **(env_kwargs or {}))
    agent = AGENT_TYPES[agent_kind](num_actions=env.action_space.n, seed=seed)
    seed_rng = np.random.default_rng(seed)

    if num_training_workloads is not None:
        if num_training_workloads < 1:
            raise ValueError("num_training_workloads must be >= 1")
        training_seeds = list(range(num_training_workloads))
    else:
        training_seeds = None

    history = {"reward": [], "epsilon": [], "avg_waiting_time": [],
               "fcfs_avg_waiting_time": [], "q_table_size": []}

    for episode in range(num_episodes):
        env.num_processes = int(seed_rng.integers(low, high + 1)) if high > low else low
        if training_seeds is not None:
            workload_seed = training_seeds[episode % len(training_seeds)]
        else:
            workload_seed = int(seed_rng.integers(0, TRAIN_SEED_LIMIT))
        obs, info = env.reset(seed=workload_seed)
        mask = info["action_mask"]
        total_reward = 0.0
        terminated = False

        while not terminated:
            action = agent.choose_action(obs, mask, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_mask = info["action_mask"]
            agent.update(obs, action, reward, next_obs, next_mask, terminated)
            obs, mask = next_obs, next_mask
            total_reward += reward

        history["epsilon"].append(agent.epsilon)
        agent.decay_epsilon()
        history["reward"].append(total_reward)
        history["avg_waiting_time"].append(env.sim.get_metrics()["avg_waiting_time"])
        # FCFS on the very same workload: a per-episode yardstick, so the learning
        # curve isn't drowned out by some workloads simply being harder than others.
        history["fcfs_avg_waiting_time"].append(
            run_fcfs(CPUSimulator(env.workload))["avg_waiting_time"])
        history["q_table_size"].append(len(agent.q_table))

        if verbose_every and (episode + 1) % verbose_every == 0:
            recent = history["reward"][-verbose_every:]
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"avg reward (last {verbose_every}): {np.mean(recent):.2f} | "
                  f"epsilon: {agent.epsilon:.3f} | "
                  f"Q-table size: {len(agent.q_table)}")

    return agent, history


def run_agent(agent, env: CPUSchedulingEnv, seed=None) -> dict:
    """Run one greedy (no exploration) episode and return the simulator metrics."""
    if env.action_space.n != agent.num_actions:
        raise ValueError(f"agent was trained for {agent.num_actions} queue slots but the "
                         f"environment has {env.action_space.n}; create the environment "
                         f"with max_queue_size={agent.num_actions}")
    obs, info = env.reset(seed=seed)
    terminated = False
    while not terminated:
        action = agent.choose_action(obs, info["action_mask"], training=False)
        obs, _, terminated, _, info = env.step(action)
    return env.sim.get_metrics()


if __name__ == "__main__":
    print(f"Training per-process Q-learning agent ({NUM_EPISODES} episodes)...\n")
    agent, history = train(verbose_every=NUM_EPISODES // 10)
    waits, fcfs = history["avg_waiting_time"], history["fcfs_avg_waiting_time"]
    print(f"\nAvg waiting, first 200 episodes: RL {np.mean(waits[:200]):.2f}"
          f" vs FCFS {np.mean(fcfs[:200]):.2f} (same workloads)")
    print(f"Avg waiting, last 200 episodes:  RL {np.mean(waits[-200:]):.2f}"
          f" vs FCFS {np.mean(fcfs[-200:]):.2f} (same workloads)")
    print(f"Final Q-table size: {len(agent.q_table)}")
