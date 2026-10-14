"""
Gym Environment Wrapper
--------------------------
Wraps CPUSimulator in the standard Gymnasium interface (reset / step)
so a reinforcement learning agent can interact with it like any other
RL environment.

This file is responsible for two RL-specific things the simulator
itself doesn't know about:
  1. STATE representation  — turning the ready queue into a fixed-size
     numeric vector the agent can read.
  2. REWARD function        — turning "what happened this step" into a
     single number that tells the agent whether that was a good or
     bad scheduling decision.

State per process (normalized to [0, 1]):
    [remaining_burst_time, waiting_time, priority]
Padded with zeros up to max_queue_size processes, so the state vector
is always the same length regardless of how many processes are
actually waiting. A real process always has remaining burst >= 1, so
an all-zero slot unambiguously means "empty".

Action: index of the process to run, in ready-queue (arrival) order.
`info["action_mask"]` marks which indices hold a real process; agents
should only pick masked-in actions. An out-of-range action is clipped
to the last real process so a careless agent can't crash the env.

Reward (per report section 4.5.1 — weighted sum of):
    - average waiting time added this step         (negative)
    - average turnaround time added this step      (negative)
    - a context-switch penalty                     (negative, only if switched)
    - a fairness bonus: Jain's Index of the waiting times still in the queue
    - a starvation penalty for each process waiting > STARVATION_THRESHOLD
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simulator import CPUSimulator
from workload_generator import generate_workload

# --- Fixed sizes so the state/action space never changes shape ---
MAX_QUEUE_SIZE = 20          # max number of processes the agent can see / choose among
FEATURES_PER_PROCESS = 3     # [remaining_burst, waiting_time, priority]

# --- Normalization constants (upper bounds, values above are clipped to 1) ---
MAX_BURST_TIME = 20.0
MAX_WAITING_TIME = 50.0
MAX_PRIORITY = 10.0

# --- Reward weights ---
W_WAITING = 1.0
W_TURNAROUND = 0.5
W_CONTEXT_SWITCH = 0.5
W_FAIRNESS = 0.5
W_STARVATION = 0.5
STARVATION_THRESHOLD = 25    # time units in the ready queue before a process counts as starving

DEFAULT_REWARD_WEIGHTS = {
    "waiting": W_WAITING,
    "turnaround": W_TURNAROUND,
    "context_switch": W_CONTEXT_SWITCH,
    "fairness": W_FAIRNESS,
    "starvation": W_STARVATION,
}


class CPUSchedulingEnv(gym.Env):
    """A Gymnasium-compatible environment wrapping CPUSimulator.

    Parameters
    ----------
    num_processes : int
        Processes per episode (each episode is a fresh workload).
    seed : int or None
        Default workload seed used by reset() when no seed is passed.
        None -> a different random workload every episode.
    workload_kwargs : dict
        Extra arguments forwarded to generate_workload (e.g.
        mean_inter_arrival, burst_distribution) to change the load.
    workload : list[Process] or None
        Use this exact workload every episode instead of generating one.
    time_slice : int or None
        None (default) -> non-preemptive: the chosen process runs to completion.
        int -> the chosen process runs at most this long, then the agent
        decides again (lets the agent preempt, like Round Robin).
    reward_weights : dict or None
        Override any of DEFAULT_REWARD_WEIGHTS.
    """

    metadata = {"render_modes": []}

    def __init__(self, num_processes: int = 8, seed: int = None,
                 workload_kwargs: dict = None, workload: list = None,
                 max_queue_size: int = MAX_QUEUE_SIZE, time_slice: int = None,
                 reward_weights: dict = None):
        super().__init__()
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be >= 1")
        if time_slice is not None and time_slice <= 0:
            raise ValueError("time_slice must be positive or None")

        self.num_processes = len(workload) if workload is not None else num_processes
        self._seed = seed
        self.workload_kwargs = dict(workload_kwargs or {})
        self.fixed_workload = workload
        self.max_queue_size = max_queue_size
        self.time_slice = time_slice
        self.reward_weights = {**DEFAULT_REWARD_WEIGHTS, **(reward_weights or {})}

        self.action_space = spaces.Discrete(max_queue_size)
        obs_size = max_queue_size * FEATURES_PER_PROCESS
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        self.sim = None       # created fresh in reset()
        self.workload = None  # the current episode's list of Process objects

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.fixed_workload is not None:
            workload = self.fixed_workload
        else:
            use_seed = seed if seed is not None else self._seed
            if use_seed is None:
                # Draw from Gymnasium's seeded RNG, so a seeded env produces a
                # reproducible sequence of workloads.
                use_seed = int(self.np_random.integers(0, 2**31 - 1))
            workload = generate_workload(num_processes=self.num_processes,
                                         seed=use_seed, **self.workload_kwargs)

        self.workload = workload
        self.sim = CPUSimulator(workload)

        # Advance past any initial idle time before the first process arrives.
        self.sim.advance_to_next_arrival()

        return self._get_observation(), {"action_mask": self.action_mask()}

    def step(self, action):
        if self.sim is None:
            raise RuntimeError("Call reset() before step().")
        if self.sim.is_done():
            raise RuntimeError("Episode is over; call reset() before step().")

        valid = min(len(self.sim.ready_queue), self.max_queue_size)
        action_index = int(np.clip(int(action), 0, valid - 1))

        _, done, info = self.sim.step(action_index=action_index,
                                      time_slice=self.time_slice)

        reward, parts = self._compute_reward(info)
        info["reward_parts"] = parts
        info["action_index"] = action_index
        info["action_was_invalid"] = action_index != int(action)

        # Skip forward in time if the queue is empty but processes remain.
        if not done:
            self.sim.advance_to_next_arrival()

        info["action_mask"] = self.action_mask()
        return self._get_observation(), reward, done, False, info

    def action_mask(self) -> np.ndarray:
        """Boolean array: True for every action index that holds a real process."""
        mask = np.zeros(self.max_queue_size, dtype=bool)
        if self.sim is not None:
            mask[:min(len(self.sim.ready_queue), self.max_queue_size)] = True
        return mask

    def _get_observation(self) -> np.ndarray:
        """Build the fixed-size, normalized state vector from the
        current ready queue."""
        obs = np.zeros(self.max_queue_size * FEATURES_PER_PROCESS, dtype=np.float32)

        queue = self.sim.ready_queue[:self.max_queue_size] if self.sim else []
        for i, proc in enumerate(queue):
            base = i * FEATURES_PER_PROCESS
            obs[base + 0] = min(proc.remaining_burst_time / MAX_BURST_TIME, 1.0)
            obs[base + 1] = min(proc.waiting_time / MAX_WAITING_TIME, 1.0)
            obs[base + 2] = min(proc.priority / MAX_PRIORITY, 1.0)

        return obs

    def _compute_reward(self, info: dict):
        """Turn the outcome of one step into a single reward value.

        Returns (reward, parts) where parts holds each weighted term, which
        makes the reward easy to debug and to unit-test."""
        w = self.reward_weights
        n = self.num_processes

        # Waiting time added this step, across ALL processes (including the
        # ones that arrived while the CPU was busy). Summed over an episode
        # this is exactly the total waiting time, so the waiting term adds up
        # to -W_WAITING * average_waiting_time.
        waiting = info["waiting_added"] / n
        # Turnaround grows for every unfinished process: the waiting ones
        # plus the one on the CPU.
        turnaround = (info["waiting_added"] + info["run_time"]) / n

        starving = sum(1 for p in self.sim.ready_queue
                       if p.waiting_time > STARVATION_THRESHOLD)

        parts = {
            "waiting": -w["waiting"] * waiting,
            "turnaround": -w["turnaround"] * turnaround,
            "context_switch": -w["context_switch"] * (1.0 if info["context_switch"] else 0.0),
            "fairness": w["fairness"] * self._current_fairness_index(),
            "starvation": -w["starvation"] * starving,
        }
        return float(sum(parts.values())), parts

    def _current_fairness_index(self) -> float:
        """Jain's Fairness Index over the waiting times of processes
        currently sitting in the ready queue."""
        waits = [p.waiting_time for p in self.sim.ready_queue]
        if not waits or sum(waits) == 0:
            return 1.0  # nobody waiting, or all waited equally (0) -> perfectly fair
        n = len(waits)
        return (sum(waits) ** 2) / (n * sum(w ** 2 for w in waits))


if __name__ == "__main__":
    # Quick manual test: run one full episode with a random (but valid)
    # policy, just to confirm the environment works end to end.
    env = CPUSchedulingEnv(num_processes=5, seed=42)
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    rng = np.random.default_rng(0)
    total_reward = 0.0
    steps = 0
    terminated = False
    while not terminated:
        action = int(rng.choice(np.flatnonzero(info["action_mask"])))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        print(f"Step {steps}: ran {info['ran_process_id']}, reward={reward:.2f}")

    print(f"\nEpisode finished in {steps} steps. Total reward: {total_reward:.2f}")
    print("Final metrics:", env.sim.get_metrics())
