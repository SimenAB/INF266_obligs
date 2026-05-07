import argparse

import numpy as np

from bandit import Bandits_final


class StandardUCB:
    def __init__(self, k, c=2.0):
        self.k = k
        self.c = c
        self.counts = np.zeros(k, dtype=int)
        self.sums = np.zeros(k, dtype=float)
        self.t = 0

    def select_action(self):
        for action in range(self.k):
            if self.counts[action] == 0:
                return action

        means = self.sums / self.counts
        bonus = self.c * np.sqrt(np.log(self.t + 1) / self.counts)
        return int(np.argmax(means + bonus))

    def update(self, action, reward):
        self.t += 1
        self.counts[action] += 1
        self.sums[action] += reward


class SlidingWindowUCB:
    def __init__(self, k, window_size=100, c=2.0):
        self.k = k
        self.window_size = window_size
        self.c = c
        self.history = []

    def select_action(self):
        recent_history = self.history[-self.window_size :]
        counts = np.zeros(self.k, dtype=int)
        sums = np.zeros(self.k, dtype=float)

        for action, reward in recent_history:
            counts[action] += 1
            sums[action] += reward

        for action in range(self.k):
            if counts[action] == 0:
                return action

        means = sums / counts
        window_length = max(1, len(recent_history))
        bonus = self.c * np.sqrt(np.log(window_length + 1) / counts)
        return int(np.argmax(means + bonus))

    def update(self, action, reward):
        self.history.append((action, reward))


class EpsilonGreedy:
    def __init__(self, k, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon
        self.counts = np.zeros(k, dtype=int)
        self.sums = np.zeros(k, dtype=float)

    def select_action(self):
        if np.random.rand() < self.epsilon or np.any(self.counts == 0):
            return int(np.random.randint(self.k))

        means = self.sums / np.maximum(1, self.counts)
        return int(np.argmax(means))

    def update(self, action, reward):
        self.counts[action] += 1
        self.sums[action] += reward


def run_experiment(agent_factory, episodes, seed):
    np.random.seed(seed)
    env = Bandits_final()
    agent = agent_factory()

    total_reward = 0.0
    total_regret = 0.0
    optimal_action_count = 0
    change_count = 0
    previous_means = env.means.copy()

    for _ in range(episodes):
        action = agent.select_action()
        _, reward, _, _, _ = env.step(action)

        if not np.array_equal(previous_means, env.means):
            change_count += 1
            previous_means = env.means.copy()

        optimal_action = int(np.argmax(env.means))
        total_reward += reward
        total_regret += float(np.max(env.means) - env.means[action])
        optimal_action_count += int(action == optimal_action)

        agent.update(action, reward)

    return {
        "reward": total_reward,
        "regret": total_regret,
        "optimal_action_rate": optimal_action_count / episodes,
        "changes": change_count,
    }


def summarize(results):
    keys = ["reward", "regret", "optimal_action_rate", "changes"]
    return {
        key: (
            float(np.mean([result[key] for result in results])),
            float(np.std([result[key] for result in results])),
        )
        for key in keys
    }


def print_single_run(agents, episodes, seed):
    print(f"One run, {episodes} episodes, seed={seed}")
    print("-" * 75)

    for name, factory in agents.items():
        result = run_experiment(factory, episodes, seed)
        print(
            f"{name:32} "
            f"reward={result['reward']:8.2f}  "
            f"regret={result['regret']:8.2f}  "
            f"optimal_rate={result['optimal_action_rate']:.3f}  "
            f"changes={result['changes']}"
        )


def print_average_runs(agents, episodes, runs, start_seed):
    print()
    print(
        f"Average over {runs} independent runs, {episodes} episodes each "
        f"(seeds {start_seed} to {start_seed + runs - 1})"
    )
    print("-" * 75)

    for name, factory in agents.items():
        results = [
            run_experiment(factory, episodes, seed)
            for seed in range(start_seed, start_seed + runs)
        ]
        summary = summarize(results)
        reward_mean, reward_std = summary["reward"]
        regret_mean, regret_std = summary["regret"]
        optimal_mean, optimal_std = summary["optimal_action_rate"]
        changes_mean, changes_std = summary["changes"]

        print(
            f"{name:32} "
            f"reward={reward_mean:8.2f} +- {reward_std:7.2f}  "
            f"regret={regret_mean:7.2f} +- {regret_std:7.2f}  "
            f"optimal_rate={optimal_mean:.3f} +- {optimal_std:.3f}  "
            f"changes={changes_mean:.2f} +- {changes_std:.2f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run non-stationary multi-armed bandit experiments."
    )
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--ucb-c", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    args = parser.parse_args()

    agents = {
        "Standard UCB": lambda: StandardUCB(k=3, c=args.ucb_c),
        f"Sliding-window UCB (w={args.window_size})": lambda: SlidingWindowUCB(
            k=3,
            window_size=args.window_size,
            c=args.ucb_c,
        ),
        f"Epsilon-greedy (eps={args.epsilon})": lambda: EpsilonGreedy(
            k=3,
            epsilon=args.epsilon,
        ),
    }

    print_single_run(agents, args.episodes, args.seed)
    print_average_runs(agents, args.episodes, args.runs, args.start_seed)


if __name__ == "__main__":
    main()
