import os
import glob
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VERSION_NAME = "DQN_v01"  # set the model version to analyze
WINDOW_SHORT = 100
WINDOW_LONG = 500


def load_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    print(type(data))
    print(data.shape)
    return data


def moving_average(values, window):
    if len(values) == 0:
        return np.array([])
    window = max(1, min(window, len(values)))
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def memory_to_episode_rows(memory):
    rows = []
    last_terminal_index = -1
    episode_counter = 0

    for i in range(memory.shape[0]):
        done = bool(memory[i, 4])
        if done:
            episode_counter += 1
            steps = i - last_terminal_index
            reward = float(memory[i, 2])  # terminal reward used by original notebook too
            rows.append((episode_counter, steps, reward))
            last_terminal_index = i

    return np.array(rows, dtype=float)


def save_plots(episodes, steps, rewards, output_prefix):
    rewards_short = moving_average(rewards, WINDOW_SHORT)
    rewards_long = moving_average(rewards, WINDOW_LONG)

    plt.figure(figsize=(16, 6))
    plt.plot(episodes, rewards_short, label=f"MA {WINDOW_SHORT}")
    plt.plot(episodes, rewards_long, label=f"MA {WINDOW_LONG}")
    plt.title("DQN AVERAGED REWARDS")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_avg_rewards.png", dpi=140)
    plt.close()

    reward_per_step = np.where(rewards > 0, rewards / steps, (10000.0 + rewards) / steps)
    reward_per_step_ma = moving_average(reward_per_step, WINDOW_SHORT)

    plt.figure(figsize=(16, 6))
    plt.plot(episodes, reward_per_step_ma, label=f"MA {WINDOW_SHORT}")
    plt.title("DQN AVERAGED REWARD PER ACTION STEP")
    plt.xlabel("Episode")
    plt.ylabel("Reward/Step")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_avg_reward_per_step.png", dpi=140)
    plt.close()

    plt.figure(figsize=(16, 6))
    plt.plot(episodes, moving_average(steps, WINDOW_SHORT), label=f"MA {WINDOW_SHORT}")
    plt.title("DQN AVERAGED STEPS PER EPISODE")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_avg_steps.png", dpi=140)
    plt.close()


def get_latest_memory_file(models_dir):
    files = glob.glob(os.path.join(models_dir, "memory_*.npy"))
    if not files:
        return None

    def key(file_path):
        match = re.search(r"memory_(\d+)\.npy$", file_path)
        return int(match.group(1)) if match else -1

    files.sort(key=key)
    return files[-1]


def print_summary(steps, rewards):
    tail = min(500, len(rewards))
    recent_rewards = rewards[-tail:]
    recent_steps = steps[-tail:]

    goals = np.count_nonzero(rewards > 0)
    crashes = np.count_nonzero(rewards <= 0)

    print("\n----- DQN PERFORMANCE SUMMARY -----")
    print("episodes_total:", int(len(rewards)))
    print("goals:", int(goals), "crashes:", int(crashes))
    print("goal_rate:", float(goals) / float(len(rewards)))
    print("avg_reward_all:", float(np.mean(rewards)))
    print("avg_reward_last_window:", float(np.mean(recent_rewards)))
    print("avg_steps_all:", float(np.mean(steps)))
    print("avg_steps_last_window:", float(np.mean(recent_steps)))
    print("best_reward:", float(np.max(rewards)))
    print("-----------------------------------")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    models_dir = os.path.join(base_dir, f"models_{VERSION_NAME}")
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Folder not found: {models_dir}")

    memory_file = get_latest_memory_file(models_dir)
    if memory_file is None:
        raise FileNotFoundError(f"No memory_*.npy files found in {models_dir}")

    print("Analyzing:", memory_file)
    memory = load_data(memory_file)
    if len(memory.shape) < 2 or memory.shape[1] < 5:
        raise ValueError("Unexpected memory format. Expected columns: state, action, reward, next_state, done")

    episodes_data = memory_to_episode_rows(memory)
    if episodes_data.shape[0] == 0:
        raise ValueError("No terminal transitions found in memory file.")

    episodes = episodes_data[:, 0]
    steps = episodes_data[:, 1]
    rewards = episodes_data[:, 2]

    analysis_dir = os.path.join(base_dir, f"analysis_{VERSION_NAME}")
    os.makedirs(analysis_dir, exist_ok=True)
    output_prefix = os.path.join(analysis_dir, f"episode_{int(episodes[-1])}")

    save_plots(episodes, steps, rewards, output_prefix)
    print_summary(steps, rewards)
    print("Saved plots:")
    print(f"- {output_prefix}_avg_rewards.png")
    print(f"- {output_prefix}_avg_reward_per_step.png")
    print(f"- {output_prefix}_avg_steps.png")
