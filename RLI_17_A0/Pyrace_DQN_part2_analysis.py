import os
import glob
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASELINE_VERSION = "DQN_v01"
IMPROVED_VERSION = "DQN_v03_part2"
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
            reward = float(memory[i, 2])
            rows.append((episode_counter, steps, reward))
            last_terminal_index = i

    return np.array(rows, dtype=float)


def get_latest_memory_file(models_dir):
    files = glob.glob(os.path.join(models_dir, "memory_*.npy"))
    if not files:
        return None

    def key(file_path):
        match = re.search(r"memory_(\d+)\.npy$", file_path)
        return int(match.group(1)) if match else -1

    files.sort(key=key)
    return files[-1]


def reward_per_step(steps, rewards):
    return np.where(rewards > 0, rewards / steps, (10000.0 + rewards) / steps)


def save_model_plots(episodes, steps, rewards, output_prefix, model_name):
    rewards_short = moving_average(rewards, WINDOW_SHORT)
    rewards_long = moving_average(rewards, WINDOW_LONG)

    plt.figure(figsize=(16, 6))
    plt.plot(episodes, rewards_short, label=f"MA {WINDOW_SHORT}")
    plt.plot(episodes, rewards_long, label=f"MA {WINDOW_LONG}")
    plt.title(f"{model_name} AVERAGED REWARDS")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_avg_rewards.png", dpi=140)
    plt.close()

    rps = reward_per_step(steps, rewards)
    rps_short = moving_average(rps, WINDOW_SHORT)
    plt.figure(figsize=(16, 6))
    plt.plot(episodes, rps_short, label=f"MA {WINDOW_SHORT}")
    plt.title(f"{model_name} AVERAGED REWARD PER ACTION STEP")
    plt.xlabel("Episode")
    plt.ylabel("Reward/Step")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_avg_reward_per_step.png", dpi=140)
    plt.close()

    steps_short = moving_average(steps, WINDOW_SHORT)
    plt.figure(figsize=(16, 6))
    plt.plot(episodes, steps_short, label=f"MA {WINDOW_SHORT}")
    plt.title(f"{model_name} AVERAGED STEPS PER EPISODE")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_avg_steps.png", dpi=140)
    plt.close()


def load_model_metrics(base_dir, version_name):
    models_dir = os.path.join(base_dir, f"models_{version_name}")
    if not os.path.exists(models_dir):
        print(f"[SKIP] Folder not found: {models_dir}")
        return None

    memory_file = get_latest_memory_file(models_dir)
    if memory_file is None:
        print(f"[SKIP] No memory_*.npy files found in {models_dir}")
        return None

    print("Analyzing:", memory_file)
    memory = load_data(memory_file)
    if len(memory.shape) < 2 or memory.shape[1] < 5:
        print(f"[SKIP] Unexpected memory format for {version_name}")
        return None

    episodes_data = memory_to_episode_rows(memory)
    if episodes_data.shape[0] == 0:
        print(f"[SKIP] No terminal transitions found for {version_name}")
        return None

    episodes = episodes_data[:, 0]
    steps = episodes_data[:, 1]
    rewards = episodes_data[:, 2]
    return episodes, steps, rewards


def save_comparison_plot(base_dir, baseline_metrics, improved_metrics):
    episodes_b, _, rewards_b = baseline_metrics
    episodes_i, _, rewards_i = improved_metrics
    rewards_b_ma = moving_average(rewards_b, WINDOW_SHORT)
    rewards_i_ma = moving_average(rewards_i, WINDOW_SHORT)

    analysis_dir = os.path.join(base_dir, "analysis_part2")
    os.makedirs(analysis_dir, exist_ok=True)
    output_file = os.path.join(analysis_dir, "comparison_avg_rewards.png")

    plt.figure(figsize=(16, 6))
    plt.plot(episodes_b, rewards_b_ma, label=f"{BASELINE_VERSION} MA {WINDOW_SHORT}")
    plt.plot(episodes_i, rewards_i_ma, label=f"{IMPROVED_VERSION} MA {WINDOW_SHORT}")
    plt.title("Part 2 Comparison - Average Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=140)
    plt.close()
    print(f"Saved comparison plot: {output_file}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    analysis_dir = os.path.join(base_dir, "analysis_part2")
    os.makedirs(analysis_dir, exist_ok=True)

    baseline_metrics = load_model_metrics(base_dir, BASELINE_VERSION)
    improved_metrics = load_model_metrics(base_dir, IMPROVED_VERSION)

    if baseline_metrics is not None:
        output_prefix = os.path.join(analysis_dir, BASELINE_VERSION)
        save_model_plots(*baseline_metrics, output_prefix, BASELINE_VERSION)
        print(f"Saved {BASELINE_VERSION} plots with prefix: {output_prefix}_*.png")

    if improved_metrics is not None:
        output_prefix = os.path.join(analysis_dir, IMPROVED_VERSION)
        save_model_plots(*improved_metrics, output_prefix, IMPROVED_VERSION)
        print(f"Saved {IMPROVED_VERSION} plots with prefix: {output_prefix}_*.png")

    if baseline_metrics is not None and improved_metrics is not None:
        save_comparison_plot(base_dir, baseline_metrics, improved_metrics)
    else:
        print("Comparison plot skipped: one or both model result folders are unavailable.")
