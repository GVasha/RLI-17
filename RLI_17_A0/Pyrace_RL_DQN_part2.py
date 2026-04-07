import os
import math
import random
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import gym_race
import torch
import torch.nn as nn
import torch.optim as optim

VERSION_NAME = "DQN_v03_part2"

REPORT_EPISODES = 500
DISPLAY_EPISODES = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OBS_MAX = 200.0


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(
            (
                np.array(state, dtype=np.float32),
                int(action),
                float(reward),
                np.array(next_state, dtype=np.float32),
                float(done),
            )
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def normalise(state):
    return state / OBS_MAX


def optimize_model():
    if len(replay_buffer) < BATCH_SIZE:
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
    states = torch.tensor(states, dtype=torch.float32, device=DEVICE)
    actions = torch.tensor(actions, dtype=torch.long, device=DEVICE).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=DEVICE)
    dones = torch.tensor(dones, dtype=torch.float32, device=DEVICE)

    current_q = model(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        next_q = model(next_states).max(1)[0]
        target_q = rewards + (1.0 - dones) * DISCOUNT_FACTOR * next_q

    loss = criterion(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def save_model(episode):
    file = f"models_{VERSION_NAME}/dqn_model_{episode}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "episode": episode,
        },
        file,
    )
    print(file, "saved")


def load_model(episode):
    file = f"models_{VERSION_NAME}/dqn_model_{episode}.pt"
    checkpoint = torch.load(file, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(file, "loaded")


def simulate(learning=True, episode_start=0):
    learning_rate = get_learning_rate(episode_start)
    explore_rate = get_explore_rate(episode_start)
    total_reward = 0
    total_rewards = []
    threshold = 1000

    max_reward = -10_000
    env.set_view(True)
    model.train(mode=learning)

    for episode in range(episode_start, NUM_EPISODES + episode_start):
        if episode > 0:
            total_rewards.append(total_reward)

            if learning and episode % REPORT_EPISODES == 0:
                plt.plot(total_rewards)
                plt.ylabel("rewards")
                plt.show(block=False)
                plt.pause(4.0)
                file = f"models_{VERSION_NAME}/memory_{episode}"
                env.save_memory(file)
                save_model(episode)
                plt.close()

        obv, _ = env.reset()
        state_0 = normalise(np.array(obv, dtype=np.float32))
        total_reward = 0
        if not learning:
            env.pyrace.mode = 2

        if episode >= threshold:
            explore_rate = 0.01

        for t in range(MAX_T):
            action = select_action(state_0, explore_rate if learning else 0.0)
            obv, reward, done, _, info = env.step(action)
            state = normalise(np.array(obv, dtype=np.float32))

            env.remember(tuple(state_0), action, reward, tuple(state), done)
            total_reward += reward

            if learning:
                replay_buffer.push(state_0, action, reward, state, done)
                optimize_model()

            state_0 = state

            if (episode % DISPLAY_EPISODES == 0) or (env.pyrace.mode == 2):
                env.set_msgs(
                    [
                        "SIMULATE PART 2",
                        f"Episode: {episode}",
                        f"Time steps: {t}",
                        f"check: {info['check']}",
                        f"dist: {info['dist']:.0f}",
                        f"crash: {info['crash']}",
                        f"Reward: {total_reward:.0f}",
                        f"Max Reward: {max_reward:.0f}",
                    ]
                )
                env.render()

            if done or t >= MAX_T - 1:
                if total_reward > max_reward:
                    max_reward = total_reward
                break

        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate


def load_and_play(episode, learning=False):
    print("Start loading model")
    load_model(episode)

    memory_file = f"models_{VERSION_NAME}/memory_{episode}.npy"
    if os.path.exists(memory_file):
        memory = load_data(memory_file)
        i = np.count_nonzero(memory[:, 4] == True)
        print("loaded memory episodes", i)

    simulate(learning, episode)


def select_action(state, explore_rate):
    if random.random() < explore_rate:
        return env.action_space.sample()

    state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_tensor)
    return int(torch.argmax(q_values, dim=1).item())


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t + 1) / DECAY_FACTOR)))


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t + 1) / DECAY_FACTOR)))


def load_data(file):
    data = np.load(file, allow_pickle=True)
    print(type(data))
    print(data.shape)
    if len(data.shape) >= 2 and data.shape[1] >= 5:
        print("episodes", np.count_nonzero(data[:, 4] == True))
    else:
        print("data preview not recognized as replay memory")
    return data


if __name__ == "__main__":
    env = gym.make(
        "Pyrace-v3",
        continuous_obs=True,
        extended_actions=True,
        reward_mode="v2",
    ).unwrapped
    print("env", type(env))

    if not os.path.exists(f"models_{VERSION_NAME}"):
        os.makedirs(f"models_{VERSION_NAME}")

    STATE_DIM = int(np.prod(env.observation_space.shape))
    NUM_ACTIONS = env.action_space.n
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
    print("state_dim", STATE_DIM, "num_actions", NUM_ACTIONS, "state_bounds", STATE_BOUNDS)

    MIN_EXPLORE_RATE = 0.001
    MIN_LEARNING_RATE = 1e-4
    DISCOUNT_FACTOR = 0.99
    DECAY_FACTOR = 5000.0
    print("decay factor", DECAY_FACTOR)

    NUM_EPISODES = 65_000
    MAX_T = 2000
    BATCH_SIZE = 64
    REPLAY_CAPACITY = 20_000

    model = DQN(STATE_DIM, NUM_ACTIONS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer(REPLAY_CAPACITY)
    print(model)

    simulate()
    # load_and_play(3500, learning=True)
    # load_and_play(3000, learning=False)
