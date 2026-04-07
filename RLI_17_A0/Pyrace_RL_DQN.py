import sys, os
import math, random
from collections import deque
import numpy as np
import matplotlib
matplotlib.use('Agg') # to avoid some "memory" errors with TkAgg backend
import matplotlib.pyplot as plt

import gymnasium as gym
import gym_race
"""
this imports race_env.py (a gym env) and pyrace_2d.py (the race game) and registers the env as "Pyrace-v1"

register(
    id='Pyrace-v1',
    entry_point='gym_race.envs:RaceEnv',
    max_episode_steps=2_000,
)
"""
import torch
import torch.nn as nn
import torch.optim as optim

VERSION_NAME = 'DQN_v01' # the name for our model

REPORT_EPISODES  = 500 # report (plot) every...
DISPLAY_EPISODES = 100 # display live game every...

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done)
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


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
    file = f'models_{VERSION_NAME}/dqn_model_{episode}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode': episode,
    }, file)
    print(file, 'saved')


def load_model(episode):
    file = f'models_{VERSION_NAME}/dqn_model_{episode}.pt'
    checkpoint = torch.load(file, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.to(DEVICE)
    model.eval()
    print(file, 'loaded')


def simulate(learning=True, episode_start=0):
    global model

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
                plt.ylabel('rewards')
                plt.show(block=False)
                plt.pause(4.0)
                file = f'models_{VERSION_NAME}/memory_{episode}'
                env.save_memory(file)
                save_model(episode)
                plt.close()

        obv, _ = env.reset()
        state_0 = np.array(obv, dtype=np.float32)
        total_reward = 0
        if not learning:
            env.pyrace.mode = 2

        if episode >= threshold:
            explore_rate = 0.01

        for t in range(MAX_T):
            action = select_action(state_0, explore_rate if learning else 0.0)
            obv, reward, done, _, info = env.step(action)
            state = np.array(obv, dtype=np.float32)
            env.remember(tuple(state_0.astype(int)), action, reward, tuple(state.astype(int)), done)
            total_reward += reward

            if learning:
                replay_buffer.push(state_0, action, reward, state, done)
                loss = optimize_model()

            state_0 = state

            if (episode % DISPLAY_EPISODES == 0) or (env.pyrace.mode == 2):
                env.set_msgs(['SIMULATE',
                            f'Episode: {episode}',
                            f'Time steps: {t}',
                            f'check: {info["check"]}',
                            f'dist: {info["dist"]}',
                            f'crash: {info["crash"]}',
                            f'Reward: {total_reward:.0f}',
                            f'Max Reward: {max_reward:.0f}'])
                env.render()
            if done or t >= MAX_T - 1:
                if total_reward > max_reward:
                    max_reward = total_reward
                break

        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate


def load_and_play(episode, learning=False):
    print("Start loading model")
    load_model(episode)

    memory_file = f'models_{VERSION_NAME}/memory_{episode}' + '.npy'
    if os.path.exists(memory_file):
        memory = load_data(memory_file)
        i = np.count_nonzero(memory[:, 4] == True)
        print('loaded memory episodes', i)

    simulate(learning, episode)


def select_action(state, explore_rate):
    if random.random() < explore_rate:
        return env.action_space.sample()

    state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_tensor)
    return int(torch.argmax(q_values, dim=1).item())


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def state_to_bucket(state):
    return tuple(int(x) for x in state)


def load_data(file):
    data = np.load(file,allow_pickle=True)
    print(type(data))
    print(data.shape)
    # print(data[-1])
    if len(data.shape) >= 2 and data.shape[1] >= 5:
        print('episodes', np.count_nonzero(data[:,4] == True))
    else:
        print('data preview not recognized as replay memory')
    return data


if __name__ == "__main__":

    env = gym.make("Pyrace-v1").unwrapped # skip the TimeLimig and OrderEnforcing default wrappers
    print('env',type(env))
    if not os.path.exists(f'models_{VERSION_NAME}'): os.makedirs(f'models_{VERSION_NAME}')

    STATE_DIM = int(np.prod(env.observation_space.shape))
    NUM_ACTIONS = env.action_space.n
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
    print('state_dim', STATE_DIM, 'num_actions', NUM_ACTIONS, 'state_bounds', STATE_BOUNDS)

    MIN_EXPLORE_RATE = 0.001
    MIN_LEARNING_RATE = 1e-4
    DISCOUNT_FACTOR = 0.99

    DECAY_FACTOR = 5000.0
    print(DECAY_FACTOR)

    NUM_EPISODES = 65_000
    MAX_T = 2000

    BATCH_SIZE = 64
    REPLAY_CAPACITY = 20_000

    model = DQN(STATE_DIM, NUM_ACTIONS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer(REPLAY_CAPACITY)
    print(model)

    simulate()  # LEARN starting from scratch...
    # load_and_play(3500, learning=True)
    #load_and_play(3000, learning=False) # e.g. 2000, 3000 ...
