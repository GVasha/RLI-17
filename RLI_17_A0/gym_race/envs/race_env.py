import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gym_race.envs.pyrace_2d import PyRace2D


class RaceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode="human", continuous_obs=False, extended_actions=False, reward_mode="v1"):
        self.continuous_obs = continuous_obs
        self.extended_actions = extended_actions
        self.reward_mode = reward_mode
        self.action_space = spaces.Discrete(4 if self.extended_actions else 3)
        if self.continuous_obs:
            self.observation_space = spaces.Box(
                np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                np.array([200.0, 200.0, 200.0, 200.0, 200.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Box(
                np.array([0, 0, 0, 0, 0], dtype=np.int32),
                np.array([10, 10, 10, 10, 10], dtype=np.int32),
                dtype=np.int32,
            )
        self.is_view = True
        self.pyrace = PyRace2D(self.is_view)
        self.memory = []
        self.render_mode = render_mode
        self.msgs = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mode = self.pyrace.mode
        del self.pyrace
        self.is_view = True
        self.msgs = []
        if isinstance(self.render_mode, int):
            mode = self.render_mode
        self.pyrace = PyRace2D(self.is_view, mode=mode)
        obs = self.pyrace.observe(continuous=self.continuous_obs)
        obs_dtype = np.float32 if self.continuous_obs else np.int32
        return np.array(obs, dtype=obs_dtype), {}

    def step(self, action):
        self.pyrace.action(action, extended_actions=self.extended_actions)
        reward = self.pyrace.evaluate(reward_mode=self.reward_mode)
        done = self.pyrace.is_done()
        obs = self.pyrace.observe(continuous=self.continuous_obs)
        obs_dtype = np.float32 if self.continuous_obs else np.int32
        return np.array(obs, dtype=obs_dtype), reward, done, False, {
            "dist": self.pyrace.car.distance,
            "check": self.pyrace.car.current_check,
            "crash": not self.pyrace.car.is_alive,
        }

    # def render(self, close=False , msgs=[], **kwargs): # gymnasium.render() does not accept other keyword arguments
    def render(self):  # gymnasium.render() does not accept other keyword arguments
        if self.is_view:
            self.pyrace.view_(self.msgs)

    def set_view(self, flag):
        self.is_view = flag

    def set_msgs(self, msgs):
        self.msgs = msgs

    def save_memory(self, file):
        # print(self.memory) # heterogeneus types
        # np.save(file, self.memory)
        np.save(file, np.array(self.memory, dtype=object))
        print(file + " saved")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
