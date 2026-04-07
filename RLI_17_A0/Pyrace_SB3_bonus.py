"""
Migrates the Pyrace car-driving agent from the hand-coded DQN to a framework-level algorithm using Stable-Baselines3.

Two algorithms:
  - DQN: discrete actions, built-in target network + replay buffer 
  - PPO: on-policy policy gradient; works with Discrete action spaces

Why SB3 improves on the Part 2 DQN:
  - SB3 DQN includes a target network that stabilises training.
  - Replay buffer in SB3 defaults to 1_000_000 transitions.
  - PPO avoids the exploration/exploitation trade-off entirely via clipped policy gradients.
"""

import os
import gymnasium as gym
import gym_race  

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor


ALGORITHM     = "DQN"      
TOTAL_STEPS   = 500_000    
EVAL_FREQ     = 10_000     
SAVE_PATH     = f"models_SB3_{ALGORITHM}"
LOG_PATH      = f"logs_SB3_{ALGORITHM}"

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(LOG_PATH,  exist_ok=True)


env      = Monitor(gym.make("Pyrace-v3").unwrapped, filename=os.path.join(LOG_PATH, "train"))
eval_env = Monitor(gym.make("Pyrace-v3").unwrapped, filename=os.path.join(LOG_PATH, "eval"))

checkpoint_cb = CheckpointCallback(
    save_freq=EVAL_FREQ,
    save_path=SAVE_PATH,
    name_prefix="pyrace_sb3"
)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(SAVE_PATH, "best"),
    log_path=LOG_PATH,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=5,
    deterministic=True,
    render=False
)

# Model 
if ALGORITHM == "DQN":
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=100_000,        
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        target_update_interval=500, 
        exploration_fraction=0.15,
        exploration_final_eps=0.01,
        train_freq=4,
        policy_kwargs=dict(net_arch=[128, 128]),  
        tensorboard_log=LOG_PATH,
        verbose=1
    )
elif ALGORITHM == "PPO":
    # PPO is on-policy: no replay buffer, uses clipped surrogate objective.
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[128, 128]),
        tensorboard_log=LOG_PATH,
        verbose=1
    )
else:
    raise ValueError(f"Unknown ALGORITHM: {ALGORITHM}. Choose 'DQN' or 'PPO'.")

print(f"\nTraining {ALGORITHM} on Pyrace-v3 for {TOTAL_STEPS:,} steps...")
print(model.policy)

# Train 
model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=[checkpoint_cb, eval_cb],
    progress_bar=True
)

model.save(os.path.join(SAVE_PATH, f"pyrace_{ALGORITHM.lower()}_final"))
print(f"\nModel saved to {SAVE_PATH}/")

print("\nRunning greedy evaluation...")
obs, _ = eval_env.reset()
total_reward = 0
for _ in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = eval_env.step(action)
    total_reward += reward
    if done:
        break
print(f"Evaluation episode reward: {total_reward:.1f}")
