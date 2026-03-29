import os
from typing import List, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from config import Config
from env import FASEnv


class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self.running = 0.0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)

        if rewards is not None:
            self.running += float(np.mean(rewards))
        if dones is not None and np.any(dones):
            self.episode_rewards.append(self.running)
            self.running = 0.0
        return True


def make_env(cfg: Config):
    def _init():
        env = FASEnv(cfg)
        return Monitor(env)

    return _init


def train_agent(cfg: Config) -> Tuple[PPO, RewardLoggerCallback]:
    vec_env = DummyVecEnv([make_env(cfg)])

    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        policy_kwargs=policy_kwargs,
        seed=cfg.seed,
        verbose=1,
    )

    cb = RewardLoggerCallback()
    model.learn(total_timesteps=cfg.total_timesteps, callback=cb)

    model_dir = os.path.join(cfg.out_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "ppo_fas_memetic")
    model.save(model_path)

    np.save(os.path.join(cfg.out_dir, "train_rewards.npy"), np.array(cb.episode_rewards, dtype=float))
    return model, cb


if __name__ == "__main__":
    from channel import set_global_seed

    cfg = Config()
    set_global_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    train_agent(cfg)