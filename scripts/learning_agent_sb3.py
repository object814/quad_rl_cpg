import os
import numpy as np
from datetime import datetime
import imageio
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.gym_env import UnitreeA1Env
from gym import Wrapper

class MaxTimestepEnv(Wrapper):
    """
    A wrapper for limiting the maximum number of timesteps per episode.
    """
    def __init__(self, env, max_timesteps):
        super().__init__(env)
        self.max_timesteps = max_timesteps
        self.current_timesteps = 0

    def reset(self, **kwargs):
        self.current_timesteps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.current_timesteps += 1

        # Terminate episode if max timesteps is reached
        if self.current_timesteps >= self.max_timesteps:
            done = True
            info["TimeLimit.truncated"] = True  # Mark as truncated due to time limit

        return obs, reward, done, info
    

class SB3PPOTrainer:
    def __init__(self, max_timesteps_per_episode=10000):
        # Environment setup
        base_env = UnitreeA1Env()
        wrapped_env = MaxTimestepEnv(base_env, max_timesteps=max_timesteps_per_episode)
        self.vec_env = DummyVecEnv([lambda: wrapped_env])  # Wrap with DummyVecEnv for SB3 compatibility

        # Training hyperparameters
        self.learning_rate = 3e-4
        self.n_steps = 2048
        self.gamma = 0.99
        self.ent_coef = 0.005
        self.vf_coef = 0.5
        self.total_timesteps = 6_000_000

        # Directory setup
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.model_dir = f"networks/saved_models_sb3/SB3_ppo_{self.timestamp}"
        self.gif_dir = f"networks/gifs/SB3_ppo_{self.timestamp}"
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.gif_dir, exist_ok=True)

        # Initialize the PPO model
        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            gamma=self.gamma,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            verbose=1,
            tensorboard_log="./runs/sb3_model_training/"
        )

        # Callback for saving models at intervals
        self.checkpoint_callback = CheckpointCallback(
            save_freq=50_000,
            save_path=self.model_dir,
            name_prefix="ppo_model"
        )

    def train(self):
        """
        Train the PPO model using SB3.
        """
        self.model.learn(total_timesteps=self.total_timesteps, callback=self.checkpoint_callback)

    def evaluate(self, num_episodes=10):
        """
        Evaluate the trained PPO model.
        """
        total_rewards = []
        for ep in range(num_episodes):
            obs = self.vec_env.reset()
            done = False
            episode_reward = 0
            timestep = 0
            frames = []

            while not done:
             

                action, _ = self.model.predict(obs)
                obs, reward, done, _ = self.vec_env.step(action)
                episode_reward += reward[0]  # SB3 uses vectorized environments; reward is a list
                timestep += 1

            total_rewards.append(episode_reward)
            #self.save_gif(frames, f"{self.gif_dir}/episode_{ep + 1}.gif")
            print(f"Episode {ep + 1}/{num_episodes} - Reward: {episode_reward}")

        print(f"Average Reward over {num_episodes} episodes: {np.mean(total_rewards)}")

    def save_gif(self, frames, filename):
        """
        Save a sequence of frames as a GIF.
        """
        if len(frames) == 0:
            print(f"[WARNING] No frames captured for {filename}")
            return
        try:
            imageio.mimsave(filename, frames, fps=10)
            print(f"[INFO] GIF saved to {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to save GIF: {e}")


if __name__ == "__main__":
    trainer = SB3PPOTrainer(max_timesteps_per_episode=10000)
    trainer.train()  # Train the model
    trainer.evaluate(num_episodes=10)  # Evaluate and save GIFs
